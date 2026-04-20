"""Single-stage IK grasp using the D405 gripper camera via fine_object_detector_node.

Passive node — sits idle until it receives Bool(True) on /visual_grasp/start.
On start it (a) moves the arm to ik.READY_POSE_P2, (b) opens the gripper,
(c) IK-steps toward the target (from /fine_detector/objects, D405 gripper
camera), closes the gripper when close enough, retracts the arm, then
publishes Bool(True) on /visual_grasp/done and returns to idle.

The READY_POSE_P2 step replaces what visual_servo_arm used to do — once the
arm is in pre-grasp pose the gripper D405 sees the object directly, so the
coarse head-camera servo stage is no longer required.

Caller (task_executor or operator) is responsible for:
  - setting fine_detector target via /fine_detector/target_object
  - activating fine_object_detector_node via /fine_detector/activate (Bool true)
  - sending /visual_grasp/start after the above
  - listening for /visual_grasp/done to know when pick is done
"""

import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
from geometry_msgs.msg import PoseStamped
from hello_helpers.hello_misc import HelloNode
import threading
import tf2_ros
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray
from .manipulation import ik_ros_utils as ik
import ikpy


class IKVisualGrasp(HelloNode):

    def __init__(self):
        HelloNode.__init__(self)

        self.delta = 0.06
        self.target_frame = 'base_link'
        self.gripper_frame = 'link_grasp_center'
        self.tf_buffer = None
        self.tf_listener = None
        self.joint_states_lock = threading.Lock()
        self.joint_state = {}

        # Grasp-target offset in base_link. x/y kept at 0 after removing the
        # previous calibration that produced a left-bias. +z=2.5 cm is a
        # heuristic to aim slightly above the detected object centroid so the
        # fingers close around the object body rather than scraping the
        # supporting surface.
        self.shift_x = 0.0
        self.shift_y = 0.0
        self.shift_z = 0.025

        self.target_object_name = None
        self.active = False
        self.picked = False

        # Mode-switch clients; wired up in main() after HelloNode.main() has
        # initialized the rclpy context. /manipulation/rotate_base leaves the
        # driver in navigation mode, and arm/lift/head joint targets only
        # execute reliably in position mode — so we flip to position on start
        # and back to navigation after pick so the subsequent nav step works.
        self.switch_pos_cli = None
        self.switch_nav_cli = None

    # ── joint states ──────────────────────────────────────────────────

    def joint_states_callback(self, msg):
        with self.joint_states_lock:
            joint_states = msg
        joint_names = [
            'joint_lift', 'joint_arm_l0', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll'
        ]
        self.joint_state = {}
        for joint_name in joint_names:
            i = joint_states.name.index(joint_name)
            self.joint_state[joint_name] = joint_states.position[i]

    # ── TF helpers ────────────────────────────────────────────────────

    def get_gripper_pose_in_base_frame(self):
        return self.tf_buffer.lookup_transform(
            self.target_frame, self.gripper_frame,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=1.0),
        )

    # ── start / stop control ─────────────────────────────────────────

    def _on_start(self, msg: Bool):
        if not msg.data:
            if self.active:
                print("visual_grasp: stop received — going idle")
                self._go_idle()
            return
        # Always re-initialize on start(True): the executor re-issues start
        # at the beginning of every pick (including pick #2 after a place),
        # and the ready-pose move must run each time regardless of any
        # stale `active` flag from a prior aborted cycle.
        self.active = True
        self.picked = False

        # Target precedence: explicit param > whatever the latest
        # /fine_detector/target_object message gave us. The executor
        # publishes that topic right before sending /visual_grasp/start, so
        # in the bringup pipeline the param is unset and we rely on the topic.
        if not self.has_parameter('target_object'):
            self.declare_parameter('target_object', '')
        param_val = self.get_parameter('target_object').value
        if param_val:
            self.target_object_name = param_val

        # The previous step in the executor's pipeline is
        # /manipulation/rotate_base, which ends in navigation mode. Arm/lift
        # joint targets don't execute in nav mode, so flip to position
        # mode before driving the arm to READY_POSE_P2.
        self._call_trigger(self.switch_pos_cli)

        print("visual_grasp: moving to ready pose (READY_POSE_P2)...")
        self.move_to_pose(ik.READY_POSE_P2, blocking=True)
        self.open_gripper()
        print(f"visual_grasp: started — tracking '{self.target_object_name}', gripper open")

    def _on_target_object(self, msg):
        """Latched-style sub: mirror whatever target the executor told the
        fine detector to track, so we don't need a parameter in bringup."""
        if msg.data:
            self.target_object_name = msg.data

    def _go_idle(self):
        print("visual_grasp: idle")
        self.active = False
        self.target_object_name = None

    # ── detection callback ───────────────────────────────────────────

    def detection_callback(self, msg: Detection3DArray):
        if not self.active or self.target_object_name is None or self.picked:
            return

        goal_pos = self._extract_target_pos(msg)
        if goal_pos is None:
            return

        try:
            gripper_transformed = self.get_gripper_pose_in_base_frame()
            gripper_pos = ik.get_xyz_from_msg(gripper_transformed)
        except Exception as e:
            print(f"Error getting gripper TF: {e}")
            return

        waypoint_pos, waypoint_orient = self.compute_waypoint_to_goal(goal_pos, gripper_pos)

        with self.joint_states_lock:
            joint_state = self.joint_state
        q_init = ik.get_current_configuration(joint_state)
        q_soln = ik.get_grasp_goal(waypoint_pos, waypoint_orient, q_init)

        ik.print_q(q_soln)
        if q_soln is not None:
            ik.move_to_configuration(self, q_soln)

            goal_pos_shifted = goal_pos + np.array([self.shift_x, self.shift_y, self.shift_z])
            dist = np.linalg.norm(goal_pos_shifted - gripper_pos)
            print(f"Distance to goal after move: {dist:.3f} m")

            if dist < self.delta:
                self.pick()

    def _extract_target_pos(self, msg: Detection3DArray):
        for det in msg.detections:
            if not det.results:
                continue
            if det.results[0].hypothesis.class_id == self.target_object_name:
                p = det.results[0].pose.pose.position
                return np.array([p.x, p.y, p.z])
        return None

    # ── waypoint computation ─────────────────────────────────────────

    def compute_waypoint_to_goal(self, goal_pos, gripper_pos):
        goal_pos = goal_pos.copy()
        goal_pos += np.array([self.shift_x, self.shift_y, self.shift_z])
        direction = np.array(goal_pos) - np.array(gripper_pos)
        distance = np.linalg.norm(direction)
        if distance > self.delta:
            waypoint_pos = np.array(gripper_pos) + self.delta * direction / distance
        else:
            waypoint_pos = np.array(goal_pos)

        waypoint_pos[2] = max(waypoint_pos[2], goal_pos[2])
        waypoint_orient = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, 0.0)
        return waypoint_pos, waypoint_orient

    # ── pick ──────────────────────────────────────────────────────────

    def pick(self):
        print("visual_grasp: within delta — closing gripper")
        self.picked = True
        self.move_to_pose({'joint_gripper_finger_left': 0.2}, blocking=True)
        self.move_to_pose({'joint_arm': 0.0}, blocking=True)
        # Hand control back to nav: the executor immediately drives the base
        # to the place anchor after /visual_grasp/done, which needs nav mode.
        self._call_trigger(self.switch_nav_cli)
        print("visual_grasp: pick complete — publishing done")
        self.done_pub.publish(Bool(data=True))
        self._go_idle()

    def open_gripper(self):
        self.move_to_pose({'joint_gripper_finger_left': 0.8}, blocking=True)

    # ── mode-switch helper ──────────────────────────────────────────

    def _call_trigger(self, client) -> bool:
        if client is None:
            return False
        if not client.wait_for_service(timeout_sec=2.0):
            print(f"visual_grasp: trigger service unavailable ({client.srv_name})")
            return False
        fut = client.call_async(Trigger.Request())
        deadline = time.time() + 4.0
        while not fut.done() and time.time() < deadline:
            time.sleep(0.01)
        if not fut.done():
            return False
        res = fut.result()
        return bool(res is not None and res.success)

    # ── node entry point ─────────────────────────────────────────────

    def main(self):
        HelloNode.main(self, 'visual_grasp', 'visual_grasp', wait_for_first_pointcloud=False)
        self.logger = self.get_logger()
        self.callback_group = ReentrantCallbackGroup()

        self.joint_states_subscriber = self.create_subscription(
            JointState, '/stretch/joint_states',
            callback=self.joint_states_callback, qos_profile=1,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.switch_pos_cli = self.create_client(
            Trigger, '/switch_to_position_mode',
            callback_group=self.callback_group,
        )
        self.switch_nav_cli = self.create_client(
            Trigger, '/switch_to_navigation_mode',
            callback_group=self.callback_group,
        )

        self.done_pub = self.create_publisher(Bool, '/visual_grasp/done', 10)

        self.create_subscription(
            Bool, '/visual_grasp/start',
            self._on_start, 10,
            callback_group=self.callback_group,
        )

        self.create_subscription(
            String, '/fine_detector/target_object',
            self._on_target_object, 10,
            callback_group=self.callback_group,
        )

        self.det_sub = self.create_subscription(
            Detection3DArray,
            '/fine_detector/objects',
            self.detection_callback,
            qos_profile=1,
            callback_group=self.callback_group,
        )

        print("visual_grasp: ready — waiting for /visual_grasp/start")


def main():
    node = IKVisualGrasp()
    try:
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        if node.active:
            node._go_idle()


if __name__ == '__main__':
    main()
