import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import FollowJointTrajectory
from hello_helpers.hello_misc import HelloNode
import threading
import tf2_ros
from tf2_geometry_msgs import TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from vision_msgs.msg import Detection3DArray
from .manipulation import ik_ros_utils as ik
import ikpy

# Make sure to run:
#   ros2 launch stretch_core stretch_driver.launch.py

class IKVisualGrasp(HelloNode):
    """IK target-following fed by the D405 fine detector, with pick.

    Same IK stepping logic as target_following.py / grasp_objects.py, but
    subscribes to /fine_detector/objects (Detection3DArray in base_link)
    instead of /object_detector/goal_pose (PoseStamped).

    Set self.target_object_name to the class in config/objects.yaml you
    want to track (e.g. 'white_bottle', 'green_cup', 'blue_cup').
    """

    def __init__(self):
        HelloNode.__init__(self)

        self.delta = 0.06
        self.target_frame = 'base_link'
        self.gripper_frame = 'link_grasp_center'
        self.tf_buffer = None
        self.tf_listener = None
        self.joint_states_lock = threading.Lock()

        self.shift_x = 0.04
        self.shift_y = -0.03
        self.shift_z = 0.03 # 0.03 for bottle and 0.01 for cup

        self.target_object_name = "yellow cup"   # ← set before running
        self.picked = False

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
        gripper_transformed = self.tf_buffer.lookup_transform(
            self.target_frame, self.gripper_frame,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=1.0),
        )
        return gripper_transformed

    # ── fine-detector callback ────────────────────────────────────────

    def detection_callback(self, msg: Detection3DArray):
        """Each fine-detector frame: extract target pos → IK step → pick when close."""
        if self.target_object_name is None or self.picked:
            return

        goal_pos = self._extract_target_pos(msg)
        if goal_pos is None:
            return

        try:
            gripper_transformed = self.get_gripper_pose_in_base_frame()
            gripper_pos = ik.get_xyz_from_msg(gripper_transformed)
        except Exception as e:
            print(f"Error getting gripper TF: {e}")
            import traceback
            traceback.print_exc()
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
        """Return the xyz np.array for self.target_object_name, or None."""
        for det in msg.detections:
            if not det.results:
                continue
            if det.results[0].hypothesis.class_id == self.target_object_name:
                p = det.results[0].pose.pose.position
                return np.array([p.x, p.y, p.z])
        return None

    # ── waypoint computation (same as target_following / grasp_objects) ─

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
        """Close gripper, retract arm, deactivate the fine detector, go idle."""
        print("Within delta threshold — picking object")
        self.picked = True
        self.move_to_pose({'gripper_aperture': -0.2}, blocking=True)
        self.move_to_pose({'joint_arm': 0.0}, blocking=True)
        self._set_fine_detector(False)
        self.target_object_name = None
        print("Pick complete — idle (ready for place)")

    # ── fine-detector helpers ─────────────────────────────────────────

    def _set_fine_detector(self, active: bool):
        msg = Bool()
        msg.data = active
        self.activate_pub.publish(msg)

    # ── ready pose ────────────────────────────────────────────────────

    def move_to_ready_pose(self):
        self.move_to_pose({
            'joint_lift': ik.READY_POSE_P2['joint_lift'],
            'joint_arm': ik.READY_POSE_P2['joint_arm_l0'],
            'joint_wrist_yaw': ik.READY_POSE_P2['joint_wrist_yaw'],
            'joint_wrist_pitch': ik.READY_POSE_P2['joint_wrist_pitch'],
            'gripper_aperture': ik.READY_POSE_P2['gripper_aperture'],
        }, blocking=True)
        self.move_to_pose({
            'joint_head_pan': ik.READY_POSE_P2['joint_head_pan'],
            'joint_head_tilt': ik.READY_POSE_P2['joint_head_tilt'],
        }, blocking=True)

    # ── handoff from visual_servo_arm ────────────────────────────────

    def open_gripper(self):
        self.move_to_pose({'gripper_aperture': 0.5}, blocking=True)

    def go_idle(self):
        """Deactivate fine detector, clear target, stop controlling."""
        print("Going idle — deactivating detector, clearing target")
        if self.started:
            self._set_fine_detector(False)
        self.target_object_name = None
        self.picked = True

    def _on_servo_reached(self, msg: Bool):
        """Stage 1 finished — start fine detection + IK grasp from current pose."""
        if not msg.data or self.started:
            return
        self.started = True
        print("Received handoff from visual_servo_arm — starting fine grasp")

        self.open_gripper()
        print("Gripper open — ready for fine approach")

        self.activate_pub = self.create_publisher(Bool, '/fine_detector/activate', 10)

        self.det_sub = self.create_subscription(
            Detection3DArray,
            '/fine_detector/objects',
            self.detection_callback,
            qos_profile=1,
            callback_group=self.callback_group,
        )

        self._set_fine_detector(True)
        print("Fine detector activated — waiting for detections")

    # ── node entry point ──────────────────────────────────────────────

    def main(self):
        HelloNode.main(self, 'visual_grasp', 'visual_grasp', wait_for_first_pointcloud=False)
        self.logger = self.get_logger()
        self.callback_group = ReentrantCallbackGroup()
        self.started = False

        self.declare_parameter('target_object', '')
        param_val = self.get_parameter('target_object').value
        if param_val:
            self.target_object_name = param_val

        self.joint_states_subscriber = self.create_subscription(
            JointState, '/stretch/joint_states',
            callback=self.joint_states_callback, qos_profile=1,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Bool, '/visual_servo/reached',
            self._on_servo_reached, 10,
            callback_group=self.callback_group,
        )
        print("Waiting for /visual_servo/reached to start fine grasp...")


def main():
    node = IKVisualGrasp()
    try:
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        if not node.picked:
            print("Aborted/cancelled — going idle")
            node.go_idle()


if __name__ == '__main__':
    main()
