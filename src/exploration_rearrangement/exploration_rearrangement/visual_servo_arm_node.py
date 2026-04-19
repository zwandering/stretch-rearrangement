"""Move the gripper near an object detected by the head D435i.

Subscribes to /detector/objects (Detection3DArray) from object_detector_node,
extracts the target object position, transforms it to base_link, applies an
(offset_x, offset_y, offset_z) shift, and IK-steps the arm toward that goal.

Same IK stepping logic as grasp_objects.py but no grasp at the end — the
node stops commanding once the gripper is within delta of the offset goal.
"""

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
import ikpy.utils.geometry

# Make sure to run:
#   ros2 launch stretch_core stretch_driver.launch.py

class IKVisualServoArm(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)

        self.delta = 0.03
        self.target_frame = 'base_link'
        self.gripper_frame = 'link_grasp_center'
        self.tf_buffer = None
        self.tf_listener = None
        self.joint_states_lock = threading.Lock()

        # offset from object position, expressed in link_grasp_center frame
        self.offset_in_gripper = np.array([-0.1, 0.0, 0.0])

        self.target_object_name = None   # ← set before running
        self.reached = False

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

    def get_goal_pose_in_base_frame(self, goal_msg):
        goal_transformed = self.tf_buffer.transform(goal_msg, self.target_frame)
        return goal_transformed

    def get_gripper_pose_in_base_frame(self):
        gripper_transformed = self.tf_buffer.lookup_transform(self.target_frame, self.gripper_frame, rclpy.time.Time())
        return gripper_transformed

    def _offset_in_base_link(self, object_pos_base):
        """Compute IK target = object_pos + offset rotated from gripper frame to base_link."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, self.gripper_frame, rclpy.time.Time())
            q = tf.transform.rotation
            # rotation matrix from gripper frame to base_link
            x, y, z, w = q.x, q.y, q.z, q.w
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
            ])
            offset_base = R @ self.offset_in_gripper
        except:
            offset_base = self.offset_in_gripper
        return object_pos_base + offset_base

    def detection_callback(self, msg: Detection3DArray):
        if self.target_object_name is None or self.reached:
            return

        goal_msg = self._extract_target_as_pose_stamped(msg)
        if goal_msg is None:
            return

        try:
            goal_transformed = self.get_goal_pose_in_base_frame(goal_msg)
            gripper_transformed = self.get_gripper_pose_in_base_frame()

            goal_pos = ik.get_xyz_from_msg(goal_transformed)
            gripper_pos = ik.get_xyz_from_msg(gripper_transformed)
        except:
            print("Error getting transforms")
            return

        target_pos = self._offset_in_base_link(goal_pos)
        waypoint_pos, waypoint_orient = self.compute_waypoint_to_goal(target_pos, gripper_pos)

        q_init = ik.get_current_configuration(self.joint_state)
        q_soln = ik.get_grasp_goal(waypoint_pos, waypoint_orient, q_init)

        ik.print_q(q_soln)
        if q_soln is not None:
            ik.move_to_configuration(self, q_soln)

            dist = np.linalg.norm(target_pos - gripper_pos)
            print(f"Distance to offset goal after move: {dist:.3f} m")

            if dist < self.delta:
                print("Arm positioned — within delta of offset goal")
                self.reached = True
                self.target_object_name = None
                self.reached_pub.publish(Bool(data=True))
                print("Idle — handed over to visual_grasp_node")

    def _extract_target_as_pose_stamped(self, msg: Detection3DArray):
        """Convert the target detection into a PoseStamped so tf_buffer.transform works."""
        for det in msg.detections:
            if not det.results:
                continue
            if det.results[0].hypothesis.class_id == self.target_object_name:
                p = det.results[0].pose.pose.position
                pose_msg = PoseStamped()
                pose_msg.header = msg.header
                pose_msg.pose.position.x = p.x
                pose_msg.pose.position.y = p.y
                pose_msg.pose.position.z = p.z
                pose_msg.pose.orientation.w = 1.0
                return pose_msg
        return None

    def compute_waypoint_to_goal(self, target_pos, gripper_pos):
        dist = np.linalg.norm(target_pos - gripper_pos)
        if dist > self.delta:
            waypoint_pos = gripper_pos + (target_pos - gripper_pos) / dist * self.delta
        else:
            waypoint_pos = target_pos

        waypoint_orient = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, 0.0)

        return waypoint_pos, waypoint_orient

    def open_gripper(self):
        self.move_to_pose({'gripper_aperture': 0.5}, blocking=True)

    def move_to_ready_pose(self):
        self.move_to_pose(ik.READY_POSE_P2, blocking=True)

    def go_idle(self):
        """Clear tracking target, stop controlling the arm."""
        print("Going idle — clearing target")
        self.target_object_name = None
        self.reached = True

    def main(self):
        HelloNode.main(self, 'visual_servo_arm', 'visual_servo_arm', wait_for_first_pointcloud=False)
        self.logger = self.get_logger()
        self.callback_group = ReentrantCallbackGroup()

        if not self.has_parameter('target_object'):
            self.declare_parameter('target_object', '')
        param_val = self.get_parameter('target_object').value
        if param_val:
            self.target_object_name = param_val

        self.joint_states_subscriber = self.create_subscription(JointState, '/stretch/joint_states', callback=self.joint_states_callback, qos_profile=1)

        self.stow_the_robot()
        self.move_to_ready_pose()
        self.open_gripper()
        print("At Ready Pose — gripper open")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.reached_pub = self.create_publisher(Bool, '/visual_servo/reached', 10)
        self.det_sub = self.create_subscription(
            Detection3DArray,
            '/detector/objects',
            self.detection_callback,
            qos_profile=1,
            callback_group=self.callback_group,
        )
        print(f"Listening to /detector/objects for '{self.target_object_name}'")


def main():
    node = IKVisualServoArm()
    try:
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        if not node.reached:
            print("Aborted/cancelled — going idle")
            node.go_idle()


if __name__ == '__main__':
    main()
