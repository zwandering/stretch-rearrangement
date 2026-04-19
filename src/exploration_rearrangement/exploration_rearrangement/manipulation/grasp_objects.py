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
import ik_ros_utils as ik
import ikpy

# Make sure to run:
#   ros2 launch stretch_core stretch_driver.launch.py

class IKTargetFollowing(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)

        self.delta = 0.06 # cm
        self.target_frame = 'base_link'
        self.gripper_frame = 'link_grasp_center'
        self.tf_buffer = None
        self.tf_listener = None
        self.joint_states_lock = threading.Lock()

        self.shift_x = 0.04
        self.shift_y = -0.03
        self.shift_z = 0.03 # 0.03 for bottle and 0.01 for cup, 
    
    def joint_states_callback(self, msg):
        # unpacks joint state messages for what works with/is expected by ikpy
        with self.joint_states_lock:
            joint_states = msg
        # extract information needed for ik_solver
        joint_names = [
            'joint_lift', 'joint_arm_l0', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll'
        ]
        self.joint_state = {}
        for joint_name in joint_names:
            i = joint_states.name.index(joint_name)
            self.joint_state[joint_name] = joint_states.position[i]

    def get_goal_pose_in_base_frame(self, goal_msg):
        goal_transformed = self.tf_buffer.transform(goal_msg, self.target_frame, timeout=rclpy.duration.Duration(seconds=1.0))

        return goal_transformed

    def get_gripper_pose_in_base_frame(self):
        gripper_transformed = self.tf_buffer.lookup_transform(self.target_frame, self.gripper_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))

        return gripper_transformed

    def goal_callback(self, goal_msg):
        # print(msg)
        # self.get_logger().info(f'Received goal pose: {msg.pose}')
        try:
            goal_transformed = self.get_goal_pose_in_base_frame(goal_msg)
            gripper_transformed = self.get_gripper_pose_in_base_frame()

            goal_pos = ik.get_xyz_from_msg(goal_transformed)
            gripper_pos = ik.get_xyz_from_msg(gripper_transformed)
        except Exception as e:
            print(f"Error getting transforms: {e}")
            import traceback
            traceback.print_exc()
            return

        waypoint_pos, waypoint_orient = self.compute_waypoint_to_goal(goal_pos, gripper_pos)

        # TODO: ------------- start --------------
        with self.joint_states_lock:
            joint_state = self.joint_state
        q_init = ik.get_current_configuration(joint_state)
        q_soln = ik.get_grasp_goal(waypoint_pos, waypoint_orient, q_init)
        # TODO: -------------- end ---------------

        # NOTE: if you find that the robot's base is moving too much, its likely that the ik solver is
        # struggling to find solutions without the base doing most of the work to achieve the waypoint pose.
        # you can adjust the `self.delta` variable to be smaller so that the displacements are smaller, and
        #   there is a valid solution without excessive base movement
        # you can also set your own triggers manually (keep delta large but use an if/else on move_to_pose()
        #   so the base only moves above a certain distance threshold
        # you can also try adjusting joint limits of the base trans/rot in `ik_ros_utils.py` to be much smaller
        # one or some combination of these should help!

        ik.print_q(q_soln)
        if q_soln is not None:
            ik.move_to_configuration(self, q_soln)
            goal_pos_shifted = np.array(goal_pos) + np.array([self.shift_x, self.shift_y, self.shift_z])
            dist = np.linalg.norm(goal_pos_shifted - gripper_pos)
            print(f"Distance to goal after move: {dist:.3f} m")
            if dist < self.delta:
                print("Within delta threshold, closing gripper")
                self.move_to_pose({'gripper_aperture': -0.2}, blocking=True)
                self.move_to_pose({'joint_arm': 0.0}, blocking=True)

    def compute_waypoint_to_goal(self, goal_pos, gripper_pos):

        # TODO: ------------- start --------------
        goal_pos = goal_pos.copy()
        goal_pos += np.array([self.shift_x, self.shift_y, self.shift_z])
        direction = np.array(goal_pos) - np.array(gripper_pos)
        distance = np.linalg.norm(direction)
        if distance > self.delta:
            # move one delta step toward the goal
            waypoint_pos = np.array(gripper_pos) + self.delta * direction / distance
        else:
            # close enough — move directly to the goal
            waypoint_pos = np.array(goal_pos)
        # TODO: -------------- end ---------------

        # use an zero rotation for the waypoint (its a point so we don't need to worry about orientation)
        waypoint_pos[2] = max(waypoint_pos[2], goal_pos[2]) # ensure the waypoint is not below the goal (so we don't accidentally move under the object and miss it)
        waypoint_orient = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, 0.0) # [roll, pitch, yaw]

        return waypoint_pos, waypoint_orient


    def move_to_ready_pose(self):
        # TODO: minor - uncomment the correct ready pose for part 1 or 2!
        #   part 1: 
        # self.move_to_pose(ik.READY_POSE_P1, blocking=True)
        #   part 2: READY_POSE_P2
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

    def main(self):
        HelloNode.main(self, 'follow_target', 'follow_target', wait_for_first_pointcloud=False)
        self.logger = self.get_logger()
        self.callback_group = ReentrantCallbackGroup()
        self.joint_states_subscriber = self.create_subscription(JointState, '/stretch/joint_states', callback=self.joint_states_callback, qos_profile=1)

        self.stow_the_robot()
        self.move_to_ready_pose()
        print("At Ready Pose")

        # TODO: ------------- start --------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/object_detector/goal_pose',
            self.goal_callback,
            qos_profile=1,
            callback_group=self.callback_group
        )
        # TODO: -------------- end ---------------


if __name__ == '__main__':
    target_follower = IKTargetFollowing()
    target_follower.main()
    target_follower.new_thread.join()
