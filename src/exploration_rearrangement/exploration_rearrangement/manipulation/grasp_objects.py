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
import ikpy.utils.geometry

# Make sure to run:
#   ros2 launch stretch_core stretch_driver.launch.py

class IKTargetFollowing(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)

        self.delta = 0.03 # cm
        self.target_frame = 'base_link'
        self.gripper_frame = 'link_grasp_center'
        self.tf_buffer = None
        self.tf_listener = None
        self.joint_states_lock = threading.Lock()
    
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
        # TODO: ------------- start --------------
        goal_transformed = self.tf_buffer.transform(goal_msg, self.target_frame)
        # TODO: -------------- end ---------------

        return goal_transformed

    def get_gripper_pose_in_base_frame(self):
        # TODO: ------------- start --------------
        gripper_transformed = self.tf_buffer.lookup_transform(self.target_frame, self.gripper_frame, rclpy.time.Time())
        # TODO: -------------- end ---------------

        return gripper_transformed

    def goal_callback(self, goal_msg):
        # print(msg)
        # self.get_logger().info(f'Received goal pose: {msg.pose}')
        try:
            goal_transformed = self.get_goal_pose_in_base_frame(goal_msg)
            gripper_transformed = self.get_gripper_pose_in_base_frame()

            goal_pos = ik.get_xyz_from_msg(goal_transformed)
            gripper_pos = ik.get_xyz_from_msg(gripper_transformed)
        except:
            print("Error getting transforms")
            return

        waypoint_pos, waypoint_orient = self.compute_waypoint_to_goal(goal_pos, gripper_pos)

        # TODO: ------------- start --------------
        q_init = ik.get_current_configuration(self.joint_state)
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

    def compute_waypoint_to_goal(self, goal_pos, gripper_pos):

        # TODO: ------------- start --------------
        dist  = np.linalg.norm(goal_pos - gripper_pos)
        if dist > self.delta:
            waypoint_pos = gripper_pos + (goal_pos - gripper_pos) / dist * self.delta
        else:
            waypoint_pos = goal_pos

        # TODO: -------------- end ---------------

        # use an zero rotation for the waypoint (its a point so we don't need to worry about orientation)
        waypoint_orient = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, 0.0) # [roll, pitch, yaw]

        return waypoint_pos, waypoint_orient


    def move_to_ready_pose(self):
        self.move_to_pose(ik.READY_POSE_P2, blocking=True)

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
        self.create_subscription(PoseStamped, '/object_detector/goal_pose', self.goal_callback, 10)
        # TODO: -------------- end ---------------



if __name__ == '__main__':
    target_follower = IKTargetFollowing()
    target_follower.main()
    target_follower.new_thread.join()
