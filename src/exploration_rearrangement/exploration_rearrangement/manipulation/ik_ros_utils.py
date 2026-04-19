import numpy as np
import ikpy.chain
from geometry_msgs.msg import Pose, PoseStamped
from tf2_geometry_msgs import TransformStamped

from .urdf_utils import MODIFIED_URDF_PATH

READY_POSE_P1 = {
    'joint_lift': 0.8,
    'joint_wrist_yaw': 1.5,
    'joint_wrist_pitch': -0.1,
    'gripper_aperture': 0.5
}

READY_POSE_P2 = {
    'joint_lift': 0.8,
    'joint_arm_l0': 0.0,
    'joint_wrist_yaw': 0.0,
    'joint_wrist_pitch': -0.1,
    'gripper_aperture': 0.5,
    'joint_head_pan': -1.6,
    'joint_head_tilt': -0.5,
}

ACTIVE_LINKS_MASK = [
    False,  # 0: base_link (fixed)
    True,   # 1: joint_base_rotation (revolute - mobile base yaw)
    True,   # 2: joint_base_translation (prismatic - mobile base x)
    False,  # 3: joint_mast (fixed)
    True,   # 4: joint_lift (prismatic - vertical lift)
    False,  # 5: joint_arm_l4 (fixed)
    True,   # 6: joint_arm_l3 (prismatic)
    True,   # 7: joint_arm_l2 (prismatic)
    True,   # 8: joint_arm_l1 (prismatic)
    True,   # 9: joint_arm_l0 (prismatic)
    True,   # 10: joint_wrist_yaw (revolute)
    False,  # 11: joint_wrist_yaw_bottom (fixed)
    True,   # 12: joint_wrist_pitch (revolute)
    True,   # 13: joint_wrist_roll (revolute)
    False,  # 14: joint_gripper_s3_body (fixed)
    False,  # 15: joint_grasp_center (fixed - end effector)
]

IK_POSITION_TOLERANCE = 1e-2

def get_xyz_from_msg(msg):
    if isinstance(msg, PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
    elif isinstance(msg, TransformStamped):
        x = msg.transform.translation.x
        y = msg.transform.translation.y
        z = msg.transform.translation.z

    return np.array((x, y, z))


chain = ikpy.chain.Chain.from_urdf_file(str(MODIFIED_URDF_PATH), active_links_mask=ACTIVE_LINKS_MASK)

for link in chain.links:
    print(f"* Link Name: {link.name}, Type: {link.joint_type}")

def get_current_configuration(joint_state):
    """
    Get current robot joint configuration from ROS state.
    
    Returns:
        list: Configuration vector matching kinematic chain structure.
                Contains positions for all 16 links (active and fixed).
    """
    # Extract joint positions from ROS joint state message
    
    def _clamp_joint_value(name, value):
        names = [l.name for l in chain.links]
        index = names.index(name)
        bounds = chain.links[index].bounds
        return min(max(value, bounds[0]), bounds[1])

    # Virtual base joints (not actual hardware joints)
    base_rotation = 0.0
    base_translation = 0.0
    
    # Hardware joint positions (clamped to limits)
    lift_position = _clamp_joint_value('joint_lift', joint_state['joint_lift'])
    arm_extension = _clamp_joint_value('joint_arm_l0', joint_state['joint_arm_l0'])
    wrist_yaw = _clamp_joint_value('joint_wrist_yaw', joint_state['joint_wrist_yaw'])
    wrist_pitch = _clamp_joint_value('joint_wrist_pitch', joint_state['joint_wrist_pitch'])
    wrist_roll = _clamp_joint_value('joint_wrist_roll', joint_state['joint_wrist_roll'])
    
    # Build configuration vector for all 16 links in chain
    # Order: base_link, link_base_rotation, link_base_translation, link_mast, link_lift,
    #        link_arm_l4, link_arm_l3, link_arm_l2, link_arm_l1, link_arm_l0,
    #        link_wrist_yaw, link_wrist_yaw_bottom, link_wrist_pitch, link_wrist_roll,
    #        link_gripper_s3_body, link_grasp_center
    return [
        0.0,              # 0: base_link (fixed origin)
        base_rotation,    # 1: virtual base rotation
        base_translation, # 2: virtual base translation
        0.0,              # 3: mast (fixed)
        lift_position,    # 4: lift joint
        0.0,              # 5: arm_l4 (fixed)
        arm_extension,    # 6-9: arm extension (4 prismatic segments with same value)
        arm_extension,
        arm_extension,
        arm_extension,
        wrist_yaw,        # 10: wrist yaw
        0.0,              # 11: wrist_yaw_bottom (fixed)
        wrist_pitch,      # 12: wrist pitch
        wrist_roll,       # 13: wrist roll
        0.0,              # 14: gripper body (fixed)
        0.0,              # 15: grasp center (fixed)
    ]

def get_grasp_goal(target_point, target_orientation, q_init):
    # previously the move_to_grasp() function from lab 2
    #   moved to it's own function without the final move_to_configuration() call for convenience in this lab
    q_soln = chain.inverse_kinematics(target_point, target_orientation, orientation_mode='all', initial_position=q_init)
    # print('Solution:', q_soln)
    print("Solution Found")

    err = np.linalg.norm(chain.forward_kinematics(q_soln)[:3, 3] - target_point)
    if not np.isclose(err, 0.0, atol=1e-2):
        print("IKPy did not find a valid solution")
        return
    # move_to_configuration(q=q_soln)
    return q_soln

def move_to_configuration(node, configuration):
    """Move robot to a specified joint configuration."""
    base_rotation = configuration[1]
    base_translation = configuration[2]
    lift_position = configuration[4]
    arm_extension = configuration[6] + configuration[7] + configuration[8] + configuration[9]
    wrist_yaw = configuration[10]
    wrist_pitch = configuration[12]
    wrist_roll = configuration[13]
    
    # Move arm and wrist joints
    node.move_to_pose('joint_lift', lift_position, blocking=True)
    node.move_to_pose(
        {
            'joint_arm': arm_extension,
            'joint_wrist_yaw': wrist_yaw,
            'joint_wrist_pitch': wrist_pitch,
            'joint_wrist_roll': wrist_roll,
        },
        blocking=True,
    )
    
    # Move mobile base (rotation then translation)
    node.move_to_pose(
        {'rotate_mobile_base': base_rotation},
        blocking=True,
    )
    node.move_to_pose(
        {'translate_mobile_base': base_translation},
        blocking=True,
    )

def print_q(q):
    if q is None:
        print('INVALID Q')

    else:
        print("IK Config")
        print("     Base Rotation:", q[1])
        print("     Base Translation:", q[2])
        print("     Lift", q[4])
        print("     Arm", q[6] + q[7] + q[8] + q[9])
        print("     Gripper Yaw:", q[10])
        print("     Gripper Pitch:", q[12])
        print("     Gripper Roll:", q[13])
