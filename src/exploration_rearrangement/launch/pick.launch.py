"""Single-stage pick — fine_object_detector_node + visual_grasp_node only.

No visual_servo_arm. visual_grasp_node now moves the arm to
ik.READY_POSE_P2 itself on /visual_grasp/start, so the gripper D405 sees
the object directly and no head-camera servo stage is needed.

Prerequisites (start in separate terminals BEFORE this launch):
  Terminal 1:  ros2 launch stretch_core stretch_driver.launch.py
  Terminal 2:  ros2 launch stretch_core d405_basic.launch.py

Then:
  Terminal 3:  ros2 launch exploration_rearrangement pick.launch.py \\
                   target_object:='yellow cup'

To trigger a grasp manually (debug — see chat for full fake-msg recipe):
  ros2 topic pub --once /fine_detector/target_object std_msgs/msg/String \\
      '{data: "yellow cup"}'
  ros2 topic pub --once /fine_detector/activate std_msgs/msg/Bool \\
      '{data: true}'
  ros2 topic pub --once /visual_grasp/start std_msgs/msg/Bool \\
      '{data: true}'
  ros2 topic echo /visual_grasp/done
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('exploration_rearrangement')
    objects_yaml = PathJoinSubstitution([pkg, 'config', 'objects.yaml'])

    yolo_model = LaunchConfiguration('yolo_model')
    target_object = LaunchConfiguration('target_object')

    args = [
        DeclareLaunchArgument(
            'yolo_model',
            default_value='yoloe-11s-seg.pt',
            description='YOLOE weights or exported .engine path',
        ),
        DeclareLaunchArgument(
            'target_object',
            default_value='yellow cup',
            description='Object class name to track (must match objects.yaml).',
        ),
    ]

    fine_detector = Node(
        package='exploration_rearrangement',
        executable='fine_object_detector_node',
        name='fine_object_detector_node',
        output='screen',
        parameters=[{
            'mode': 'robot',
            'objects_yaml': objects_yaml,
            'model_path': yolo_model,
            'enable_head': False,
        }],
    )

    visual_grasp = Node(
        package='exploration_rearrangement',
        executable='visual_grasp_node',
        name='visual_grasp_node',
        output='screen',
        parameters=[{'target_object': target_object}],
    )

    return LaunchDescription(args + [fine_detector, visual_grasp])
