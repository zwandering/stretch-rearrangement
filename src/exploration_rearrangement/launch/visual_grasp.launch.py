"""Visual-grasp pipeline — fine detector + servo + grasp nodes.

All nodes are passive and controlled by task_executor_node via topic signals.

Prerequisites (start these in separate terminals BEFORE this launch file):

  Terminal 1:  ros2 launch stretch_core stretch_driver.launch.py
  Terminal 2:  ros2 launch stretch_core d435i_low_resolution.launch.py
  Terminal 3:  ros2 launch stretch_core d405_basic.launch.py

Then:

  Terminal 4:  ros2 launch exploration_rearrangement visual_grasp.launch.py

This starts:
  - fine_object_detector_node  (dual camera: D405 + D435i, starts idle)
  - visual_servo_arm_node      (Stage 1: coarse IK via head camera, waits for /visual_servo/start)
  - visual_grasp_node          (Stage 2: fine IK + grasp via gripper camera, waits for /visual_grasp/start)

task_executor_node orchestrates the pick sequence:
  PICK_INIT → activate detector, ready pose, open gripper
  PICK_SERVO → /visual_servo/start → wait /visual_servo/reached
  PICK_GRASP → /visual_grasp/start → wait /visual_grasp/done
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
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
        }],
    )

    visual_servo_arm = Node(
        package='exploration_rearrangement',
        executable='visual_servo_arm_node',
        name='visual_servo_arm_node',
        output='screen',
        parameters=[{'target_object': target_object}],
    )

    visual_grasp = Node(
        package='exploration_rearrangement',
        executable='visual_grasp_node',
        name='visual_grasp_node',
        output='screen',
        parameters=[{'target_object': target_object}],
    )

    return LaunchDescription(args + [
        fine_detector,
        visual_servo_arm,
        visual_grasp,
    ])
