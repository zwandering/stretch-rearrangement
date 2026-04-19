"""Visual-grasp pipeline — all detection + servo nodes in one launch.

Prerequisites (start these in separate terminals BEFORE this launch file):

  Terminal 1:  ros2 launch stretch_core stretch_driver.launch.py
  Terminal 2:  ros2 launch stretch_core d435i_low_resolution.launch.py
  Terminal 3:  ros2 launch stretch_core d405_basic.launch.py

Then:

  Terminal 4:  ros2 launch exploration_rearrangement visual_grasp.launch.py

This starts:
  - object_detector_node      (head D435i → /detector/objects)
  - fine_object_detector_node  (gripper D405 → /fine_detector/objects, starts idle)
  - visual_servo_arm_node      (Stage 1: coarse IK approach via head camera)
  - visual_grasp_node          (Stage 2: fine IK + grasp via gripper camera)

Stage 2 nodes (fine_object_detector + visual_grasp) are included but
visual_grasp_node must be launched separately when Stage 1 finishes
(see launch arg ``stage``).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('exploration_rearrangement')
    objects_yaml = PathJoinSubstitution([pkg, 'config', 'objects.yaml'])

    yolo_model = LaunchConfiguration('yolo_model')
    stage = LaunchConfiguration('stage')
    output_frame = LaunchConfiguration('head_output_frame')
    target_object = LaunchConfiguration('target_object')

    args = [
        DeclareLaunchArgument(
            'yolo_model',
            default_value='yoloe-11s-seg.pt',
            description='YOLOE weights or exported .engine path',
        ),
        DeclareLaunchArgument(
            'stage',
            default_value='all',
            description=(
                'Which stage to launch: "1" = coarse approach only, '
                '"2" = fine grasp only, "all" = both stages.'
            ),
        ),
        DeclareLaunchArgument(
            'head_output_frame',
            default_value='base_link',
            description=(
                'Output frame for head detector. Use "map" if SLAM is '
                'running, "base_link" otherwise.'
            ),
        ),
        DeclareLaunchArgument(
            'target_object',
            default_value='yellow cup',
            description='Object class name to track (must match objects.yaml).',
        ),
    ]

    # --- Stage 1 nodes: head detector + visual_servo_arm ---

    stage1_condition = IfCondition(PythonExpression([
        "'", stage, "' == '1' or '", stage, "' == 'all'",
    ]))

    head_detector = Node(
        package='exploration_rearrangement',
        executable='object_detector_node',
        name='object_detector_node',
        output='screen',
        parameters=[{
            'mode': 'robot',
            'objects_yaml': objects_yaml,
            'model_path': yolo_model,
            'output_frame': output_frame,
        }],
        condition=stage1_condition,
    )

    visual_servo_arm = Node(
        package='exploration_rearrangement',
        executable='visual_servo_arm_node',
        name='visual_servo_arm_node',
        output='screen',
        parameters=[{'target_object': target_object}],
        condition=stage1_condition,
    )

    # --- Stage 2 nodes: fine detector + visual_grasp ---

    stage2_condition = IfCondition(PythonExpression([
        "'", stage, "' == '2' or '", stage, "' == 'all'",
    ]))

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
        condition=stage2_condition,
    )

    visual_grasp = Node(
        package='exploration_rearrangement',
        executable='visual_grasp_node',
        name='visual_grasp_node',
        output='screen',
        parameters=[{'target_object': target_object}],
        condition=stage2_condition,
    )

    return LaunchDescription(args + [
        head_detector,
        visual_servo_arm,
        fine_detector,
        visual_grasp,
    ])
