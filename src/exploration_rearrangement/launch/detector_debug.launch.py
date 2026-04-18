"""Standalone YOLOE detector + optional RealSense D435i, for debugging.

Usage (detector subscribes to an already-running D435i):

    ros2 launch realsense2_camera rs_launch.py \
        enable_color:=true enable_depth:=true align_depth.enable:=true
    ros2 launch exploration_rearrangement detector_debug.launch.py

Or let this launch file start the camera too:

    ros2 launch exploration_rearrangement detector_debug.launch.py \
        start_realsense:=true

Parameters:
    model_path       Path/name of YOLOE weights. Use the exported engine
                     (``yoloe-11s-seg.engine``) for speed; the raw .pt works too.
    conf_threshold   YOLOE confidence threshold (default 0.25).
    center_log_path  JSONL path to append per-frame center records to.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('exploration_rearrangement')
    objects_yaml = PathJoinSubstitution([pkg, 'config', 'objects.yaml'])

    model_path = LaunchConfiguration('model_path')
    conf = LaunchConfiguration('conf_threshold')
    center_log = LaunchConfiguration('center_log_path')
    start_rs = LaunchConfiguration('start_realsense')

    args = [
        DeclareLaunchArgument('model_path', default_value='yoloe-11s-seg.pt'),
        DeclareLaunchArgument('conf_threshold', default_value='0.25'),
        DeclareLaunchArgument('center_log_path', default_value=''),
        DeclareLaunchArgument('start_realsense', default_value='false',
                              description='If true, also launch realsense2_camera.'),
    ]

    realsense = GroupAction(
        condition=IfCondition(start_rs),
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('realsense2_camera'),
                '/launch/rs_launch.py',
            ]),
            launch_arguments={
                'enable_color': 'true',
                'enable_depth': 'true',
                'align_depth.enable': 'true',
                'pointcloud.enable': 'false',
            }.items(),
        )],
    )

    detector = Node(
        package='exploration_rearrangement',
        executable='object_detector_node',
        name='object_detector_node',
        output='screen',
        parameters=[{
            'mode': 'debug',
            'model_path': model_path,
            'objects_yaml': objects_yaml,
            'conf_threshold': conf,
            'center_log_path': center_log,
            'publish_debug_image': True,
        }],
    )

    return LaunchDescription(args + [realsense, detector])
