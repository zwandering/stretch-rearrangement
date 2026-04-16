"""Exploration + mapping only. Useful for the first pass to build a map you can save."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('exploration_rearrangement')
    slam_params = PathJoinSubstitution([pkg, 'config', 'slam_params.yaml'])
    nav2_params = PathJoinSubstitution([pkg, 'config', 'nav2_params.yaml'])
    objects_yaml = PathJoinSubstitution([pkg, 'config', 'objects.yaml'])
    rviz_cfg = PathJoinSubstitution([pkg, 'rviz', 'rearrangement.rviz'])

    args = [
        DeclareLaunchArgument('run_rviz', default_value='true'),
    ]

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('slam_toolbox'),
            '/launch/online_async_launch.py',
        ]),
        launch_arguments={
            'slam_params_file': slam_params, 'use_sim_time': 'false',
        }.items(),
    )
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('nav2_bringup'),
            '/launch/navigation_launch.py',
        ]),
        launch_arguments={
            'use_sim_time': 'false', 'params_file': nav2_params, 'autostart': 'true',
        }.items(),
    )

    exploration = Node(
        package='exploration_rearrangement', executable='exploration_node',
        name='exploration_node', output='screen',
        parameters=[{'enabled_on_start': True}],
    )
    detector = Node(
        package='exploration_rearrangement', executable='object_detector_node',
        name='object_detector_node', output='screen',
        parameters=[{'objects_yaml': objects_yaml}],
    )
    head_scan = Node(
        package='exploration_rearrangement', executable='head_scan_node',
        name='head_scan_node', output='screen',
        parameters=[{'enabled_on_start': True}],
    )
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_cfg], output='log',
    )

    return LaunchDescription(args + [slam, nav2, exploration, detector, head_scan, rviz])
