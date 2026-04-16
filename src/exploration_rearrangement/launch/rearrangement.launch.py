"""Rearrangement only: assumes a map is already loaded via map_server, or SLAM is
running in localization mode. Skips the exploration/mapping pipeline."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('exploration_rearrangement')
    regions_yaml = PathJoinSubstitution([pkg, 'config', 'regions.yaml'])
    objects_yaml = PathJoinSubstitution([pkg, 'config', 'objects.yaml'])
    tasks_yaml = PathJoinSubstitution([pkg, 'config', 'tasks.yaml'])

    planner_backend = LaunchConfiguration('planner_backend')

    args = [
        DeclareLaunchArgument('planner_backend', default_value='greedy'),
    ]

    detector = Node(
        package='exploration_rearrangement', executable='object_detector_node',
        name='object_detector_node', output='screen',
        parameters=[{'objects_yaml': objects_yaml}],
    )
    regions = Node(
        package='exploration_rearrangement', executable='region_manager_node',
        name='region_manager_node', output='screen',
        parameters=[{'regions_yaml': regions_yaml}],
    )
    head_scan = Node(
        package='exploration_rearrangement', executable='head_scan_node',
        name='head_scan_node', output='screen',
    )
    manipulation = Node(
        package='exploration_rearrangement', executable='manipulation_node',
        name='manipulation_node', output='screen',
    )
    planner = Node(
        package='exploration_rearrangement', executable='task_planner_node',
        name='task_planner_node', output='screen',
        parameters=[{
            'planner_backend': planner_backend,
            'tasks_yaml': tasks_yaml,
            'regions_yaml': regions_yaml,
        }],
    )
    executor = Node(
        package='exploration_rearrangement', executable='task_executor_node',
        name='task_executor_node', output='screen',
        parameters=[{
            'planner_backend': planner_backend,
            'tasks_yaml': tasks_yaml,
            'regions_yaml': regions_yaml,
            'start_on_launch': False,
        }],
    )

    return LaunchDescription(args + [
        detector, regions, head_scan, manipulation, planner, executor,
    ])
