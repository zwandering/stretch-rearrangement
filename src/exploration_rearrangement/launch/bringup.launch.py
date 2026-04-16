"""Full system bringup: SLAM Toolbox + Nav2 + all six project nodes + RViz."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('exploration_rearrangement')

    slam_params = PathJoinSubstitution([pkg, 'config', 'slam_params.yaml'])
    nav2_params = PathJoinSubstitution([pkg, 'config', 'nav2_params.yaml'])
    regions_yaml = PathJoinSubstitution([pkg, 'config', 'regions.yaml'])
    objects_yaml = PathJoinSubstitution([pkg, 'config', 'objects.yaml'])
    tasks_yaml = PathJoinSubstitution([pkg, 'config', 'tasks.yaml'])
    rviz_cfg = PathJoinSubstitution([pkg, 'rviz', 'rearrangement.rviz'])

    planner_backend = LaunchConfiguration('planner_backend')
    run_slam = LaunchConfiguration('run_slam')
    run_nav2 = LaunchConfiguration('run_nav2')
    run_rviz = LaunchConfiguration('run_rviz')
    start_on_launch = LaunchConfiguration('start_on_launch')

    args = [
        DeclareLaunchArgument('planner_backend', default_value='greedy',
                              description='greedy | vlm'),
        DeclareLaunchArgument('run_slam', default_value='true'),
        DeclareLaunchArgument('run_nav2', default_value='true'),
        DeclareLaunchArgument('run_rviz', default_value='true'),
        DeclareLaunchArgument('start_on_launch', default_value='false',
                              description='auto-trigger executor after launch'),
    ]

    slam = GroupAction(
        condition=None,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('slam_toolbox'),
                '/launch/online_async_launch.py',
            ]),
            launch_arguments={
                'slam_params_file': slam_params,
                'use_sim_time': 'false',
            }.items(),
        )],
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('nav2_bringup'),
            '/launch/navigation_launch.py',
        ]),
        launch_arguments={
            'use_sim_time': 'false',
            'params_file': nav2_params,
            'autostart': 'true',
        }.items(),
    )

    exploration = Node(
        package='exploration_rearrangement', executable='exploration_node',
        name='exploration_node', output='screen',
        parameters=[{
            'enabled_on_start': False,
            'min_cluster_size': 8,
            'alpha_dist': 1.0,
            'beta_info': 0.05,
        }],
    )
    detector = Node(
        package='exploration_rearrangement', executable='object_detector_node',
        name='object_detector_node', output='screen',
        parameters=[{
            'objects_yaml': objects_yaml,
        }],
    )
    regions = Node(
        package='exploration_rearrangement', executable='region_manager_node',
        name='region_manager_node', output='screen',
        parameters=[{'regions_yaml': regions_yaml}],
    )
    head_scan = Node(
        package='exploration_rearrangement', executable='head_scan_node',
        name='head_scan_node', output='screen',
        parameters=[{'enabled_on_start': False}],
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
            'start_on_launch': start_on_launch,
        }],
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_cfg], output='log',
    )

    return LaunchDescription(args + [
        slam, nav2,
        exploration, detector, regions, head_scan, manipulation, planner, executor,
        rviz,
    ])
