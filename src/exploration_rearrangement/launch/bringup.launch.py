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

    run_slam = LaunchConfiguration('run_slam')
    run_nav2 = LaunchConfiguration('run_nav2')
    run_rviz = LaunchConfiguration('run_rviz')
    start_on_launch = LaunchConfiguration('start_on_launch')
    yolo_model = LaunchConfiguration('yolo_model')
    vlm_model = LaunchConfiguration('vlm_model')
    vlm_api_key_env = LaunchConfiguration('vlm_api_key_env')
    instruction_topic = LaunchConfiguration('instruction_topic')

    args = [
        DeclareLaunchArgument('run_slam', default_value='true'),
        DeclareLaunchArgument('run_nav2', default_value='true'),
        DeclareLaunchArgument('run_rviz', default_value='true'),
        DeclareLaunchArgument('start_on_launch', default_value='false',
                              description='auto-trigger executor after launch'),
        DeclareLaunchArgument('yolo_model', default_value='yoloe-11s-seg.pt',
                              description='YOLOE weights or exported .engine path'),
        DeclareLaunchArgument('vlm_model', default_value='gemini-2.5-flash',
                              description='Gemini model id (OpenAI-compatible endpoint)'),
        DeclareLaunchArgument('vlm_api_key_env', default_value='GEMINI_API_KEY',
                              description='env var name holding the Gemini API key'),
        DeclareLaunchArgument('instruction_topic', default_value='/instruction/text',
                              description='topic to subscribe for operator instructions'),
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
            'mode': 'robot',
            'objects_yaml': objects_yaml,
            'model_path': yolo_model,
        }],
    )
    fine_detector = Node(
        package='exploration_rearrangement', executable='fine_object_detector_node',
        name='fine_object_detector_node', output='screen',
        parameters=[{
            'mode': 'robot',
            'objects_yaml': objects_yaml,
            'model_path': yolo_model,
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
            'regions_yaml': regions_yaml,
            'vlm_model': vlm_model,
            'vlm_api_key_env': vlm_api_key_env,
            'instruction_topic': instruction_topic,
        }],
    )
    executor = Node(
        package='exploration_rearrangement', executable='task_executor_node',
        name='task_executor_node', output='screen',
        parameters=[{
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
        exploration, detector, fine_detector, regions, head_scan,
        manipulation, planner, executor,
        rviz,
    ])
