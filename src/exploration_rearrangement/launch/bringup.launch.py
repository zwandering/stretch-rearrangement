"""Stage 3: online bringup.

Wraps ``stretch_nav2 navigation.launch.py`` against a saved map (built in
stage 1) plus the project's brain — region manager, VLM planner, executor,
manipulation, detector, and the upstream nav coordinator. Detection keeps
running so any object that's been moved since the snapshot gets corrected
when it re-enters the FOV.

Operator workflow once this is up:
  - In RViz, click "2D Pose Estimate" to localize against the saved map.
  - ``ros2 topic pub --once /instruction/text std_msgs/msg/String '{data: "..."}'``
  - ``ros2 service call /executor/start std_srvs/srv/Trigger``
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('exploration_rearrangement')

    objects_yaml = PathJoinSubstitution([pkg, 'config', 'objects.yaml'])
    regions_yaml = PathJoinSubstitution([pkg, 'config', 'regions.yaml'])
    rviz_cfg = PathJoinSubstitution([pkg, 'rviz', 'rearrangement.rviz'])
    default_map = PathJoinSubstitution([pkg, 'maps', 'asangium.yaml'])

    map_arg = LaunchConfiguration('map')
    objects_snapshot = LaunchConfiguration('objects_snapshot')
    run_rviz = LaunchConfiguration('run_rviz')
    start_on_launch = LaunchConfiguration('start_on_launch')
    yolo_model = LaunchConfiguration('yolo_model')
    vlm_model = LaunchConfiguration('vlm_model')
    vlm_api_key_env = LaunchConfiguration('vlm_api_key_env')
    instruction_topic = LaunchConfiguration('instruction_topic')

    args = [
        DeclareLaunchArgument(
            'map', default_value=default_map,
            description='Saved map yaml (built via stage 1).'),
        DeclareLaunchArgument(
            'objects_snapshot', default_value='',
            description=('Optional YAML of object map-frame positions to seed '
                         'the planner with. Live /detector/objects keeps '
                         'overriding once the run starts.')),
        DeclareLaunchArgument('run_rviz', default_value='true'),
        DeclareLaunchArgument('start_on_launch', default_value='false',
                              description='auto-arm executor 3s after launch'),
        DeclareLaunchArgument('yolo_model', default_value='yoloe-11s-seg.pt',
                              description='YOLOE weights or exported .engine path'),
        DeclareLaunchArgument('vlm_model', default_value='gemini-2.5-flash',
                              description='Gemini model id (OpenAI-compatible endpoint)'),
        DeclareLaunchArgument('vlm_api_key_env', default_value='GEMINI_API_KEY',
                              description='env var name holding the Gemini API key'),
        DeclareLaunchArgument('instruction_topic', default_value='/instruction/text',
                              description='topic the planner subscribes to'),
    ]

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('stretch_nav2'),
            '/launch/navigation.launch.py',
        ]),
        launch_arguments={'map': map_arg}.items(),
    )

    navigation = Node(
        package='exploration_rearrangement', executable='navigation_node',
        name='navigation_coordinator', output='screen',
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
            'enable_head': False,
        }],
    )

    visual_grasp = Node(
        package='exploration_rearrangement', executable='visual_grasp_node',
        name='visual_grasp_node', output='screen',
    )

    regions = Node(
        package='exploration_rearrangement', executable='region_manager_node',
        name='region_manager_node', output='screen',
        parameters=[{'regions_yaml': regions_yaml}],
    )

    planner = Node(
        package='exploration_rearrangement', executable='task_planner_node',
        name='task_planner_node', output='screen',
        parameters=[{
            'regions_yaml': regions_yaml,
            'vlm_model': vlm_model,
            'vlm_api_key_env': vlm_api_key_env,
            'instruction_topic': instruction_topic,
            'objects_snapshot_yaml': objects_snapshot,
        }],
    )

    executor = Node(
        package='exploration_rearrangement', executable='task_executor_node',
        name='task_executor_node', output='screen',
        parameters=[{'start_on_launch': start_on_launch}],
    )

    manipulation = Node(
        package='exploration_rearrangement', executable='manipulation_node',
        name='manipulation_node', output='screen',
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_cfg], output='log',
        condition=IfCondition(run_rviz),
    )

    return LaunchDescription(args + [
        nav2,
        navigation,
        detector, fine_detector,
        regions,
        planner, executor,
        manipulation, visual_grasp,
        rviz,
    ])
