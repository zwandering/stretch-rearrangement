"""End-to-end sim launch — fake_sim_node replaces SLAM / Nav2 / stretch_driver.

Runs the real detector, planner, and executor on top of synthetic TF, map,
camera, Nav2 action and manipulation actions. Produces
/tmp/rearrangement_metrics.json when the state machine finishes.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('exploration_rearrangement')
    regions_yaml = PathJoinSubstitution([pkg, 'config', 'regions.yaml'])
    objects_yaml = PathJoinSubstitution([pkg, 'config', 'objects.yaml'])
    tasks_yaml = PathJoinSubstitution([pkg, 'config', 'tasks.yaml'])

    planner_backend = LaunchConfiguration('planner_backend')
    start_on_launch = LaunchConfiguration('start_on_launch')

    args = [
        DeclareLaunchArgument('planner_backend', default_value='greedy',
                              description='greedy | vlm'),
        DeclareLaunchArgument('start_on_launch', default_value='true'),
    ]

    sim = Node(
        package='exploration_rearrangement', executable='fake_sim_node',
        name='fake_sim_node', output='screen',
    )
    detector = Node(
        package='exploration_rearrangement', executable='object_detector_node',
        name='object_detector_node', output='screen',
        parameters=[{
            'objects_yaml': objects_yaml,
            'merge_dist_m': 0.6,
            'ema_alpha': 0.5,
        }],
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
            'explore_timeout_s': 20.0,
            'wait_after_explore_s': 3.0,
            'min_objects_required': 3,
            'pick_standoff_m': 0.6,
        }],
    )
    exploration = Node(
        package='exploration_rearrangement', executable='exploration_node',
        name='exploration_node', output='screen',
        parameters=[{
            'enabled_on_start': False,
            'min_cluster_size': 8,
        }],
    )

    return LaunchDescription(args + [
        sim, detector, regions, planner, executor, exploration,
    ])
