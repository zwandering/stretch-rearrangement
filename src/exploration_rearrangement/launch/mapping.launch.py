"""Stage 1: offline mapping + object detection.

Operator drives the robot manually while ``stretch_nav2 offline_mapping``
builds a SLAM map and the YOLOE detector latches every target object's
``map``-frame position. When the map is good and every object has been seen,
the operator runs:

    ros2 service call /detector/snapshot std_srvs/srv/Trigger
    ros2 run nav2_map_server map_saver_cli -f <name>

to persist a ``<name>.{pgm,yaml}`` and the matching object snapshot YAML.
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
    rviz_cfg = PathJoinSubstitution([pkg, 'rviz', 'rearrangement.rviz'])

    run_rviz = LaunchConfiguration('run_rviz')
    yolo_model = LaunchConfiguration('yolo_model')
    objects_snapshot = LaunchConfiguration('objects_snapshot')

    args = [
        DeclareLaunchArgument('run_rviz', default_value='true'),
        DeclareLaunchArgument('yolo_model', default_value='yoloe-11s-seg.pt',
                              description='YOLOE weights or exported .engine path'),
        DeclareLaunchArgument(
            'objects_snapshot',
            default_value='/tmp/objects_snapshot.yaml',
            description=('Path the detector writes its EMA-smoothed object '
                         'positions to when /detector/snapshot is called '
                         '(or on shutdown).'),
        ),
    ]

    offline_mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('stretch_nav2'),
            '/launch/offline_mapping.launch.py',
        ]),
    )

    detector = Node(
        package='exploration_rearrangement', executable='object_detector_node',
        name='object_detector_node', output='screen',
        parameters=[{
            'mode': 'robot',
            'objects_yaml': objects_yaml,
            'model_path': yolo_model,
            'objects_snapshot_path': objects_snapshot,
        }],
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_cfg], output='log',
        condition=IfCondition(run_rviz),
    )

    return LaunchDescription(args + [offline_mapping, detector, rviz])
