"""Stage 1 — detection nodes only (Terminal 3 of 3).

Stage 1 runs as three independent terminals so each subsystem can be
restarted on its own while teleop-mapping:

  Terminal 1:  ros2 launch stretch_nav2 offline_mapping.launch.py
               (this already includes the Stretch driver — do NOT also
                start ``stretch_core stretch_driver.launch.py``)

  Terminal 2:  ros2 launch realsense2_camera rs_launch.py \\
                   enable_color:=true enable_depth:=true \\
                   align_depth.enable:=true pointcloud.enable:=true
               (or whichever realsense launch you normally use onboard)

  Terminal 3:  ros2 launch exploration_rearrangement mapping.launch.py
               (this file — runs the YOLOE detector that latches each
                target object's map-frame position)

When the SLAM map looks complete and every target object has been seen
by the detector at least once, the operator runs:

    ros2 service call /detector/snapshot std_srvs/srv/Trigger
    ros2 run nav2_map_server map_saver_cli -f <name>

to persist ``<name>.{pgm,yaml}`` and the matching object snapshot YAML.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('exploration_rearrangement')
    objects_yaml = PathJoinSubstitution([pkg, 'config', 'objects.yaml'])

    yolo_model = LaunchConfiguration('yolo_model')
    objects_snapshot = LaunchConfiguration('objects_snapshot')

    args = [
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

    return LaunchDescription(args + [detector])
