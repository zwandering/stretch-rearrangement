"""TF2 / geometry helpers."""

from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return 0.0, 0.0, float(np.sin(half)), float(np.cos(half))


def lookup_pose(
    tf_buffer: Buffer,
    target_frame: str,
    source_frame: str,
    timeout_sec: float = 0.5,
) -> Optional[TransformStamped]:
    try:
        return tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=timeout_sec),
        )
    except TransformException:
        return None


def robot_pose_in_map(
    node: Node,
    tf_buffer: Buffer,
    base_frame: str = 'base_link',
    map_frame: str = 'map',
) -> Optional[PoseStamped]:
    tf = lookup_pose(tf_buffer, map_frame, base_frame)
    if tf is None:
        return None
    ps = PoseStamped()
    ps.header.stamp = node.get_clock().now().to_msg()
    ps.header.frame_id = map_frame
    ps.pose.position.x = tf.transform.translation.x
    ps.pose.position.y = tf.transform.translation.y
    ps.pose.position.z = tf.transform.translation.z
    ps.pose.orientation = tf.transform.rotation
    return ps


def transform_point_to_frame(
    tf_buffer: Buffer,
    point_xyz: Tuple[float, float, float],
    source_frame: str,
    target_frame: str,
    stamp=None,
) -> Optional[Tuple[float, float, float]]:
    from geometry_msgs.msg import PointStamped
    pt = PointStamped()
    pt.header.frame_id = source_frame
    pt.header.stamp = stamp if stamp is not None else rclpy.time.Time().to_msg()
    pt.point.x = float(point_xyz[0])
    pt.point.y = float(point_xyz[1])
    pt.point.z = float(point_xyz[2])
    try:
        tf = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=0.5),
        )
    except TransformException:
        return None
    out = do_transform_point(pt, tf)
    return out.point.x, out.point.y, out.point.z


def euclidean_2d(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
