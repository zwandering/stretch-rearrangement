"""RGB-D object detection via HSV segmentation → 3D map poses."""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .utils.color_segmentation import (
    Detection2D,
    annotate,
    load_color_specs,
    pixel_to_camera,
    segment_all,
)
from .utils.transform_utils import transform_point_to_frame


class ObjectDetectorNode(Node):

    def __init__(self) -> None:
        super().__init__('object_detector_node')

        self.declare_parameter('objects_yaml', '')
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('info_topic', '/camera/color/camera_info')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('merge_dist_m', 0.3)
        self.declare_parameter('ema_alpha', 0.3)
        self.declare_parameter('publish_debug_image', True)

        yaml_path = self.get_parameter('objects_yaml').value
        if not yaml_path:
            self.get_logger().warn('objects_yaml not set; using defaults')
            self.specs = load_color_specs({'objects': _DEFAULT_OBJECTS})
        else:
            with open(Path(yaml_path), 'r') as f:
                self.specs = load_color_specs(yaml.safe_load(f))
        self.get_logger().info(
            f'Loaded {len(self.specs)} color specs: '
            f'{[s.name for s in self.specs]}'
        )

        self.camera_frame = self.get_parameter('camera_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.merge_dist = float(self.get_parameter('merge_dist_m').value)
        self.ema = float(self.get_parameter('ema_alpha').value)
        self.pub_debug = bool(self.get_parameter('publish_debug_image').value)

        self.bridge = CvBridge()
        self.camera_info: Optional[CameraInfo] = None
        self.objects: Dict[str, PoseStamped] = {}
        self.last_debug_image: Optional[np.ndarray] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cb = ReentrantCallbackGroup()
        # RealSense publishes image streams with BEST_EFFORT reliability; match it.
        self.create_subscription(
            CameraInfo, self.get_parameter('info_topic').value,
            self._on_info, qos_profile_sensor_data, callback_group=cb,
        )
        rgb_sub = Subscriber(
            self, Image, self.get_parameter('rgb_topic').value,
            qos_profile=qos_profile_sensor_data,
        )
        depth_sub = Subscriber(
            self, Image, self.get_parameter('depth_topic').value,
            qos_profile=qos_profile_sensor_data,
        )
        self.sync = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=5, slop=0.1,
        )
        self.sync.registerCallback(self._on_rgbd)

        self.marker_pub = self.create_publisher(
            MarkerArray, '/detected_objects', 10,
        )
        self.debug_img_pub = self.create_publisher(
            Image, '/detector/debug_image', 2,
        )

        self.create_service(
            Trigger, '/detector/clear', self._on_clear, callback_group=cb,
        )
        self.create_timer(1.0, self._publish_markers, callback_group=cb)
        self.get_logger().info('ObjectDetectorNode ready.')

    def _on_info(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _on_clear(self, req, res):
        self.objects.clear()
        res.success = True
        res.message = 'cleared'
        return res

    def _on_rgbd(self, rgb: Image, depth: Image) -> None:
        if self.camera_info is None:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
            if depth.encoding in ('16UC1', 'mono16'):
                depth_img = self.bridge.imgmsg_to_cv2(
                    depth, desired_encoding='16UC1',
                )
            else:
                depth_img = self.bridge.imgmsg_to_cv2(
                    depth, desired_encoding='passthrough',
                )
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        dets: List[Detection2D] = segment_all(bgr, self.specs)
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        for d in dets:
            cam_xyz = pixel_to_camera(
                depth_img, d.center_px[0], d.center_px[1], fx, fy, cx, cy,
            )
            if cam_xyz is None:
                continue
            map_xyz = transform_point_to_frame(
                self.tf_buffer, cam_xyz,
                self.camera_frame, self.map_frame, rgb.header.stamp,
            )
            if map_xyz is None:
                continue
            self._update_object(d.label, map_xyz, rgb.header.stamp)

        if self.pub_debug:
            debug_bgr = annotate(bgr, dets)
            self.last_debug_image = debug_bgr
            msg = self.bridge.cv2_to_imgmsg(debug_bgr, encoding='bgr8')
            msg.header = rgb.header
            self.debug_img_pub.publish(msg)

    def _update_object(self, label: str, xyz, stamp) -> None:
        ps = PoseStamped()
        ps.header.frame_id = self.map_frame
        ps.header.stamp = stamp
        if label in self.objects:
            prev = self.objects[label].pose.position
            dx = xyz[0] - prev.x
            dy = xyz[1] - prev.y
            if np.hypot(dx, dy) < self.merge_dist:
                ps.pose.position.x = (1 - self.ema) * prev.x + self.ema * xyz[0]
                ps.pose.position.y = (1 - self.ema) * prev.y + self.ema * xyz[1]
                ps.pose.position.z = (1 - self.ema) * prev.z + self.ema * xyz[2]
            else:
                ps.pose.position.x = float(xyz[0])
                ps.pose.position.y = float(xyz[1])
                ps.pose.position.z = float(xyz[2])
        else:
            ps.pose.position.x = float(xyz[0])
            ps.pose.position.y = float(xyz[1])
            ps.pose.position.z = float(xyz[2])
        ps.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.objects[label] = ps

    def _publish_markers(self) -> None:
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        for i, (label, ps) in enumerate(self.objects.items()):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = label
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose = ps.pose
            m.scale.x = m.scale.y = m.scale.z = 0.1
            m.color = _color_for(label)
            ma.markers.append(m)

            txt = Marker()
            txt.header = m.header
            txt.ns = label + '_label'
            txt.id = i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose = ps.pose
            txt.pose.position.z += 0.2
            txt.scale.z = 0.1
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            txt.text = label
            ma.markers.append(txt)
        self.marker_pub.publish(ma)


def _color_for(label: str) -> ColorRGBA:
    palette = {
        'blue_bottle': ColorRGBA(r=0.1, g=0.3, b=1.0, a=1.0),
        'red_box':     ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0),
        'yellow_cup':  ColorRGBA(r=1.0, g=0.9, b=0.1, a=1.0),
    }
    return palette.get(label, ColorRGBA(r=0.7, g=0.7, b=0.7, a=1.0))


_DEFAULT_OBJECTS = [
    {'name': 'blue_bottle', 'hsv_low': [100, 120, 60], 'hsv_high': [130, 255, 255]},
    {'name': 'red_box', 'hsv_low': [0, 120, 70], 'hsv_high': [10, 255, 255],
     'hsv_low_2': [170, 120, 70], 'hsv_high_2': [180, 255, 255]},
    {'name': 'yellow_cup', 'hsv_low': [20, 120, 120], 'hsv_high': [35, 255, 255]},
]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
