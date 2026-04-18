"""Gripper-camera fine-detection node for close-range manipulation.

Sits idle until the task executor tells it to start. Activation is a
``std_msgs/Bool`` on ``/fine_detector/activate``:
  * True  → start running YOLOE on every RGB-D pair from the D405,
  * False → stop publishing and release CPU.

While active, every frame goes through YOLOE; the highest-confidence
detection per known class is back-projected into 3D, transformed into
``base_link`` (default ``output_frame``), EMA-smoothed per class, and
published as a ``Detection3DArray`` on ``/fine_detector/objects`` — giving
the manipulation node an accurate, constantly-refreshed pose for whichever
object sits in the gripper's field of view.

This node is deliberately separate from ``object_detector_node``:
  * different camera (D405 on gripper, vs D435i on head),
  * different output frame (``base_link`` so manipulation uses the pose
    directly with no SLAM dependency at arm-extension time),
  * only active during the pick/place phases — no idle CPU burn during nav.

No 3D dedup is applied here: the wrist camera's field of view is tight and
the close-range perspective makes cross-class confusion (which motivated
the head detector's dedup) unlikely; per-class best-conf is enough.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from .object_detector_node import (
    _Det,
    _annotate,
    _load_objects,
    _load_yolo_model,
    _parse_yolo_tracks,
)
from .utils.depth_projection import pixel_to_camera
from .utils.transform_utils import transform_point_to_frame


# Stretch 3's gripper D405 sits under /gripper_camera in the stretch_core
# launch; debug falls back to the same standalone realsense bringup that the
# head detector's debug mode uses (so a single D435i on a bench still works).
_ROBOT_TOPICS = {
    'rgb':   '/gripper_camera/color/image_raw',
    'depth': '/gripper_camera/aligned_depth_to_color/image_raw',
    'info':  '/gripper_camera/color/camera_info',
    'frame': 'gripper_camera_color_optical_frame',
}
_DEBUG_TOPICS = {
    'rgb':   '/camera/camera/color/image_raw',
    'depth': '/camera/camera/aligned_depth_to_color/image_raw',
    'info':  '/camera/camera/color/camera_info',
    'frame': 'camera_color_optical_frame',
}


def _topics_for(mode: str) -> Dict[str, str]:
    return _DEBUG_TOPICS if mode == 'debug' else _ROBOT_TOPICS


def _ema_update(
    prev: Optional[Tuple[float, float, float]],
    new: Tuple[float, float, float],
    alpha: float,
    teleport_dist: float = 0.0,
) -> Tuple[float, float, float]:
    """Blend prev → new by `alpha`. If the xy step exceeds `teleport_dist`
    (>0), replace instead of smoothing — matches the head detector's
    teleport-recovery behaviour."""
    new_t = (float(new[0]), float(new[1]), float(new[2]))
    if prev is None:
        return new_t
    if teleport_dist > 0.0:
        dx = new_t[0] - prev[0]
        dy = new_t[1] - prev[1]
        if (dx * dx + dy * dy) ** 0.5 >= teleport_dist:
            return new_t
    a = alpha
    return (
        (1 - a) * prev[0] + a * new_t[0],
        (1 - a) * prev[1] + a * new_t[1],
        (1 - a) * prev[2] + a * new_t[2],
    )


class FineObjectDetectorNode(Node):

    def __init__(self) -> None:
        super().__init__('fine_object_detector_node')

        self.declare_parameter('mode', 'robot')
        self.declare_parameter('model_path', 'yoloe-11s-seg.pt')
        self.declare_parameter('objects_yaml', '')
        self.declare_parameter('conf_threshold', 0.30)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('device', '')
        self.declare_parameter('ema_alpha', 0.3)
        self.declare_parameter('merge_dist_m', 0.3)  # teleport-replace threshold
        self.declare_parameter('clear_state_on_activate', True)
        self.declare_parameter('publish_debug_image', True)

        self.declare_parameter('activate_topic', '/fine_detector/activate')
        self.declare_parameter('detections_topic', '/fine_detector/objects')
        self.declare_parameter('debug_image_topic', '/fine_detector/debug_image')

        self.mode = str(self.get_parameter('mode').value).lower()
        if self.mode not in ('robot', 'debug'):
            self.get_logger().warn(f"unknown mode '{self.mode}', defaulting to 'robot'")
            self.mode = 'robot'

        topic_defaults = _topics_for(self.mode)
        self.declare_parameter('rgb_topic',    topic_defaults['rgb'])
        self.declare_parameter('depth_topic',  topic_defaults['depth'])
        self.declare_parameter('info_topic',   topic_defaults['info'])
        self.declare_parameter('camera_frame', topic_defaults['frame'])
        default_out = 'base_link' if self.mode == 'robot' else topic_defaults['frame']
        self.declare_parameter('output_frame', default_out)

        self.camera_frame = str(self.get_parameter('camera_frame').value)
        self.output_frame = str(self.get_parameter('output_frame').value)
        self.conf_th = float(self.get_parameter('conf_threshold').value)
        self.iou_th = float(self.get_parameter('iou_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.ema = float(self.get_parameter('ema_alpha').value)
        self.merge_dist = float(self.get_parameter('merge_dist_m').value)
        self.clear_on_activate = bool(self.get_parameter('clear_state_on_activate').value)
        self.pub_debug = bool(self.get_parameter('publish_debug_image').value)
        device = str(self.get_parameter('device').value) or None

        objects_yaml = str(self.get_parameter('objects_yaml').value)
        self.object_defs, self.prompts, self.prompt_to_name = _load_objects(
            Path(objects_yaml) if objects_yaml else None, self.get_logger(),
        )
        self.get_logger().info(
            f'fine detector classes: {[o["name"] for o in self.object_defs]}'
        )

        self.model = _load_yolo_model(
            Path(self.get_parameter('model_path').value),
            self.prompts,
            device,
            self.get_logger(),
        )

        self.bridge = CvBridge()
        self.camera_info: Optional[CameraInfo] = None
        self.active: bool = False
        self.smoothed: Dict[str, Tuple[float, float, float]] = {}
        self.last_confs: Dict[str, float] = {}
        self._tf_warn_once = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cb = ReentrantCallbackGroup()

        rgb_topic = str(self.get_parameter('rgb_topic').value)
        depth_topic = str(self.get_parameter('depth_topic').value)
        info_topic = str(self.get_parameter('info_topic').value)
        self.create_subscription(
            CameraInfo, info_topic, self._on_info,
            qos_profile_sensor_data, callback_group=cb,
        )
        rgb_sub = Subscriber(self, Image, rgb_topic, qos_profile=qos_profile_sensor_data)
        depth_sub = Subscriber(self, Image, depth_topic, qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=5, slop=0.1,
        )
        self.sync.registerCallback(self._on_rgbd)

        activate_topic = str(self.get_parameter('activate_topic').value)
        self.create_subscription(
            Bool, activate_topic, self._on_activate, 10, callback_group=cb,
        )

        self.detections_pub = self.create_publisher(
            Detection3DArray, str(self.get_parameter('detections_topic').value), 10,
        )
        self.debug_img_pub = self.create_publisher(
            Image, str(self.get_parameter('debug_image_topic').value), 2,
        )

        self.get_logger().info(
            f'fine_object_detector_node ready | mode={self.mode} '
            f'rgb={rgb_topic} → {self.output_frame}, activate on {activate_topic}'
        )

    # --- callbacks --------------------------------------------------------

    def _on_info(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _on_activate(self, msg: Bool) -> None:
        want = bool(msg.data)
        if want and not self.active:
            if self.clear_on_activate:
                self.smoothed.clear()
                self.last_confs.clear()
            self._tf_warn_once = False
            self.active = True
            self.get_logger().info('fine detector: activated')
        elif not want and self.active:
            self.active = False
            self.get_logger().info('fine detector: deactivated')

    def _on_rgbd(self, rgb: Image, depth: Image) -> None:
        if not self.active or self.camera_info is None:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
            if depth.encoding in ('16UC1', 'mono16'):
                depth_img = self.bridge.imgmsg_to_cv2(depth, desired_encoding='16UC1')
            else:
                depth_img = self.bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        try:
            results = self.model.predict(
                bgr, conf=self.conf_th, iou=self.iou_th,
                imgsz=self.imgsz, verbose=False,
            )
        except Exception as e:
            self.get_logger().warn(f'yolo predict error: {e}')
            return

        dets = _parse_yolo_tracks(results, self.prompts, self.prompt_to_name)
        # Highest-confidence detection per known class; no cross-class 3D dedup.
        best_per_name: Dict[str, _Det] = {}
        for d in dets:
            cur = best_per_name.get(d.name)
            if cur is None or d.conf > cur.conf:
                best_per_name[d.name] = d

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx0 = self.camera_info.k[2]
        cy0 = self.camera_info.k[5]

        for name, d in best_per_name.items():
            u, v = d.center
            cam_xyz = pixel_to_camera(depth_img, u, v, fx, fy, cx0, cy0)
            if cam_xyz is None:
                continue
            if self.output_frame == self.camera_frame:
                world_xyz = cam_xyz
            else:
                world_xyz = transform_point_to_frame(
                    self.tf_buffer, cam_xyz,
                    self.camera_frame, self.output_frame, rgb.header.stamp,
                )
                if world_xyz is None:
                    if not self._tf_warn_once:
                        self.get_logger().warn(
                            f'TF {self.camera_frame}→{self.output_frame} unavailable'
                        )
                        self._tf_warn_once = True
                    continue
            self.smoothed[name] = _ema_update(
                self.smoothed.get(name), world_xyz, self.ema,
                teleport_dist=self.merge_dist,
            )
            self.last_confs[name] = float(d.conf)

        self._publish_detections(rgb.header.stamp)

        if self.pub_debug:
            out_msg = self.bridge.cv2_to_imgmsg(_annotate(bgr, dets), encoding='bgr8')
            out_msg.header = rgb.header
            self.debug_img_pub.publish(out_msg)

    # --- publishers -------------------------------------------------------

    def _publish_detections(self, stamp) -> None:
        arr = Detection3DArray()
        arr.header.stamp = stamp
        arr.header.frame_id = self.output_frame
        for name, xyz in self.smoothed.items():
            det = Detection3D()
            det.header = arr.header
            det.id = name
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = name
            hyp.hypothesis.score = float(self.last_confs.get(name, 0.0))
            hyp.pose.pose.position.x = float(xyz[0])
            hyp.pose.pose.position.y = float(xyz[1])
            hyp.pose.pose.position.z = float(xyz[2])
            hyp.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            det.results.append(hyp)
            det.bbox.center = hyp.pose.pose
            arr.detections.append(det)
        self.detections_pub.publish(arr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FineObjectDetectorNode()
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
