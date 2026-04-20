"""Dual-camera fine-detection node for close-range manipulation.

Sits idle until the task executor tells it to start. Activation is a
``std_msgs/Bool`` on ``/fine_detector/activate``:
  * True  → start running YOLOE on every RGB-D pair from BOTH the D405
            gripper camera AND the D435i head camera,
  * False → stop publishing and release CPU.

While active, each camera stream is processed independently. They share
the same YOLOE model (inference calls are serialized by a lock) but
maintain their own per-class EMA-smoothed 3D state and their own set of
output topics:
  * Gripper:  /fine_detector/objects, /fine_detector/bboxes_3d,
              /fine_detector/bboxes_3d_markers, /fine_detector/debug_image
  * Head:     /fine_detector/head_objects, /fine_detector/head_bboxes_3d,
              /fine_detector/head_bboxes_3d_markers,
              /fine_detector/head_debug_image

Both streams default to ``base_link`` as output frame so the manipulation
consumer can cross-reference them without a SLAM dependency. Set
``enable_head:=false`` to fall back to gripper-only operation.

The node also subscribes to ``/fine_detector/target_object``
(``std_msgs/String``). If set to a known class name, both cameras'
published output is filtered down to just that class — detection still
runs over every prompt internally. Sending an empty string clears the
filter; sending an unknown name is rejected with a warning and the
previous target is kept. The valid name set is the ``objects:`` list in
``config/objects.yaml`` (the same list baked into the exported YOLOE
model at ``set_up_yolo_e`` time).

This node is deliberately separate from ``object_detector_node``:
  * different activation window (only pick/place phases — no idle CPU
    burn during navigation),
  * different output frame (``base_link``, not ``map``),
  * no cross-class 3D dedup — both cameras rely on per-class best-conf.
    The full-scene, map-frame, dedup-heavy pipeline stays in
    ``object_detector_node``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import (
    BoundingBox3D, BoundingBox3DArray,
    Detection3D, Detection3DArray, ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray

from .object_detector_node import (
    _Det,
    _annotate,
    _bbox3d_line_list_points,
    _color_for,
    _load_objects,
    _load_yolo_model,
    _parse_yolo_tracks,
)
from .utils.depth_projection import (
    estimate_bbox_3d, estimate_bbox_3d_from_mask, pixel_to_camera,
)
from .utils.transform_utils import lookup_pose, transform_point_to_frame


# D405 on gripper (robot) / standalone realsense on bench (debug).
_GRIPPER_ROBOT_TOPICS = {
    'rgb':   '/gripper_camera/color/image_rect_raw',
    'depth': '/gripper_camera/aligned_depth_to_color/image_raw',
    'info':  '/gripper_camera/color/camera_info',
    'frame': 'gripper_camera_color_optical_frame',
}
_GRIPPER_DEBUG_TOPICS = {
    'rgb':   '/camera/camera/color/image_raw',
    'depth': '/camera/camera/aligned_depth_to_color/image_raw',
    'info':  '/camera/camera/color/camera_info',
    'frame': 'camera_color_optical_frame',
}
# D435i on head (robot). Stretch driver exposes it under /camera/* (not the
# realsense2_camera default /camera/camera/*). In debug we only have a
# single realsense bench bringup, so fall back to the same topics as the
# gripper debug — in that case set enable_head:=false, or override the
# head_*_topic params to point at a second camera.
_HEAD_ROBOT_TOPICS = {
    'rgb':   '/camera/color/image_raw',
    'depth': '/camera/aligned_depth_to_color/image_raw',
    'info':  '/camera/color/camera_info',
    'frame': 'camera_color_optical_frame',
}
_HEAD_DEBUG_TOPICS = _GRIPPER_DEBUG_TOPICS


def _gripper_topics_for(mode: str) -> Dict[str, str]:
    return _GRIPPER_DEBUG_TOPICS if mode == 'debug' else _GRIPPER_ROBOT_TOPICS


def _head_topics_for(mode: str) -> Dict[str, str]:
    return _HEAD_DEBUG_TOPICS if mode == 'debug' else _HEAD_ROBOT_TOPICS


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


@dataclass
class _CamSource:
    """Per-camera input topics + output publishers + 3D state."""
    tag: str
    rgb_topic: str
    depth_topic: str
    info_topic: str
    camera_frame: str
    output_frame: str
    camera_info: Optional[CameraInfo] = None
    smoothed: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    smoothed_sizes: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    orientations: Dict[str, Quaternion] = field(default_factory=dict)
    last_confs: Dict[str, float] = field(default_factory=dict)
    tf_warn_once: bool = False
    detections_pub: Any = None
    bboxes_pub: Any = None
    bbox_markers_pub: Any = None
    debug_img_pub: Any = None
    conf_th: float = 0.0


class FineObjectDetectorNode(Node):

    def __init__(self) -> None:
        super().__init__('fine_object_detector_node')

        self.declare_parameter('mode', 'robot')
        self.declare_parameter('model_path', 'yoloe-11s-seg.pt')
        self.declare_parameter('objects_yaml', '')
        self.declare_parameter('conf_threshold', 0.30)
        self.declare_parameter('head_conf_threshold', 0.10)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('device', '')
        self.declare_parameter('ema_alpha', 1.0)
        self.declare_parameter('merge_dist_m', 0.3)  # teleport-replace threshold
        self.declare_parameter('bbox_line_width', 0.005)
        self.declare_parameter('clear_state_on_activate', True)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('enable_head', True)

        self.declare_parameter('activate_topic', '/fine_detector/activate')
        self.declare_parameter('target_object_topic', '/fine_detector/target_object')
        self.declare_parameter('detections_topic', '/fine_detector/objects')
        self.declare_parameter('bboxes_topic', '/fine_detector/bboxes_3d')
        self.declare_parameter('bbox_markers_topic', '/fine_detector/bboxes_3d_markers')
        self.declare_parameter('debug_image_topic', '/fine_detector/debug_image')
        self.declare_parameter('head_detections_topic', '/fine_detector/head_objects')
        self.declare_parameter('head_bboxes_topic', '/fine_detector/head_bboxes_3d')
        self.declare_parameter('head_bbox_markers_topic', '/fine_detector/head_bboxes_3d_markers')
        self.declare_parameter('head_debug_image_topic', '/fine_detector/head_debug_image')

        self.mode = str(self.get_parameter('mode').value).lower()
        if self.mode not in ('robot', 'debug'):
            self.get_logger().warn(f"unknown mode '{self.mode}', defaulting to 'robot'")
            self.mode = 'robot'

        gripper_defaults = _gripper_topics_for(self.mode)
        head_defaults = _head_topics_for(self.mode)
        self.declare_parameter('rgb_topic',         gripper_defaults['rgb'])
        self.declare_parameter('depth_topic',       gripper_defaults['depth'])
        self.declare_parameter('info_topic',        gripper_defaults['info'])
        self.declare_parameter('camera_frame',      gripper_defaults['frame'])
        self.declare_parameter('head_rgb_topic',    head_defaults['rgb'])
        self.declare_parameter('head_depth_topic',  head_defaults['depth'])
        self.declare_parameter('head_info_topic',   head_defaults['info'])
        self.declare_parameter('head_camera_frame', head_defaults['frame'])
        default_out = 'base_link' if self.mode == 'robot' else gripper_defaults['frame']
        head_default_out = 'base_link' if self.mode == 'robot' else head_defaults['frame']
        self.declare_parameter('output_frame', default_out)
        self.declare_parameter('head_output_frame', head_default_out)

        self.conf_th = float(self.get_parameter('conf_threshold').value)
        self.iou_th = float(self.get_parameter('iou_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.ema = float(self.get_parameter('ema_alpha').value)
        self.merge_dist = float(self.get_parameter('merge_dist_m').value)
        self.bbox_line_width = float(self.get_parameter('bbox_line_width').value)
        self.clear_on_activate = bool(self.get_parameter('clear_state_on_activate').value)
        self.pub_debug = bool(self.get_parameter('publish_debug_image').value)
        self.enable_head = bool(self.get_parameter('enable_head').value)
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
        # Gripper + head share one YOLOE model; serialize inference so the
        # two RGBD callbacks never call predict() concurrently.
        self._model_lock = threading.Lock()

        self.bridge = CvBridge()
        self.active: bool = False
        self.known_names = {o['name'] for o in self.object_defs}
        self.target_name: Optional[str] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cb = ReentrantCallbackGroup()

        # --- Build per-camera sources -------------------------------------
        self.sources: List[_CamSource] = []

        gripper_src = _CamSource(
            tag='gripper',
            rgb_topic=str(self.get_parameter('rgb_topic').value),
            depth_topic=str(self.get_parameter('depth_topic').value),
            info_topic=str(self.get_parameter('info_topic').value),
            camera_frame=str(self.get_parameter('camera_frame').value),
            output_frame=str(self.get_parameter('output_frame').value),
        )
        gripper_src.detections_pub = self.create_publisher(
            Detection3DArray, str(self.get_parameter('detections_topic').value), 10,
        )
        gripper_src.bboxes_pub = self.create_publisher(
            BoundingBox3DArray, str(self.get_parameter('bboxes_topic').value), 10,
        )
        gripper_src.bbox_markers_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter('bbox_markers_topic').value), 10,
        )
        gripper_src.debug_img_pub = self.create_publisher(
            Image, str(self.get_parameter('debug_image_topic').value), 2,
        )
        gripper_src.conf_th = self.conf_th
        self.sources.append(gripper_src)

        if self.enable_head:
            head_src = _CamSource(
                tag='head',
                rgb_topic=str(self.get_parameter('head_rgb_topic').value),
                depth_topic=str(self.get_parameter('head_depth_topic').value),
                info_topic=str(self.get_parameter('head_info_topic').value),
                camera_frame=str(self.get_parameter('head_camera_frame').value),
                output_frame=str(self.get_parameter('head_output_frame').value),
            )
            head_src.detections_pub = self.create_publisher(
                Detection3DArray, str(self.get_parameter('head_detections_topic').value), 10,
            )
            head_src.bboxes_pub = self.create_publisher(
                BoundingBox3DArray, str(self.get_parameter('head_bboxes_topic').value), 10,
            )
            head_src.bbox_markers_pub = self.create_publisher(
                MarkerArray, str(self.get_parameter('head_bbox_markers_topic').value), 10,
            )
            head_src.debug_img_pub = self.create_publisher(
                Image, str(self.get_parameter('head_debug_image_topic').value), 2,
            )
            head_src.conf_th = float(self.get_parameter('head_conf_threshold').value)
            self.sources.append(head_src)

        # Synchronizers must be stored, otherwise they get garbage-collected.
        self._syncs: List[ApproximateTimeSynchronizer] = []
        for src in self.sources:
            self.create_subscription(
                CameraInfo, src.info_topic,
                lambda msg, s=src: self._on_info(s, msg),
                qos_profile_sensor_data, callback_group=cb,
            )
            rgb_sub = Subscriber(
                self, Image, src.rgb_topic, qos_profile=qos_profile_sensor_data,
            )
            depth_sub = Subscriber(
                self, Image, src.depth_topic, qos_profile=qos_profile_sensor_data,
            )
            sync = ApproximateTimeSynchronizer(
                [rgb_sub, depth_sub], queue_size=5, slop=0.1,
            )
            sync.registerCallback(
                lambda rgb, depth, s=src: self._on_rgbd(s, rgb, depth)
            )
            self._syncs.append(sync)

        activate_topic = str(self.get_parameter('activate_topic').value)
        self.create_subscription(
            Bool, activate_topic, self._on_activate, 10, callback_group=cb,
        )
        target_topic = str(self.get_parameter('target_object_topic').value)
        self.create_subscription(
            String, target_topic, self._on_target_object, 10, callback_group=cb,
        )

        tags = ','.join(s.tag for s in self.sources)
        self.get_logger().info(
            f'fine_object_detector_node ready | mode={self.mode} sources=[{tags}] '
            f'activate on {activate_topic}, target on {target_topic}'
        )

    # --- callbacks --------------------------------------------------------

    def _on_info(self, src: _CamSource, msg: CameraInfo) -> None:
        src.camera_info = msg

    def _on_activate(self, msg: Bool) -> None:
        want = bool(msg.data)
        if want and not self.active:
            if self.clear_on_activate:
                for src in self.sources:
                    src.smoothed.clear()
                    src.smoothed_sizes.clear()
                    src.orientations.clear()
                    src.last_confs.clear()
                    src.tf_warn_once = False
            self.active = True
            self.get_logger().info(
                f'fine detector: activated ({len(self.sources)} source(s))'
            )
        elif not want and self.active:
            self.active = False
            self.get_logger().info('fine detector: deactivated')

    def _on_target_object(self, msg: String) -> None:
        name = (msg.data or '').strip()
        if not name:
            if self.target_name is not None:
                self.get_logger().info('fine detector: target filter cleared')
            self.target_name = None
            return
        if name not in self.known_names:
            self.get_logger().warn(
                f"fine detector: target '{name}' not in exported-model class list "
                f"{sorted(self.known_names)} — keeping previous target "
                f"{self.target_name!r}"
            )
            return
        if name != self.target_name:
            self.get_logger().info(f"fine detector: target set to '{name}'")
        self.target_name = name

    def _on_rgbd(self, src: _CamSource, rgb: Image, depth: Image) -> None:
        if not self.active or src.camera_info is None:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
            if depth.encoding in ('16UC1', 'mono16'):
                depth_img = self.bridge.imgmsg_to_cv2(depth, desired_encoding='16UC1')
            else:
                depth_img = self.bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'[{src.tag}] cv_bridge error: {e}')
            return

        try:
            with self._model_lock:
                results = self.model.predict(
                    bgr, conf=src.conf_th, iou=self.iou_th,
                    imgsz=self.imgsz, verbose=False,
                )
        except Exception as e:
            self.get_logger().warn(f'[{src.tag}] yolo predict error: {e}')
            return

        dets = _parse_yolo_tracks(results, self.prompts, self.prompt_to_name)
        best_per_name: Dict[str, _Det] = {}
        for d in dets:
            cur = best_per_name.get(d.name)
            if cur is None or d.conf > cur.conf:
                best_per_name[d.name] = d

        fx = src.camera_info.k[0]
        fy = src.camera_info.k[4]
        cx0 = src.camera_info.k[2]
        cy0 = src.camera_info.k[5]

        cam_to_out_q: Optional[Quaternion] = None
        if src.output_frame != src.camera_frame:
            tf = lookup_pose(
                self.tf_buffer, src.output_frame, src.camera_frame,
                timeout_sec=0.1,
            )
            if tf is not None:
                cam_to_out_q = tf.transform.rotation

        for name, d in best_per_name.items():
            u, v = d.center
            if d.mask is not None:
                bbox_xyz = estimate_bbox_3d_from_mask(
                    depth_img, d.mask, fx, fy, cx0, cy0,
                )
            else:
                bbox_xyz = estimate_bbox_3d(
                    depth_img, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
                    fx, fy, cx0, cy0,
                )
            if bbox_xyz is not None:
                cam_xyz = bbox_xyz[0]
            else:
                cam_xyz = pixel_to_camera(depth_img, u, v, fx, fy, cx0, cy0)
                if cam_xyz is None:
                    continue
            if src.output_frame == src.camera_frame:
                world_xyz = cam_xyz
            else:
                world_xyz = transform_point_to_frame(
                    self.tf_buffer, cam_xyz,
                    src.camera_frame, src.output_frame, rgb.header.stamp,
                )
                if world_xyz is None:
                    if not src.tf_warn_once:
                        self.get_logger().warn(
                            f'[{src.tag}] TF {src.camera_frame}→{src.output_frame} unavailable'
                        )
                        src.tf_warn_once = True
                    continue
            src.smoothed[name] = _ema_update(
                src.smoothed.get(name), world_xyz, self.ema,
                teleport_dist=self.merge_dist,
            )
            src.last_confs[name] = float(d.conf)

            if bbox_xyz is not None:
                src.smoothed_sizes[name] = _ema_update(
                    src.smoothed_sizes.get(name), bbox_xyz[1], self.ema,
                )
            if cam_to_out_q is not None:
                src.orientations[name] = cam_to_out_q

        self._publish_detections(src, rgb.header.stamp)
        self._publish_bboxes_3d(src, rgb.header.stamp)

        if self.pub_debug:
            out_msg = self.bridge.cv2_to_imgmsg(_annotate(bgr, dets), encoding='bgr8')
            out_msg.header = rgb.header
            src.debug_img_pub.publish(out_msg)

    # --- publishers -------------------------------------------------------

    def _publish_detections(self, src: _CamSource, stamp) -> None:
        arr = Detection3DArray()
        arr.header.stamp = stamp
        arr.header.frame_id = src.output_frame
        for name, xyz in src.smoothed.items():
            if self.target_name is not None and name != self.target_name:
                continue
            det = Detection3D()
            det.header = arr.header
            det.id = name
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = name
            hyp.hypothesis.score = float(src.last_confs.get(name, 0.0))
            hyp.pose.pose.position.x = float(xyz[0])
            hyp.pose.pose.position.y = float(xyz[1])
            hyp.pose.pose.position.z = float(xyz[2])
            orient = src.orientations.get(name, Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
            hyp.pose.pose.orientation = orient
            det.results.append(hyp)
            det.bbox.center = hyp.pose.pose
            size = src.smoothed_sizes.get(name)
            if size is not None:
                det.bbox.size = Vector3(x=size[0], y=size[1], z=size[2])
            arr.detections.append(det)
        src.detections_pub.publish(arr)

    def _publish_bboxes_3d(self, src: _CamSource, stamp) -> None:
        arr = BoundingBox3DArray()
        arr.header.stamp = stamp
        arr.header.frame_id = src.output_frame
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        marker_id = 0
        for name, xyz in src.smoothed.items():
            if self.target_name is not None and name != self.target_name:
                continue
            size = src.smoothed_sizes.get(name)
            if size is None:
                continue
            box = BoundingBox3D()
            box.center.position.x = float(xyz[0])
            box.center.position.y = float(xyz[1])
            box.center.position.z = float(xyz[2])
            box.center.orientation = src.orientations.get(
                name, Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            )
            box.size = Vector3(x=size[0], y=size[1], z=size[2])
            arr.boxes.append(box)

            m = Marker()
            m.header.frame_id = src.output_frame
            m.header.stamp = stamp
            m.ns = f'{name}_{src.tag}_fine_bbox3d'
            m.id = marker_id
            marker_id += 1
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.pose = box.center
            m.scale.x = self.bbox_line_width
            m.color = _color_for(name)
            m.points = _bbox3d_line_list_points(size)
            markers.markers.append(m)
        src.bboxes_pub.publish(arr)
        src.bbox_markers_pub.publish(markers)


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
