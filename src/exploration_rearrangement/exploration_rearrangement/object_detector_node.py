"""YOLOE-based RGB-D object detection + tracking → 3D poses.

Modes
-----
- ``robot`` (default): subscribe to the Stretch head D435i under
  ``/camera/...`` and publish object poses in ``map``.
- ``debug``:  subscribe to a standalone D435i launched via ``realsense2_camera``
  under ``/camera/camera/...`` and publish poses in
  ``camera_color_optical_frame`` (no SLAM/TF required). Use this for
  bench-side testing of the detector alone.

Model
-----
Uses the smallest YOLOE variant (``yoloe-11s-seg.pt``) by default. If
``model_path`` points to a ``.engine`` / ``.onnx`` file, we assume the text
prompts were baked in at export time (see ``set_up_yolo_e.py``); otherwise we
load the ``.pt`` weights and call ``set_classes`` at startup.

Tracking is Ultralytics' built-in tracker (BYTETrack) via
``model.track(..., persist=True)``. Each frame we pick the top-confidence
detection for every named object and update an EMA-smoothed 3D pose; the
running center record (per-object 2D pixel + 3D world + track id) is
published to ``/detector/centers`` every frame and can be appended to a
JSONL log for offline analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray, PoseStamped, Quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from .utils.depth_projection import pixel_to_camera
from .utils.transform_utils import transform_point_to_frame


# ---------------------------------------------------------------------------
# Per-mode topic defaults
# ---------------------------------------------------------------------------

_ROBOT_TOPICS = {
    'rgb':   '/camera/color/image_raw',
    'depth': '/camera/aligned_depth_to_color/image_raw',
    'info':  '/camera/color/camera_info',
    'frame': 'camera_color_optical_frame',
}
# realsense2_camera (ROS 2 Jazzy) default namespace: /camera/camera/...
_DEBUG_TOPICS = {
    'rgb':   '/camera/camera/color/image_raw',
    'depth': '/camera/camera/aligned_depth_to_color/image_raw',
    'info':  '/camera/camera/color/camera_info',
    'frame': 'camera_color_optical_frame',
}


def _topics_for(mode: str) -> Dict[str, str]:
    return _DEBUG_TOPICS if mode == 'debug' else _ROBOT_TOPICS


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ObjectDetectorNode(Node):

    def __init__(self) -> None:
        super().__init__('object_detector_node')

        # --- Parameters -----------------------------------------------------
        self.declare_parameter('mode', 'robot')                 # robot | debug
        self.declare_parameter('model_path', 'yoloe-11s-seg.pt')
        self.declare_parameter('objects_yaml', '')
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('tracker', 'bytetrack.yaml')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('device', '')                    # '' → auto
        self.declare_parameter('merge_dist_m', 0.3)
        self.declare_parameter('dedup_dist_m', 0.25)
        self.declare_parameter('ema_alpha', 0.3)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('center_log_path', '')           # JSONL if non-empty

        self.mode = str(self.get_parameter('mode').value).lower()
        if self.mode not in ('robot', 'debug'):
            self.get_logger().warn(f"unknown mode '{self.mode}', defaulting to 'robot'")
            self.mode = 'robot'

        topic_defaults = _topics_for(self.mode)
        self.declare_parameter('rgb_topic',    topic_defaults['rgb'])
        self.declare_parameter('depth_topic',  topic_defaults['depth'])
        self.declare_parameter('info_topic',   topic_defaults['info'])
        self.declare_parameter('camera_frame', topic_defaults['frame'])
        # In debug mode there's no /map; stay in camera frame.
        default_out = 'map' if self.mode == 'robot' else topic_defaults['frame']
        self.declare_parameter('output_frame', default_out)

        self.camera_frame = str(self.get_parameter('camera_frame').value)
        self.output_frame = str(self.get_parameter('output_frame').value)
        self.merge_dist = float(self.get_parameter('merge_dist_m').value)
        self.dedup_dist = float(self.get_parameter('dedup_dist_m').value)
        self.ema = float(self.get_parameter('ema_alpha').value)
        self.pub_debug = bool(self.get_parameter('publish_debug_image').value)
        self.conf_th = float(self.get_parameter('conf_threshold').value)
        self.iou_th = float(self.get_parameter('iou_threshold').value)
        self.tracker_cfg = str(self.get_parameter('tracker').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        device = str(self.get_parameter('device').value) or None
        log_path = str(self.get_parameter('center_log_path').value)
        self.center_log: Optional[Path] = Path(log_path) if log_path else None

        # --- Object / prompt config ----------------------------------------
        objects_yaml = str(self.get_parameter('objects_yaml').value)
        self.object_defs, self.prompts, self.prompt_to_name = _load_objects(
            Path(objects_yaml) if objects_yaml else None, self.get_logger(),
        )
        self.get_logger().info(
            f"objects={[o['name'] for o in self.object_defs]}  "
            f"prompts={self.prompts}",
        )

        # --- YOLOE model ---------------------------------------------------
        self.model = _load_yolo_model(
            Path(self.get_parameter('model_path').value),
            self.prompts,
            device,
            self.get_logger(),
        )

        # --- ROS plumbing --------------------------------------------------
        self.bridge = CvBridge()
        self.camera_info: Optional[CameraInfo] = None
        self.objects: Dict[str, PoseStamped] = {}          # smoothed world pose per name
        self.last_pixel_centers: Dict[str, Tuple[int, int]] = {}
        self.last_track_ids: Dict[str, int] = {}
        self.last_confs: Dict[str, float] = {}

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

        self.marker_pub = self.create_publisher(MarkerArray, '/detected_objects', 10)
        self.centers_pub = self.create_publisher(PoseArray, '/detector/centers', 10)
        self.objects_pub = self.create_publisher(
            Detection3DArray, '/detector/objects', 10,
        )
        self.debug_img_pub = self.create_publisher(Image, '/detector/debug_image', 2)

        self.create_service(Trigger, '/detector/clear', self._on_clear, callback_group=cb)
        self.create_timer(1.0, self._publish_markers, callback_group=cb)

        self.get_logger().info(
            f"ready | mode={self.mode} rgb={rgb_topic} depth={depth_topic} "
            f"output_frame={self.output_frame}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_info(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _on_clear(self, req, res):
        self.objects.clear()
        self.last_pixel_centers.clear()
        self.last_track_ids.clear()
        res.success = True
        res.message = 'cleared'
        return res

    def _on_rgbd(self, rgb: Image, depth: Image) -> None:
        if self.camera_info is None:
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
            results = self.model.track(
                bgr, persist=True, tracker=self.tracker_cfg,
                conf=self.conf_th, iou=self.iou_th, imgsz=self.imgsz,
                verbose=False,
            )
        except Exception as e:
            self.get_logger().warn(f'yolo track error: {e}')
            return

        dets = _parse_yolo_tracks(results, self.prompts, self.prompt_to_name)
        # Keep the highest-confidence detection per named object.
        best_per_name: Dict[str, _Det] = {}
        for d in dets:
            cur = best_per_name.get(d.name)
            if cur is None or d.conf > cur.conf:
                best_per_name[d.name] = d

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx0 = self.camera_info.k[2]
        cy0 = self.camera_info.k[5]

        # Project every per-class top detection to 3D (camera frame) first.
        candidates: List[Tuple[float, str, _Det, Tuple[float, float, float]]] = []
        for name, d in best_per_name.items():
            u, v = d.center
            cam_xyz = pixel_to_camera(depth_img, u, v, fx, fy, cx0, cy0)
            if cam_xyz is None:
                continue
            candidates.append((float(d.conf), name, d, cam_xyz))

        # Simple 3D dedup — if two different-class candidates land within
        # dedup_dist_m of each other in the camera frame (same physical
        # object firing multiple prompts), keep only the higher-confidence one.
        kept = _dedup_candidates_3d(
            candidates, self.dedup_dist, logger=self.get_logger(),
        )

        for _, name, d, cam_xyz in kept:
            u, v = d.center
            if self.output_frame == self.camera_frame:
                world_xyz = cam_xyz
            else:
                world_xyz = transform_point_to_frame(
                    self.tf_buffer, cam_xyz,
                    self.camera_frame, self.output_frame, rgb.header.stamp,
                )
                if world_xyz is None:
                    continue
            self._update_object(name, world_xyz, rgb.header.stamp)
            self.last_pixel_centers[name] = (u, v)
            self.last_confs[name] = float(d.conf)
            if d.track_id is not None:
                self.last_track_ids[name] = d.track_id

        # /detector/centers — poses in a single PoseArray, same order as objects.
        pa = PoseArray()
        pa.header.stamp = rgb.header.stamp
        pa.header.frame_id = self.output_frame
        for ps in self.objects.values():
            pa.poses.append(ps.pose)
        self.centers_pub.publish(pa)

        # /detector/objects — labeled Detection3DArray, one entry per class.
        self._publish_object_detections(rgb.header.stamp)

        if self.center_log is not None and self.objects:
            self._log_centers(rgb.header.stamp)

        if self.pub_debug:
            debug_bgr = _annotate(bgr, dets)
            out_msg = self.bridge.cv2_to_imgmsg(debug_bgr, encoding='bgr8')
            out_msg.header = rgb.header
            self.debug_img_pub.publish(out_msg)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _update_object(self, name: str, xyz, stamp) -> None:
        ps = PoseStamped()
        ps.header.frame_id = self.output_frame
        ps.header.stamp = stamp
        if name in self.objects:
            prev = self.objects[name].pose.position
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
        self.objects[name] = ps

    def _publish_markers(self) -> None:
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        for i, (name, ps) in enumerate(self.objects.items()):
            m = Marker()
            m.header.frame_id = self.output_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = name
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose = ps.pose
            m.scale.x = m.scale.y = m.scale.z = 0.1
            m.color = _color_for(name)
            ma.markers.append(m)

            txt = Marker()
            txt.header = m.header
            txt.ns = name + '_label'
            txt.id = i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose = ps.pose
            txt.pose.position.z += 0.2
            txt.scale.z = 0.1
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            tid = self.last_track_ids.get(name)
            txt.text = f'{name}#{tid}' if tid is not None else name
            ma.markers.append(txt)
        self.marker_pub.publish(ma)

    def _publish_object_detections(self, stamp) -> None:
        arr = Detection3DArray()
        arr.header.stamp = stamp
        arr.header.frame_id = self.output_frame
        for name, ps in self.objects.items():
            det = Detection3D()
            det.header = arr.header
            det.id = name
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = name
            hyp.hypothesis.score = float(self.last_confs.get(name, 0.0))
            hyp.pose.pose = ps.pose
            det.results.append(hyp)
            det.bbox.center = ps.pose
            arr.detections.append(det)
        self.objects_pub.publish(arr)

    def _log_centers(self, stamp) -> None:
        record = {
            'ts': float(stamp.sec) + float(stamp.nanosec) * 1e-9,
            'frame': self.output_frame,
            'objects': {},
        }
        for name, ps in self.objects.items():
            record['objects'][name] = {
                'xyz': [ps.pose.position.x, ps.pose.position.y, ps.pose.position.z],
                'uv': list(self.last_pixel_centers.get(name, (-1, -1))),
                'track_id': int(self.last_track_ids.get(name, -1)),
            }
        try:
            with open(self.center_log, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except OSError as e:
            self.get_logger().warn(f'center_log write failed: {e}')


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

class _Det:
    __slots__ = ('name', 'prompt', 'conf', 'center', 'bbox', 'track_id')

    def __init__(self, name, prompt, conf, center, bbox, track_id):
        self.name = name
        self.prompt = prompt
        self.conf = conf
        self.center = center     # (u, v) pixel
        self.bbox = bbox         # (x, y, w, h)
        self.track_id = track_id


def _parse_yolo_tracks(
    results, prompts: List[str], prompt_to_name: Dict[int, str],
) -> List[_Det]:
    out: List[_Det] = []
    if not results:
        return out
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return out
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    ids_t = r.boxes.id
    ids = ids_t.cpu().numpy().astype(int) if ids_t is not None else [None] * len(clss)
    for (x1, y1, x2, y2), conf, cls, tid in zip(xyxy, confs, clss, ids):
        name = prompt_to_name.get(int(cls))
        if name is None:
            continue
        cu = int(0.5 * (x1 + x2))
        cv = int(0.5 * (y1 + y2))
        out.append(_Det(
            name=name,
            prompt=prompts[int(cls)] if int(cls) < len(prompts) else str(cls),
            conf=float(conf),
            center=(cu, cv),
            bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            track_id=int(tid) if tid is not None else None,
        ))
    return out


def _dedup_candidates_3d(
    candidates: List[Tuple[float, str, "_Det", Tuple[float, float, float]]],
    dedup_dist: float,
    logger=None,
) -> List[Tuple[float, str, "_Det", Tuple[float, float, float]]]:
    """Greedy 3D NMS across class labels: keep highest-conf detection, drop
    any subsequent detection within dedup_dist of an already-kept one."""
    if dedup_dist <= 0.0 or len(candidates) <= 1:
        return list(candidates)
    ordered = sorted(candidates, key=lambda t: -t[0])
    kept: List[Tuple[float, str, "_Det", Tuple[float, float, float]]] = []
    thresh_sq = dedup_dist * dedup_dist
    for entry in ordered:
        _, name, _, xyz = entry
        too_close_to = None
        for k in kept:
            kxyz = k[3]
            dsq = (
                (xyz[0] - kxyz[0]) ** 2
                + (xyz[1] - kxyz[1]) ** 2
                + (xyz[2] - kxyz[2]) ** 2
            )
            if dsq < thresh_sq:
                too_close_to = k
                break
        if too_close_to is None:
            kept.append(entry)
        elif logger is not None:
            logger.debug(
                f'dedup: dropping {name} (conf={entry[0]:.2f}) — within '
                f'{dedup_dist}m of {too_close_to[1]} (conf={too_close_to[0]:.2f})'
            )
    return kept


def _annotate(bgr: np.ndarray, dets: List[_Det]) -> np.ndarray:
    out = bgr.copy()
    for d in dets:
        x, y, w, h = d.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, d.center, 4, (0, 0, 255), -1)
        tag = f'{d.name} {d.conf:.2f}'
        if d.track_id is not None:
            tag += f' #{d.track_id}'
        cv2.putText(out, tag, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


_MARKER_PALETTE: Dict[str, Tuple[float, float, float]] = {
    'white_bottle': (1.0, 1.0, 1.0),
    'green_cup':    (0.1, 0.9, 0.1),
    'blue_cup':     (0.1, 0.3, 1.0),
}


def _color_for(name: str) -> ColorRGBA:
    r, g, b = _MARKER_PALETTE.get(name, (0.7, 0.7, 0.7))
    return ColorRGBA(r=r, g=g, b=b, a=1.0)


def _load_objects(
    yaml_path: Optional[Path], logger,
) -> Tuple[List[dict], List[str], Dict[int, str]]:
    if yaml_path is None or not yaml_path.exists():
        logger.warn('objects_yaml not found; using built-in defaults')
        cfg = {'objects': _DEFAULT_OBJECTS}
    else:
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
    defs = cfg.get('objects', [])
    prompts: List[str] = []
    prompt_to_name: Dict[int, str] = {}
    idx = 0
    for entry in defs:
        name = entry['name']
        entry_prompts = entry.get('prompts') or [name.replace('_', ' ')]
        for p in entry_prompts:
            prompts.append(p)
            prompt_to_name[idx] = name
            idx += 1
    return defs, prompts, prompt_to_name


def _load_yolo_model(path: Path, prompts: List[str], device: Optional[str], logger):
    """Load YOLOE from .pt (runtime prompts) or any exported artifact (baked prompts).

    Exported formats include .engine / .onnx / .torchscript / .mnn /
    *_openvino_model/ / *_ncnn_model/ / *_paddle_model/ — anything that
    ``model.export(format=...)`` produces. All of those are loaded via
    ``YOLO(path, task='segment')``; only the raw .pt checkpoint needs
    ``YOLOE.set_classes`` at runtime.
    """
    from ultralytics import YOLO, YOLOE  # lazy import
    spath = str(path)
    is_native_pt = spath.lower().endswith('.pt')
    if is_native_pt:
        logger.info(f'loading YOLOE weights: {spath}  classes={prompts}')
        model = YOLOE(spath)
        model.set_classes(prompts, model.get_text_pe(prompts))
    else:
        logger.info(f'loading exported YOLOE: {spath} (prompts baked at export)')
        model = YOLO(spath, task='segment')
    if device:
        try:
            model.to(device)
        except Exception as e:
            logger.warn(f'model.to({device}) failed: {e}; continuing on default device')
    return model


_DEFAULT_OBJECTS = [
    {'name': 'white_bottle', 'prompts': ['white bottle']},
    {'name': 'green_cup',    'prompts': ['green cup']},
    {'name': 'blue_cup',     'prompts': ['blue cup']},
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
