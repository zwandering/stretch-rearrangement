"""Instruction-driven task planner: natural-language → Gemini VLM → pick/place PoseArray."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rclpy
import yaml
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion
from nav_msgs.msg import OccupancyGrid
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from .planners import (
    DetectedObject,
    PickPlaceTask,
    PlannerInput,
    RegionInfo,
    VLMPlanner,
    VLMPlanError,
)
from .region_manager_node import point_in_polygon
from .utils.transform_utils import robot_pose_in_map, yaw_to_quat


class TaskPlannerNode(Node):

    def __init__(self) -> None:
        super().__init__('task_planner_node')

        self.declare_parameter('regions_yaml', '')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.declare_parameter('vlm_model', 'gemini-2.5-flash')
        self.declare_parameter(
            'vlm_base_url',
            'https://generativelanguage.googleapis.com/v1beta/openai/',
        )
        self.declare_parameter('vlm_api_key_env', 'GEMINI_API_KEY')
        self.declare_parameter('vlm_max_retries', 5)
        self.declare_parameter('vlm_retry_base_sec', 1.0)

        self.declare_parameter('min_detections_before_plan', 3)
        self.declare_parameter('instruction_topic', '/instruction/text')
        self.declare_parameter('place_anchor_z', 0.0)
        self.declare_parameter('objects_snapshot_yaml', '')

        self.map_frame: str = self.get_parameter('map_frame').value
        self.base_frame: str = self.get_parameter('base_frame').value
        self.min_detections = int(self.get_parameter('min_detections_before_plan').value)
        self.place_anchor_z = float(self.get_parameter('place_anchor_z').value)

        self.regions: Dict[str, RegionInfo] = self._load_regions(
            self.get_parameter('regions_yaml').value,
        )
        self.get_logger().info(f'Loaded regions: {list(self.regions)}')

        self.planner = VLMPlanner(
            model=str(self.get_parameter('vlm_model').value),
            base_url=str(self.get_parameter('vlm_base_url').value),
            api_key_env=str(self.get_parameter('vlm_api_key_env').value),
            use_image=False,
            max_retries=int(self.get_parameter('vlm_max_retries').value),
            retry_base_sec=float(self.get_parameter('vlm_retry_base_sec').value),
            logger=self.get_logger(),
        )

        self.latest_detections: Dict[str, Tuple[float, float, float, float]] = {}
        self._seed_detections_from_snapshot(
            self.get_parameter('objects_snapshot_yaml').value,
        )
        self.latest_map: Optional[OccupancyGrid] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cb = ReentrantCallbackGroup()

        detections_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            Detection3DArray, '/detector/objects',
            self._on_detections, detections_qos, callback_group=cb,
        )

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            OccupancyGrid, '/map',
            self._on_map, map_qos, callback_group=cb,
        )

        instruction_topic: str = str(self.get_parameter('instruction_topic').value)
        self.create_subscription(
            String, instruction_topic,
            self._on_instruction, 10, callback_group=cb,
        )

        self.plan_pub = self.create_publisher(
            PoseArray, '/planner/pick_place_plan', 10,
        )
        self.markers_pub = self.create_publisher(
            MarkerArray, '/planner/plan_visualization', 10,
        )

        self.latest_plan: List[PickPlaceTask] = []
        self.get_logger().info(
            f'TaskPlannerNode ready. Listening on {instruction_topic}.'
        )

    # --- setup helpers ---------------------------------------------------

    def _load_regions(self, path: str) -> Dict[str, RegionInfo]:
        if not path:
            self.get_logger().warn('regions_yaml not provided; region classification disabled.')
            return {}
        with open(Path(path), 'r') as f:
            cfg = yaml.safe_load(f) or {}
        out: Dict[str, RegionInfo] = {}
        for entry in cfg.get('regions', []):
            poly = [tuple(p) for p in entry['polygon']]
            anchor = tuple(entry.get(
                'place_anchor',
                [sum(p[0] for p in poly) / len(poly),
                 sum(p[1] for p in poly) / len(poly), 0.0],
            ))
            out[entry['name']] = RegionInfo(entry['name'], poly, anchor)
        return out

    def _seed_detections_from_snapshot(self, path: str) -> None:
        if not path:
            return
        p = Path(path)
        if not p.exists():
            self.get_logger().warn(
                f'objects_snapshot_yaml={p} does not exist; skipping seed.'
            )
            return
        try:
            with open(p, 'r') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().warn(f'failed to read snapshot {p}: {e}')
            return
        objs = cfg.get('objects', {}) or {}
        for label, entry in objs.items():
            try:
                self.latest_detections[str(label)] = (
                    float(entry['x']),
                    float(entry['y']),
                    float(entry.get('z', 0.0)),
                    float(entry.get('conf', 0.0)),
                )
            except (KeyError, TypeError, ValueError) as e:
                self.get_logger().warn(f'snapshot entry {label!r} bad ({e}); skipping.')
        self.get_logger().info(
            f'Seeded {len(self.latest_detections)} object(s) from snapshot {p}.'
        )

    # --- subscriptions ---------------------------------------------------

    def _on_detections(self, msg: Detection3DArray) -> None:
        latest: Dict[str, Tuple[float, float, float, float]] = {}
        for det in msg.detections:
            label = det.id
            if not label and det.results:
                label = det.results[0].hypothesis.class_id
            if not label:
                continue
            pos = det.bbox.center.position
            score = det.results[0].hypothesis.score if det.results else 0.0
            latest[label] = (
                float(pos.x), float(pos.y), float(pos.z), float(score),
            )
        self.latest_detections = latest

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def _on_instruction(self, msg: String) -> None:
        text = (msg.data or '').strip()
        if not text:
            self.get_logger().warn('Empty instruction received; ignoring.')
            return
        self.get_logger().info(f'Received instruction: {text!r}')
        self._run_plan(text)

    # --- planning --------------------------------------------------------

    def _run_plan(self, instruction: str) -> None:
        if not self.regions:
            self.get_logger().error('No regions loaded; cannot plan.')
            self._publish_empty_plan()
            return

        snapshot = dict(self.latest_detections)
        if len(snapshot) < self.min_detections:
            self.get_logger().warn(
                f'Only {len(snapshot)}/{self.min_detections} objects seen; '
                'proceeding but plan may be incomplete.'
            )

        detected: List[DetectedObject] = []
        obj_z: Dict[str, float] = {}
        for label, (x, y, z, _score) in snapshot.items():
            region = self._which_region((x, y))
            detected.append(DetectedObject(
                label=label, pose_xy=(x, y), current_region=region, z=z,
            ))
            obj_z[label] = z

        if not detected:
            self.get_logger().warn('No detections to plan over; publishing empty plan.')
            self._publish_empty_plan()
            return

        robot_xy = self._robot_xy()
        inp = PlannerInput(
            objects=detected,
            regions=self.regions,
            goal_assignment={},
            robot_xy=robot_xy,
            instruction=instruction,
        )

        try:
            plan = self.planner.plan(inp)
        except VLMPlanError as e:
            self.get_logger().error(f'VLM planning failed: {e}')
            self._publish_empty_plan()
            return
        except Exception as e:
            self.get_logger().error(f'Unexpected planner error: {e}')
            self._publish_empty_plan()
            return

        self.latest_plan = plan
        summary = ', '.join(
            f'{i}:{t.object_label}→{t.target_region}' for i, t in enumerate(plan)
        ) or '(empty)'
        self.get_logger().info(f'Plan: {summary}')

        self._publish_plan(plan, obj_z)
        self._publish_plan_markers(plan)

    def _robot_xy(self) -> Tuple[float, float]:
        pose = robot_pose_in_map(
            self, self.tf_buffer, self.base_frame, self.map_frame,
        )
        if pose is None:
            self.get_logger().warn(
                f'TF {self.map_frame}→{self.base_frame} unavailable; using (0,0).'
            )
            return (0.0, 0.0)
        return (pose.pose.position.x, pose.pose.position.y)

    def _which_region(self, xy: Tuple[float, float]) -> Optional[str]:
        for name, r in self.regions.items():
            if point_in_polygon(xy[0], xy[1], r.polygon):
                return name
        return None

    # --- publishing ------------------------------------------------------

    def _publish_empty_plan(self) -> None:
        pa = PoseArray()
        pa.header.frame_id = self.map_frame
        pa.header.stamp = self.get_clock().now().to_msg()
        self.plan_pub.publish(pa)

    def _publish_plan(
        self, plan: List[PickPlaceTask], obj_z: Dict[str, float],
    ) -> None:
        pa = PoseArray()
        pa.header.frame_id = self.map_frame
        pa.header.stamp = self.get_clock().now().to_msg()

        for task in plan:
            pick_pose = Pose()
            pick_pose.position = Point(
                x=float(task.pick_xy[0]),
                y=float(task.pick_xy[1]),
                z=float(obj_z.get(task.object_label, 0.0)),
            )
            pick_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            pa.poses.append(pick_pose)

            region = self.regions.get(task.target_region)
            yaw = region.place_anchor[2] if region else 0.0
            qx, qy, qz, qw = yaw_to_quat(yaw)
            place_pose = Pose()
            place_pose.position = Point(
                x=float(task.place_xy[0]),
                y=float(task.place_xy[1]),
                z=self.place_anchor_z,
            )
            place_pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            pa.poses.append(place_pose)

        self.plan_pub.publish(pa)

    def _publish_plan_markers(self, plan: List[PickPlaceTask]) -> None:
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        stamp = self.get_clock().now().to_msg()
        for i, t in enumerate(plan):
            line = Marker()
            line.header.frame_id = self.map_frame
            line.header.stamp = stamp
            line.ns = 'plan'
            line.id = i
            line.type = Marker.ARROW
            line.action = Marker.ADD
            line.scale.x = 0.04
            line.scale.y = 0.08
            line.scale.z = 0.1
            line.color = ColorRGBA(r=1.0, g=0.4, b=0.0, a=0.9)
            line.points.append(Point(
                x=float(t.pick_xy[0]), y=float(t.pick_xy[1]), z=0.05,
            ))
            line.points.append(Point(
                x=float(t.place_xy[0]), y=float(t.place_xy[1]), z=0.05,
            ))
            ma.markers.append(line)

            txt = Marker()
            txt.header = line.header
            txt.ns = 'plan_label'
            txt.id = i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            mx = 0.5 * (t.pick_xy[0] + t.place_xy[0])
            my = 0.5 * (t.pick_xy[1] + t.place_xy[1])
            txt.pose.position = Point(x=mx, y=my, z=0.25)
            txt.scale.z = 0.18
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            txt.text = f'{i}: {t.object_label} → {t.target_region}'
            ma.markers.append(txt)
        self.markers_pub.publish(ma)

    def get_plan(self) -> List[PickPlaceTask]:
        return list(self.latest_plan)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskPlannerNode()
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
