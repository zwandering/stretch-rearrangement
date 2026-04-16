"""Task planner node: wraps GreedyPlanner / VLMPlanner as a ROS 2 service."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .planners import (
    DetectedObject,
    GreedyPlanner,
    PickPlaceTask,
    PlannerBackend,
    PlannerInput,
    RegionInfo,
    VLMPlanner,
)
from .region_manager_node import point_in_polygon
from .utils.transform_utils import robot_pose_in_map, yaw_to_quat


class TaskPlannerNode(Node):

    def __init__(self) -> None:
        super().__init__('task_planner_node')

        self.declare_parameter('planner_backend', 'greedy')
        self.declare_parameter('tasks_yaml', '')
        self.declare_parameter('regions_yaml', '')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.declare_parameter('vlm_model', 'gemini-2.5-flash')
        self.declare_parameter(
            'vlm_base_url',
            'https://generativelanguage.googleapis.com/v1beta/openai/',
        )
        self.declare_parameter('vlm_api_key_env', 'GEMINI_API_KEY')
        self.declare_parameter('vlm_use_image', True)
        self.declare_parameter('vlm_max_retries', 2)

        self.map_frame: str = self.get_parameter('map_frame').value
        self.base_frame: str = self.get_parameter('base_frame').value
        self.backend: PlannerBackend = self._build_backend()
        self.get_logger().info(f'Planner backend = {self.backend.name}')

        self.goal_assignment: Dict[str, str] = self._load_tasks(
            self.get_parameter('tasks_yaml').value,
        )
        self.regions: Dict[str, RegionInfo] = self._load_regions(
            self.get_parameter('regions_yaml').value,
        )
        self.get_logger().info(
            f'Goal assignment: {self.goal_assignment} | regions: {list(self.regions)}'
        )

        self.bridge = CvBridge()
        self.latest_objects: Dict[str, PoseStamped] = {}
        self.latest_debug_image = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cb = ReentrantCallbackGroup()
        self.create_subscription(
            MarkerArray, '/detected_objects',
            self._on_detections, 10, callback_group=cb,
        )
        self.create_subscription(
            Image, '/detector/debug_image',
            self._on_debug_image, 2, callback_group=cb,
        )
        self.plan_pub = self.create_publisher(
            MarkerArray, '/planner/plan_visualization', 10,
        )
        self.create_service(
            Trigger, '/planner/compute',
            self._on_compute, callback_group=cb,
        )
        self.latest_plan: List[PickPlaceTask] = []

        self.get_logger().info('TaskPlannerNode ready.')

    # --- setup helpers ---------------------------------------------------

    def _build_backend(self) -> PlannerBackend:
        name = str(self.get_parameter('planner_backend').value).lower()
        if name == 'vlm':
            import logging
            return VLMPlanner(
                model=str(self.get_parameter('vlm_model').value),
                base_url=str(self.get_parameter('vlm_base_url').value),
                api_key_env=str(self.get_parameter('vlm_api_key_env').value),
                use_image=bool(self.get_parameter('vlm_use_image').value),
                max_retries=int(self.get_parameter('vlm_max_retries').value),
                logger=self.get_logger(),  # rclpy logger is usable enough (info/warn/...)
            )
        return GreedyPlanner()

    def _load_tasks(self, path: str) -> Dict[str, str]:
        if not path:
            return {}
        with open(Path(path), 'r') as f:
            cfg = yaml.safe_load(f) or {}
        return dict(cfg.get('assignments', {}))

    def _load_regions(self, path: str) -> Dict[str, RegionInfo]:
        if not path:
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

    # --- subscriptions ---------------------------------------------------

    def _on_detections(self, msg: MarkerArray) -> None:
        latest: Dict[str, PoseStamped] = {}
        for m in msg.markers:
            if m.type != Marker.CUBE:
                continue
            ps = PoseStamped()
            ps.header = m.header
            ps.pose = m.pose
            latest[m.ns] = ps
        self.latest_objects = latest

    def _on_debug_image(self, msg: Image) -> None:
        try:
            self.latest_debug_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8',
            )
        except Exception:
            self.latest_debug_image = None

    # --- service handler -------------------------------------------------

    def _on_compute(self, req, res):
        pose = robot_pose_in_map(
            self, self.tf_buffer, self.base_frame, self.map_frame,
        )
        if pose is None:
            res.success = False
            res.message = 'robot pose tf unavailable'
            return res

        detected = []
        for label, ps in self.latest_objects.items():
            xy = (ps.pose.position.x, ps.pose.position.y)
            region = self._which_region(xy)
            detected.append(DetectedObject(
                label=label, pose_xy=xy, current_region=region,
                z=ps.pose.position.z,
            ))

        inp = PlannerInput(
            objects=detected,
            regions=self.regions,
            goal_assignment=self.goal_assignment,
            robot_xy=(pose.pose.position.x, pose.pose.position.y),
            context_image_bgr=self.latest_debug_image,
        )

        try:
            plan = self.backend.plan(inp)
        except Exception as e:
            self.get_logger().error(f'Planner error: {e}')
            res.success = False
            res.message = f'planner error: {e}'
            return res

        self.latest_plan = plan
        self._publish_plan_markers(plan)

        summary = ', '.join(
            f'{i}:{t.object_label}→{t.target_region}' for i, t in enumerate(plan)
        ) or '(empty)'
        res.success = True
        res.message = f'[{self.backend.name}] plan: {summary}'
        self.get_logger().info(res.message)
        return res

    # --- utilities -------------------------------------------------------

    def _which_region(self, xy: Tuple[float, float]) -> Optional[str]:
        for name, r in self.regions.items():
            if point_in_polygon(xy[0], xy[1], r.polygon):
                return name
        return None

    def get_plan(self) -> List[PickPlaceTask]:
        return list(self.latest_plan)

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
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.05
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
        self.plan_pub.publish(ma)


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
