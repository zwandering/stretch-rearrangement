"""Semantic region management: polygons in map frame → region lookup + place poses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from .utils.transform_utils import yaw_to_quat


@dataclass
class Region:
    name: str
    polygon: List[Tuple[float, float]]
    place_anchor: Tuple[float, float, float]  # x, y, yaw
    color: Tuple[float, float, float] = (0.2, 0.7, 1.0)


def point_in_polygon(px: float, py: float, poly: List[Tuple[float, float]]) -> bool:
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > py) != (yj > py):
            x_intersect = xi + (py - yi) * (xj - xi) / (yj - yi + 1e-12)
            if px < x_intersect:
                inside = not inside
        j = i
    return inside


def polygon_centroid(poly: List[Tuple[float, float]]) -> Tuple[float, float]:
    arr = np.asarray(poly)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


class RegionManagerNode(Node):

    def __init__(self) -> None:
        super().__init__('region_manager_node')

        self.declare_parameter('regions_yaml', '')
        self.declare_parameter('map_frame', 'map')

        self.map_frame = self.get_parameter('map_frame').value
        yaml_path = self.get_parameter('regions_yaml').value

        self.regions: Dict[str, Region] = {}
        if yaml_path:
            self._load(Path(yaml_path))
        else:
            self.get_logger().warn('regions_yaml not set; using default 4-quadrant layout')
            self._load_default()

        self.get_logger().info(
            f'Loaded {len(self.regions)} regions: {list(self.regions.keys())}'
        )

        self.marker_pub = self.create_publisher(
            MarkerArray, '/regions/visualization', 10,
        )
        cb = ReentrantCallbackGroup()
        self.create_timer(2.0, self._publish_markers, callback_group=cb)

        from std_srvs.srv import Trigger
        self.create_service(
            Trigger, '/regions/reload', self._on_reload, callback_group=cb,
        )

        self.get_logger().info('RegionManagerNode ready.')

    def _load(self, path: Path) -> None:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        for entry in cfg.get('regions', []):
            self.regions[entry['name']] = Region(
                name=entry['name'],
                polygon=[tuple(p) for p in entry['polygon']],
                place_anchor=tuple(entry.get(
                    'place_anchor',
                    list(polygon_centroid(entry['polygon'])) + [0.0],
                )),
                color=tuple(entry.get('color', [0.2, 0.7, 1.0])),
            )

    def _load_default(self) -> None:
        defaults = [
            ('A', [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]),
            ('B', [(-2.0, 0.0), (0.0, 0.0), (0.0, 2.0), (-2.0, 2.0)]),
            ('C', [(0.0, -2.0), (2.0, -2.0), (2.0, 0.0), (0.0, 0.0)]),
            ('D', [(-2.0, -2.0), (0.0, -2.0), (0.0, 0.0), (-2.0, 0.0)]),
        ]
        palette = [(1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0), (1.0, 1.0, 0.5)]
        for (name, poly), c in zip(defaults, palette):
            cx, cy = polygon_centroid(poly)
            self.regions[name] = Region(name, poly, (cx, cy, 0.0), c)

    def _on_reload(self, req, res):
        yaml_path = self.get_parameter('regions_yaml').value
        if yaml_path:
            self.regions.clear()
            self._load(Path(yaml_path))
            res.success = True
            res.message = f'reloaded {len(self.regions)} regions'
        else:
            res.success = False
            res.message = 'no regions_yaml set'
        return res

    # Programmatic API used by other nodes (same process) or via service wrappers.
    def which_region(self, x: float, y: float) -> Optional[str]:
        for name, r in self.regions.items():
            if point_in_polygon(x, y, r.polygon):
                return name
        return None

    def place_pose(self, region_name: str) -> Optional[PoseStamped]:
        r = self.regions.get(region_name)
        if r is None:
            return None
        ps = PoseStamped()
        ps.header.frame_id = self.map_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ax, ay, ayaw = r.place_anchor
        ps.pose.position.x = float(ax)
        ps.pose.position.y = float(ay)
        qx, qy, qz, qw = yaw_to_quat(float(ayaw))
        ps.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        return ps

    def pick_approach_pose(
        self,
        target_xy: Tuple[float, float],
        robot_xy: Tuple[float, float],
        standoff_m: float = 0.55,
    ) -> PoseStamped:
        dx = target_xy[0] - robot_xy[0]
        dy = target_xy[1] - robot_xy[1]
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        ux, uy = dx / norm, dy / norm
        ax = target_xy[0] - ux * standoff_m
        ay = target_xy[1] - uy * standoff_m
        yaw = float(np.arctan2(dy, dx))
        ps = PoseStamped()
        ps.header.frame_id = self.map_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = ax
        ps.pose.position.y = ay
        qx, qy, qz, qw = yaw_to_quat(yaw)
        ps.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        return ps

    def _publish_markers(self) -> None:
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        now = self.get_clock().now().to_msg()
        for i, r in enumerate(self.regions.values()):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = now
            m.ns = 'region_boundary'
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.04
            m.color = ColorRGBA(
                r=float(r.color[0]), g=float(r.color[1]),
                b=float(r.color[2]), a=0.9,
            )
            for x, y in r.polygon + [r.polygon[0]]:
                m.points.append(Point(x=float(x), y=float(y), z=0.01))
            ma.markers.append(m)

            t = Marker()
            t.header = m.header
            t.ns = 'region_label'
            t.id = i
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            cx, cy = polygon_centroid(r.polygon)
            t.pose.position = Point(x=cx, y=cy, z=0.3)
            t.scale.z = 0.4
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            t.text = r.name
            ma.markers.append(t)
        self.marker_pub.publish(ma)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RegionManagerNode()
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
