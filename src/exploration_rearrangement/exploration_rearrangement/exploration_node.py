"""Frontier-based exploration using Nav2 NavigateToPose."""

from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import ColorRGBA, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .utils.frontier_utils import Frontier, extract_frontiers, score_frontier
from .utils.transform_utils import robot_pose_in_map, yaw_to_quat


class ExplorationNode(Node):

    def __init__(self) -> None:
        super().__init__('exploration_node')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('min_cluster_size', 8)
        self.declare_parameter('goal_tolerance_m', 0.5)
        self.declare_parameter('alpha_dist', 1.0)
        self.declare_parameter('beta_info', 0.05)
        self.declare_parameter('replan_period_s', 3.0)
        self.declare_parameter('goal_timeout_s', 60.0)
        self.declare_parameter('enabled_on_start', False)

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.min_cluster = int(self.get_parameter('min_cluster_size').value)
        self.goal_tol = float(self.get_parameter('goal_tolerance_m').value)
        self.alpha_dist = float(self.get_parameter('alpha_dist').value)
        self.beta_info = float(self.get_parameter('beta_info').value)
        self.goal_timeout = float(self.get_parameter('goal_timeout_s').value)

        self.enabled: bool = bool(self.get_parameter('enabled_on_start').value)
        self.current_map: Optional[OccupancyGrid] = None
        self.active_goal_future = None
        self.goal_start_time = None
        self.last_goal_xy: Optional[Tuple[float, float]] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        cb_reent = ReentrantCallbackGroup()
        cb_excl = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            OccupancyGrid, '/map', self._on_map, map_qos, callback_group=cb_reent,
        )
        self.frontier_pub = self.create_publisher(
            MarkerArray, '/exploration/frontiers', 10,
        )
        self.status_pub = self.create_publisher(
            String, '/exploration/status', 10,
        )

        self.nav_client = ActionClient(
            self, NavigateToPose, '/navigate_to_pose',
            callback_group=cb_reent,
        )

        self.create_service(
            Trigger, '/exploration/start',
            self._on_start, callback_group=cb_excl,
        )
        self.create_service(
            Trigger, '/exploration/stop',
            self._on_stop, callback_group=cb_excl,
        )

        self.create_timer(
            float(self.get_parameter('replan_period_s').value),
            self._tick, callback_group=cb_excl,
        )
        self.get_logger().info('ExplorationNode ready.')

    def _on_start(self, req, res):
        self.enabled = True
        res.success = True
        res.message = 'exploration started'
        self.get_logger().info('Exploration enabled.')
        return res

    def _on_stop(self, req, res):
        self.enabled = False
        if self.active_goal_future is not None:
            try:
                self.active_goal_future.cancel()
            except Exception:
                pass
        res.success = True
        res.message = 'exploration stopped'
        self.get_logger().info('Exploration disabled.')
        return res

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.current_map = msg

    def _tick(self) -> None:
        if not self.enabled or self.current_map is None:
            return

        if self.active_goal_future is not None:
            elapsed = (self.get_clock().now().nanoseconds
                       - self.goal_start_time) * 1e-9
            if elapsed < self.goal_timeout:
                if self.last_goal_xy is not None:
                    pose = robot_pose_in_map(
                        self, self.tf_buffer, self.base_frame, self.map_frame,
                    )
                    if pose is not None:
                        d = ((pose.pose.position.x - self.last_goal_xy[0]) ** 2
                             + (pose.pose.position.y - self.last_goal_xy[1]) ** 2) ** 0.5
                        if d < self.goal_tol:
                            self.get_logger().info(
                                f'Reached frontier within {d:.2f} m, selecting next.'
                            )
                            self.active_goal_future = None
                return
            else:
                self.get_logger().warn('Goal timeout; picking a new frontier.')
                self.active_goal_future = None

        pose = robot_pose_in_map(
            self, self.tf_buffer, self.base_frame, self.map_frame,
        )
        if pose is None:
            return
        robot_xy = (pose.pose.position.x, pose.pose.position.y)
        frontiers = extract_frontiers(self.current_map, self.min_cluster)
        self._publish_markers(frontiers)
        if not frontiers:
            self.status_pub.publish(String(data='done'))
            self.get_logger().info('No frontiers remaining — exploration done.')
            self.enabled = False
            return

        best = min(
            frontiers,
            key=lambda f: score_frontier(
                f, robot_xy, self.alpha_dist, self.beta_info,
            ),
        )
        self._send_goal(best, robot_xy)

    def _send_goal(self, frontier: Frontier, robot_xy: Tuple[float, float]) -> None:
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Nav2 action server unavailable.')
            return
        goal = PoseStamped()
        goal.header.frame_id = self.map_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        gx, gy = frontier.centroid_world
        goal.pose.position.x = float(gx)
        goal.pose.position.y = float(gy)
        yaw = float(
            __import__('numpy').arctan2(gy - robot_xy[1], gx - robot_xy[0]),
        )
        qx, qy, qz, qw = yaw_to_quat(yaw)
        goal.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal
        self.get_logger().info(
            f'Sending frontier goal ({gx:.2f}, {gy:.2f}), size={frontier.size}'
        )
        self.active_goal_future = self.nav_client.send_goal_async(nav_goal)
        self.active_goal_future.add_done_callback(self._on_goal_response)
        self.goal_start_time = self.get_clock().now().nanoseconds
        self.last_goal_xy = (float(gx), float(gy))
        self.status_pub.publish(String(data='navigating'))

    def _on_goal_response(self, future) -> None:
        try:
            gh = future.result()
        except Exception as e:
            self.get_logger().warn(f'Goal send failed: {e}')
            self.active_goal_future = None
            return
        if not gh.accepted:
            self.get_logger().warn('Goal rejected by Nav2.')
            self.active_goal_future = None
            return
        res_future = gh.get_result_async()
        res_future.add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future) -> None:
        try:
            _ = future.result()
        except Exception as e:
            self.get_logger().warn(f'Nav2 result error: {e}')
        self.active_goal_future = None

    def _publish_markers(self, frontiers: List[Frontier]) -> None:
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'frontiers'
        m.id = 0
        m.type = Marker.SPHERE_LIST
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.15
        m.color = ColorRGBA(r=0.0, g=1.0, b=0.8, a=0.8)
        for f in frontiers:
            p = Point(x=float(f.centroid_world[0]),
                      y=float(f.centroid_world[1]), z=0.05)
            m.points.append(p)
        ma.markers.append(m)
        self.frontier_pub.publish(ma)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExplorationNode()
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
