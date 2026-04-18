"""Lightweight fake simulator for end-to-end testing without Gazebo.

Replaces stretch_driver + camera + Nav2 + SLAM with a single Python node:
  * TF: map -> odom -> base_link (dynamic); map -> camera_color_optical_frame (top-down)
  * /map (OccupancyGrid, TRANSIENT_LOCAL)
  * /odom, /joint_states
  * /camera/color/image_raw + /camera/aligned_depth_to_color/image_raw + /camera/color/camera_info
    rendered from a bird's-eye virtual camera, drawing colored blobs for each scene object.
  * /navigate_to_pose action — interpolates the robot pose toward the goal.
  * /manipulation/pick|place actions — fake hardware: pick removes the closest
    object and "attaches" it; place teleports it to just in front of the robot.
  * Trigger services for head scan, mode switching, stow — all no-ops.

Scene objects roughly match config/regions.yaml quadrants: A(+x,+y), B(-x,+y),
C(+x,-y), D(-x,-y). Initial placements are chosen so the default tasks.yaml
forces all three objects to migrate.
"""

import math
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from geometry_msgs.msg import Quaternion, TransformStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


OBJECT_COLORS_BGR = {
    'white_bottle': (245, 245, 245),
    'green_cup':    (40, 220, 40),
    'blue_cup':     (220, 60, 40),
}


class FakeSimNode(Node):

    def __init__(self) -> None:
        super().__init__('fake_sim_node')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('image_w', 640)
        self.declare_parameter('image_h', 640)
        self.declare_parameter('fx', 200.0)
        self.declare_parameter('fy', 200.0)
        self.declare_parameter('cam_z', 3.0)
        self.declare_parameter('nav_speed_m_s', 1.5)
        self.declare_parameter('nav_yaw_rate', 2.5)

        self.map_frame = str(self.get_parameter('map_frame').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.cam_frame = str(self.get_parameter('camera_frame').value)
        self.W = int(self.get_parameter('image_w').value)
        self.H = int(self.get_parameter('image_h').value)
        self.fx = float(self.get_parameter('fx').value)
        self.fy = float(self.get_parameter('fy').value)
        self.img_cx = self.W / 2.0
        self.img_cy = self.H / 2.0
        self.cam_z = float(self.get_parameter('cam_z').value)
        self.nav_speed = float(self.get_parameter('nav_speed_m_s').value)
        self.nav_yaw_rate = float(self.get_parameter('nav_yaw_rate').value)

        self.rx = 0.0
        self.ry = 0.0
        self.ryaw = 0.0

        # Scene: all three objects start in A (+x,+y) so every task has to move.
        #   white_bottle (A) -> goal C
        #   green_cup    (C) -> goal A   (reversed from goal for motion)
        #   blue_cup     (A) -> goal D
        self.objects: Dict[str, Optional[Tuple[float, float, float]]] = {
            'white_bottle': (1.5, 1.5, 0.4),
            'green_cup':    (1.5, -1.5, 0.4),
            'blue_cup':     (2.5, 0.5, 0.4),
        }
        self.held_object: Optional[str] = None

        self.bridge = CvBridge()
        cb = ReentrantCallbackGroup()

        self.tfb = TransformBroadcaster(self)
        self.static_tfb = StaticTransformBroadcaster(self)
        self._publish_static_tfs()

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.rgb_pub = self.create_publisher(Image, '/camera/color/image_raw', 2)
        self.depth_pub = self.create_publisher(
            Image, '/camera/aligned_depth_to_color/image_raw', 2)
        self.info_pub = self.create_publisher(
            CameraInfo, '/camera/color/camera_info', 2)
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)

        # No-op trigger services (replace stretch_driver / head_scan_node).
        self.create_service(Trigger, '/head/start_scan',
                            self._noop_trigger, callback_group=cb)
        self.create_service(Trigger, '/head/stop_scan',
                            self._noop_trigger, callback_group=cb)
        self.create_service(Trigger, '/switch_to_navigation_mode',
                            self._noop_trigger, callback_group=cb)
        self.create_service(Trigger, '/switch_to_position_mode',
                            self._noop_trigger, callback_group=cb)
        self.create_service(Trigger, '/stow_the_robot',
                            self._noop_trigger, callback_group=cb)

        self.nav_server = ActionServer(
            self, NavigateToPose, '/navigate_to_pose',
            execute_callback=self._execute_nav,
            goal_callback=lambda _goal: GoalResponse.ACCEPT,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=cb,
        )
        self.pick_server = ActionServer(
            self, FollowJointTrajectory, '/manipulation/pick',
            execute_callback=self._execute_pick,
            goal_callback=lambda _goal: GoalResponse.ACCEPT,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=cb,
        )
        self.place_server = ActionServer(
            self, FollowJointTrajectory, '/manipulation/place',
            execute_callback=self._execute_place,
            goal_callback=lambda _goal: GoalResponse.ACCEPT,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=cb,
        )

        self.occ_grid = self._build_map()

        self.create_timer(0.05, self._broadcast_dynamic_tf, callback_group=cb)
        self.create_timer(0.05, self._publish_odom, callback_group=cb)
        self.create_timer(0.1,  self._publish_camera, callback_group=cb)
        self.create_timer(1.0,  self._publish_map, callback_group=cb)
        self.create_timer(0.1,  self._publish_joint_states, callback_group=cb)

        self.get_logger().info(
            f'FakeSimNode ready — 3 objects, bird\'s-eye camera {self.W}x{self.H}.'
        )

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _noop_trigger(self, _req, res):
        res.success = True
        res.message = 'ok'
        return res

    def _publish_static_tfs(self) -> None:
        now = self.get_clock().now().to_msg()

        t1 = TransformStamped()
        t1.header.stamp = now
        t1.header.frame_id = self.map_frame
        t1.child_frame_id = self.odom_frame
        t1.transform.rotation.w = 1.0

        # map -> camera_color_optical_frame: camera at (0,0,cam_z) pointing -z.
        # In optical convention the camera x-axis is right, y is down, z forward.
        # That means child axes expressed in the parent (map) are:
        #   x_cam -> +x_world (east)
        #   y_cam -> -y_world (south)
        #   z_cam -> -z_world (down)
        # A 180 deg rotation about x — quaternion (1, 0, 0, 0).
        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = self.map_frame
        t2.child_frame_id = self.cam_frame
        t2.transform.translation.z = self.cam_z
        t2.transform.rotation.x = 1.0
        t2.transform.rotation.w = 0.0

        self.static_tfb.sendTransform([t1, t2])

    def _build_map(self) -> OccupancyGrid:
        res = 0.1
        w, h = 80, 80  # 8 m x 8 m centered on origin
        origin_x, origin_y = -4.0, -4.0
        grid = np.zeros((h, w), dtype=np.int8)
        grid[0, :] = 100
        grid[-1, :] = 100
        grid[:, 0] = 100
        grid[:, -1] = 100
        msg = OccupancyGrid()
        msg.header.frame_id = self.map_frame
        msg.info.resolution = res
        msg.info.width = w
        msg.info.height = h
        msg.info.origin.position.x = origin_x
        msg.info.origin.position.y = origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = [int(v) for v in grid.flatten().tolist()]
        return msg

    # ------------------------------------------------------------------
    # publishers
    # ------------------------------------------------------------------

    def _broadcast_dynamic_tf(self) -> None:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = float(self.rx)
        t.transform.translation.y = float(self.ry)
        qx, qy, qz, qw = _yaw_to_quat(self.ryaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tfb.sendTransform(t)

    def _publish_odom(self) -> None:
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame
        msg.pose.pose.position.x = float(self.rx)
        msg.pose.pose.position.y = float(self.ry)
        qx, qy, qz, qw = _yaw_to_quat(self.ryaw)
        msg.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        self.odom_pub.publish(msg)

    def _publish_map(self) -> None:
        self.occ_grid.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self.occ_grid)

    def _publish_joint_states(self) -> None:
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = [
            'joint_lift', 'wrist_extension', 'joint_wrist_yaw',
            'joint_head_pan', 'joint_head_tilt',
            'joint_gripper_finger_left',
        ]
        js.position = [0.5, 0.0, 0.0, 0.0, -0.7, 0.0]
        self.js_pub.publish(js)

    def _publish_camera(self) -> None:
        rgb = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        depth = np.zeros((self.H, self.W), dtype=np.uint16)
        stamp = self.get_clock().now().to_msg()

        for label, pos in self.objects.items():
            if pos is None:
                continue
            xw, yw, zw = pos
            # Project into bird's-eye camera (see _publish_static_tfs for geometry).
            xc = xw
            yc = -yw
            zc = self.cam_z - zw
            if zc <= 0.05:
                continue
            u = int(round(self.fx * xc / zc + self.img_cx))
            v = int(round(self.fy * yc / zc + self.img_cy))
            if not (0 <= u < self.W and 0 <= v < self.H):
                continue
            color = OBJECT_COLORS_BGR[label]
            cv2.circle(rgb, (u, v), 30, color, -1)
            cv2.circle(depth, (u, v), 30, int(zc * 1000), -1)

        info = CameraInfo()
        info.header.frame_id = self.cam_frame
        info.header.stamp = stamp
        info.width = self.W
        info.height = self.H
        info.k = [self.fx, 0.0, self.img_cx,
                  0.0, self.fy, self.img_cy,
                  0.0, 0.0, 1.0]
        info.p = [self.fx, 0.0, self.img_cx, 0.0,
                  0.0, self.fy, self.img_cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        self.info_pub.publish(info)

        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='bgr8')
        rgb_msg.header.frame_id = self.cam_frame
        rgb_msg.header.stamp = stamp
        self.rgb_pub.publish(rgb_msg)

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='16UC1')
        depth_msg.header.frame_id = self.cam_frame
        depth_msg.header.stamp = stamp
        self.depth_pub.publish(depth_msg)

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def _execute_nav(self, goal_handle):
        g = goal_handle.request.pose.pose
        gx = float(g.position.x)
        gy = float(g.position.y)
        gyaw = _quat_to_yaw(
            g.orientation.x, g.orientation.y, g.orientation.z, g.orientation.w)
        self.get_logger().info(
            f'Nav goal: ({gx:.2f}, {gy:.2f}, yaw={gyaw:.2f})')

        dt = 0.05
        deadline = time.time() + 20.0  # upper bound
        while rclpy.ok() and time.time() < deadline:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return NavigateToPose.Result()
            dx = gx - self.rx
            dy = gy - self.ry
            d = math.hypot(dx, dy)
            dyaw = _angle_wrap(gyaw - self.ryaw)
            if d < 0.04 and abs(dyaw) < 0.05:
                break
            step = min(d, self.nav_speed * dt)
            if d > 1e-3:
                self.rx += step * dx / d
                self.ry += step * dy / d
            yaw_step = max(-self.nav_yaw_rate * dt,
                           min(self.nav_yaw_rate * dt, dyaw))
            self.ryaw = _angle_wrap(self.ryaw + yaw_step)
            time.sleep(dt)

        self.rx, self.ry, self.ryaw = gx, gy, gyaw
        goal_handle.succeed()
        return NavigateToPose.Result()

    def _execute_pick(self, goal_handle):
        res = FollowJointTrajectory.Result()
        best_label, best_d = None, 1e9
        for label, pos in self.objects.items():
            if pos is None:
                continue
            d = math.hypot(pos[0] - self.rx, pos[1] - self.ry)
            if d < best_d:
                best_d, best_label = d, label
        time.sleep(0.5)
        if best_label is not None and best_d < 1.5:
            self.held_object = best_label
            self.objects[best_label] = None
            self.get_logger().info(
                f'PICK: picked {best_label} at dist={best_d:.2f} m')
            res.error_code = 0
            goal_handle.succeed()
        else:
            self.get_logger().warn(
                f'PICK: no object within reach (nearest={best_label} at {best_d:.2f})')
            res.error_code = -1
            goal_handle.abort()
        return res

    def _execute_place(self, goal_handle):
        res = FollowJointTrajectory.Result()
        time.sleep(0.5)
        if self.held_object is None:
            self.get_logger().warn('PLACE: nothing held')
            res.error_code = -1
            goal_handle.abort()
            return res
        # Drop in front of the robot (short reach).
        px = self.rx + 0.4 * math.cos(self.ryaw)
        py = self.ry + 0.4 * math.sin(self.ryaw)
        self.objects[self.held_object] = (px, py, 0.4)
        self.get_logger().info(
            f'PLACE: {self.held_object} -> ({px:.2f}, {py:.2f})')
        self.held_object = None
        res.error_code = 0
        goal_handle.succeed()
        return res


# ---------------------------------------------------------------------
# math helpers
# ---------------------------------------------------------------------

def _yaw_to_quat(yaw: float):
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s, c)


def _angle_wrap(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeSimNode()
    executor = MultiThreadedExecutor(num_threads=8)
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
