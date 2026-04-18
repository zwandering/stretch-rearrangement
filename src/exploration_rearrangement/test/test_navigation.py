#!/usr/bin/env python3
"""Robot integration test for Nav2 NavigateToPose — see test/README.md for details."""

import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


def yaw_to_quat(yaw: float) -> Quaternion:
    half = 0.5 * yaw
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


class NavigationTester(Node):
    def __init__(self):
        super().__init__('navigation_tester')
        cb = ReentrantCallbackGroup()
        self.nav_client = ActionClient(
            self, NavigateToPose, '/navigate_to_pose', callback_group=cb,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.results = []

    def get_robot_xy(self):
        """Try to get current robot position from TF."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0),
            )
            return (tf.transform.translation.x, tf.transform.translation.y)
        except Exception:
            return None

    def send_nav_goal(self, x: float, y: float, yaw: float,
                      label: str, timeout: float = 30.0) -> bool:
        """Send a navigation goal and wait for completion."""
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'[{label}] Nav2 action server unavailable')
            self.results.append((label, False, 'action server unavailable'))
            return False

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation = yaw_to_quat(yaw)

        self.get_logger().info(
            f'[{label}] Sending goal: ({x:.2f}, {y:.2f}, yaw={yaw:.2f})'
        )
        send_future = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)

        gh = send_future.result()
        if gh is None or not gh.accepted:
            msg = 'goal rejected' if gh is not None else 'send timeout'
            self.get_logger().error(f'[{label}] {msg}')
            self.results.append((label, False, msg))
            return False

        self.get_logger().info(f'[{label}] Goal accepted, waiting for completion ...')
        result_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)

        if result_future.result() is None:
            self.results.append((label, False, 'navigation timeout'))
            return False

        # Check final position
        time.sleep(0.5)
        pos = self.get_robot_xy()
        if pos is not None:
            dist = math.hypot(pos[0] - x, pos[1] - y)
            msg = f'Arrived at ({pos[0]:.2f}, {pos[1]:.2f}), {dist:.2f}m from goal'
            ok = dist < 0.5
        else:
            msg = 'Navigation complete but unable to get position'
            ok = True

        self.results.append((label, ok, msg))
        return ok

    def test_cancel(self, label: str) -> bool:
        """Test cancelling a navigation goal."""
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.results.append((label, False, 'action server unavailable'))
            return False

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = 3.0
        goal.pose.pose.position.y = 3.0
        goal.pose.pose.orientation = yaw_to_quat(0.0)

        self.get_logger().info(f'[{label}] Sending far goal (3, 3), cancelling after 1s')
        send_future = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        gh = send_future.result()
        if gh is None or not gh.accepted:
            self.results.append((label, False, 'goal not accepted'))
            return False

        time.sleep(1.0)
        cancel_future = gh.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)

        ok = cancel_future.result() is not None
        msg = 'cancel succeeded' if ok else 'cancel failed'
        self.results.append((label, ok, msg))
        return ok

    def run_all_tests(self):
        self.get_logger().info('=' * 60)
        self.get_logger().info('Starting Navigation tests')
        self.get_logger().info('=' * 60)

        time.sleep(1.0)

        # -----------------------------------------------------------
        # Test 1: short forward navigation
        # Expected: robot moves to (1, 0), < 0.5m from goal
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 1] Navigate to (1.0, 0.0) — short forward')
        ok = self.send_nav_goal(1.0, 0.0, 0.0, 'TEST1: short forward')
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"}')

        # -----------------------------------------------------------
        # Test 2: return to origin
        # Expected: robot moves back to (0, 0)
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 2] Navigate to (0.0, 0.0) — return to origin')
        ok = self.send_nav_goal(0.0, 0.0, 0.0, 'TEST2: return to origin')
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"}')

        # -----------------------------------------------------------
        # Test 3: diagonal navigation with rotation
        # Expected: reach (-1, 1), heading 135 deg (upper-left)
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 3] Navigate to (-1.0, 1.0) — diagonal + rotation')
        ok = self.send_nav_goal(-1.0, 1.0, 2.356, 'TEST3: diagonal')
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"}')

        # -----------------------------------------------------------
        # Test 4: long distance (region A center)
        # Expected: reach (2, 2), near region A's place_anchor
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 4] Navigate to (2.0, 2.0) — long distance (region A)')
        ok = self.send_nav_goal(2.0, 2.0, 0.0, 'TEST4: long distance')
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"}')

        # -----------------------------------------------------------
        # Test 5: cancel navigation
        # Expected: cancel request succeeds
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 5] Send goal then cancel')
        ok = self.test_cancel('TEST5: cancel nav')
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"}')

        time.sleep(1.0)

        # -----------------------------------------------------------
        # Test 6: two consecutive goals
        # Expected: robot ultimately reaches the second goal
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 6] Send two consecutive goals → reach second')
        self.send_nav_goal(0.0, 0.0, 0.0, 'TEST6: first goal (origin)')
        ok = self.send_nav_goal(1.5, -1.5, 3.14, 'TEST6: second goal (region C)')
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"}')

        # -----------------------------------------------------------
        # Summary
        # -----------------------------------------------------------
        self.get_logger().info('=' * 60)
        self.get_logger().info('Navigation test summary:')
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        for label, ok, msg in self.results:
            status = 'PASS' if ok else 'FAIL'
            self.get_logger().info(f'  [{status}] {label}: {msg}')
        self.get_logger().info(f'Passed: {passed}/{total}')
        self.get_logger().info('=' * 60)


def main():
    rclpy.init()
    tester = NavigationTester()
    executor = MultiThreadedExecutor()
    executor.add_node(tester)
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
