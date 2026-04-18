#!/usr/bin/env python3
"""Robot integration test for ManipulationNode (pick/place) — see test/README.md for details."""

import math
import sys
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def yaw_to_quat(yaw: float) -> Quaternion:
    half = 0.5 * yaw
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


class ManipulationTester(Node):
    def __init__(self):
        super().__init__('manipulation_tester')
        cb = ReentrantCallbackGroup()
        self.pick_client = ActionClient(
            self, FollowJointTrajectory, '/manipulation/pick', callback_group=cb,
        )
        self.place_client = ActionClient(
            self, FollowJointTrajectory, '/manipulation/place', callback_group=cb,
        )
        self.nav_client = ActionClient(
            self, NavigateToPose, '/navigate_to_pose', callback_group=cb,
        )
        self.stow_cli = self.create_client(Trigger, '/manipulation/stow', callback_group=cb)
        self.results = []

    def call_trigger(self, client, label: str) -> bool:
        if not client.wait_for_service(timeout_sec=5.0):
            self.results.append((label, False, 'service unavailable'))
            return False
        fut = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        res = fut.result()
        if res is None:
            self.results.append((label, False, 'call timeout'))
            return False
        self.results.append((label, res.success, res.message))
        return res.success

    def send_manip_action(self, client, label: str, timeout: float = 20.0) -> int:
        """Send manipulation action, return error_code. -999 means comm failure."""
        if not client.wait_for_server(timeout_sec=5.0):
            self.results.append((label, False, 'action server unavailable'))
            return -999

        goal = FollowJointTrajectory.Goal()
        fut = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        gh = fut.result()
        if gh is None or not gh.accepted:
            msg = 'goal rejected' if gh else 'send timeout'
            self.results.append((label, False, msg))
            return -999

        result_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_fut, timeout_sec=timeout)
        res = result_fut.result()
        if res is None:
            self.results.append((label, False, 'execution timeout'))
            return -999

        code = res.result.error_code
        ok = (code == 0)
        self.results.append((label, ok, f'error_code={code}'))
        return code

    def navigate_to(self, x: float, y: float, yaw: float, timeout: float = 30.0) -> bool:
        """Helper: navigate to the given position."""
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            return False
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation = yaw_to_quat(yaw)
        fut = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        gh = fut.result()
        if gh is None or not gh.accepted:
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=timeout)
        return rf.result() is not None

    def run_all_tests(self):
        self.get_logger().info('=' * 60)
        self.get_logger().info('Starting ManipulationNode tests')
        self.get_logger().info('=' * 60)

        time.sleep(1.0)

        # -----------------------------------------------------------
        # Test 1: stow
        # Expected: success=True, arm retracts to safe position
        # Risk: stretch_driver not running, stow service path mismatch
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 1] Calling /manipulation/stow ...')
        ok = self.call_trigger(self.stow_cli, 'TEST1: stow')
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"}')

        # -----------------------------------------------------------
        # Test 2: pick (navigate near object first)
        # In sim, blue_bottle is at (1.5, 1.5); navigate nearby then pick
        # Expected: error_code=0 (successful grasp)
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 2] Navigate near object → pick ...')
        self.get_logger().info('  Navigating to (1.0, 1.5) ...')
        nav_ok = self.navigate_to(1.0, 1.5, 0.0)
        self.get_logger().info(f'  Nav: {"success" if nav_ok else "failed"}')
        if nav_ok:
            code = self.send_manip_action(self.pick_client, 'TEST2: pick')
            self.get_logger().info(
                f'  Result: {"PASS" if code == 0 else "FAIL"} — error_code={code}'
            )
        else:
            self.results.append(('TEST2: pick', False, 'Navigation failed, cannot test pick'))
            self.get_logger().info('  Result: FAIL — navigation failed')

        # -----------------------------------------------------------
        # Test 3: place (navigate to target region then place)
        # If TEST2 grasped an object, navigate to region C (1.5, -1.5) and place
        # Expected: error_code=0
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 3] Navigate to target region → place ...')
        self.get_logger().info('  Navigating to (1.5, -1.5) ...')
        nav_ok = self.navigate_to(1.5, -1.5, 3.14)
        self.get_logger().info(f'  Nav: {"success" if nav_ok else "failed"}')
        if nav_ok:
            code = self.send_manip_action(self.place_client, 'TEST3: place')
            self.get_logger().info(
                f'  Result: {"PASS" if code == 0 else "FAIL"} — error_code={code}'
            )
        else:
            self.results.append(('TEST3: place', False, 'Navigation failed, cannot test place'))
            self.get_logger().info('  Result: FAIL — navigation failed')

        # -----------------------------------------------------------
        # Test 4: two consecutive stow calls — idempotency
        # Expected: both succeed
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 4] Two consecutive stow calls ...')
        ok1 = self.call_trigger(self.stow_cli, 'TEST4: stow #1')
        ok2 = self.call_trigger(self.stow_cli, 'TEST4: stow #2')
        self.get_logger().info(
            f'  Result: {"PASS" if ok1 and ok2 else "FAIL"}'
        )

        # -----------------------------------------------------------
        # Test 5: pick far from objects — failure handling
        # Navigate far from all objects, then pick
        # Sim expected: error_code=-1 (no object in range)
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 5] Pick far from objects (should fail) ...')
        self.navigate_to(-2.5, -2.5, 0.0)
        code = self.send_manip_action(self.pick_client, 'TEST5: pick (far from objects)')
        # Pick far from objects should fail (code != 0)
        expected_fail = (code != 0)
        # Overwrite record: failure is the expected behavior
        if self.results and self.results[-1][0] == 'TEST5: pick (far from objects)':
            self.results[-1] = (
                'TEST5: pick (far from objects)',
                expected_fail,
                f'error_code={code}, expected failure {"OK" if expected_fail else "ERROR: should not succeed"}',
            )
        self.get_logger().info(
            f'  Result: {"PASS" if expected_fail else "FAIL"} — expected pick failure, actual code={code}'
        )

        # -----------------------------------------------------------
        # Test 6: place with nothing held — failure handling
        # Sim expected: error_code=-1
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 6] Place with nothing held (should fail) ...')
        code = self.send_manip_action(self.place_client, 'TEST6: place (nothing held)')
        expected_fail = (code != 0)
        if self.results and self.results[-1][0] == 'TEST6: place (nothing held)':
            self.results[-1] = (
                'TEST6: place (nothing held)',
                expected_fail,
                f'error_code={code}, expected failure {"OK" if expected_fail else "ERROR"}',
            )
        self.get_logger().info(
            f'  Result: {"PASS" if expected_fail else "FAIL"} — expected place failure, actual code={code}'
        )

        # -----------------------------------------------------------
        # Test 7: full flow nav→pick→nav→place
        # Navigate to red_box (1.5,-1.5), pick, then navigate to region A and place
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 7] Full flow: nav→pick→nav→place ...')
        # 7a: navigate near red_box
        self.get_logger().info('  7a: Navigating near red_box (1.0, -1.5) ...')
        if not self.navigate_to(1.0, -1.5, 0.0):
            self.results.append(('TEST7: full flow', False, '7a navigation failed'))
            self.get_logger().info('  Result: FAIL')
        else:
            # 7b: pick
            self.get_logger().info('  7b: pick ...')
            code = self.send_manip_action(self.pick_client, 'TEST7b: pick')
            if code != 0:
                self.results.append(('TEST7: full flow', False, f'7b pick failed code={code}'))
                self.get_logger().info('  Result: FAIL — pick failed')
            else:
                # 7c: navigate to region A
                self.get_logger().info('  7c: Navigating to region A (1.5, 1.5) ...')
                if not self.navigate_to(1.5, 1.5, 0.0):
                    self.results.append(('TEST7: full flow', False, '7c navigation failed'))
                    self.get_logger().info('  Result: FAIL — navigation to place point failed')
                else:
                    # 7d: place
                    self.get_logger().info('  7d: place ...')
                    code = self.send_manip_action(self.place_client, 'TEST7d: place')
                    ok = (code == 0)
                    self.results.append(('TEST7: full flow', ok, f'place code={code}'))
                    self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"}')

        # -----------------------------------------------------------
        # Summary
        # -----------------------------------------------------------
        self.get_logger().info('=' * 60)
        self.get_logger().info('ManipulationNode test summary:')
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        for label, ok, msg in self.results:
            status = 'PASS' if ok else 'FAIL'
            self.get_logger().info(f'  [{status}] {label}: {msg}')
        self.get_logger().info(f'Passed: {passed}/{total}')
        self.get_logger().info('=' * 60)


def main():
    rclpy.init()
    tester = ManipulationTester()
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
