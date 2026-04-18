#!/usr/bin/env python3
"""Robot integration test for ExplorationNode — see test/README.md for details."""

import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray


class ExplorationTester(Node):
    def __init__(self):
        super().__init__('exploration_tester')
        self.status_msgs = []
        self.frontier_markers_received = False
        self.frontier_marker_count = 0

        self.create_subscription(
            String, '/exploration/status', self._on_status, 10,
        )
        self.create_subscription(
            MarkerArray, '/exploration/frontiers', self._on_frontiers, 10,
        )
        self.start_cli = self.create_client(Trigger, '/exploration/start')
        self.stop_cli = self.create_client(Trigger, '/exploration/stop')
        self.results = []

    def _on_status(self, msg):
        self.status_msgs.append(msg.data)
        self.get_logger().info(f'  [STATUS] {msg.data}')

    def _on_frontiers(self, msg):
        self.frontier_markers_received = True
        sphere_markers = [m for m in msg.markers if m.type == 7]  # SPHERE_LIST
        if sphere_markers:
            self.frontier_marker_count = sum(len(m.points) for m in sphere_markers)

    def spin_for(self, seconds: float):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def call_trigger(self, client, label: str) -> bool:
        if not client.wait_for_service(timeout_sec=5.0):
            self.results.append((label, False, 'service unavailable'))
            return False
        fut = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        res = fut.result()
        if res is None:
            self.results.append((label, False, 'call timeout'))
            return False
        self.results.append((label, res.success, res.message))
        return res.success

    def run_all_tests(self):
        self.get_logger().info('=' * 60)
        self.get_logger().info('Starting ExplorationNode tests')
        self.get_logger().info('=' * 60)

        time.sleep(2.0)

        # -----------------------------------------------------------
        # Test 1: /exploration/start service
        # Expected: success=True
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 1] Calling /exploration/start ...')
        ok = self.call_trigger(self.start_cli, 'TEST1: start')
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"}')

        # -----------------------------------------------------------
        # Test 2: check frontiers and status output after starting
        # Expected: receive frontier markers or status messages
        # In sim, most of the map is known, so frontiers may exhaust quickly → "done"
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 2] Waiting for exploration output (15s) ...')
        self.status_msgs.clear()
        self.frontier_markers_received = False
        self.spin_for(15.0)
        got_status = len(self.status_msgs) > 0
        got_frontiers = self.frontier_markers_received
        ok = got_status or got_frontiers
        msg = (f'status messages: {self.status_msgs[:5]}, '
               f'frontier markers: {"yes" if got_frontiers else "no"} '
               f'({self.frontier_marker_count} frontier points)')
        self.results.append(('TEST2: exploration output', ok, msg))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"} — {msg}')

        # -----------------------------------------------------------
        # Test 3: /exploration/stop service
        # Expected: success=True, exploration stops
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 3] Calling /exploration/stop ...')
        ok = self.call_trigger(self.stop_cli, 'TEST3: stop')
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"}')

        # -----------------------------------------------------------
        # Test 4: no new status messages after stop (unless already "done")
        # Wait 8s, check for new "navigating" messages
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 4] Verifying no new navigation after stop (8s) ...')
        self.status_msgs.clear()
        self.spin_for(8.0)
        new_navigating = [s for s in self.status_msgs if s == 'navigating']
        ok = len(new_navigating) == 0
        msg = f'navigating messages after stop: {len(new_navigating)}'
        self.results.append(('TEST4: quiet after stop', ok, msg))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"} — {msg}')

        # -----------------------------------------------------------
        # Test 5: rapid start/stop toggle (3 rounds)
        # Expected: no crashes, all calls succeed
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 5] Rapid start/stop toggle (3 rounds) ...')
        all_ok = True
        for i in range(3):
            ok1 = self.call_trigger(self.start_cli, f'TEST5: start #{i}')
            self.spin_for(3.0)
            ok2 = self.call_trigger(self.stop_cli, f'TEST5: stop #{i}')
            self.spin_for(1.0)
            if not (ok1 and ok2):
                all_ok = False
        self.get_logger().info(
            f'  Result: {"PASS" if all_ok else "FAIL"} — 3 rounds start/stop complete'
        )

        # -----------------------------------------------------------
        # Test 6: verify exploration completes in simulation
        # fake_sim_node's map is mostly known, frontiers should exhaust quickly
        # Expected: receive "done" status (or very few frontiers)
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 6] Starting exploration, waiting for completion or timeout (30s) ...')
        self.status_msgs.clear()
        self.call_trigger(self.start_cli, 'TEST6: start')
        self.spin_for(30.0)
        got_done = 'done' in self.status_msgs
        ok = got_done or self.frontier_marker_count == 0
        msg = (f'"done" status: {"yes" if got_done else "no"}, '
               f'final frontier points: {self.frontier_marker_count}')
        self.results.append(('TEST6: exploration complete', ok, msg))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"} — {msg}')

        self.call_trigger(self.stop_cli, 'cleanup: stop')

        # -----------------------------------------------------------
        # Summary
        # -----------------------------------------------------------
        self.get_logger().info('=' * 60)
        self.get_logger().info('ExplorationNode test summary:')
        main_results = [r for r in self.results
                        if r[0].startswith('TEST') and '#' not in r[0]]
        passed = sum(1 for _, ok, _ in main_results if ok)
        total = len(main_results)
        for label, ok, msg in main_results:
            status = 'PASS' if ok else 'FAIL'
            self.get_logger().info(f'  [{status}] {label}: {msg}')
        self.get_logger().info(f'Passed: {passed}/{total}')
        self.get_logger().info('=' * 60)


def main():
    rclpy.init()
    tester = ExplorationTester()
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
