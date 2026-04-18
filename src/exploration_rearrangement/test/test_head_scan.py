#!/usr/bin/env python3
"""Robot integration test for HeadScanNode — see test/README.md for details."""

import sys
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class HeadScanTester(Node):
    def __init__(self):
        super().__init__('head_scan_tester')
        self.start_cli = self.create_client(Trigger, '/head/start_scan')
        self.stop_cli = self.create_client(Trigger, '/head/stop_scan')
        self.once_cli = self.create_client(Trigger, '/head/scan_once')
        self.results = []

    def call_trigger(self, client, label: str, timeout: float = 10.0) -> bool:
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'[{label}] service unavailable, timeout 5s')
            self.results.append((label, False, 'service unavailable'))
            return False
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result() is None:
            self.get_logger().error(f'[{label}] call timeout')
            self.results.append((label, False, 'call timeout'))
            return False
        res = future.result()
        self.results.append((label, res.success, res.message))
        return res.success

    def run_all_tests(self):
        self.get_logger().info('=' * 60)
        self.get_logger().info('Starting HeadScanNode tests')
        self.get_logger().info('=' * 60)

        # -----------------------------------------------------------
        # Test 1: start_scan
        # Expected: success=True, head begins periodic rotation
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 1] Calling /head/start_scan ...')
        ok = self.call_trigger(self.start_cli, 'TEST1: start_scan')
        self.get_logger().info(
            f'  Result: {"PASS" if ok else "FAIL"} — expected success=True'
        )

        # Wait a few seconds for periodic scanning to execute several waypoints
        self.get_logger().info('  Waiting 12s, observe periodic head movement ...')
        time.sleep(12.0)

        # -----------------------------------------------------------
        # Test 2: stop_scan
        # Expected: success=True, head stops rotating
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 2] Calling /head/stop_scan ...')
        ok = self.call_trigger(self.stop_cli, 'TEST2: stop_scan')
        self.get_logger().info(
            f'  Result: {"PASS" if ok else "FAIL"} — expected success=True, head stopped'
        )

        time.sleep(2.0)

        # -----------------------------------------------------------
        # Test 3: scan_once (while stopped)
        # Expected: success=True, head completes one full sweep then returns
        # Note: scan_once is blocking, needs longer timeout
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 3] Calling /head/scan_once (while stopped) ...')
        ok = self.call_trigger(self.once_cli, 'TEST3: scan_once (stopped)', timeout=30.0)
        self.get_logger().info(
            f'  Result: {"PASS" if ok else "FAIL"} — expected success=True, full sweep'
        )

        time.sleep(1.0)

        # -----------------------------------------------------------
        # Test 4: rapid start→stop→start→stop toggle
        # Expected: all calls succeed, no crashes
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 4] Rapid start→stop toggle (3 rounds) ...')
        all_ok = True
        for i in range(3):
            ok1 = self.call_trigger(self.start_cli, f'TEST4: start #{i}')
            time.sleep(1.0)
            ok2 = self.call_trigger(self.stop_cli, f'TEST4: stop #{i}')
            time.sleep(0.5)
            if not (ok1 and ok2):
                all_ok = False
        self.get_logger().info(
            f'  Result: {"PASS" if all_ok else "FAIL"} — expected all start/stop succeed'
        )

        # -----------------------------------------------------------
        # Test 5: two consecutive scan_once calls
        # Expected: both succeed, no residual state causing failure
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 5] Two consecutive scan_once calls ...')
        ok1 = self.call_trigger(self.once_cli, 'TEST5: scan_once #1', timeout=30.0)
        ok2 = self.call_trigger(self.once_cli, 'TEST5: scan_once #2', timeout=30.0)
        self.get_logger().info(
            f'  Result: {"PASS" if ok1 and ok2 else "FAIL"} — expected both succeed'
        )

        # -----------------------------------------------------------
        # Summary
        # -----------------------------------------------------------
        self.get_logger().info('=' * 60)
        self.get_logger().info('HeadScanNode test summary:')
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        for label, ok, msg in self.results:
            status = 'PASS' if ok else 'FAIL'
            self.get_logger().info(f'  [{status}] {label}: {msg}')
        self.get_logger().info(f'Passed: {passed}/{total}')
        self.get_logger().info('=' * 60)


def main():
    rclpy.init()
    tester = HeadScanTester()
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
