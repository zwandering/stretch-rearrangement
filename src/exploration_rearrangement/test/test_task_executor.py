#!/usr/bin/env python3
"""Robot integration test for TaskExecutorNode — see test/README.md for details."""

import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


class TaskExecutorTester(Node):
    def __init__(self):
        super().__init__('task_executor_tester')
        self.state_history: List[str] = []
        self.current_state: Optional[str] = None

        self.create_subscription(
            String, '/executor/state', self._on_state, 10,
        )
        self.start_cli = self.create_client(Trigger, '/executor/start')
        self.abort_cli = self.create_client(Trigger, '/executor/abort')
        self.results = []

    def _on_state(self, msg):
        if msg.data != self.current_state:
            self.state_history.append(msg.data)
            self.current_state = msg.data
            self.get_logger().info(f'  [STATE] → {msg.data}')

    def spin_for(self, seconds: float):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def call_trigger(self, client, label: str, timeout: float = 10.0) -> bool:
        if not client.wait_for_service(timeout_sec=5.0):
            self.results.append((label, False, 'service unavailable'))
            return False
        fut = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        res = fut.result()
        if res is None:
            self.results.append((label, False, 'call timeout'))
            return False
        self.results.append((label, res.success, res.message))
        return res.success

    def wait_for_state(self, target: str, timeout_s: float = 60.0) -> bool:
        """Wait for state to become target, or timeout."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.current_state == target:
                return True
            rclpy.spin_once(self, timeout_sec=0.5)
        return False

    def wait_for_terminal_state(self, timeout_s: float = 180.0) -> Optional[str]:
        """Wait for DONE or FAILED."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.current_state in ('DONE', 'FAILED'):
                return self.current_state
            rclpy.spin_once(self, timeout_sec=0.5)
        return None

    def run_all_tests(self):
        self.get_logger().info('=' * 60)
        self.get_logger().info('Starting TaskExecutorNode tests')
        self.get_logger().info('=' * 60)

        time.sleep(2.0)

        # -----------------------------------------------------------
        # Test 1: initial state
        # Executor should be in IDLE after launch (start_on_launch=false).
        # We may not receive IDLE via topic (state only published on transitions),
        # but can verify it hasn't started doing anything.
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 1] Verifying initial state (should be IDLE) ...')
        self.spin_for(3.0)
        ok = self.current_state is None or self.current_state == 'IDLE'
        msg = f'Current state: {self.current_state or "none (IDLE not published)"}'
        self.results.append(('TEST1: initial IDLE', ok, msg))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"} — {msg}')

        # -----------------------------------------------------------
        # Test 2: /executor/start service
        # Expected: success=True, state machine begins running
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 2] Calling /executor/start ...')
        self.state_history.clear()
        ok = self.call_trigger(self.start_cli, 'TEST2: start')
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"}')

        # -----------------------------------------------------------
        # Test 3: monitor state transitions
        # Expected: at least HEAD_SCAN and EXPLORE visited
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 3] Waiting for state transitions (30s) ...')
        self.spin_for(30.0)
        expected_states = {'HEAD_SCAN', 'EXPLORE'}
        seen = set(self.state_history)
        hit = expected_states & seen
        ok = len(hit) >= 1
        msg = f'Visited states: {self.state_history}, expected to include: {expected_states}'
        self.results.append(('TEST3: state transitions', ok, msg))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"} — {msg}')

        # -----------------------------------------------------------
        # Test 4: /executor/abort
        # Expected: state becomes FAILED
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 4] Calling /executor/abort ...')
        ok = self.call_trigger(self.abort_cli, 'TEST4: abort')
        self.spin_for(2.0)
        is_failed = self.current_state == 'FAILED'
        self.results.append(('TEST4: abort→FAILED', is_failed,
                              f'State after abort: {self.current_state}'))
        self.get_logger().info(
            f'  Result: {"PASS" if is_failed else "FAIL"} — State: {self.current_state}'
        )

        # -----------------------------------------------------------
        # Test 5: start after abort (should be ineffective, not in IDLE)
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 5] Trying start after abort ...')
        prev_state = self.current_state
        ok = self.call_trigger(self.start_cli, 'TEST5: start after abort')
        self.spin_for(2.0)
        still_failed = self.current_state == 'FAILED'
        self.results.append(('TEST5: start after abort', still_failed,
                              f'State still {self.current_state} (expected unchanged)'))
        self.get_logger().info(
            f'  Result: {"PASS" if still_failed else "FAIL"} — '
            f'State: {self.current_state}'
        )

        # -----------------------------------------------------------
        # Test 6: full run (requires executor node restart)
        # Since executor is now FAILED, a full test requires restarting the node.
        # Here we just check for a metrics file from a prior run.
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 6] Checking metrics file ...')
        self.get_logger().info(
            '  Note: Full run test requires sim.launch.py start_on_launch:=true'
        )
        metrics_path = Path('/tmp/rearrangement_metrics.json')
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
                has_start = metrics.get('t_start') is not None
                has_end = metrics.get('t_end') is not None
                ok = has_start and has_end
                msg = (f'backend={metrics.get("backend")}, '
                       f'success={metrics.get("success")}, '
                       f'pick={metrics.get("pick_successes")}/{metrics.get("pick_attempts")}, '
                       f'place={metrics.get("place_successes")}/{metrics.get("place_attempts")}')
            except Exception as e:
                ok = False
                msg = f'Parse error: {e}'
        else:
            ok = False
            msg = 'File not found (complete a full run first)'
        self.results.append(('TEST6: metrics file', ok, msg))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"} — {msg}')

        # -----------------------------------------------------------
        # Summary
        # -----------------------------------------------------------
        self.get_logger().info('=' * 60)
        self.get_logger().info('TaskExecutorNode test summary:')
        main_results = [r for r in self.results if r[0].startswith('TEST')]
        passed = sum(1 for _, ok, _ in main_results if ok)
        total = len(main_results)
        for label, ok, msg in main_results:
            status = 'PASS' if ok else 'FAIL'
            self.get_logger().info(f'  [{status}] {label}: {msg}')
        self.get_logger().info(f'Passed: {passed}/{total}')
        self.get_logger().info('=' * 60)

        self.get_logger().info('')
        self.get_logger().info('Hint: To run a full end-to-end test, use:')
        self.get_logger().info(
            '  ros2 launch exploration_rearrangement sim.launch.py start_on_launch:=true'
        )
        self.get_logger().info('  Then monitor /executor/state topic and /tmp/rearrangement_metrics.json')


def main():
    rclpy.init()
    tester = TaskExecutorTester()
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
