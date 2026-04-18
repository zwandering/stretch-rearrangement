#!/usr/bin/env python3
"""End-to-end simulation integration test — see test/README.md for details."""

import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class E2ETester(Node):
    def __init__(self):
        super().__init__('e2e_sim_tester')
        self.state_history: List[str] = []
        self.current_state: Optional[str] = None
        self.detected_labels = set()
        self.start_time = time.time()

        self.create_subscription(String, '/executor/state', self._on_state, 10)
        self.create_subscription(
            MarkerArray, '/detected_objects', self._on_detections, 10,
        )
        self.results = []

    def _on_state(self, msg):
        if msg.data != self.current_state:
            elapsed = time.time() - self.start_time
            self.state_history.append(msg.data)
            self.current_state = msg.data
            self.get_logger().info(f'  [{elapsed:.1f}s] STATE → {msg.data}')

    def _on_detections(self, msg):
        for m in msg.markers:
            if m.type == Marker.CUBE:
                self.detected_labels.add(m.ns)

    def spin_for(self, seconds: float):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.2)

    def wait_for_terminal(self, timeout_s: float = 300.0) -> Optional[str]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.current_state in ('DONE', 'FAILED'):
                return self.current_state
            rclpy.spin_once(self, timeout_sec=0.5)
        return None

    def run_all_tests(self):
        self.get_logger().info('=' * 60)
        self.get_logger().info('End-to-end simulation integration test')
        self.get_logger().info('Waiting for system to finish or timeout (max 300s) ...')
        self.get_logger().info('=' * 60)

        terminal = self.wait_for_terminal(timeout_s=300.0)
        total_time = time.time() - self.start_time
        seen = set(self.state_history)

        self.get_logger().info(f'Total time: {total_time:.1f}s')
        self.get_logger().info(f'Visited states: {self.state_history}')

        # -----------------------------------------------------------
        # Test 1: entered HEAD_SCAN / EXPLORE
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 1] Entered HEAD_SCAN / EXPLORE ...')
        ok = 'HEAD_SCAN' in seen or 'EXPLORE' in seen
        self.results.append(('TEST1: exploration phase', ok,
                              f'HEAD_SCAN={"yes" if "HEAD_SCAN" in seen else "no"}, '
                              f'EXPLORE={"yes" if "EXPLORE" in seen else "no"}'))

        # -----------------------------------------------------------
        # Test 2: detected 3 objects
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 2] Object detection ...')
        ok = len(self.detected_labels) >= 3
        self.results.append(('TEST2: object detection', ok,
                              f'Detected: {self.detected_labels}'))

        # -----------------------------------------------------------
        # Test 3: entered PLAN state
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 3] Entered PLAN state ...')
        ok = 'PLAN' in seen
        self.results.append(('TEST3: PLAN', ok, f'PLAN state: {"yes" if ok else "no"}'))

        # -----------------------------------------------------------
        # Test 4: executed pick/place
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 4] Executing pick/place ...')
        has_pick = 'PICK' in seen
        has_place = 'PLACE' in seen
        ok = has_pick and has_place
        self.results.append(('TEST4: pick/place', ok,
                              f'PICK={"yes" if has_pick else "no"}, '
                              f'PLACE={"yes" if has_place else "no"}'))

        # -----------------------------------------------------------
        # Test 5: final state is DONE
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 5] Final state ...')
        ok = terminal == 'DONE'
        self.results.append(('TEST5: DONE', ok, f'Final: {terminal or "timeout"}'))

        # -----------------------------------------------------------
        # Test 6: metrics file validation
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 6] metrics file ...')
        metrics_path = Path('/tmp/rearrangement_metrics.json')
        if metrics_path.exists():
            try:
                m = json.loads(metrics_path.read_text())
                pick_rate = (m.get('pick_successes', 0) /
                             max(m.get('pick_attempts', 1), 1))
                place_rate = (m.get('place_successes', 0) /
                              max(m.get('place_attempts', 1), 1))
                task_results = m.get('task_results', [])
                task_success = sum(1 for t in task_results if t.get('success'))
                ok = m.get('success', False)
                msg = (f'success={m.get("success")}, '
                       f'pick={m.get("pick_successes")}/{m.get("pick_attempts")} '
                       f'({pick_rate:.0%}), '
                       f'place={m.get("place_successes")}/{m.get("place_attempts")} '
                       f'({place_rate:.0%}), '
                       f'tasks={task_success}/{len(task_results)}')
            except Exception as e:
                ok = False
                msg = f'Parse error: {e}'
        else:
            ok = False
            msg = 'File not found'
        self.results.append(('TEST6: metrics', ok, msg))

        # -----------------------------------------------------------
        # Test 7: total time
        # Should complete within 120s in simulation
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 7] Total time ...')
        ok = total_time < 180.0 and terminal is not None
        self.results.append(('TEST7: time', ok,
                              f'{total_time:.1f}s (threshold 180s)'))

        # -----------------------------------------------------------
        # Summary
        # -----------------------------------------------------------
        self.get_logger().info('=' * 60)
        self.get_logger().info('End-to-end simulation test summary:')
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        for label, ok, msg in self.results:
            status = 'PASS' if ok else 'FAIL'
            self.get_logger().info(f'  [{status}] {label}: {msg}')
        self.get_logger().info(f'Passed: {passed}/{total}')
        self.get_logger().info('State transition sequence: ' + ' → '.join(self.state_history))
        self.get_logger().info('=' * 60)


def main():
    rclpy.init()
    tester = E2ETester()
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
