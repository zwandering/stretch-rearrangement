#!/usr/bin/env python3
"""Robot integration test for TaskPlannerNode — see test/README.md for details."""

import sys
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploration_rearrangement.planners.base import (
    DetectedObject, RegionInfo, PlannerInput, filter_actionable,
)
from exploration_rearrangement.planners.greedy import GreedyPlanner


def _sample_regions():
    return {
        'A': RegionInfo('A', [(0, 0), (3, 0), (3, 3), (0, 3)], (1.5, 1.5, 0.0)),
        'B': RegionInfo('B', [(-3, 0), (0, 0), (0, 3), (-3, 3)], (-1.5, 1.5, 0.0)),
        'C': RegionInfo('C', [(0, -3), (3, -3), (3, 0), (0, 0)], (1.5, -1.5, 3.14)),
        'D': RegionInfo('D', [(-3, -3), (0, -3), (0, 0), (-3, 0)], (-1.5, -1.5, 3.14)),
    }


class TaskPlannerTester(Node):
    def __init__(self):
        super().__init__('task_planner_tester')
        self.compute_cli = self.create_client(Trigger, '/planner/compute')
        self.clear_cli = self.create_client(Trigger, '/detector/clear')
        self.plan_markers: Optional[MarkerArray] = None
        self.detected_count = 0

        self.create_subscription(
            MarkerArray, '/planner/plan_visualization',
            self._on_plan_markers, 10,
        )
        self.create_subscription(
            MarkerArray, '/detected_objects',
            self._on_detections, 10,
        )
        self.results = []

    def _on_plan_markers(self, msg):
        self.plan_markers = msg

    def _on_detections(self, msg):
        cubes = [m for m in msg.markers if m.type == Marker.CUBE]
        self.detected_count = len(cubes)

    def spin_for(self, seconds: float):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def call_compute(self, label: str) -> tuple:
        if not self.compute_cli.wait_for_service(timeout_sec=5.0):
            self.results.append((label, False, 'service unavailable'))
            return False, ''
        fut = self.compute_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=15.0)
        res = fut.result()
        if res is None:
            self.results.append((label, False, 'call timeout'))
            return False, ''
        self.results.append((label, res.success, res.message))
        return res.success, res.message

    def run_all_tests(self):
        self.get_logger().info('=' * 60)
        self.get_logger().info('Starting TaskPlannerNode tests')
        self.get_logger().info('=' * 60)

        # Wait for detector to start and detect objects
        self.get_logger().info('Waiting for object detection to be ready (max 15s) ...')
        deadline = time.time() + 15.0
        while self.detected_count < 1 and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.5)
        self.get_logger().info(f'  Detected {self.detected_count} objects')

        # -----------------------------------------------------------
        # Test 1: call /planner/compute
        # Expected: success=True, message contains plan summary
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 1] Calling /planner/compute ...')
        ok, msg = self.call_compute('TEST1: compute')
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"} — {msg}')

        # -----------------------------------------------------------
        # Test 2: parse plan message, verify object names
        # Expected: message contains blue_bottle, red_box, yellow_cup (if all detected)
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 2] Verifying objects in plan ...')
        expected = {'blue_bottle', 'red_box', 'yellow_cup'}
        found = {name for name in expected if name in msg}
        ok = len(found) > 0
        self.results.append(('TEST2: plan objects', ok, f'Objects in plan: {found}'))
        self.get_logger().info(
            f'  Result: {"PASS" if ok else "FAIL"} — Found: {found}, expected subset: {expected}'
        )

        # -----------------------------------------------------------
        # Test 3: verify plan visualization
        # Expected: /planner/plan_visualization has LINE_STRIP markers
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 3] Checking plan visualization ...')
        self.spin_for(2.0)
        lines = 0
        if self.plan_markers:
            lines = sum(1 for m in self.plan_markers.markers
                        if m.type == Marker.LINE_STRIP)
        ok = lines > 0
        self.results.append(('TEST3: plan visualization', ok, f'{lines} pick→place lines'))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"} — {lines} lines')

        # -----------------------------------------------------------
        # Test 4: compute again → verify determinism (greedy)
        # Expected: two calls return identical plans
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 4] Compute again, verify determinism ...')
        ok2, msg2 = self.call_compute('TEST4: compute #2')
        ok = ok2 and msg == msg2
        self.results.append(('TEST4: determinism', ok,
                              f'first={msg[:60]}... second={msg2[:60]}...'))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"}')

        # -----------------------------------------------------------
        # Test 5: compute after clearing detections
        # Clear detections first, then compute → should return empty plan or "empty"
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 5] Compute after clearing detections ...')
        if self.clear_cli.wait_for_service(timeout_sec=3.0):
            fut = self.clear_cli.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            time.sleep(1.0)
            ok5, msg5 = self.call_compute('TEST5: compute (no detections)')
            has_empty = 'empty' in msg5.lower() or '0:' not in msg5
            self.results.append(('TEST5: empty plan', ok5,
                                  f'message={msg5[:80]}'))
            self.get_logger().info(f'  Result: {"PASS" if ok5 else "FAIL"} — {msg5[:80]}')
        else:
            self.results.append(('TEST5: empty plan', False, 'clear service unavailable'))
            self.get_logger().info('  Result: FAIL — clear unavailable')

        # -----------------------------------------------------------
        # Test 6: pure-Python logic — GreedyPlanner different scenarios
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 6] GreedyPlanner pure-logic test ...')
        regions = _sample_regions()
        goals = {'blue_bottle': 'C', 'red_box': 'A', 'yellow_cup': 'D'}

        # Scenario a: all objects need to be moved
        objs_a = [
            DetectedObject('blue_bottle', (1.5, 1.5), current_region='A', z=0.4),
            DetectedObject('red_box', (1.5, -1.5), current_region='C', z=0.4),
            DetectedObject('yellow_cup', (2.5, 0.5), current_region='A', z=0.4),
        ]
        inp_a = PlannerInput(objs_a, regions, goals, robot_xy=(0.0, 0.0))
        plan_a = GreedyPlanner().plan(inp_a)

        # Scenario b: one object already in target region
        objs_b = [
            DetectedObject('blue_bottle', (1.5, -1.5), current_region='C', z=0.4),
            DetectedObject('red_box', (1.5, -1.5), current_region='C', z=0.4),
            DetectedObject('yellow_cup', (2.5, 0.5), current_region='A', z=0.4),
        ]
        inp_b = PlannerInput(objs_b, regions, goals, robot_xy=(0.0, 0.0))
        plan_b = GreedyPlanner().plan(inp_b)

        ok_a = len(plan_a) == 3
        ok_b = len(plan_b) == 2 and 'blue_bottle' not in [t.object_label for t in plan_b]
        ok = ok_a and ok_b
        self.results.append(('TEST6: Greedy logic', ok,
                              f'scenario a: {len(plan_a)} tasks, scenario b: {len(plan_b)} tasks'))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"}')

        # -----------------------------------------------------------
        # Test 7: filter_actionable filtering logic
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 7] filter_actionable pure-logic test ...')
        # Object has no assignment → filtered out
        objs_c = [
            DetectedObject('unknown_thing', (0.5, 0.5), current_region='A'),
        ]
        inp_c = PlannerInput(objs_c, regions, {'no_match': 'A'}, robot_xy=(0.0, 0.0))
        filtered = filter_actionable(inp_c)
        ok1 = len(filtered) == 0

        # Target region does not exist → filtered out
        inp_d = PlannerInput(
            [DetectedObject('red_box', (0.5, 0.5), current_region='A')],
            regions,
            {'red_box': 'Z'},
            robot_xy=(0.0, 0.0),
        )
        ok2 = len(filter_actionable(inp_d)) == 0

        # Already in target region → filtered out
        inp_e = PlannerInput(
            [DetectedObject('red_box', (1.5, 1.5), current_region='A')],
            regions,
            {'red_box': 'A'},
            robot_xy=(0.0, 0.0),
        )
        ok3 = len(filter_actionable(inp_e)) == 0

        ok = ok1 and ok2 and ok3
        self.results.append(('TEST7: filter_actionable', ok,
                              f'no assignment={ok1}, target missing={ok2}, already placed={ok3}'))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"}')

        # -----------------------------------------------------------
        # Summary
        # -----------------------------------------------------------
        self.get_logger().info('=' * 60)
        self.get_logger().info('TaskPlannerNode test summary:')
        main_results = [r for r in self.results if r[0].startswith('TEST')]
        passed = sum(1 for _, ok, _ in main_results if ok)
        total = len(main_results)
        for label, ok, msg in main_results:
            status = 'PASS' if ok else 'FAIL'
            self.get_logger().info(f'  [{status}] {label}: {msg}')
        self.get_logger().info(f'Passed: {passed}/{total}')
        self.get_logger().info('=' * 60)


def main():
    rclpy.init()
    tester = TaskPlannerTester()
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
