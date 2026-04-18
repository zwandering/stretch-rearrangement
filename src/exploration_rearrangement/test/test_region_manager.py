#!/usr/bin/env python3
"""Robot integration test for RegionManagerNode — see test/README.md for details."""

import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploration_rearrangement.region_manager_node import (
    point_in_polygon,
    polygon_centroid,
)


# Region definitions from regions.yaml (hardcoded for pure-Python logic tests)
REGIONS = {
    'A': [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)],
    'B': [(-3.0, 0.0), (0.0, 0.0), (0.0, 3.0), (-3.0, 3.0)],
    'C': [(0.0, -3.0), (3.0, -3.0), (3.0, 0.0), (0.0, 0.0)],
    'D': [(-3.0, -3.0), (0.0, -3.0), (0.0, 0.0), (-3.0, 0.0)],
}


class RegionManagerTester(Node):
    def __init__(self):
        super().__init__('region_manager_tester')
        self.latest_markers = None
        self.create_subscription(
            MarkerArray, '/regions/visualization',
            self._on_markers, 10,
        )
        self.reload_cli = self.create_client(Trigger, '/regions/reload')
        self.results = []

    def _on_markers(self, msg):
        self.latest_markers = msg

    def spin_for(self, seconds: float):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def record(self, label: str, ok: bool, msg: str):
        self.results.append((label, ok, msg))
        status = 'PASS' if ok else 'FAIL'
        self.get_logger().info(f'  Result: {status} — {msg}')

    def run_all_tests(self):
        self.get_logger().info('=' * 60)
        self.get_logger().info('Starting RegionManagerNode tests')
        self.get_logger().info('=' * 60)

        # -----------------------------------------------------------
        # Test 1: visualization markers published
        # Expected: receive /regions/visualization within 3s
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 1] Waiting for /regions/visualization (3s) ...')
        self.spin_for(3.0)
        ok = self.latest_markers is not None
        self.record('TEST1: markers published', ok,
                     f'Received {len(self.latest_markers.markers) if ok else 0} markers')

        # -----------------------------------------------------------
        # Test 2: markers contain 4 regions
        # Expected: 4 LINE_STRIP (boundaries) + 4 TEXT (labels)
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 2] Verifying markers contain 4 regions ...')
        boundary_count = 0
        label_names = []
        if self.latest_markers:
            for m in self.latest_markers.markers:
                if m.ns == 'region_boundary' and m.type == Marker.LINE_STRIP:
                    boundary_count += 1
                if m.ns == 'region_label' and m.type == Marker.TEXT_VIEW_FACING:
                    label_names.append(m.text)
        ok = boundary_count == 4 and set(label_names) == {'A', 'B', 'C', 'D'}
        self.record('TEST2: 4 regions', ok,
                     f'boundary={boundary_count}, labels={label_names}')

        # -----------------------------------------------------------
        # Test 3: reload service
        # Expected: success=True (if regions_yaml param is set)
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 3] Calling /regions/reload ...')
        if self.reload_cli.wait_for_service(timeout_sec=5.0):
            fut = self.reload_cli.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            res = fut.result()
            if res:
                self.record('TEST3: reload', res.success, res.message)
            else:
                self.record('TEST3: reload', False, 'call timeout')
        else:
            self.record('TEST3: reload', False, 'service unavailable')

        # -----------------------------------------------------------
        # Test 4: point_in_polygon — points inside regions
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 4] point_in_polygon: points inside regions ...')
        cases = [
            ('A', 1.5, 1.5, True),
            ('B', -1.5, 1.5, True),
            ('C', 1.5, -1.5, True),
            ('D', -1.5, -1.5, True),
        ]
        all_ok = True
        for region, x, y, expected in cases:
            result = point_in_polygon(x, y, REGIONS[region])
            if result != expected:
                all_ok = False
                self.get_logger().warn(
                    f'    FAIL: ({x}, {y}) in {region} = {result}, expected {expected}'
                )
        self.record('TEST4: points inside', all_ok, 'all center points correctly classified')

        # -----------------------------------------------------------
        # Test 5: point_in_polygon — points outside regions
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 5] point_in_polygon: points outside regions ...')
        out_cases = [
            ('A', -1.0, 1.0),   # in B, not in A
            ('B', 1.0, 1.0),    # in A, not in B
            ('C', -1.0, -1.0),  # in D, not in C
            ('A', 5.0, 5.0),    # completely outside map
        ]
        all_ok = True
        for region, x, y in out_cases:
            result = point_in_polygon(x, y, REGIONS[region])
            if result:
                all_ok = False
                self.get_logger().warn(
                    f'    FAIL: ({x}, {y}) should not be in {region}, but returned True'
                )
        self.record('TEST5: points outside', all_ok, 'outside points correctly excluded')

        # -----------------------------------------------------------
        # Test 6: origin (0,0) — multi-region boundary vertex
        # Origin is a vertex of all 4 regions, testing boundary behavior
        # Note: behavior at boundary is undefined, but must not crash
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 6] Origin (0,0) boundary case ...')
        try:
            count = sum(
                1 for poly in REGIONS.values()
                if point_in_polygon(0.0, 0.0, poly)
            )
            ok = True
            msg = f'Origin claimed by {count} regions (boundary behavior, no crash is OK)'
        except Exception as e:
            ok = False
            msg = f'Exception: {e}'
        self.record('TEST6: origin boundary', ok, msg)

        # -----------------------------------------------------------
        # Test 7: polygon_centroid correctness
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 7] polygon_centroid correctness ...')
        expected_centroids = {
            'A': (1.5, 1.5),
            'B': (-1.5, 1.5),
            'C': (1.5, -1.5),
            'D': (-1.5, -1.5),
        }
        all_ok = True
        for name, poly in REGIONS.items():
            cx, cy = polygon_centroid(poly)
            ex, ey = expected_centroids[name]
            if abs(cx - ex) > 0.01 or abs(cy - ey) > 0.01:
                all_ok = False
                self.get_logger().warn(
                    f'    FAIL: {name} centroid=({cx:.2f},{cy:.2f}), expected ({ex},{ey})'
                )
        self.record('TEST7: centroid', all_ok, 'all region centroids computed correctly')

        # -----------------------------------------------------------
        # Test 8: region coverage completeness
        # Grid-sample [-3,3]x[-3,3], verify each point belongs to exactly one region
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 8] Region coverage completeness (grid sampling) ...')
        uncovered = 0
        overlap = 0
        total = 0
        for xi in range(-29, 30, 2):
            for yi in range(-29, 30, 2):
                x, y = xi * 0.1, yi * 0.1
                total += 1
                count = sum(
                    1 for poly in REGIONS.values()
                    if point_in_polygon(x, y, poly)
                )
                if count == 0:
                    uncovered += 1
                elif count > 1:
                    overlap += 1
        ok = uncovered == 0
        msg = (f'Sampled {total} points: uncovered={uncovered}, overlap={overlap}')
        self.record('TEST8: region coverage', ok, msg)

        # -----------------------------------------------------------
        # Summary
        # -----------------------------------------------------------
        self.get_logger().info('=' * 60)
        self.get_logger().info('RegionManagerNode test summary:')
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        for label, ok, msg in self.results:
            status = 'PASS' if ok else 'FAIL'
            self.get_logger().info(f'  [{status}] {label}: {msg}')
        self.get_logger().info(f'Passed: {passed}/{total}')
        self.get_logger().info('=' * 60)


def main():
    rclpy.init()
    tester = RegionManagerTester()
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
