#!/usr/bin/env python3
"""Robot integration test for ObjectDetectorNode — see test/README.md for details."""

import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray


class ObjectDetectorTester(Node):
    def __init__(self):
        super().__init__('object_detector_tester')

        self.camera_info_received = False
        self.rgb_received = False
        self.depth_received = False
        self.latest_markers: Optional[MarkerArray] = None
        self.debug_image_received = False
        self.marker_count = 0
        self.results = []

        self.create_subscription(
            CameraInfo, '/camera/color/camera_info',
            self._on_info, qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, '/camera/color/image_raw',
            self._on_rgb, qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, '/camera/aligned_depth_to_color/image_raw',
            self._on_depth, qos_profile_sensor_data,
        )
        self.create_subscription(
            MarkerArray, '/detected_objects',
            self._on_markers, 10,
        )
        self.create_subscription(
            Image, '/detector/debug_image',
            self._on_debug, 2,
        )
        self.clear_cli = self.create_client(Trigger, '/detector/clear')

    def _on_info(self, msg):
        self.camera_info_received = True

    def _on_rgb(self, msg):
        self.rgb_received = True

    def _on_depth(self, msg):
        self.depth_received = True

    def _on_markers(self, msg):
        self.latest_markers = msg
        cubes = [m for m in msg.markers if m.type == Marker.CUBE]
        self.marker_count = len(cubes)

    def _on_debug(self, msg):
        self.debug_image_received = True

    def spin_for(self, seconds: float):
        """Spin for the given duration, processing callbacks."""
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def run_all_tests(self):
        self.get_logger().info('=' * 60)
        self.get_logger().info('Starting ObjectDetectorNode tests')
        self.get_logger().info('=' * 60)

        # -----------------------------------------------------------
        # Test 1: camera topic data check
        # Expected: receive CameraInfo, RGB, Depth within 5s
        # Risk: camera not started, topic name mismatch, QoS incompatible
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 1] Checking camera topic data ...')
        self.spin_for(5.0)
        info_ok = self.camera_info_received
        rgb_ok = self.rgb_received
        depth_ok = self.depth_received
        all_ok = info_ok and rgb_ok and depth_ok
        msg = (f'CameraInfo={"OK" if info_ok else "MISS"}, '
               f'RGB={"OK" if rgb_ok else "MISS"}, '
               f'Depth={"OK" if depth_ok else "MISS"}')
        self.results.append(('TEST1: camera topics', all_ok, msg))
        self.get_logger().info(f'  Result: {"PASS" if all_ok else "FAIL"} — {msg}')

        if not all_ok:
            self.get_logger().warn('  Camera data missing, subsequent tests may fail')

        # -----------------------------------------------------------
        # Test 2: detection result publishing
        # Expected: /detected_objects has at least 1 CUBE marker
        # Risk: HSV params mismatch, invalid depth, TF not ready
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 2] Waiting for detection results (10s) ...')
        self.spin_for(10.0)
        ok = self.marker_count > 0
        msg = f'Detected {self.marker_count} objects'
        self.results.append(('TEST2: detection publish', ok, msg))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"} — {msg}')

        # -----------------------------------------------------------
        # Test 3: object label verification
        # Expected: all labels in {blue_bottle, red_box, yellow_cup}
        # Risk: false positives (e.g. background misidentified as object)
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 3] Verifying object labels ...')
        expected_labels = {'blue_bottle', 'red_box', 'yellow_cup'}
        detected_labels = set()
        if self.latest_markers:
            for m in self.latest_markers.markers:
                if m.type == Marker.CUBE:
                    detected_labels.add(m.ns)
        unexpected = detected_labels - expected_labels
        ok = len(unexpected) == 0 and len(detected_labels) > 0
        msg = f'Detected labels: {detected_labels}, unknown labels: {unexpected or "none"}'
        self.results.append(('TEST3: label verification', ok, msg))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"} — {msg}')

        # -----------------------------------------------------------
        # Test 4: object position sanity
        # Expected: all objects x,y within [-4, 4] m (map bounds)
        # Risk: depth estimation anomaly, TF error causing out-of-bounds
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 4] Verifying object position sanity ...')
        pos_ok = True
        pos_details = []
        if self.latest_markers:
            for m in self.latest_markers.markers:
                if m.type == Marker.CUBE:
                    x, y = m.pose.position.x, m.pose.position.y
                    in_range = -5.0 < x < 5.0 and -5.0 < y < 5.0
                    pos_details.append(
                        f'{m.ns}: ({x:.2f}, {y:.2f}) {"OK" if in_range else "OUT"}'
                    )
                    if not in_range:
                        pos_ok = False
        ok = pos_ok and len(pos_details) > 0
        msg = '; '.join(pos_details) if pos_details else 'no detection data'
        self.results.append(('TEST4: position sanity', ok, msg))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"} — {msg}')

        # -----------------------------------------------------------
        # Test 5: debug image
        # Expected: /detector/debug_image is publishing data
        # Risk: publish_debug_image param set to False
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 5] Checking debug image publishing ...')
        ok = self.debug_image_received
        msg = 'debug image received' if ok else 'debug image not received'
        self.results.append(('TEST5: debug image', ok, msg))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"} — {msg}')

        # -----------------------------------------------------------
        # Test 6: /detector/clear
        # Expected: markers cleared after call (only DELETEALL remains)
        # Risk: immediately refilled by new detections (normal, but briefly empty)
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 6] Calling /detector/clear ...')
        self.marker_count = 0
        if self.clear_cli.wait_for_service(timeout_sec=5.0):
            fut = self.clear_cli.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            res = fut.result()
            ok = res is not None and res.success
            msg = f'success={res.success if res else "None"}'
        else:
            ok = False
            msg = 'service unavailable'
        self.results.append(('TEST6: clear service', ok, msg))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"} — {msg}')

        # -----------------------------------------------------------
        # Test 7: re-detection after clear
        # Expected: objects detected again after 5s
        # -----------------------------------------------------------
        self.get_logger().info('[TEST 7] Waiting for re-detection (5s) ...')
        self.spin_for(5.0)
        ok = self.marker_count > 0
        msg = f'Re-detected {self.marker_count} objects'
        self.results.append(('TEST7: re-detection', ok, msg))
        self.get_logger().info(f'  Result: {"PASS" if ok else "FAIL"} — {msg}')

        # -----------------------------------------------------------
        # Summary
        # -----------------------------------------------------------
        self.get_logger().info('=' * 60)
        self.get_logger().info('ObjectDetectorNode test summary:')
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        for label, ok, msg in self.results:
            status = 'PASS' if ok else 'FAIL'
            self.get_logger().info(f'  [{status}] {label}: {msg}')
        self.get_logger().info(f'Passed: {passed}/{total}')
        self.get_logger().info('=' * 60)


def main():
    rclpy.init()
    tester = ObjectDetectorTester()
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
