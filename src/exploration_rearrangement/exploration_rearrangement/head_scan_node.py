"""Periodic head pan/tilt scan to enlarge the detector's field of coverage."""

from typing import List, Tuple

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class HeadScanNode(Node):

    def __init__(self) -> None:
        super().__init__('head_scan_node')

        self.declare_parameter(
            'trajectory_action',
            '/stretch_controller/follow_joint_trajectory',
        )
        self.declare_parameter('period_s', 8.0)
        self.declare_parameter('enabled_on_start', False)
        self.declare_parameter('pan_waypoints',
                               [-1.2, -0.6, 0.0, 0.6, 1.2])
        self.declare_parameter('tilt_angle', -0.55)

        self.enabled: bool = bool(self.get_parameter('enabled_on_start').value)
        self.pan_waypoints: List[float] = [
            float(v) for v in self.get_parameter('pan_waypoints').value
        ]
        self.tilt_angle: float = float(self.get_parameter('tilt_angle').value)
        self.period: float = float(self.get_parameter('period_s').value)

        cb = ReentrantCallbackGroup()
        self.client = ActionClient(
            self, FollowJointTrajectory,
            self.get_parameter('trajectory_action').value,
            callback_group=cb,
        )

        self.create_service(
            Trigger, '/head/start_scan',
            self._on_start, callback_group=cb,
        )
        self.create_service(
            Trigger, '/head/stop_scan',
            self._on_stop, callback_group=cb,
        )
        self.create_service(
            Trigger, '/head/scan_once',
            self._on_scan_once, callback_group=cb,
        )

        self._idx = 0
        self.create_timer(self.period, self._tick, callback_group=cb)
        self.get_logger().info(
            f'HeadScanNode ready; waypoints={self.pan_waypoints}, '
            f'tilt={self.tilt_angle}, enabled={self.enabled}'
        )

    def _on_start(self, req, res):
        self.enabled = True
        res.success = True
        res.message = 'head scanning'
        return res

    def _on_stop(self, req, res):
        self.enabled = False
        res.success = True
        res.message = 'head idle'
        return res

    def _on_scan_once(self, req, res):
        ok = self._send_sequence(self.pan_waypoints)
        res.success = ok
        res.message = 'sweep complete' if ok else 'sweep failed'
        return res

    def _tick(self) -> None:
        if not self.enabled:
            return
        pan = self.pan_waypoints[self._idx % len(self.pan_waypoints)]
        self._idx += 1
        self._send_waypoint(pan, self.tilt_angle)

    def _send_waypoint(self, pan: float, tilt: float, time_s: float = 1.5) -> bool:
        if not self.client.wait_for_server(timeout_sec=2.0):
            return False
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = ['joint_head_pan', 'joint_head_tilt']
        pt = JointTrajectoryPoint()
        pt.positions = [float(pan), float(tilt)]
        pt.time_from_start.sec = int(time_s)
        pt.time_from_start.nanosec = int((time_s - int(time_s)) * 1e9)
        traj.points.append(pt)
        goal.trajectory = traj
        fut = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        gh = fut.result()
        if gh is None or not gh.accepted:
            return False
        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=time_s + 3.0)
        return res_fut.result() is not None

    def _send_sequence(self, pans: List[float]) -> bool:
        ok = True
        for p in pans:
            ok = ok and self._send_waypoint(p, self.tilt_angle, 1.2)
        return ok


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HeadScanNode()
    executor = MultiThreadedExecutor()
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
