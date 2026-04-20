"""Stretch 3 pick/place primitives via FollowJointTrajectory action.

Joints used (Stretch 3 driver in position mode):
    joint_lift                 — vertical lift [m]
    wrist_extension            — total telescoping arm extension [m]
    joint_wrist_yaw            — wrist yaw [rad]
    joint_wrist_pitch          — wrist pitch [rad]
    joint_wrist_roll           — wrist roll [rad]
    joint_gripper_finger_left  — driven finger [rad] (positive = open)
    joint_head_pan             — head pan [rad]
    joint_head_tilt            — head tilt [rad]
    translate_mobile_base      — incremental forward translation [m] (position mode)
    rotate_mobile_base         — incremental yaw rotation [rad] (position mode)
"""

import time
from typing import List, Optional

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point, PoseStamped
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


OPEN_GRIPPER = 0.25
CLOSED_GRIPPER = -0.10


class ManipulationNode(Node):

    def __init__(self) -> None:
        super().__init__('manipulation_node')

        self.declare_parameter(
            'trajectory_action',
            '/stretch_controller/follow_joint_trajectory',
        )
        self.declare_parameter('switch_to_position_srv', '/switch_to_position_mode')
        self.declare_parameter('switch_to_navigation_srv', '/switch_to_navigation_mode')
        self.declare_parameter('stow_srv', '/stow_the_robot')
        self.declare_parameter('pick_height_m', 0.75)
        # place is just "drop into the bucket above the place_anchor": lift
        # the gripper to drop_height_m (a few cm above the bucket rim),
        # extend the arm over the bucket, open the gripper, retract.
        # Default 0.55 m clears a ~45 cm bucket rim by ~10 cm.
        self.declare_parameter('drop_height_m', 0.55)
        self.declare_parameter('arm_extend_m', 0.30)

        cb = ReentrantCallbackGroup()
        self.traj_client = ActionClient(
            self, FollowJointTrajectory,
            self.get_parameter('trajectory_action').value,
            callback_group=cb,
        )
        self.switch_pos_cli = self.create_client(
            Trigger, self.get_parameter('switch_to_position_srv').value,
            callback_group=cb,
        )
        self.switch_nav_cli = self.create_client(
            Trigger, self.get_parameter('switch_to_navigation_srv').value,
            callback_group=cb,
        )
        self.stow_cli = self.create_client(
            Trigger, self.get_parameter('stow_srv').value,
            callback_group=cb,
        )

        self.pick_server = ActionServer(
            self, FollowJointTrajectory, '/manipulation/pick',
            execute_callback=self._exec_pick,
            goal_callback=lambda g: GoalResponse.ACCEPT,
            cancel_callback=lambda g: CancelResponse.ACCEPT,
            callback_group=cb,
        )
        self.place_server = ActionServer(
            self, FollowJointTrajectory, '/manipulation/place',
            execute_callback=self._exec_place,
            goal_callback=lambda g: GoalResponse.ACCEPT,
            cancel_callback=lambda g: CancelResponse.ACCEPT,
            callback_group=cb,
        )
        self.create_service(
            Trigger, '/manipulation/stow', self._on_stow, callback_group=cb,
        )

        self.get_logger().info('ManipulationNode ready.')

    # --- Action handlers -------------------------------------------------

    def _exec_pick(self, goal_handle: ServerGoalHandle):
        """Execute pick sequence.

        The incoming FollowJointTrajectory goal is treated as a carrier:
        point[0].positions[0] is used only as a sentinel; this primitive
        runs a canonical open→lower→extend→close→lift→retract→stow sequence.
        """
        self._call_trigger(self.switch_pos_cli)
        ok = (
            self._send_joints([('joint_head_pan', 0.0), ('joint_head_tilt', -0.7)])
            and self._send_joints([
                ('joint_wrist_yaw', 0.0),
                ('joint_wrist_pitch', 0.0),
                ('joint_wrist_roll', 0.0),
                ('joint_gripper_finger_left', OPEN_GRIPPER),
            ])
            and self._send_joints([
                ('joint_lift', float(self.get_parameter('pick_height_m').value)),
            ])
            and self._send_joints([
                ('wrist_extension', float(self.get_parameter('arm_extend_m').value)),
            ])
            and self._send_joints([
                ('joint_gripper_finger_left', CLOSED_GRIPPER),
            ], time_from_start=1.0)
            and self._send_joints([
                ('joint_lift', float(self.get_parameter('pick_height_m').value) + 0.10),
            ])
            and self._send_joints([('wrist_extension', 0.0)])
        )
        self._call_trigger(self.stow_cli)
        self._call_trigger(self.switch_nav_cli)

        result = FollowJointTrajectory.Result()
        if ok:
            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        else:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
        return result

    def _exec_place(self, goal_handle: ServerGoalHandle):
        """Drop the held object into a bucket above the current place_anchor.

        The base has already been parked at the place_anchor's xy by nav,
        so all we do here is: lift the gripper to drop_height_m (a few cm
        above the bucket rim, default 0.55 m for a ~45 cm bucket), extend
        the arm over the bucket, open the gripper to release, retract,
        stow. The plan's place pose is not used by this server — only the
        regions.yaml place_anchor (which nav already drove to) determines
        where the bucket should be.
        """
        self._call_trigger(self.switch_pos_cli)
        ok = (
            self._send_joints([
                ('joint_lift', float(self.get_parameter('drop_height_m').value)),
            ])
            and self._send_joints([
                ('wrist_extension', float(self.get_parameter('arm_extend_m').value)),
            ])
            and self._send_joints([
                ('joint_gripper_finger_left', OPEN_GRIPPER),
            ], time_from_start=1.0)
            and self._send_joints([('wrist_extension', 0.0)])
        )
        self._call_trigger(self.stow_cli)
        self._call_trigger(self.switch_nav_cli)

        result = FollowJointTrajectory.Result()
        if ok:
            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        else:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
        return result

    def _on_stow(self, req, res):
        ok = self._call_trigger(self.stow_cli)
        res.success = ok
        res.message = 'stow dispatched' if ok else 'stow failed'
        return res

    # --- Primitives ------------------------------------------------------

    def _send_joints(
        self,
        joints: List[tuple],
        time_from_start: float = 2.0,
    ) -> bool:
        if not self.traj_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn('Joint trajectory action not available.')
            return False
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = [j[0] for j in joints]
        pt = JointTrajectoryPoint()
        pt.positions = [float(j[1]) for j in joints]
        sec = int(time_from_start)
        nsec = int((time_from_start - sec) * 1e9)
        pt.time_from_start.sec = sec
        pt.time_from_start.nanosec = nsec
        traj.points.append(pt)
        goal.trajectory = traj

        fut = self.traj_client.send_goal_async(goal)
        if not self._wait_for_future(fut, timeout_sec=5.0):
            return False
        gh = fut.result()
        if gh is None or not gh.accepted:
            return False
        result_fut = gh.get_result_async()
        if not self._wait_for_future(result_fut, timeout_sec=time_from_start + 4.0):
            return False
        res = result_fut.result()
        if res is None:
            return False
        return res.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL

    def _call_trigger(self, client) -> bool:
        if not client.wait_for_service(timeout_sec=2.0):
            return False
        fut = client.call_async(Trigger.Request())
        if not self._wait_for_future(fut, timeout_sec=4.0):
            return False
        res = fut.result()
        return bool(res is not None and res.success)

    def _wait_for_future(self, fut, timeout_sec: float) -> bool:
        # Don't call rclpy.spin_until_future_complete from inside an action
        # callback — the node is already owned by MultiThreadedExecutor,
        # and re-entering spin deadlocks the executor. Poll instead; the
        # executor's other threads keep servicing the future.
        deadline = time.time() + timeout_sec
        while not fut.done() and time.time() < deadline:
            time.sleep(0.01)
        return fut.done()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ManipulationNode()
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
