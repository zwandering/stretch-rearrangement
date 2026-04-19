"""Plan ↔ navigation ↔ manipulation glue.

Subscribes to a pick/place plan from ``task_planner_node``, drives the base
through it via the upstream ``navigation_node`` 3-topic protocol, and fires
``/manipulation/{pick,place}`` action goals at each handoff.

State machine
-------------
::

    IDLE
      └ /executor/start service        → AWAIT_PLAN  (stows arm)
    AWAIT_PLAN
      └ /planner/pick_place_plan       → DISPATCH
    DISPATCH
      ├ publish PoseArray to /nav/goals
      ├ step_index = 0
      └ publish "proceed"              → AWAIT_ARRIVED
    AWAIT_ARRIVED
      └ /nav/arrived_flag = "arrived"  →
            (step_index even) PICK, (step_index odd) PLACE
    PICK
      ├ success → "proceed", step_index += 1, → AWAIT_ARRIVED (or DONE)
      └ failure → re-publish /nav/goals starting at step_index+2,
                  step_index += 2, "proceed", → AWAIT_ARRIVED (or DONE)
    PLACE
      ├ success → record task ok, "proceed", step_index += 1, → AWAIT_ARRIVED
      └ failure → record task fail, "proceed", step_index += 1, → AWAIT_ARRIVED
    DONE / FAILED
      └ writes metrics, idles
"""

import json
import time
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional

import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseArray
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


class State(Enum):
    IDLE = auto()
    AWAIT_PLAN = auto()
    DISPATCH = auto()
    AWAIT_ARRIVED = auto()
    PICK = auto()
    PLACE = auto()
    DONE = auto()
    FAILED = auto()


class TaskExecutorNode(Node):

    def __init__(self) -> None:
        super().__init__('task_executor_node')

        self.declare_parameter('metrics_path', '/tmp/rearrangement_metrics.json')
        self.declare_parameter('start_on_launch', False)
        self.declare_parameter('manipulation_server_timeout_s', 3.0)

        self.metrics_path = Path(self.get_parameter('metrics_path').value)
        self.manip_server_timeout = float(
            self.get_parameter('manipulation_server_timeout_s').value
        )

        self.state: State = State.IDLE
        self.state_entered_ns: int = self.get_clock().now().nanoseconds

        # Plan + cursor.
        self.plan_poses: List[Pose] = []
        self.plan_header_frame: str = 'map'
        self.step_index: int = 0

        # Manipulation action progress flag: None | 'pending' | 'done' | 'failed'.
        self.active_manip: Optional[str] = None

        self.metrics = {
            't_start': None,
            't_plan_received': None,
            't_end': None,
            'task_results': [],
            'pick_attempts': 0,
            'pick_successes': 0,
            'place_attempts': 0,
            'place_successes': 0,
        }

        cb = ReentrantCallbackGroup()

        self.create_subscription(
            PoseArray, '/planner/pick_place_plan',
            self._on_plan, 10, callback_group=cb,
        )
        self.create_subscription(
            String, '/nav/arrived_flag',
            self._on_arrived, 10, callback_group=cb,
        )

        self.nav_goals_pub = self.create_publisher(PoseArray, '/nav/goals', 10)
        self.nav_control_pub = self.create_publisher(String, '/nav/control_flag', 10)
        self.status_pub = self.create_publisher(String, '/executor/state', 10)

        self.pick_client = ActionClient(
            self, FollowJointTrajectory, '/manipulation/pick', callback_group=cb,
        )
        self.place_client = ActionClient(
            self, FollowJointTrajectory, '/manipulation/place', callback_group=cb,
        )
        self.stow_cli = self.create_client(
            Trigger, '/manipulation/stow', callback_group=cb,
        )

        self.create_service(
            Trigger, '/executor/start', self._on_start, callback_group=cb,
        )
        self.create_service(
            Trigger, '/executor/abort', self._on_abort, callback_group=cb,
        )

        self.create_timer(0.5, self._tick, callback_group=cb)

        if bool(self.get_parameter('start_on_launch').value):
            self.get_logger().info('start_on_launch=true; arming executor in 3s...')
            self.create_timer(3.0, self._auto_begin, callback_group=cb)

        self.get_logger().info('TaskExecutorNode ready.')

    # --- services / triggers --------------------------------------------

    def _on_start(self, req, res):
        self._begin()
        res.success = True
        res.message = f'executor armed (state={self.state.name})'
        return res

    def _on_abort(self, req, res):
        self.get_logger().warn('Executor abort requested.')
        self._publish_control('stop')
        self.active_manip = None
        self._finish(success=False)
        res.success = True
        res.message = 'aborted'
        return res

    def _auto_begin(self) -> None:
        if self.state == State.IDLE:
            self._begin()

    def _begin(self) -> None:
        if self.state != State.IDLE:
            self.get_logger().info(
                f'_begin() called in state {self.state.name}; ignoring.'
            )
            return
        self.metrics['t_start'] = time.time()
        # Stow the arm once before the run so nothing fouls during nav.
        self._call_trigger(self.stow_cli)
        self._goto(State.AWAIT_PLAN)
        self.get_logger().info(
            'Waiting for /planner/pick_place_plan. Publish an instruction on '
            '/instruction/text to drive the planner.'
        )

    # --- subscriptions ---------------------------------------------------

    def _on_plan(self, msg: PoseArray) -> None:
        if self.state != State.AWAIT_PLAN:
            self.get_logger().info(
                f'Plan received in state {self.state.name}; ignoring.'
            )
            return
        if not msg.poses:
            self.get_logger().warn('Plan is empty; finishing as success.')
            self._finish(success=True)
            return
        if len(msg.poses) % 2 != 0:
            self.get_logger().error(
                f'Plan has {len(msg.poses)} poses (odd); expected pick/place pairs. '
                'Aborting.'
            )
            self._finish(success=False)
            return

        self.plan_poses = list(msg.poses)
        self.plan_header_frame = msg.header.frame_id or 'map'
        self.step_index = 0
        self.metrics['t_plan_received'] = time.time()
        self.get_logger().info(
            f'Plan accepted: {len(self.plan_poses) // 2} pick/place pair(s).'
        )
        self._goto(State.DISPATCH)
        self._dispatch_remaining()

    def _on_arrived(self, msg: String) -> None:
        if msg.data.strip().lower() != 'arrived':
            return
        if self.state != State.AWAIT_ARRIVED:
            self.get_logger().info(
                f'arrived in state {self.state.name}; ignoring (step_index={self.step_index}).'
            )
            return
        if self.step_index >= len(self.plan_poses):
            self.get_logger().warn(
                'arrived after plan exhausted; finishing.'
            )
            self._finish(success=True)
            return
        # Even step → just arrived at a pick pose; odd → at a place pose.
        if self.step_index % 2 == 0:
            self._goto(State.PICK)
        else:
            self._goto(State.PLACE)
        # active_manip starts None so _tick will dispatch the goal.
        self.active_manip = None

    # --- state machine --------------------------------------------------

    def _goto(self, s: State) -> None:
        self.get_logger().info(f'State: {self.state.name} → {s.name}')
        self.state = s
        self.state_entered_ns = self.get_clock().now().nanoseconds
        self.status_pub.publish(String(data=s.name))

    def _tick(self) -> None:
        s = self.state
        if s == State.PICK:
            self._do_manipulation(self.pick_client, is_pick=True)
        elif s == State.PLACE:
            self._do_manipulation(self.place_client, is_pick=False)

    def _do_manipulation(self, client: ActionClient, is_pick: bool) -> None:
        if self.active_manip is None:
            if not client.wait_for_server(timeout_sec=self.manip_server_timeout):
                kind = 'pick' if is_pick else 'place'
                self.get_logger().error(
                    f'/manipulation/{kind} action server unavailable; failing this step.'
                )
                self.active_manip = 'failed'
                return
            if is_pick:
                self.metrics['pick_attempts'] += 1
            else:
                self.metrics['place_attempts'] += 1
            self.active_manip = 'pending'
            fut = client.send_goal_async(FollowJointTrajectory.Goal())
            fut.add_done_callback(self._on_manip_goal_response)
            return

        if self.active_manip == 'pending':
            return

        if self.active_manip == 'done':
            self.active_manip = None
            if is_pick:
                self.metrics['pick_successes'] += 1
                self._after_step(success=True, was_pick=True)
            else:
                self.metrics['place_successes'] += 1
                self._record_pair_result(success=True)
                self._after_step(success=True, was_pick=False)
            return

        if self.active_manip == 'failed':
            self.active_manip = None
            if not is_pick:
                self._record_pair_result(success=False)
            self._after_step(success=False, was_pick=is_pick)
            return

    def _on_manip_goal_response(self, future) -> None:
        try:
            gh = future.result()
        except Exception as e:
            self.get_logger().warn(f'manipulation send_goal failed: {e}')
            self.active_manip = 'failed'
            return
        if not gh.accepted:
            self.get_logger().warn('manipulation goal rejected.')
            self.active_manip = 'failed'
            return
        rf = gh.get_result_async()
        rf.add_done_callback(self._on_manip_result)

    def _on_manip_result(self, future) -> None:
        try:
            res = future.result()
            code = res.result.error_code
            self.active_manip = 'done' if code == 0 else 'failed'
        except Exception as e:
            self.get_logger().warn(f'manipulation result error: {e}')
            self.active_manip = 'failed'

    # --- step bookkeeping ----------------------------------------------

    def _after_step(self, success: bool, was_pick: bool) -> None:
        """Advance step_index and tell nav what to do next."""
        if was_pick and not success:
            # Skip the orphan place pose: re-issue /nav/goals from step_index+2.
            self.step_index += 2
            if self.step_index >= len(self.plan_poses):
                # Tell nav we're done; mark overall failure if any pair failed.
                self._publish_control('proceed')
                self._finish_based_on_results()
                return
            self._goto(State.DISPATCH)
            self._dispatch_remaining()
            return

        # Normal success or place-failure path: just nudge to next pose.
        self.step_index += 1
        if self.step_index >= len(self.plan_poses):
            self._publish_control('proceed')  # nav goes IDLE
            self._finish_based_on_results()
            return
        self._publish_control('proceed')
        self._goto(State.AWAIT_ARRIVED)

    def _dispatch_remaining(self) -> None:
        """Publish the slice of plan_poses from step_index onwards to nav and proceed."""
        remaining = self.plan_poses[self.step_index:]
        out = PoseArray()
        out.header.frame_id = self.plan_header_frame
        out.header.stamp = self.get_clock().now().to_msg()
        out.poses = remaining
        # Republishing /nav/goals resets nav's internal goal_index to 0.
        # Latch in two stages: send goals first, then proceed once the nav node
        # has had a chance to flip into AWAITING_GO.
        self.nav_goals_pub.publish(out)
        self._publish_control('proceed')
        self._goto(State.AWAIT_ARRIVED)

    def _publish_control(self, cmd: str) -> None:
        self.nav_control_pub.publish(String(data=cmd))

    def _record_pair_result(self, success: bool) -> None:
        # step_index points at the place pose we just tried; the matching
        # pick was at step_index - 1, and the pair index is step_index // 2.
        pair_index = self.step_index // 2
        self.metrics['task_results'].append({
            'pair_index': int(pair_index),
            'success': bool(success),
            'ts': time.time(),
        })

    def _finish_based_on_results(self) -> None:
        any_failure = any(
            not r.get('success', False) for r in self.metrics['task_results']
        )
        self._finish(success=not any_failure)

    def _finish(self, success: bool) -> None:
        self.metrics['t_end'] = time.time()
        self.metrics['success'] = bool(success)
        try:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.write_text(json.dumps(self.metrics, indent=2))
            self.get_logger().info(f'Metrics written to {self.metrics_path}')
        except Exception as e:
            self.get_logger().warn(f'Could not write metrics: {e}')
        self._goto(State.DONE if success else State.FAILED)

    def _call_trigger(self, client) -> bool:
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f'Trigger service unavailable; skipping call.')
            return False
        client.call_async(Trigger.Request())
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskExecutorNode()
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
