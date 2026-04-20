"""Plan ↔ navigation ↔ manipulation glue.

Subscribes to a pick/place plan from ``task_planner_node``, drives the base
through it via the upstream ``navigation_node`` 3-topic protocol, and
orchestrates a single-stage visual grasp via topic signals.

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
      ├ activate fine detector, set target, send /visual_grasp/start
      │   (visual_grasp_node moves to READY_POSE_P2 itself, then IK-grasps
      │    using gripper D405 + /fine_detector/objects)
      └ /visual_grasp/done = True      → deactivate detector, advance step
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
from std_msgs.msg import Bool, String
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


# visual_grasp_node does an IK-stepping loop with no internal timeout;
# bound how long we let it run before declaring this pick a failure and
# moving to the next pair.
PICK_TIMEOUT_S = 60.0

# /manipulation/place is a scripted lift→extend→open→retract→stow chain;
# ~10–15 s in practice. 30 s catches a hung action server.
PLACE_TIMEOUT_S = 30.0


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

        self.plan_poses: List[Pose] = []
        self.plan_header_frame: str = 'map'
        self.pick_labels: List[str] = []
        self.step_index: int = 0

        # For old-style /manipulation/place action
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

        # --- Navigation protocol ---
        self.create_subscription(
            PoseArray, '/planner/pick_place_plan',
            self._on_plan, 10, callback_group=cb,
        )
        self.create_subscription(
            String, '/planner/pick_labels',
            self._on_pick_labels, 10, callback_group=cb,
        )
        self.create_subscription(
            String, '/nav/arrived_flag',
            self._on_arrived, 10, callback_group=cb,
        )
        self.nav_goals_pub = self.create_publisher(PoseArray, '/nav/goals', 10)
        self.nav_control_pub = self.create_publisher(String, '/nav/control_flag', 10)
        self.status_pub = self.create_publisher(String, '/executor/state', 10)

        # --- Visual pick orchestration (single-stage via visual_grasp_node) ---
        self.fine_activate_pub = self.create_publisher(Bool, '/fine_detector/activate', 10)
        self.fine_target_pub = self.create_publisher(String, '/fine_detector/target_object', 10)
        self.grasp_start_pub = self.create_publisher(Bool, '/visual_grasp/start', 10)

        self.create_subscription(
            Bool, '/visual_grasp/done',
            self._on_grasp_done, 10, callback_group=cb,
        )

        # --- Manipulation node (stow + rotate + place) ---
        self.place_client = ActionClient(
            self, FollowJointTrajectory, '/manipulation/place', callback_group=cb,
        )
        self.stow_cli = self.create_client(
            Trigger, '/manipulation/stow', callback_group=cb,
        )
        self.rotate_cli = self.create_client(
            Trigger, '/manipulation/rotate_base', callback_group=cb,
        )

        self.create_service(
            Trigger, '/executor/start', self._on_start, callback_group=cb,
        )
        self.create_service(
            Trigger, '/executor/abort', self._on_abort, callback_group=cb,
        )

        # --- Debug: force the state machine into an arbitrary state ---
        # Publish e.g. 'PICK', 'PLACE', 'AWAIT_ARRIVED', or 'PICK:0' to
        # also override step_index. Useful for skipping nav and exercising
        # one stage in isolation.
        self.create_subscription(
            String, '/executor/set_state',
            self._on_set_state, 10, callback_group=cb,
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
        self.grasp_start_pub.publish(Bool(data=False))
        self.fine_activate_pub.publish(Bool(data=False))
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
        if self.step_index % 2 == 0:
            self._goto(State.PICK)
            self._begin_pick()
        else:
            self._goto(State.PLACE)
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
            self._check_pick_timeout()
        elif s == State.PLACE:
            self._do_place()
            self._check_place_timeout()

    def _state_elapsed_s(self) -> float:
        return (self.get_clock().now().nanoseconds - self.state_entered_ns) / 1e9

    def _check_pick_timeout(self) -> None:
        if self._state_elapsed_s() < PICK_TIMEOUT_S:
            return
        self.get_logger().warn(
            f'Pick: timeout after {PICK_TIMEOUT_S:.0f}s — visual_grasp did not '
            'complete; treating as failure and skipping place.'
        )
        self.grasp_start_pub.publish(Bool(data=False))
        self.fine_activate_pub.publish(Bool(data=False))
        self._record_pair_result(success=False)
        self._after_step(success=False, was_pick=True)

    def _check_place_timeout(self) -> None:
        if self.active_manip != 'pending':
            return
        if self._state_elapsed_s() < PLACE_TIMEOUT_S:
            return
        self.get_logger().warn(
            f'Place: timeout after {PLACE_TIMEOUT_S:.0f}s — manipulation/place '
            'did not return; treating as failure.'
        )
        self.active_manip = 'failed'

    # --- visual pick orchestration (single stage via visual_grasp_node) ---

    def _begin_pick(self) -> None:
        """PICK: rotate base, activate fine detector, fire /visual_grasp/start.

        Step order:
          1. Rotate the base by ``pre_grasp_rotation_rad`` (default +90°) so
             the arm side faces the object — nav parks the body facing the
             object but Stretch's arm extends from the side.
          2. Set fine detector target + activate it.
          3. /visual_grasp/start. visual_grasp_node moves the arm to
             ik.READY_POSE_P2 itself and IK-steps from there.
        """
        self.metrics['pick_attempts'] += 1

        if not self._call_trigger_blocking(self.rotate_cli, timeout_sec=15.0):
            self.get_logger().warn(
                'Pick: pre-grasp base rotation failed; continuing anyway.'
            )

        target = self._current_pick_target()
        self.fine_target_pub.publish(String(data=target))
        self.fine_activate_pub.publish(Bool(data=True))
        self.get_logger().info(f'Pick: fine detector activated, target={target!r}')

        self.grasp_start_pub.publish(Bool(data=True))
        self.get_logger().info('Pick: visual_grasp started')

    def _current_pick_target(self) -> str:
        """Object label for the current pick step.

        Picks live at even step indices, so the label index is
        ``step_index // 2``. The planner publishes labels on
        ``/planner/pick_labels`` (JSON-encoded list); if that's missing we
        fall back to the ``target_object`` parameter.
        """
        idx = self.step_index // 2
        if 0 <= idx < len(self.pick_labels):
            return self.pick_labels[idx]
        if not self.has_parameter('target_object'):
            self.declare_parameter('target_object', 'yellow cup')
        return str(self.get_parameter('target_object').value)

    def _on_pick_labels(self, msg: String) -> None:
        try:
            labels = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(
                f'Could not parse /planner/pick_labels: {e}; data={msg.data!r}'
            )
            return
        if not isinstance(labels, list) or not all(isinstance(s, str) for s in labels):
            self.get_logger().warn(
                f'/planner/pick_labels payload not a list[str]: {labels!r}'
            )
            return
        self.pick_labels = labels
        self.get_logger().info(f'Received pick labels: {self.pick_labels}')

    def _on_grasp_done(self, msg: Bool) -> None:
        if not msg.data or self.state != State.PICK:
            return
        self.get_logger().info('Pick: visual_grasp done — object grasped')

        self.fine_activate_pub.publish(Bool(data=False))

        self.metrics['pick_successes'] += 1
        self._after_step(success=True, was_pick=True)

    # --- place (kept as /manipulation/place action) ---------------------

    def _do_place(self) -> None:
        if self.active_manip is None:
            if not self.place_client.wait_for_server(timeout_sec=self.manip_server_timeout):
                self.get_logger().error(
                    '/manipulation/place action server unavailable; failing this step.'
                )
                self.active_manip = 'failed'
                return
            self.metrics['place_attempts'] += 1
            self.active_manip = 'pending'
            fut = self.place_client.send_goal_async(FollowJointTrajectory.Goal())
            fut.add_done_callback(self._on_place_goal_response)
            return

        if self.active_manip == 'pending':
            return

        if self.active_manip == 'done':
            self.active_manip = None
            self.metrics['place_successes'] += 1
            self._record_pair_result(success=True)
            self._after_step(success=True, was_pick=False)
            return

        if self.active_manip == 'failed':
            self.active_manip = None
            self._record_pair_result(success=False)
            self._after_step(success=False, was_pick=False)
            return

    def _on_place_goal_response(self, future) -> None:
        try:
            gh = future.result()
        except Exception as e:
            self.get_logger().warn(f'place send_goal failed: {e}')
            self.active_manip = 'failed'
            return
        if not gh.accepted:
            self.get_logger().warn('place goal rejected.')
            self.active_manip = 'failed'
            return
        rf = gh.get_result_async()
        rf.add_done_callback(self._on_place_result)

    def _on_place_result(self, future) -> None:
        try:
            res = future.result()
            code = res.result.error_code
            self.active_manip = 'done' if code == 0 else 'failed'
        except Exception as e:
            self.get_logger().warn(f'place result error: {e}')
            self.active_manip = 'failed'

    # --- step bookkeeping ----------------------------------------------

    def _after_step(self, success: bool, was_pick: bool) -> None:
        """Advance step_index and tell nav what to do next."""
        if was_pick:
            self.fine_activate_pub.publish(Bool(data=False))
            self.grasp_start_pub.publish(Bool(data=False))
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

    def _call_trigger_blocking(self, client, timeout_sec: float = 10.0) -> bool:
        # Synchronous Trigger call. Polls the future instead of re-entering
        # rclpy.spin (which deadlocks under MultiThreadedExecutor when
        # called from within a callback).
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Trigger service unavailable; skipping call.')
            return False
        fut = client.call_async(Trigger.Request())
        deadline = time.time() + timeout_sec
        while not fut.done() and time.time() < deadline:
            time.sleep(0.01)
        if not fut.done():
            return False
        res = fut.result()
        return bool(res is not None and res.success)

    # --- debug: state override -----------------------------------------

    def _on_set_state(self, msg: String) -> None:
        """Force the state machine into an arbitrary state.

        Payload format: ``STATE`` or ``STATE:step_index``. Examples:
          - ``PICK``           → jump to PICK at the current step_index
          - ``PLACE:1``        → set step_index=1 then jump to PLACE
          - ``DISPATCH``       → re-dispatch /nav/goals from step_index
          - ``DONE`` / ``FAILED`` → finalize and write metrics
        """
        payload = msg.data.strip()
        if not payload:
            return
        parts = payload.split(':', 1)
        name = parts[0].strip().upper()
        try:
            new_state = State[name]
        except KeyError:
            self.get_logger().warn(
                f'/executor/set_state: unknown state {name!r}; valid: '
                f'{[s.name for s in State]}'
            )
            return
        if len(parts) == 2 and parts[1].strip():
            try:
                self.step_index = int(parts[1].strip())
            except ValueError:
                self.get_logger().warn(
                    f'/executor/set_state: bad step_index in {payload!r}'
                )
                return

        self.get_logger().warn(
            f'/executor/set_state: forcing {self.state.name} → {new_state.name} '
            f'(step_index={self.step_index})'
        )
        self.active_manip = None
        self._goto(new_state)

        if new_state == State.DISPATCH:
            self._dispatch_remaining()
        elif new_state == State.PICK:
            self._begin_pick()
        elif new_state == State.DONE:
            self._finish(success=True)
        elif new_state == State.FAILED:
            self._finish(success=False)
        # PLACE: _tick will fire _do_place on the next timer cycle.
        # AWAIT_PLAN / AWAIT_ARRIVED / IDLE: no entry action.


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
