"""
NavigationModule coordinator node.

Vendored from AuWeerachai/MobileManipulation NavigationModule, commit
e4b53e523f560f41674bfcc0a209f6ba5730239c. The only local edit is the
``main(args=None)`` signature so this entry point matches the rest of the
package's console_scripts.

Role in the pipeline:
  - task_planner_node publishes geometry_msgs/PoseArray on /nav/goals,
    containing pick0, place0, pick1, place1, ... target poses (x/y in the
    map frame). Orientation is ignored; this node computes a yaw that
    faces each goal.
  - task_executor_node publishes std_msgs/String on /nav/control_flag with
    payload "proceed" (go / advance to next goal) or "stop" (abort).
  - This node drives with Nav2 (via stretch_nav2.BasicNavigator). When the
    robot is within STOP_DISTANCE_M of the current goal, Nav2 is cancelled
    and this node publishes std_msgs/String "arrived" on
    /nav/arrived_flag so manipulation can take over.

State machine:
  IDLE            -- nothing to do
  AWAITING_GO     -- goals received, waiting for "proceed" (or retry of same goal)
  NAVIGATING      -- Nav2 goal in flight, monitoring distance to target
  HANDOFF         -- within STOP_DISTANCE_M, Nav2 cancelled, arrived flag sent,
                     waiting for manipulation to finish and send "proceed"

Assumptions:
  - Nav2 is already running against a saved map (e.g., maps/asangium.yaml).
    Bring it up with the stretch_nav2 navigation launch file before this node.
  - AMCL has been given an initial pose (e.g., RViz "2D Pose Estimate").
  - TF tree includes map -> base_link (AMCL publishes this once localised).
"""

import math
import time
from typing import List, Optional

import rclpy
import tf2_ros
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from stretch_nav2.robot_navigator import BasicNavigator, TaskResult

# -----------------------------------------------------------------------------
# Tuning knobs
# -----------------------------------------------------------------------------

# Distance from the current goal (in meters) at which we cancel Nav2 and hand
# off to manipulation.
STOP_DISTANCE_M = 1.0

# How often we poll TF to measure distance-to-goal while navigating (Hz).
DISTANCE_CHECK_HZ = 5.0

# After we call navigator.cancelTask() we give Nav2 a short grace window to
# actually zero the base velocity before we change state to HANDOFF. Without
# this, the controller may still be ticking when we publish "arrived", and the
# base can coast / trigger a recovery behaviour that looks like the robot
# wandering off to a random place.
CANCEL_SETTLE_S = 0.3

# Topic names. Override at runtime with --ros-args --remap if you want.
GOALS_TOPIC = "/nav/goals"            # planner -> me  (PoseArray)
CONTROL_TOPIC = "/nav/control_flag"   # executor -> me (String: "proceed"/"stop")
ARRIVED_TOPIC = "/nav/arrived_flag"   # me -> executor (String: "arrived")

# TF frames.
MAP_FRAME = "map"
BASE_FRAME = "base_link"


# -----------------------------------------------------------------------------
# States
# -----------------------------------------------------------------------------

IDLE = "IDLE"
AWAITING_GO = "AWAITING_GO"
NAVIGATING = "NAVIGATING"
HANDOFF = "HANDOFF"


def make_pose_stamped(
    x: float, y: float, z: float, yaw: float, frame_id: str, stamp_node: Node
) -> PoseStamped:
    """Build a PoseStamped at (x, y, z) with a yaw-only orientation."""
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    ps.header.stamp = stamp_node.get_clock().now().to_msg()
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.position.z = float(z)
    ps.pose.orientation.x = 0.0
    ps.pose.orientation.y = 0.0
    ps.pose.orientation.z = math.sin(yaw / 2.0)
    ps.pose.orientation.w = math.cos(yaw / 2.0)
    return ps


class NavigationCoordinator(Node):
    """Coordinator node wired between the planner, executor, and Nav2."""

    def __init__(self, navigator: BasicNavigator) -> None:
        super().__init__("navigation_coordinator")
        self.navigator = navigator

        self.goals: List[Pose] = []
        self.goal_index: int = 0
        self.state: str = IDLE

        # IMPORTANT: we deliberately do NOT use a ReentrantCallbackGroup here.
        # The timer callback contains a short time.sleep() after cancelTask to
        # let Nav2 stop the base. With a reentrant group + MultiThreadedExecutor
        # a second tick can fire during that sleep, see state==NAVIGATING, take
        # the isTaskComplete() branch, and call _handoff() before the first
        # tick wakes up — the first tick then calls _handoff() a second time
        # and goal_index ends up skipping every other goal. By leaving the
        # callback_group argument off, these callbacks all share the node's
        # default (mutually exclusive) group, so they are serialised even
        # under a MultiThreadedExecutor.

        self.goals_sub = self.create_subscription(
            PoseArray, GOALS_TOPIC, self._goals_callback, 10
        )
        self.control_sub = self.create_subscription(
            String, CONTROL_TOPIC, self._control_callback, 10
        )
        self.arrived_pub = self.create_publisher(String, ARRIVED_TOPIC, 10)

        # TF listener to monitor map -> base_link while driving.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Distance-monitor timer.
        self.timer = self.create_timer(
            1.0 / DISTANCE_CHECK_HZ, self._timer_callback
        )

        self.get_logger().info(
            f"NavigationCoordinator ready.\n"
            f"  Listening on goals   : {GOALS_TOPIC}  (geometry_msgs/PoseArray)\n"
            f"  Listening on control : {CONTROL_TOPIC}  (std_msgs/String 'proceed'/'stop')\n"
            f"  Publishing arrived   : {ARRIVED_TOPIC}  (std_msgs/String 'arrived')\n"
            f"  Stand-off distance   : {STOP_DISTANCE_M:.2f} m"
        )

    # -------------------------------------------------------------------------
    # Subscribers
    # -------------------------------------------------------------------------

    def _goals_callback(self, msg: PoseArray) -> None:
        """Latch a new goal array and reset to AWAITING_GO."""
        if len(msg.poses) == 0:
            self.get_logger().warning("Received empty PoseArray on %s; ignoring." % GOALS_TOPIC)
            return

        # If we were mid-navigation, cancel before replacing goals.
        if self.state == NAVIGATING:
            self.get_logger().info("New goals arrived mid-navigation; cancelling current Nav2 task.")
            self.navigator.cancelTask()

        self.goals = list(msg.poses)
        self.goal_index = 0
        self.state = AWAITING_GO
        self.get_logger().info(
            f"Received {len(self.goals)} goal(s). Waiting for 'proceed' on {CONTROL_TOPIC}."
        )
        for i, p in enumerate(self.goals):
            self.get_logger().info(
                f"  goal[{i}] = ({p.position.x:.2f}, {p.position.y:.2f}, {p.position.z:.2f})"
            )

    def _control_callback(self, msg: String) -> None:
        """React to proceed / stop from the executor."""
        cmd = msg.data.strip().lower()
        self.get_logger().info(f"Control flag received: '{cmd}' (state={self.state})")

        if cmd == "proceed":
            self._handle_proceed()
        elif cmd == "stop":
            self._handle_stop()
        else:
            self.get_logger().warning(
                f"Unknown control flag '{msg.data}'. Expected 'proceed' or 'stop'."
            )

    # -------------------------------------------------------------------------
    # Control-flag handlers
    # -------------------------------------------------------------------------

    def _handle_proceed(self) -> None:
        if self.state == AWAITING_GO:
            self._send_goal(self.goal_index)
            return
        if self.state == HANDOFF:
            if self.goal_index >= len(self.goals):
                self.get_logger().info("All goals completed. Returning to IDLE.")
                self.state = IDLE
                return
            self._send_goal(self.goal_index)
            return
        if self.state == NAVIGATING:
            self.get_logger().warning("'proceed' received while already NAVIGATING; ignoring.")
            return
        if self.state == IDLE:
            self.get_logger().warning(
                "'proceed' received but no goals are loaded. Publish to %s first." % GOALS_TOPIC
            )

    def _handle_stop(self) -> None:
        if self.state == NAVIGATING:
            self.get_logger().info("'stop' received; cancelling Nav2 and reverting to AWAITING_GO.")
            self.navigator.cancelTask()
            time.sleep(CANCEL_SETTLE_S)
            self.state = AWAITING_GO  # next 'proceed' retries the same goal_index
            return
        if self.state == HANDOFF:
            self.get_logger().info("'stop' received during HANDOFF; no Nav2 task to cancel.")
            self.state = AWAITING_GO
            return
        self.get_logger().info(f"'stop' received in state {self.state}; nothing to cancel.")

    # -------------------------------------------------------------------------
    # Goal dispatch
    # -------------------------------------------------------------------------

    def _send_goal(self, index: int) -> None:
        """Send goals[index] to Nav2 with a yaw facing the goal from current base pose."""
        if index >= len(self.goals):
            self.get_logger().info("Goal index past end of array; nothing to send.")
            self.state = IDLE
            return

        target = self.goals[index]
        yaw = self._yaw_toward(target.position.x, target.position.y)
        goal_ps = make_pose_stamped(
            target.position.x, target.position.y, target.position.z, yaw, MAP_FRAME, self
        )

        self.navigator.goToPose(goal_ps)
        self.state = NAVIGATING
        self.get_logger().info(
            f"[goal {index + 1}/{len(self.goals)}] Driving to "
            f"({target.position.x:.2f}, {target.position.y:.2f}) "
            f"with yaw={yaw:.2f} rad"
        )

    # -------------------------------------------------------------------------
    # TF helpers
    # -------------------------------------------------------------------------

    def _lookup_base_xy(self) -> Optional[tuple]:
        """Return (x, y) of base_link in the map frame, or None if TF unavailable."""
        try:
            tf = self.tf_buffer.lookup_transform(
                MAP_FRAME,
                BASE_FRAME,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
            return tf.transform.translation.x, tf.transform.translation.y
        except Exception as exc:
            # tf2 raises several distinct exception types; any of them means
            # "not yet available" while AMCL is still settling.
            self.get_logger().debug(f"TF {MAP_FRAME}->{BASE_FRAME} not ready: {exc}")
            return None

    def _yaw_toward(self, tx: float, ty: float) -> float:
        """Yaw pointing from the current base_link position toward (tx, ty)."""
        xy = self._lookup_base_xy()
        if xy is None:
            return 0.0
        rx, ry = xy
        return math.atan2(ty - ry, tx - rx)

    def _distance_to_current_goal(self) -> Optional[float]:
        """Euclidean distance from base_link to goals[goal_index] in map frame."""
        if not self.goals or self.goal_index >= len(self.goals):
            return None
        target = self.goals[self.goal_index]
        xy = self._lookup_base_xy()
        if xy is None:
            return None
        rx, ry = xy
        return math.hypot(target.position.x - rx, target.position.y - ry)

    # -------------------------------------------------------------------------
    # Periodic distance monitor
    # -------------------------------------------------------------------------

    def _timer_callback(self) -> None:
        if self.state != NAVIGATING:
            return

        # If Nav2 finished on its own (SUCCEEDED or FAILED) before we hit the
        # 1 m stand-off, treat that as arrival too — hand off and move on.
        if self.navigator.isTaskComplete():
            result = self.navigator.getResult()
            self.get_logger().info(
                f"Nav2 task completed with result={result} before {STOP_DISTANCE_M:.2f}m stand-off; handing off."
            )
            self._handoff()
            return

        d = self._distance_to_current_goal()
        if d is None:
            return

        if d <= STOP_DISTANCE_M:
            self.get_logger().info(
                f"Within {STOP_DISTANCE_M:.2f} m of goal (d={d:.2f} m). "
                f"Cancelling Nav2 and handing off."
            )
            # Claim the handoff BEFORE any blocking call. If we left state as
            # NAVIGATING here, a concurrent tick during cancelTask + sleep
            # would see NAVIGATING, take the isTaskComplete() branch, and
            # handoff a second time — which used to skip every other goal.
            self.state = HANDOFF
            self.navigator.cancelTask()
            # Let the Nav2 controller actually stop the base before we publish
            # the arrived flag. Without this, the robot can keep drifting for
            # a moment and look like it's heading somewhere random.
            time.sleep(CANCEL_SETTLE_S)
            self._handoff()

    def _handoff(self) -> None:
        """Publish 'arrived' and advance to HANDOFF waiting for next 'proceed'.

        Tracks the index we last handed off on so that repeated calls for the
        same arrival are idempotent. The (previously) double-_handoff race
        would otherwise cause every other goal to be skipped.
        """
        if getattr(self, "_last_handoff_index", -1) == self.goal_index:
            self.get_logger().debug(
                f"_handoff() called twice for goal_index={self.goal_index}; "
                f"ignoring the duplicate."
            )
            return
        self._last_handoff_index = self.goal_index

        arrived = String()
        arrived.data = "arrived"
        self.arrived_pub.publish(arrived)

        self.goal_index += 1
        self.state = HANDOFF
        self.get_logger().info(
            f"Handoff sent on {ARRIVED_TOPIC}. "
            f"Waiting for executor's 'proceed' "
            f"(next goal index will be {self.goal_index}/{len(self.goals)})."
        )


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main(args=None) -> None:
    rclpy.init(args=args)

    # Create the Nav2 helper (is itself a Node) and wait for Nav2 to be active.
    # Assumes someone has already set the initial pose (e.g., via RViz's
    # "2D Pose Estimate" button or by calling navigator.setInitialPose()).
    navigator = BasicNavigator()
    navigator.get_logger().info("Waiting for Nav2 to become active...")
    navigator.waitUntilNav2Active()

    coordinator = NavigationCoordinator(navigator)

    # Only the coordinator goes on our MultiThreadedExecutor. BasicNavigator
    # spins its own node internally inside goToPose / isTaskComplete /
    # cancelTask via rclpy.spin_until_future_complete(self, ...). Adding it to
    # an external executor causes a double-spin race: cancels sometimes do not
    # propagate, the action-client state gets corrupted, and the robot can
    # drive off to a "random place" after the last goal because Nav2 never
    # actually stopped. Keep navigator out of the executor.
    executor = MultiThreadedExecutor()
    executor.add_node(coordinator)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.destroy_node()
        navigator.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
