"""Top-level state machine coordinating explore / detect / plan / execute."""

import json
import time
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
import yaml
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .planners import (
    DetectedObject,
    PickPlaceTask,
    PlannerBackend,
    PlannerInput,
    RegionInfo,
    VLMPlanner,
)
from .region_manager_node import point_in_polygon, polygon_centroid
from .utils.transform_utils import robot_pose_in_map, yaw_to_quat


class State(Enum):
    IDLE = auto()
    HEAD_SCAN = auto()
    EXPLORE = auto()
    WAIT_OBJECTS = auto()
    PLAN = auto()
    NAV_TO_PICK = auto()
    PICK = auto()
    NAV_TO_PLACE = auto()
    PLACE = auto()
    DONE = auto()
    FAILED = auto()


class TaskExecutorNode(Node):

    def __init__(self) -> None:
        super().__init__('task_executor_node')

        self.declare_parameter('planner_backend', 'greedy')
        self.declare_parameter('tasks_yaml', '')
        self.declare_parameter('regions_yaml', '')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('explore_timeout_s', 180.0)
        self.declare_parameter('min_objects_required', 3)
        self.declare_parameter('wait_after_explore_s', 6.0)
        self.declare_parameter('pick_standoff_m', 0.55)
        self.declare_parameter('metrics_path', '/tmp/rearrangement_metrics.json')
        self.declare_parameter('start_on_launch', False)
        self.declare_parameter('vlm_model', 'gemini-2.5-flash')
        self.declare_parameter('vlm_base_url',
            'https://generativelanguage.googleapis.com/v1beta/openai/')
        self.declare_parameter('vlm_api_key_env', 'GEMINI_API_KEY')
        self.declare_parameter('vlm_use_image', True)

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.explore_timeout = float(self.get_parameter('explore_timeout_s').value)
        self.min_objects = int(self.get_parameter('min_objects_required').value)
        self.wait_explore = float(self.get_parameter('wait_after_explore_s').value)
        self.standoff = float(self.get_parameter('pick_standoff_m').value)
        self.metrics_path = Path(self.get_parameter('metrics_path').value)

        self.state: State = State.IDLE
        self.state_entered_ns: int = self.get_clock().now().nanoseconds
        self.active_future = None
        self.active_action_goal_handle = None
        self.exec_queue: List[PickPlaceTask] = []
        self.current_task: Optional[PickPlaceTask] = None

        self.latest_objects: Dict[str, PoseStamped] = {}
        self.regions: Dict[str, RegionInfo] = self._load_regions(
            self.get_parameter('regions_yaml').value,
        )
        self.goal_assignment: Dict[str, str] = self._load_tasks(
            self.get_parameter('tasks_yaml').value,
        )
        self.planner: PlannerBackend = self._build_backend()

        self.metrics = {
            'backend': self.planner.name,
            't_start': None,
            't_explore_done': None,
            't_all_objects_seen': None,
            't_end': None,
            'first_detection': {},
            'task_results': [],
            'pick_attempts': 0,
            'pick_successes': 0,
            'place_attempts': 0,
            'place_successes': 0,
        }

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cb = ReentrantCallbackGroup()
        self.create_subscription(
            MarkerArray, '/detected_objects',
            self._on_detections, 10, callback_group=cb,
        )
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose',
                                       callback_group=cb)
        self.pick_client = ActionClient(self, FollowJointTrajectory,
                                        '/manipulation/pick', callback_group=cb)
        self.place_client = ActionClient(self, FollowJointTrajectory,
                                         '/manipulation/place', callback_group=cb)
        self.start_explore_cli = self.create_client(
            Trigger, '/exploration/start', callback_group=cb)
        self.stop_explore_cli = self.create_client(
            Trigger, '/exploration/stop', callback_group=cb)
        self.start_head_cli = self.create_client(
            Trigger, '/head/start_scan', callback_group=cb)
        self.stop_head_cli = self.create_client(
            Trigger, '/head/stop_scan', callback_group=cb)
        self.stow_cli = self.create_client(
            Trigger, '/manipulation/stow', callback_group=cb)

        self.status_pub = self.create_publisher(String, '/executor/state', 10)

        self.create_service(Trigger, '/executor/start',
                            self._on_start, callback_group=cb)
        self.create_service(Trigger, '/executor/abort',
                            self._on_abort, callback_group=cb)

        self.create_timer(0.5, self._tick, callback_group=cb)

        if bool(self.get_parameter('start_on_launch').value):
            self.get_logger().info('start_on_launch=true; launching in 3s...')
            self.create_timer(3.0, lambda: self._begin(), callback_group=cb)

        self.get_logger().info(
            f'TaskExecutorNode ready (backend={self.planner.name}).'
        )

    # --- setup -----------------------------------------------------------

    def _build_backend(self) -> PlannerBackend:
        return VLMPlanner(
            model=str(self.get_parameter('vlm_model').value),
            base_url=str(self.get_parameter('vlm_base_url').value),
            api_key_env=str(self.get_parameter('vlm_api_key_env').value),
            use_image=bool(self.get_parameter('vlm_use_image').value),
        )

    def _load_tasks(self, path: str) -> Dict[str, str]:
        if not path:
            return {}
        with open(Path(path), 'r') as f:
            return dict((yaml.safe_load(f) or {}).get('assignments', {}))

    def _load_regions(self, path: str) -> Dict[str, RegionInfo]:
        if not path:
            return {}
        with open(Path(path), 'r') as f:
            cfg = yaml.safe_load(f) or {}
        out: Dict[str, RegionInfo] = {}
        for entry in cfg.get('regions', []):
            poly = [tuple(p) for p in entry['polygon']]
            anchor = tuple(entry.get(
                'place_anchor',
                list(polygon_centroid(poly)) + [0.0],
            ))
            out[entry['name']] = RegionInfo(entry['name'], poly, anchor)
        return out

    # --- services --------------------------------------------------------

    def _on_start(self, req, res):
        self._begin()
        res.success = True
        res.message = 'executor started'
        return res

    def _on_abort(self, req, res):
        self.get_logger().warn('Executor abort requested.')
        self._call_trigger(self.stop_explore_cli)
        self._call_trigger(self.stop_head_cli)
        self._goto(State.FAILED)
        res.success = True
        res.message = 'aborted'
        return res

    def _begin(self):
        if self.state != State.IDLE:
            return
        self.metrics['t_start'] = time.time()
        self.get_logger().info('=== Starting exploration + rearrangement ===')
        self._goto(State.HEAD_SCAN)

    # --- subscriptions ---------------------------------------------------

    def _on_detections(self, msg: MarkerArray):
        now_s = time.time()
        for m in msg.markers:
            if m.type != Marker.CUBE:
                continue
            ps = PoseStamped()
            ps.header = m.header
            ps.pose = m.pose
            self.latest_objects[m.ns] = ps
            if m.ns not in self.metrics['first_detection']:
                self.metrics['first_detection'][m.ns] = now_s - (
                    self.metrics['t_start'] or now_s
                )

    # --- state machine --------------------------------------------------

    def _goto(self, s: State):
        self.get_logger().info(f'State: {self.state.name} → {s.name}')
        self.state = s
        self.state_entered_ns = self.get_clock().now().nanoseconds
        self.status_pub.publish(String(data=s.name))

    def _elapsed_in_state(self) -> float:
        return (self.get_clock().now().nanoseconds - self.state_entered_ns) * 1e-9

    def _tick(self):
        s = self.state
        if s == State.IDLE or s == State.DONE or s == State.FAILED:
            return
        if s == State.HEAD_SCAN:
            # Stow arm first so nothing fouls on the mobile base while we drive.
            self._call_trigger(self.stow_cli)
            self._call_trigger(self.start_head_cli)
            self._goto(State.EXPLORE)
            return
        if s == State.EXPLORE:
            if self._elapsed_in_state() < 1.5:
                return
            if self.active_future is None:
                self._call_trigger(self.start_explore_cli)
                self.active_future = True  # sentinel
            if (len(self.latest_objects) >= self.min_objects
                    or self._elapsed_in_state() > self.explore_timeout):
                self._call_trigger(self.stop_explore_cli)
                self._call_trigger(self.stop_head_cli)
                self.metrics['t_explore_done'] = time.time()
                self.active_future = None
                self._goto(State.WAIT_OBJECTS)
            return
        if s == State.WAIT_OBJECTS:
            if self._elapsed_in_state() < self.wait_explore:
                return
            self.metrics['t_all_objects_seen'] = time.time()
            self._goto(State.PLAN)
            return
        if s == State.PLAN:
            self._do_plan()
            return
        if s == State.NAV_TO_PICK:
            self._poll_nav_then(State.PICK)
            return
        if s == State.PICK:
            self._do_manipulation(self.pick_client, success_next=State.NAV_TO_PLACE,
                                  is_pick=True)
            return
        if s == State.NAV_TO_PLACE:
            self._poll_nav_then(State.PLACE)
            return
        if s == State.PLACE:
            self._do_manipulation(self.place_client, success_next=None, is_pick=False)
            return

    def _do_plan(self):
        pose = robot_pose_in_map(self, self.tf_buffer,
                                 self.base_frame, self.map_frame)
        if pose is None:
            self.get_logger().warn('No robot pose yet; retrying PLAN...')
            return
        detected = []
        for label, ps in self.latest_objects.items():
            xy = (ps.pose.position.x, ps.pose.position.y)
            detected.append(DetectedObject(
                label=label, pose_xy=xy, z=ps.pose.position.z,
                current_region=self._which_region(xy),
            ))
        inp = PlannerInput(
            objects=detected,
            regions=self.regions,
            goal_assignment=self.goal_assignment,
            robot_xy=(pose.pose.position.x, pose.pose.position.y),
            context_image_bgr=None,
        )
        plan = self.planner.plan(inp)
        self.exec_queue = plan
        if not plan:
            self.get_logger().info('Nothing to do — all objects already in place.')
            self._finish(success=True)
            return
        self.get_logger().info(
            f'Planned {len(plan)} tasks: '
            + ', '.join(f'{t.object_label}→{t.target_region}' for t in plan)
        )
        self._start_next_task()

    def _start_next_task(self):
        if not self.exec_queue:
            self._finish(success=True)
            return
        self.current_task = self.exec_queue.pop(0)
        approach = self._pick_approach_pose(self.current_task.pick_xy)
        self._send_nav(approach)
        self._goto(State.NAV_TO_PICK)

    def _pick_approach_pose(self, target_xy: Tuple[float, float]) -> PoseStamped:
        pose = robot_pose_in_map(self, self.tf_buffer,
                                 self.base_frame, self.map_frame)
        robot_xy = (pose.pose.position.x, pose.pose.position.y) if pose else (0.0, 0.0)
        dx = target_xy[0] - robot_xy[0]
        dy = target_xy[1] - robot_xy[1]
        norm = np.hypot(dx, dy) or 1.0
        ux, uy = dx / norm, dy / norm
        ax = target_xy[0] - ux * self.standoff
        ay = target_xy[1] - uy * self.standoff
        yaw = float(np.arctan2(dy, dx))
        ps = PoseStamped()
        ps.header.frame_id = self.map_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(ax)
        ps.pose.position.y = float(ay)
        qx, qy, qz, qw = yaw_to_quat(yaw)
        ps.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        return ps

    def _place_goal_pose(self, region_name: str) -> PoseStamped:
        r = self.regions[region_name]
        ax, ay, ayaw = r.place_anchor
        ps = PoseStamped()
        ps.header.frame_id = self.map_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(ax)
        ps.pose.position.y = float(ay)
        qx, qy, qz, qw = yaw_to_quat(float(ayaw))
        ps.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        return ps

    def _send_nav(self, target: PoseStamped):
        if not self.nav_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('Nav2 action server unavailable.')
            self._finish(success=False)
            return
        goal = NavigateToPose.Goal()
        goal.pose = target
        self.active_future = self.nav_client.send_goal_async(goal)
        self.active_future.add_done_callback(self._on_nav_goal_response)

    def _on_nav_goal_response(self, future):
        try:
            gh = future.result()
        except Exception as e:
            self.get_logger().warn(f'Nav goal send failed: {e}')
            self.active_future = None
            return
        if not gh.accepted:
            self.get_logger().warn('Nav goal rejected.')
            self.active_future = 'failed'
            return
        self.active_action_goal_handle = gh
        rf = gh.get_result_async()
        rf.add_done_callback(self._on_nav_result)

    def _on_nav_result(self, future):
        try:
            _ = future.result()
            self.active_future = 'done'
        except Exception as e:
            self.get_logger().warn(f'Nav result error: {e}')
            self.active_future = 'failed'

    def _poll_nav_then(self, next_state: State):
        if self.active_future == 'done':
            self.active_future = None
            self._goto(next_state)
        elif self.active_future == 'failed':
            self.get_logger().warn('Nav failed; skipping task.')
            self.active_future = None
            self._record_task_result(success=False)
            self._start_next_task()

    def _do_manipulation(self, client: ActionClient, success_next: Optional[State],
                         is_pick: bool):
        if self.active_future is None:
            if not client.wait_for_server(timeout_sec=3.0):
                self.get_logger().error('Manipulation action unavailable.')
                self.active_future = 'failed'
                return
            goal = FollowJointTrajectory.Goal()
            if is_pick:
                self.metrics['pick_attempts'] += 1
            else:
                self.metrics['place_attempts'] += 1
            self.active_future = client.send_goal_async(goal)
            self.active_future.add_done_callback(self._on_manip_goal_response)
            return
        if self.active_future == 'done':
            self.active_future = None
            if is_pick:
                self.metrics['pick_successes'] += 1
                target = self._place_goal_pose(self.current_task.target_region)
                self._send_nav(target)
                self._goto(State.NAV_TO_PLACE)
            else:
                self.metrics['place_successes'] += 1
                self._record_task_result(success=True)
                self._start_next_task()
        elif self.active_future == 'failed':
            self.active_future = None
            self._record_task_result(success=False)
            self._start_next_task()

    def _on_manip_goal_response(self, future):
        try:
            gh = future.result()
        except Exception:
            self.active_future = 'failed'
            return
        if not gh.accepted:
            self.active_future = 'failed'
            return
        rf = gh.get_result_async()
        rf.add_done_callback(self._on_manip_result)

    def _on_manip_result(self, future):
        try:
            res = future.result()
            code = res.result.error_code
            self.active_future = 'done' if code == 0 else 'failed'
        except Exception:
            self.active_future = 'failed'

    def _record_task_result(self, success: bool):
        if self.current_task is None:
            return
        self.metrics['task_results'].append({
            'object': self.current_task.object_label,
            'target': self.current_task.target_region,
            'success': bool(success),
            'ts': time.time(),
        })
        self.current_task = None

    def _finish(self, success: bool):
        self.metrics['t_end'] = time.time()
        self.metrics['success'] = bool(success)
        try:
            self.metrics_path.write_text(json.dumps(self.metrics, indent=2))
            self.get_logger().info(f'Metrics written to {self.metrics_path}')
        except Exception as e:
            self.get_logger().warn(f'Could not write metrics: {e}')
        self._goto(State.DONE if success else State.FAILED)

    def _which_region(self, xy: Tuple[float, float]) -> Optional[str]:
        for name, r in self.regions.items():
            if point_in_polygon(xy[0], xy[1], r.polygon):
                return name
        return None

    def _call_trigger(self, client) -> bool:
        if not client.wait_for_service(timeout_sec=1.0):
            return False
        fut = client.call_async(Trigger.Request())
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
