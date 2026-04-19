"""Mock publisher for task_planner_node debugging.

Publishes the three inputs task_planner_node subscribes to, with the QoS
profiles the node expects:

  * /detector/objects  (Detection3DArray, BEST_EFFORT, 10 Hz default)
  * /map               (OccupancyGrid, TRANSIENT_LOCAL+RELIABLE, latched once)
  * static TF ``map -> base_link``  (so robot_pose_in_map returns a real xy)

Instructions (/instruction/text) are normally sent by the operator with
``ros2 topic pub``. If ``auto_publish_first_instruction`` is true the node
also fires the scenario's first instruction automatically after a short
delay, which is handy for scripted smoke tests.

Usage:
  ros2 run exploration_rearrangement fake_planner_inputs \\
      --ros-args -p scenario:=quadrants_mixed

  # In another terminal:
  ros2 topic pub --once /instruction/text std_msgs/String \\
      "{data: 'move the white bottle to region C and the cups to region A'}"

Scenario fixtures live in ``config/planner_test_scenarios.yaml`` — edit that
file (or pass ``-p scenarios_yaml:=<path>``) to add/modify cases.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster
from vision_msgs.msg import (
    Detection3D, Detection3DArray, ObjectHypothesisWithPose,
)


class FakePlannerInputs(Node):

    def __init__(self) -> None:
        super().__init__('fake_planner_inputs')

        self.declare_parameter('scenarios_yaml', '')
        self.declare_parameter('scenario', '')
        self.declare_parameter('detection_rate_hz', 5.0)
        self.declare_parameter('detections_topic', '/detector/objects')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('instruction_topic', '/instruction/text')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('auto_publish_first_instruction', False)
        self.declare_parameter('auto_publish_delay_sec', 2.0)

        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)

        cfg_path_param = str(self.get_parameter('scenarios_yaml').value)
        cfg_path = Path(cfg_path_param) if cfg_path_param else _find_scenarios_yaml()
        if cfg_path is None or not cfg_path.exists():
            raise SystemExit(
                f'planner_test_scenarios.yaml not found (param={cfg_path_param!r})'
            )
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        self.get_logger().info(f'Scenarios loaded from {cfg_path}')

        scenario_name = str(self.get_parameter('scenario').value)
        if not scenario_name:
            scenario_name = str(cfg.get('default_scenario', ''))
        scenarios: Dict[str, Dict[str, Any]] = cfg.get('scenarios', {}) or {}
        if scenario_name not in scenarios:
            raise SystemExit(
                f'scenario {scenario_name!r} not in {sorted(scenarios.keys())}'
            )
        self.scenario_name = scenario_name
        self.scenario: Dict[str, Any] = scenarios[scenario_name]
        self.get_logger().info(
            f'Scenario "{scenario_name}": '
            f'{self.scenario.get("description", "(no description)")}'
        )

        self.map_msg = self._build_map(cfg.get('map', {}))

        self.static_tfb = StaticTransformBroadcaster(self)
        if bool(self.get_parameter('publish_tf').value):
            self._publish_static_tf(self.scenario.get('robot_xy', [0.0, 0.0]))

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        det_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.map_pub = self.create_publisher(
            OccupancyGrid, str(self.get_parameter('map_topic').value), map_qos,
        )
        self.det_pub = self.create_publisher(
            Detection3DArray,
            str(self.get_parameter('detections_topic').value),
            det_qos,
        )
        self.inst_pub = self.create_publisher(
            String, str(self.get_parameter('instruction_topic').value), 10,
        )

        self._publish_map_once()

        hz = max(1e-3, float(self.get_parameter('detection_rate_hz').value))
        self.create_timer(1.0 / hz, self._publish_detections)

        self._once_timer = None
        if bool(self.get_parameter('auto_publish_first_instruction').value):
            delay = float(self.get_parameter('auto_publish_delay_sec').value)
            self.get_logger().info(
                f'Will auto-publish the first instruction after {delay:.1f}s'
            )
            self._once_timer = self.create_timer(
                delay, self._publish_first_instruction_once,
            )

        n_obj = len(self.scenario.get('objects', []) or [])
        insts = self.scenario.get('instructions', []) or []
        self.get_logger().info(
            f'Ready: {n_obj} fake detections @ {hz:.1f} Hz, '
            f'{len(insts)} sample instruction(s). '
            f'Publish your own on {self.get_parameter("instruction_topic").value}.'
        )
        for i, text in enumerate(insts):
            self.get_logger().info(f'  sample instruction #{i}: {text!r}')

    # --- builders --------------------------------------------------------

    def _build_map(self, map_cfg: Dict[str, Any]) -> OccupancyGrid:
        res = float(map_cfg.get('resolution', 0.1))
        w = int(map_cfg.get('width', 80))
        h = int(map_cfg.get('height', 80))
        origin = map_cfg.get('origin') or [-4.0, -4.0]
        msg = OccupancyGrid()
        msg.header.frame_id = self.map_frame
        msg.info.resolution = res
        msg.info.width = w
        msg.info.height = h
        msg.info.origin.position.x = float(origin[0])
        msg.info.origin.position.y = float(origin[1])
        msg.info.origin.orientation.w = 1.0
        msg.data = [0] * (w * h)
        return msg

    def _publish_static_tf(self, xy: List[float]) -> None:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = float(xy[0])
        t.transform.translation.y = float(xy[1])
        t.transform.rotation.w = 1.0
        self.static_tfb.sendTransform(t)

    # --- publishers ------------------------------------------------------

    def _publish_map_once(self) -> None:
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self.map_msg)

    def _publish_detections(self) -> None:
        arr = Detection3DArray()
        arr.header.frame_id = self.map_frame
        arr.header.stamp = self.get_clock().now().to_msg()
        for obj in self.scenario.get('objects', []) or []:
            label = str(obj['label'])
            xyz = obj.get('xyz', [0.0, 0.0, 0.0])
            size = obj.get('size', [0.08, 0.08, 0.15])
            score = float(obj.get('score', 0.8))
            det = Detection3D()
            det.header = arr.header
            det.id = label
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score = score
            hyp.pose.pose.position.x = float(xyz[0])
            hyp.pose.pose.position.y = float(xyz[1])
            hyp.pose.pose.position.z = float(xyz[2])
            hyp.pose.pose.orientation.w = 1.0
            det.results.append(hyp)
            det.bbox.center = hyp.pose.pose
            det.bbox.size.x = float(size[0])
            det.bbox.size.y = float(size[1])
            det.bbox.size.z = float(size[2])
            arr.detections.append(det)
        self.det_pub.publish(arr)

    def _publish_first_instruction_once(self) -> None:
        if self._once_timer is not None:
            self._once_timer.cancel()
        insts = self.scenario.get('instructions', []) or []
        if not insts:
            self.get_logger().warn(
                'scenario has no instructions; auto-publish skipped'
            )
            return
        text = str(insts[0])
        self.get_logger().info(f'auto-publishing instruction: {text!r}')
        self.inst_pub.publish(String(data=text))


def _find_scenarios_yaml() -> Optional[Path]:
    """Mirror set_up_yolo_e's search strategy so the node works in both
    installed (colcon share/) and source (symlink-install) trees."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory('exploration_rearrangement'))
        cand = share / 'config' / 'planner_test_scenarios.yaml'
        if cand.exists():
            return cand
    except Exception:
        pass
    here = Path(__file__).resolve().parent
    for cand in (
        here.parent.parent / 'config' / 'planner_test_scenarios.yaml',
        Path.cwd() / 'src' / 'exploration_rearrangement' / 'config' / 'planner_test_scenarios.yaml',
        Path.cwd() / 'config' / 'planner_test_scenarios.yaml',
    ):
        if cand.exists():
            return cand
    return None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakePlannerInputs()
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
