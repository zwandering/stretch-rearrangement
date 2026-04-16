"""End-to-end integration test — runs both planners against the real YAML configs."""

from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exploration_rearrangement.planners.base import (  # noqa: E402
    DetectedObject, RegionInfo, PlannerInput,
)
from exploration_rearrangement.planners.greedy import GreedyPlanner  # noqa: E402
from exploration_rearrangement.planners.vlm import VLMPlanner  # noqa: E402


CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'


def _load_regions() -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / 'regions.yaml').read_text())
    out = {}
    for entry in cfg['regions']:
        poly = [tuple(p) for p in entry['polygon']]
        anchor = tuple(entry['place_anchor'])
        out[entry['name']] = RegionInfo(entry['name'], poly, anchor)
    return out


def _load_tasks() -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / 'tasks.yaml').read_text())
    return dict(cfg['assignments'])


def _simulated_scene():
    # Mimic what detector + region classifier would produce in a real scene.
    return [
        DetectedObject('blue_bottle', (-1.2,  1.2), current_region='B', z=0.7),
        DetectedObject('red_box',     (-1.5, -1.5), current_region='D', z=0.7),
        DetectedObject('yellow_cup',  ( 1.8,  0.5), current_region='A', z=0.7),
    ]


def test_real_yaml_loads_cleanly():
    regions = _load_regions()
    tasks = _load_tasks()
    assert set(regions.keys()) == {'A', 'B', 'C', 'D'}
    assert set(tasks.keys()) == {'blue_bottle', 'red_box', 'yellow_cup'}
    for r in regions.values():
        assert len(r.polygon) >= 3
        assert len(r.place_anchor) == 3


def test_greedy_against_real_configs():
    regions = _load_regions()
    tasks = _load_tasks()
    objs = _simulated_scene()
    inp = PlannerInput(
        objects=objs, regions=regions, goal_assignment=tasks,
        robot_xy=(0.0, 0.0),
    )
    plan = GreedyPlanner().plan(inp)
    assert len(plan) == 3
    assert {t.object_label for t in plan} == {'blue_bottle', 'red_box', 'yellow_cup'}
    # Every task routes to the correct goal region
    for t in plan:
        assert t.target_region == tasks[t.object_label]
        assert t.place_xy == regions[t.target_region].place_anchor[:2]
    # Order indices are contiguous
    assert [t.order_index for t in plan] == [0, 1, 2]


def test_vlm_backend_wraps_greedy_when_offline(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    regions = _load_regions()
    tasks = _load_tasks()
    objs = _simulated_scene()
    inp = PlannerInput(objs, regions, tasks, robot_xy=(0.0, 0.0))
    plan = VLMPlanner().plan(inp)
    assert len(plan) == 3


def test_skip_objects_already_in_goal_region():
    regions = _load_regions()
    tasks = _load_tasks()
    # Pretend blue_bottle is already in C (its goal).
    objs = [
        DetectedObject('blue_bottle', (1.5, -1.5), current_region='C', z=0.7),
        DetectedObject('red_box',     (-1.5, -1.5), current_region='D', z=0.7),
        DetectedObject('yellow_cup',  (1.8,  0.5), current_region='A', z=0.7),
    ]
    inp = PlannerInput(objs, regions, tasks, robot_xy=(0.0, 0.0))
    plan = GreedyPlanner().plan(inp)
    assert len(plan) == 2
    assert {t.object_label for t in plan} == {'red_box', 'yellow_cup'}


def test_greedy_plan_is_deterministic():
    regions = _load_regions()
    tasks = _load_tasks()
    objs = _simulated_scene()
    inp = PlannerInput(objs, regions, tasks, robot_xy=(0.0, 0.0))
    p1 = GreedyPlanner().plan(inp)
    p2 = GreedyPlanner().plan(inp)
    assert [t.object_label for t in p1] == [t.object_label for t in p2]
