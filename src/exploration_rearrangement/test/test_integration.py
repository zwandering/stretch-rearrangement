"""End-to-end integration test — runs the VLM planner against the real YAML configs."""

from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exploration_rearrangement.planners.base import (  # noqa: E402
    DetectedObject, RegionInfo, PlannerInput,
)
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


def _simulated_scene():
    return [
        DetectedObject('white_bottle', (-1.2,  1.2), current_region='B', z=0.7),
        DetectedObject('green_cup',     (-1.5, -1.5), current_region='D', z=0.7),
        DetectedObject('blue_cup',  ( 1.8,  0.5), current_region='A', z=0.7),
    ]


class _FakeResp:
    def __init__(self, content):
        self.choices = [type('C', (), {
            'message': type('M', (), {'content': content})()
        })()]


def test_real_yaml_loads_cleanly():
    regions = _load_regions()
    assert set(regions.keys()) == {'A', 'B', 'C', 'D'}
    for r in regions.values():
        assert len(r.polygon) >= 3
        assert len(r.place_anchor) == 3


def test_vlm_instruction_mode_with_real_regions(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'stub')
    regions = _load_regions()
    objs = _simulated_scene()
    inp = PlannerInput(
        objects=objs, regions=regions, goal_assignment={},
        robot_xy=(0.0, 0.0),
        instruction='move white bottle to C, green cup to A, blue cup to D',
    )

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return _FakeResp(
                        '{"reasoning": "instruction follow", "tasks": ['
                        '{"object_label": "white_bottle", "target_region": "C", "order_index": 0},'
                        '{"object_label": "green_cup",     "target_region": "A", "order_index": 1},'
                        '{"object_label": "blue_cup",      "target_region": "D", "order_index": 2}'
                        ']}'
                    )

    planner = VLMPlanner(log_path=None)
    planner._client = _FakeClient()
    plan = planner.plan(inp)
    assert len(plan) == 3
    assert [t.object_label for t in plan] == ['white_bottle', 'green_cup', 'blue_cup']
    for t in plan:
        assert t.place_xy == regions[t.target_region].place_anchor[:2]
    assert [t.order_index for t in plan] == [0, 1, 2]
