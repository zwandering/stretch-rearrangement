"""Unit tests for planner backends (greedy + VLM fallback)."""

import os
from pathlib import Path
import sys

# Make the package importable when running pytest outside colcon.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exploration_rearrangement.planners.base import (  # noqa: E402
    DetectedObject, RegionInfo, PlannerInput, filter_actionable, euclidean,
)
from exploration_rearrangement.planners.greedy import GreedyPlanner  # noqa: E402
from exploration_rearrangement.planners.vlm import VLMPlanner  # noqa: E402


def _sample_regions():
    return {
        'A': RegionInfo('A', [(0, 0), (2, 0), (2, 2), (0, 2)], (1.0,  1.0, 0.0)),
        'B': RegionInfo('B', [(-2, 0), (0, 0), (0, 2), (-2, 2)], (-1.0,  1.0, 0.0)),
        'C': RegionInfo('C', [(0, -2), (2, -2), (2, 0), (0, 0)], (1.0, -1.0, 3.14)),
        'D': RegionInfo('D', [(-2, -2), (0, -2), (0, 0), (-2, 0)], (-1.0, -1.0, 3.14)),
    }


def test_filter_actionable_skips_objects_already_placed():
    regions = _sample_regions()
    objs = [
        DetectedObject('red_box',     (-0.5, 1.2), current_region='B'),
        DetectedObject('blue_bottle', (1.2,  0.5), current_region='A'),  # already in A
    ]
    goals = {'red_box': 'A', 'blue_bottle': 'A'}
    inp = PlannerInput(objs, regions, goals, robot_xy=(0.0, 0.0))
    actionable = filter_actionable(inp)
    assert [o.label for o in actionable] == ['red_box']


def test_filter_actionable_drops_unassigned_and_unknown_region():
    regions = _sample_regions()
    objs = [
        DetectedObject('red_box',     (-0.5, 1.2), current_region='B'),
        DetectedObject('unknown_obj', (0.3,  0.3), current_region='A'),
    ]
    goals = {'red_box': 'Z', 'unknown_obj': 'A'}
    inp = PlannerInput(objs, regions, goals, robot_xy=(0.0, 0.0))
    assert filter_actionable(inp) == []


def test_greedy_plan_length_and_order():
    regions = _sample_regions()
    objs = [
        DetectedObject('red_box',     (-0.5, 1.2), current_region='B'),
        DetectedObject('blue_bottle', (1.8,  0.5), current_region='A'),
        DetectedObject('yellow_cup',  (1.2,  1.6), current_region='A'),
    ]
    goals = {'red_box': 'A', 'blue_bottle': 'C', 'yellow_cup': 'D'}
    inp = PlannerInput(objs, regions, goals, robot_xy=(0.0, 0.0))
    plan = GreedyPlanner().plan(inp)
    assert len(plan) == 3
    assert [t.order_index for t in plan] == [0, 1, 2]
    assert {t.object_label for t in plan} == {'red_box', 'blue_bottle', 'yellow_cup'}
    for t in plan:
        assert t.place_xy == regions[t.target_region].place_anchor[:2]


def test_greedy_first_pick_is_nearest_to_robot():
    regions = _sample_regions()
    objs = [
        DetectedObject('far', (1.9,  1.9), current_region='A'),
        DetectedObject('near', (0.1,  0.1), current_region='A'),
    ]
    goals = {'far': 'C', 'near': 'D'}
    inp = PlannerInput(objs, regions, goals, robot_xy=(0.0, 0.0))
    plan = GreedyPlanner().plan(inp)
    assert plan[0].object_label == 'near'


def test_greedy_empty_when_everything_placed():
    regions = _sample_regions()
    objs = [
        DetectedObject('red_box', (-0.5, 1.2), current_region='B'),
    ]
    goals = {'red_box': 'B'}  # already in B
    inp = PlannerInput(objs, regions, goals, robot_xy=(0.0, 0.0))
    assert GreedyPlanner().plan(inp) == []


def test_vlm_falls_back_to_greedy_when_no_api_key(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    regions = _sample_regions()
    objs = [
        DetectedObject('red_box', (-0.5, 1.2), current_region='B'),
    ]
    goals = {'red_box': 'A'}
    inp = PlannerInput(objs, regions, goals, robot_xy=(0.0, 0.0))
    plan = VLMPlanner().plan(inp)
    assert len(plan) == 1
    assert plan[0].object_label == 'red_box'
    assert plan[0].target_region == 'A'


def test_vlm_parses_valid_json(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'stub')
    regions = _sample_regions()
    objs = [
        DetectedObject('red_box',     (-0.5, 1.2), current_region='B'),
        DetectedObject('blue_bottle', (1.8,  0.5), current_region='A'),
    ]
    goals = {'red_box': 'A', 'blue_bottle': 'C'}
    inp = PlannerInput(objs, regions, goals, robot_xy=(0.0, 0.0))

    planner = VLMPlanner(log_path=None)

    class _FakeChoice:
        def __init__(self, content):
            self.message = type('M', (), {'content': content})

    class _FakeResp:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return _FakeResp(
                        '{"reasoning": "test", "tasks": ['
                        '{"object_label": "blue_bottle", "target_region": "C", "order_index": 0},'
                        '{"object_label": "red_box",     "target_region": "A", "order_index": 1}'
                        ']}'
                    )

    planner._client = _FakeClient()
    plan = planner.plan(inp)
    assert [t.object_label for t in plan] == ['blue_bottle', 'red_box']
    assert [t.order_index for t in plan] == [0, 1]


def test_vlm_auto_appends_missing_objects(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'stub')
    regions = _sample_regions()
    objs = [
        DetectedObject('red_box',     (-0.5, 1.2), current_region='B'),
        DetectedObject('blue_bottle', (1.8,  0.5), current_region='A'),
    ]
    goals = {'red_box': 'A', 'blue_bottle': 'C'}
    inp = PlannerInput(objs, regions, goals, robot_xy=(0.0, 0.0))

    class _FakeChoice:
        def __init__(self, content):
            self.message = type('M', (), {'content': content})

    class _FakeResp:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    # Only returns red_box; blue_bottle missing
                    return _FakeResp(
                        '{"reasoning": "partial", "tasks": ['
                        '{"object_label": "red_box", "target_region": "A", "order_index": 0}'
                        ']}'
                    )

    planner = VLMPlanner(log_path=None)
    planner._client = _FakeClient()
    plan = planner.plan(inp)
    assert {t.object_label for t in plan} == {'red_box', 'blue_bottle'}


def test_vlm_falls_back_on_invalid_json(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'stub')
    regions = _sample_regions()
    objs = [DetectedObject('red_box', (-0.5, 1.2), current_region='B')]
    goals = {'red_box': 'A'}
    inp = PlannerInput(objs, regions, goals, robot_xy=(0.0, 0.0))

    class _BadClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    class _C:
                        choices = [type('X', (), {
                            'message': type('M', (), {'content': 'not json'})()
                        })()]
                    return _C()

    planner = VLMPlanner(log_path=None, max_retries=0)
    planner._client = _BadClient()
    plan = planner.plan(inp)
    # Must still succeed via greedy fallback
    assert len(plan) == 1 and plan[0].object_label == 'red_box'


def test_euclidean_symmetry():
    assert euclidean((0, 0), (3, 4)) == 5.0
    assert euclidean((3, 4), (0, 0)) == 5.0
