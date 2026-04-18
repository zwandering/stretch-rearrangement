"""Unit tests for the VLM planner backend."""

from pathlib import Path
import sys

import pytest

# Make the package importable when running pytest outside colcon.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exploration_rearrangement.planners.base import (  # noqa: E402
    DetectedObject, RegionInfo, PlannerInput, filter_actionable, euclidean,
)
from exploration_rearrangement.planners.vlm import (  # noqa: E402
    VLMPlanner, VLMPlanError,
)


def _sample_regions():
    return {
        'A': RegionInfo('A', [(0, 0), (2, 0), (2, 2), (0, 2)], (1.0,  1.0, 0.0)),
        'B': RegionInfo('B', [(-2, 0), (0, 0), (0, 2), (-2, 2)], (-1.0,  1.0, 0.0)),
        'C': RegionInfo('C', [(0, -2), (2, -2), (2, 0), (0, 0)], (1.0, -1.0, 3.14)),
        'D': RegionInfo('D', [(-2, -2), (0, -2), (0, 0), (-2, 0)], (-1.0, -1.0, 3.14)),
    }


class _FakeChoice:
    def __init__(self, content):
        self.message = type('M', (), {'content': content})


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _fake_client_returning(content: str):
    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return _FakeResp(content)
    return _FakeClient()


# --- filter_actionable ---------------------------------------------------


def test_filter_actionable_skips_objects_already_placed():
    regions = _sample_regions()
    objs = [
        DetectedObject('green_cup',     (-0.5, 1.2), current_region='B'),
        DetectedObject('white_bottle', (1.2,  0.5), current_region='A'),
    ]
    goals = {'green_cup': 'A', 'white_bottle': 'A'}
    inp = PlannerInput(objs, regions, goals, robot_xy=(0.0, 0.0))
    actionable = filter_actionable(inp)
    assert [o.label for o in actionable] == ['green_cup']


def test_filter_actionable_drops_unassigned_and_unknown_region():
    regions = _sample_regions()
    objs = [
        DetectedObject('green_cup',     (-0.5, 1.2), current_region='B'),
        DetectedObject('unknown_obj', (0.3,  0.3), current_region='A'),
    ]
    goals = {'green_cup': 'Z', 'unknown_obj': 'A'}
    inp = PlannerInput(objs, regions, goals, robot_xy=(0.0, 0.0))
    assert filter_actionable(inp) == []


# --- VLM planner ---------------------------------------------------------


def test_vlm_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    regions = _sample_regions()
    objs = [DetectedObject('green_cup', (-0.5, 1.2), current_region='B')]
    goals = {'green_cup': 'A'}
    inp = PlannerInput(objs, regions, goals, robot_xy=(0.0, 0.0))
    with pytest.raises(VLMPlanError):
        VLMPlanner(log_path=None).plan(inp)


def test_vlm_parses_valid_json_assigned_mode(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'stub')
    regions = _sample_regions()
    objs = [
        DetectedObject('green_cup',     (-0.5, 1.2), current_region='B'),
        DetectedObject('white_bottle', (1.8,  0.5), current_region='A'),
    ]
    goals = {'green_cup': 'A', 'white_bottle': 'C'}
    inp = PlannerInput(objs, regions, goals, robot_xy=(0.0, 0.0))

    planner = VLMPlanner(log_path=None)
    planner._client = _fake_client_returning(
        '{"reasoning": "test", "tasks": ['
        '{"object_label": "white_bottle", "target_region": "C", "order_index": 0},'
        '{"object_label": "green_cup",     "target_region": "A", "order_index": 1}'
        ']}'
    )
    plan = planner.plan(inp)
    assert [t.object_label for t in plan] == ['white_bottle', 'green_cup']
    assert [t.order_index for t in plan] == [0, 1]


def test_vlm_auto_appends_missing_objects_in_assigned_mode(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'stub')
    regions = _sample_regions()
    objs = [
        DetectedObject('green_cup',     (-0.5, 1.2), current_region='B'),
        DetectedObject('white_bottle', (1.8,  0.5), current_region='A'),
    ]
    goals = {'green_cup': 'A', 'white_bottle': 'C'}
    inp = PlannerInput(objs, regions, goals, robot_xy=(0.0, 0.0))

    planner = VLMPlanner(log_path=None)
    planner._client = _fake_client_returning(
        '{"reasoning": "partial", "tasks": ['
        '{"object_label": "green_cup", "target_region": "A", "order_index": 0}'
        ']}'
    )
    plan = planner.plan(inp)
    assert {t.object_label for t in plan} == {'green_cup', 'white_bottle'}


def test_vlm_raises_on_invalid_json(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'stub')
    regions = _sample_regions()
    objs = [DetectedObject('green_cup', (-0.5, 1.2), current_region='B')]
    goals = {'green_cup': 'A'}
    inp = PlannerInput(objs, regions, goals, robot_xy=(0.0, 0.0))

    planner = VLMPlanner(log_path=None, max_retries=0, retry_base_sec=0.0)
    planner._client = _fake_client_returning('not json')
    with pytest.raises(VLMPlanError):
        planner.plan(inp)


def test_vlm_instruction_mode_builds_plan(monkeypatch):
    """With an instruction and empty goal_assignment, VLM derives goals from the response."""
    monkeypatch.setenv('GEMINI_API_KEY', 'stub')
    regions = _sample_regions()
    objs = [
        DetectedObject('green_cup',     (-0.5, 1.2), current_region='B'),
        DetectedObject('white_bottle', (1.8,  0.5), current_region='A'),
        DetectedObject('blue_cup',      (-1.2, -1.2), current_region='D'),
    ]
    inp = PlannerInput(
        objects=objs, regions=regions, goal_assignment={},
        robot_xy=(0.0, 0.0),
        instruction='move the white bottle to C, blue cup to A, green cup stays',
    )

    planner = VLMPlanner(log_path=None)
    planner._client = _fake_client_returning(
        '{"reasoning": "per instruction", "tasks": ['
        '{"object_label": "white_bottle", "target_region": "C", "order_index": 0},'
        '{"object_label": "blue_cup",      "target_region": "A", "order_index": 1}'
        ']}'
    )
    plan = planner.plan(inp)
    assert [t.object_label for t in plan] == ['white_bottle', 'blue_cup']
    assert plan[0].target_region == 'C'
    assert plan[1].target_region == 'A'
    assert [t.order_index for t in plan] == [0, 1]


def test_vlm_instruction_mode_skips_object_already_in_target(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'stub')
    regions = _sample_regions()
    objs = [
        DetectedObject('green_cup', (-0.5, 1.2), current_region='B'),
        DetectedObject('white_bottle', (1.2, 0.5), current_region='A'),
    ]
    inp = PlannerInput(
        objs, regions, {}, robot_xy=(0.0, 0.0),
        instruction='put white bottle in A and green cup in A',
    )

    planner = VLMPlanner(log_path=None)
    planner._client = _fake_client_returning(
        '{"reasoning": "noop for bottle", "tasks": ['
        '{"object_label": "white_bottle", "target_region": "A", "order_index": 0},'
        '{"object_label": "green_cup",    "target_region": "A", "order_index": 1}'
        ']}'
    )
    plan = planner.plan(inp)
    assert [t.object_label for t in plan] == ['green_cup']


# --- geometry helper -----------------------------------------------------


def test_euclidean_symmetry():
    assert euclidean((0, 0), (3, 4)) == 5.0
    assert euclidean((3, 4), (0, 0)) == 5.0
