"""Abstract planner backend + shared data classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DetectedObject:
    label: str
    pose_xy: Tuple[float, float]
    current_region: Optional[str] = None
    z: float = 0.0


@dataclass
class RegionInfo:
    name: str
    polygon: List[Tuple[float, float]]
    place_anchor: Tuple[float, float, float]  # x, y, yaw

    @property
    def center(self) -> Tuple[float, float]:
        arr = np.asarray(self.polygon)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())


@dataclass
class PickPlaceTask:
    object_label: str
    target_region: str
    pick_xy: Tuple[float, float]
    place_xy: Tuple[float, float]
    order_index: int = 0
    reasoning: str = ''


@dataclass
class PlannerInput:
    objects: List[DetectedObject]
    regions: Dict[str, RegionInfo]
    goal_assignment: Dict[str, str]              # object_label → region name
    robot_xy: Tuple[float, float]
    context_image_bgr: Optional[np.ndarray] = None
    instruction: Optional[str] = None            # natural-language operator instruction


class PlannerBackend(ABC):

    name: str = 'abstract'

    @abstractmethod
    def plan(self, inp: PlannerInput) -> List[PickPlaceTask]:
        ...


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def filter_actionable(inp: PlannerInput) -> List[DetectedObject]:
    """Objects that (a) have an assignment and (b) are not already in their goal region."""
    todo: List[DetectedObject] = []
    for obj in inp.objects:
        target = inp.goal_assignment.get(obj.label)
        if target is None:
            continue
        if obj.current_region == target:
            continue
        if target not in inp.regions:
            continue
        todo.append(obj)
    return todo
