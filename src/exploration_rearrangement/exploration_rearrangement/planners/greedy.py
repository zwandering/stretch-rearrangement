"""Greedy nearest-neighbor planner (baseline)."""

from typing import List

from .base import (
    PickPlaceTask,
    PlannerBackend,
    PlannerInput,
    euclidean,
    filter_actionable,
)


class GreedyPlanner(PlannerBackend):

    name = 'greedy'

    def plan(self, inp: PlannerInput) -> List[PickPlaceTask]:
        remaining = filter_actionable(inp)
        robot_xy = inp.robot_xy
        ordered: List[PickPlaceTask] = []
        idx = 0
        while remaining:
            best = None
            best_cost = float('inf')
            for obj in remaining:
                target = inp.goal_assignment[obj.label]
                place_xy = inp.regions[target].place_anchor[:2]
                cost = euclidean(robot_xy, obj.pose_xy) + euclidean(obj.pose_xy, place_xy)
                if cost < best_cost:
                    best_cost = cost
                    best = (obj, target, place_xy)
            assert best is not None
            obj, target, place_xy = best
            ordered.append(PickPlaceTask(
                object_label=obj.label,
                target_region=target,
                pick_xy=obj.pose_xy,
                place_xy=place_xy,
                order_index=idx,
                reasoning=f'greedy step {idx}: cost={best_cost:.2f}',
            ))
            idx += 1
            robot_xy = place_xy
            remaining = [o for o in remaining if o.label != obj.label]
        return ordered
