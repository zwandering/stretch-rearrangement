"""VLM-driven planner using Gemini through the OpenAI SDK."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .base import (
    DetectedObject,
    PickPlaceTask,
    PlannerBackend,
    PlannerInput,
    filter_actionable,
)
from .greedy import GreedyPlanner


DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/'
DEFAULT_MODEL = 'gemini-2.5-flash'

SYSTEM_PROMPT = """You are a mobile-manipulation task planner for a Stretch 3 robot.
You are given:
  • the robot's current (x, y) position in the map frame,
  • a list of detected target objects with (x, y) positions and their current semantic region,
  • a list of semantic regions (name, polygon vertices, placement anchor),
  • a goal assignment mapping each object label to a desired region.

Produce an ordered pick-and-place plan that moves every object to its goal region while
minimizing total travel distance. Skip objects already in their goal region. Respect these rules:
  1) Do not include objects whose label is not in the goal assignment.
  2) Output STRICT JSON, no prose, matching the schema:
     {
       "reasoning": "<short explanation>",
       "tasks": [
         {"object_label": "<label>", "target_region": "<region name>", "order_index": <int>}
       ]
     }
  3) order_index must start at 0 and increase by 1.
  4) Every object in the goal assignment that is not already placed must appear exactly once.
"""


class VLMPlanner(PlannerBackend):

    name = 'vlm'

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = 'GEMINI_API_KEY',
        use_image: bool = True,
        max_retries: int = 2,
        log_path: Optional[str] = '/tmp/rearrangement_vlm_log.jsonl',
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.use_image = use_image
        self.max_retries = max_retries
        self.log_path = Path(log_path) if log_path else None
        self.logger = logger or logging.getLogger('VLMPlanner')
        self._fallback = GreedyPlanner()
        self._client = None

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            self.logger.warning(
                f'{self.api_key_env} not set; VLMPlanner will fall back to greedy.'
            )
            return False
        try:
            from openai import OpenAI
        except Exception as e:
            self.logger.warning(f'openai package missing: {e}; falling back to greedy.')
            return False
        self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return True

    def plan(self, inp: PlannerInput) -> List[PickPlaceTask]:
        if not self._ensure_client():
            return self._fallback.plan(inp)

        actionable = filter_actionable(inp)
        if not actionable:
            return []

        prompt_text = self._build_text_prompt(inp, actionable)
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': self._build_user_content(prompt_text, inp)},
        ]

        last_err: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={'type': 'json_object'},
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                plan_json = json.loads(content)
                tasks = self._json_to_tasks(plan_json, inp, actionable)
                self._log(prompt_text, plan_json, success=True)
                return tasks
            except Exception as e:
                last_err = f'{type(e).__name__}: {e}'
                self.logger.warning(
                    f'VLM attempt {attempt + 1} failed: {last_err}'
                )
                time.sleep(1.0 * (2 ** attempt))

        self.logger.warning(
            f'VLMPlanner giving up after {self.max_retries + 1} attempts '
            f'({last_err}); falling back to greedy.'
        )
        self._log(prompt_text, None, success=False, error=last_err)
        return self._fallback.plan(inp)

    # --- helpers ---------------------------------------------------------

    def _build_text_prompt(
        self, inp: PlannerInput, actionable: List[DetectedObject],
    ) -> str:
        regions_desc = []
        for name, r in inp.regions.items():
            cx, cy = r.center
            regions_desc.append({
                'name': name,
                'center': [round(cx, 3), round(cy, 3)],
                'polygon': [[round(x, 3), round(y, 3)] for x, y in r.polygon],
                'place_anchor': [round(v, 3) for v in r.place_anchor],
            })
        objects_desc = [
            {
                'label': o.label,
                'pose': [round(o.pose_xy[0], 3), round(o.pose_xy[1], 3)],
                'current_region': o.current_region,
                'goal_region': inp.goal_assignment.get(o.label),
            }
            for o in inp.objects
        ]
        actionable_labels = [o.label for o in actionable]

        return (
            'Robot pose (map frame): '
            f'[{round(inp.robot_xy[0], 3)}, {round(inp.robot_xy[1], 3)}]\n\n'
            'Detected objects:\n'
            f'{json.dumps(objects_desc, indent=2)}\n\n'
            'Regions:\n'
            f'{json.dumps(regions_desc, indent=2)}\n\n'
            'Goal assignment (label → region):\n'
            f'{json.dumps(inp.goal_assignment, indent=2)}\n\n'
            f'Objects still to place: {actionable_labels}\n\n'
            'Return the JSON plan.'
        )

    def _build_user_content(self, text: str, inp: PlannerInput):
        if not (self.use_image and inp.context_image_bgr is not None):
            return text
        ok, buf = cv2.imencode('.jpg', inp.context_image_bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return text
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        return [
            {'type': 'text', 'text': text},
            {'type': 'image_url',
             'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
        ]

    def _json_to_tasks(
        self,
        plan_json: dict,
        inp: PlannerInput,
        actionable: List[DetectedObject],
    ) -> List[PickPlaceTask]:
        raw = plan_json.get('tasks', [])
        reason = plan_json.get('reasoning', '')
        obj_by_label = {o.label: o for o in actionable}
        tasks: List[PickPlaceTask] = []
        seen: set = set()
        for i, entry in enumerate(raw):
            label = entry.get('object_label')
            target = entry.get('target_region')
            if label not in obj_by_label:
                continue
            if target not in inp.regions:
                continue
            if label in seen:
                continue
            seen.add(label)
            obj = obj_by_label[label]
            place_xy = inp.regions[target].place_anchor[:2]
            tasks.append(PickPlaceTask(
                object_label=label,
                target_region=target,
                pick_xy=obj.pose_xy,
                place_xy=place_xy,
                order_index=int(entry.get('order_index', i)),
                reasoning=reason,
            ))
        if len(tasks) < len(actionable):
            for obj in actionable:
                if obj.label in seen:
                    continue
                target = inp.goal_assignment[obj.label]
                place_xy = inp.regions[target].place_anchor[:2]
                tasks.append(PickPlaceTask(
                    object_label=obj.label,
                    target_region=target,
                    pick_xy=obj.pose_xy,
                    place_xy=place_xy,
                    order_index=len(tasks),
                    reasoning=reason + ' [auto-appended]',
                ))
        tasks.sort(key=lambda t: t.order_index)
        for i, t in enumerate(tasks):
            t.order_index = i
        return tasks

    def _log(self, prompt: str, response, success: bool, error: Optional[str] = None):
        if self.log_path is None:
            return
        try:
            record = {
                'ts': time.time(),
                'model': self.model,
                'success': success,
                'prompt': prompt,
                'response': response,
                'error': error,
            }
            with self.log_path.open('a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception:
            pass
