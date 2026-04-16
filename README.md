# Stretch 3 — Autonomous Exploration-Driven Object Rearrangement

Implementation of the 16-762 team project: the Stretch 3 mobile manipulator
explores an unknown indoor environment, builds a 2-D map, detects three target
objects, partitions the map into semantic regions (A/B/C/D) and rearranges each
object into its goal region.

## Architecture

```
+-------------------+   +-----------------+   +------------------+
|  exploration_node |   | object_detector |   |  region_manager  |
|  (frontier → Nav2)|   | (HSV + depth)   |   | (polygons, place)|
+---------+---------+   +---------+-------+   +---------+--------+
          │                       │                     │
          ▼                       ▼                     ▼
                 +--------------------------------+
                 |        task_executor_node      |
                 |  (state machine orchestrator)  |
                 +---------+--------------+-------+
                           │              │
               +-----------▼--+      +----▼-----------+
               | task_planner |      |  manipulation  |
               | greedy / VLM |      |  (pick/place)  |
               +--------------+      +----------------+
```

## Building

```bash
cd stretch_rearrangement_ws
colcon build --symlink-install
source install/setup.bash
```

Python deps (install in the same env that runs ROS 2):

```bash
pip install openai numpy opencv-python pyyaml
```

## Running

### 1. Driver (Hello Robot)

```bash
ros2 launch stretch_core stretch_driver.launch.py
```

### 2. Full system (SLAM + Nav2 + our nodes + RViz)

Greedy planner (default):

```bash
ros2 launch exploration_rearrangement bringup.launch.py
```

VLM planner (Gemini via OpenAI SDK):

```bash
export GEMINI_API_KEY=your_key_here
ros2 launch exploration_rearrangement bringup.launch.py \
  planner_backend:=vlm
```

### 3. Kick off the task

```bash
ros2 service call /executor/start std_srvs/srv/Trigger
```

State transitions will stream to `/executor/state`. Metrics are written to
`/tmp/rearrangement_metrics.json` when the run finishes; VLM prompt/response
pairs are appended to `/tmp/rearrangement_vlm_log.jsonl`.

## Switching planners

- `planner_backend:=greedy` — deterministic nearest-neighbor baseline.
- `planner_backend:=vlm` — sends structured scene + the detector's annotated
  debug image to Gemini via `https://generativelanguage.googleapis.com/v1beta/openai/`
  and parses a JSON plan. Falls back to greedy on API / JSON failure.

Override the VLM model: `vlm_model:=gemini-2.5-pro` (or `gemini-2.0-flash`).

## Config files

| File | Purpose |
|---|---|
| `config/objects.yaml` | HSV ranges per target object |
| `config/regions.yaml` | Map-frame polygons + place anchors per region |
| `config/tasks.yaml`   | `label → goal_region` assignment |
| `config/slam_params.yaml` | SLAM Toolbox async mapping |
| `config/nav2_params.yaml` | Nav2 pipeline tuned for Stretch |

## Evaluation artifacts

`/tmp/rearrangement_metrics.json`
```json
{
  "backend": "vlm",
  "t_start": 1712345678.0,
  "t_explore_done": 1712345770.3,
  "first_detection": {"blue_bottle": 23.5, "red_box": 41.2, "yellow_cup": 55.9},
  "pick_attempts": 3, "pick_successes": 3,
  "place_attempts": 3, "place_successes": 3,
  "task_results": [
    {"object": "blue_bottle", "target": "C", "success": true, "ts": ...}
  ],
  "t_end": ..., "success": true
}
```

These map directly to the four evaluation metrics in the proposal:
task success rate, exploration efficiency, manipulation success rate,
end-to-end completion time.

## Ablation: greedy vs VLM

Run the same physical scene twice:

```bash
ros2 launch exploration_rearrangement bringup.launch.py planner_backend:=greedy
# save /tmp/rearrangement_metrics.json → metrics_greedy.json

ros2 launch exploration_rearrangement bringup.launch.py planner_backend:=vlm
# save /tmp/rearrangement_metrics.json → metrics_vlm.json
```

Compare `t_end - t_start`, pick/place success rates, and `task_results`.
