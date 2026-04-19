# Detection + VLM Planner Runbook

Scope: this document covers only the three nodes I own and have signed off
on — **head object detector**, **fine (wrist) object detector**, and the
**instruction-driven VLM planner**. Everything else in the stack (drivers,
SLAM, Nav2, manipulation, executor, sim) is in
[`INTEGRATION_RUNBOOK.md`](INTEGRATION_RUNBOOK.md) and is not confirmed
complete.

```
  RGB-D (head D435i)  ─►  object_detector_node  ─►  /detector/objects ──┐
                                                                         │
  operator (CLI)     ─►  /instruction/text      ─►  task_planner_node  ◄┤
                                                      │                  │
                                  /planner/pick_place_plan  (PoseArray)  │
                                                                         │
  RGB-D (wrist D405)  ─►  fine_object_detector_node  ─►  /fine_detector/objects
                                    ▲
                                    │ /fine_detector/activate (Bool)
                                    │ (executor flips True during PICK)
```

---

## 1. Build + verify

```bash
cd ~/Homework/16-762/project/stretch_rearrangement_ws
colcon build --packages-select exploration_rearrangement --symlink-install
source install/setup.bash
```

Verify the three executables are registered:

```bash
ros2 pkg executables exploration_rearrangement | grep -E \
    'object_detector_node|fine_object_detector_node|task_planner_node'
# → exploration_rearrangement object_detector_node
#   exploration_rearrangement fine_object_detector_node
#   exploration_rearrangement task_planner_node
```

The planner is Gemini-only (no greedy fallback). Export the key once:

```bash
export GEMINI_API_KEY="..."   # put in ~/.bashrc so all terminals inherit it
```

---

## 2. Configuration files

### 2a. `config/objects.yaml` — YOLOE prompts (both detectors)

Names defined here are the labels that appear in `/detector/objects` and
`/fine_detector/objects`, and the names the VLM planner reasons about.
Each entry maps one `name` to a list of `prompts` used as YOLOE text
prompts:

```yaml
objects:
  - name: white_bottle
    prompts: ["white water bottle", "white plastic bottle"]
  - name: green_cup
    prompts: ["green cup", "green mug"]
  - name: blue_cup
    prompts: ["blue cup", "blue mug"]
```

Guidelines:
- Multiple synonyms allowed; all map back to the same `name`.
- Prefer short concrete phrases; avoid attributes the vision backbone
  wasn't trained on (e.g. brand names).
- **Order matters when using an exported model.** `.pt` weights get
  `set_classes(...)` called at startup, but `.engine` / `.onnx` /
  `_openvino_model` bake the prompt list at export time. If you edit
  `objects.yaml` and the current model is exported, re-run:
  ```bash
  ros2 run exploration_rearrangement set_up_yolo_e
  ```
  Otherwise class indices line up with the wrong names. See the detector
  parameter tables below.

### 2b. `config/regions.yaml` — region polygons + place anchors (planner)

Defines the regions the planner classifies detections into and the
robot's place-pose anchor per region:

```yaml
regions:
  A:
    polygon: [[0, 0], [3, 0], [3, 3], [0, 3]]   # CCW, map frame
    place_anchor: [1.5, 1.5, 0.0]               # [x, y, yaw] for placing
  B:
    polygon: [[-3, 0], [0, 0], [0, 3], [-3, 3]]
    place_anchor: [-1.5, 1.5, 0.0]
  ...
```

Guidelines:
- CCW polygon in **map** frame (matches `point_in_polygon` convention).
- `place_anchor.yaw` is the heading the robot faces when placing;
  typically point the gripper toward the region centre.
- For new environments, read (x, y) from the RViz bottom status bar while
  hovering the map, and drop the coordinates in.
- Rebuild after editing (`colcon build --packages-select
  exploration_rearrangement --symlink-install`).

---

## 3. Standalone / debug startup for the three nodes

Each of the three can be run in isolation. The contracts below are what
each node **expects on its inputs** and **guarantees on its outputs**.

### 3a. `object_detector_node` — head-mounted, always-on

**Purpose.** Continuously detect the tracked objects (names from
`config/objects.yaml`) from the head D435i's RGB + aligned depth,
back-project each detection to 3D, EMA-smooth per object in the world
frame, and publish the world-frame pose + 3D bbox + debug imagery.

**Expected inputs** (robot-mode defaults — override via parameters):

| Topic | Type | Notes |
|-------|------|-------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | BGR/RGB, sensor QoS |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | 16UC1 (millimetres), time-aligned with color |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | intrinsics |
| TF `camera_color_optical_frame → map` | — | required to report poses in `map`; in `debug` mode we stay in the camera frame and TF isn't needed |

In `debug` mode the defaults switch to the standalone realsense2_camera
namespace (`/camera/camera/...`) and `output_frame` stays at
`camera_color_optical_frame` — you get valid 3D poses without SLAM.

**Key parameters:**

| Param | Default | Purpose |
|-------|---------|---------|
| `mode` | `robot` | `robot` (Stretch) or `debug` (standalone D435i) |
| `model_path` | `yoloe-11s-seg.pt` | `.pt` loads + calls `set_classes` at startup; `.engine` / `.onnx` / `_openvino_model` assumes prompts were baked at export time (see `set_up_yolo_e.py`) |
| `objects_yaml` | *(auto-discover from package share)* | Object→prompts map. When empty, the node auto-finds `config/objects.yaml` in the installed share, then in the source tree. Critical: when you use an exported model, the prompt list **must** match what was baked in — otherwise class indices line up with the wrong names. |
| `conf_threshold` | `0.25` | YOLOE minimum confidence |
| `merge_dist_m` | `0.3` | Teleport-vs-EMA threshold. If a new detection is within this of the cached pose for the same name → associate + EMA; else replace (treated as the object moved). |
| `dedup_iou_threshold` | `0.3` | Cross-class 3D NMS. Multiple prompts firing on the same physical object → keep only the highest-conf candidate. |
| `ema_alpha` | `1.0` | `1.0` = replace outright (no smoothing). Drop to e.g. `0.3` for temporal smoothing. |
| `bbox_line_width` | `0.005` | RViz bbox wireframe width (m) |

**Outputs:**

| Topic | Type | Contents |
|-------|------|----------|
| `/detector/objects` | `vision_msgs/Detection3DArray` | `output_frame` (default `map`); per-name smoothed pose + 3D bbox center/size/orientation. This is the contract the planner consumes. |
| `/detector/bboxes_3d` | `vision_msgs/BoundingBox3DArray` | Same bboxes, standalone message |
| `/detector/bboxes_3d_markers` | `visualization_msgs/MarkerArray` | Translucent CUBEs for RViz |
| `/detector/centers` | `geometry_msgs/PoseArray` | Raw centers only (no label), one per frame |
| `/detected_objects` | `visualization_msgs/MarkerArray` | CUBE + label text markers (1 Hz) |
| `/detector/debug_image` | `sensor_msgs/Image` | 2D bbox + track-id overlay — the only way to visually confirm YOLOE is firing |
| `/detector/clear` | `std_srvs/srv/Trigger` | Reset smoothed state |

**Standalone bring-up** (bench D435i on a laptop, no robot, no SLAM):

```bash
# Option A — this launch file also starts the camera:
ros2 launch exploration_rearrangement detector_debug.launch.py \
    start_realsense:=true \
    model_path:=yoloe-11s-seg.pt \
    run_rviz:=true            # opens the detector_debug.rviz config

# Option B — you already have a D435i publishing via realsense2_camera:
ros2 launch exploration_rearrangement detector_debug.launch.py \
    model_path:=yoloe-11s-seg.pt

# Verify:
ros2 topic echo /detector/objects --once      # Detection3DArray; frame=camera_color_optical_frame in debug mode
ros2 run rqt_image_view rqt_image_view /detector/debug_image
```

---

### 3b. `fine_object_detector_node` — gripper-mounted, gated

**Purpose.** Idle until `/fine_detector/activate` receives `True`. While
active, every D405 RGB-D pair runs through YOLOE, the top-confidence
detection per known class is back-projected to 3D, transformed into
`base_link`, and published — giving the manipulation node a refined,
continuously-refreshed pose for whichever object is in the wrist's FOV.
Goes silent again on `False` (releases CPU during navigation).

This node shares `_parse_yolo_tracks` with `object_detector_node` — both
rasterize YOLOE's segmentation masks at `orig_shape` via `cv2.fillPoly`,
so the letterbox-misalignment fix applies uniformly.

**Expected inputs** (robot-mode defaults):

| Topic | Type | Notes |
|-------|------|-------|
| `/gripper_camera/color/image_raw` | `sensor_msgs/Image` | D405 RGB |
| `/gripper_camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | D405 depth, aligned to color |
| `/gripper_camera/color/camera_info` | `sensor_msgs/CameraInfo` | intrinsics |
| `/fine_detector/activate` | `std_msgs/Bool` | gate: `True` = start, `False` = stop |
| TF `gripper_camera_color_optical_frame → base_link` | — | required to output in `base_link` |

**Key parameters:**

| Param | Default | Purpose |
|-------|---------|---------|
| `conf_threshold` | `0.30` | Slightly stricter than head detector (less clutter at close range) |
| `ema_alpha` | `1.0` | Same semantics as head detector — default is replace-outright |
| `merge_dist_m` | `0.3` | Teleport threshold |
| `clear_state_on_activate` | `true` | On each `True`, forget previous EMA — a fresh grasp approach shouldn't inherit a stale pose from a previous pick |
| `output_frame` | `base_link` (robot) | `camera_color_optical_frame` in debug mode |

No cross-class 3D dedup (the wrist FOV is too narrow to need it).

**Outputs** (published only while `active=True`):

| Topic | Type | Contents |
|-------|------|----------|
| `/fine_detector/objects` | `vision_msgs/Detection3DArray` | frame `base_link`; refined per-class pose + bbox. Manipulation reads this to servo the grasp. |
| `/fine_detector/bboxes_3d` | `vision_msgs/BoundingBox3DArray` | standalone bbox message |
| `/fine_detector/bboxes_3d_markers` | `visualization_msgs/MarkerArray` | RViz CUBEs |
| `/fine_detector/debug_image` | `sensor_msgs/Image` | 2D overlay |

**Standalone bring-up + manual activation** (bench test with any D435i/D405):

```bash
# In robot mode it reads /gripper_camera/...; in debug mode it reads the
# same standalone realsense2_camera namespace as the head detector — so
# you can validate the fine-detection pipeline with a single D435i on a desk.
ros2 run exploration_rearrangement fine_object_detector_node --ros-args \
    -p mode:=debug -p model_path:=yoloe-11s-seg.pt

# Another terminal — flip it on:
ros2 topic pub --once /fine_detector/activate std_msgs/Bool "{data: true}"
ros2 topic echo /fine_detector/objects
# Confirm refined poses arrive at ~5-10 Hz in the configured output_frame.

# Off:
ros2 topic pub --once /fine_detector/activate std_msgs/Bool "{data: false}"
```

---

### 3c. `task_planner_node` — instruction-driven VLM planner

**Purpose.** Keep a live dictionary of detections (label → xyz + conf),
classify each object into one of the regions in `regions.yaml`, and on each
`/instruction/text` message, ask Gemini (via OpenAI-compatible SDK) to
return an ordered pick / place plan, then publish it as a flat
`geometry_msgs/PoseArray` with alternating pick, place poses. No greedy
fallback — on API failure the node retries with exponential backoff (base
1 s, cap 30 s, up to 5 retries) and, if all retries fail, publishes an
empty `PoseArray` plus a warning. The node stays responsive for the next
instruction regardless.

**Expected inputs:**

| Topic / resource | Type | Notes |
|------------------|------|-------|
| `/detector/objects` | `vision_msgs/Detection3DArray` | BEST_EFFORT QoS; latest pose per label overwrites the dict each frame |
| `/map` | `nav_msgs/OccupancyGrid` | TRANSIENT_LOCAL + RELIABLE; cached for prompt metadata (resolution, size, origin). Obstacle-aware place-pose validation is a future use — the node only logs the metadata today. |
| `/instruction/text` | `std_msgs/String` | Each publish triggers one plan attempt against a snapshot of the dict |
| TF `map → base_link` | — | Robot xy included in the Gemini prompt (affects travel-time ordering). If missing, planner warns and sends `(0, 0)`. |
| `$GEMINI_API_KEY` | env var | Read at first `plan()` call. Env var name is configurable via `vlm_api_key_env`. |

**Key parameters:**

| Param | Default | Purpose |
|-------|---------|---------|
| `regions_yaml` | *(auto-discover)* | Path to the polygons + place anchors. Auto-found in package share if empty. |
| `vlm_model` | `gemini-2.5-flash` | Any OpenAI-compatible model id |
| `vlm_base_url` | `https://generativelanguage.googleapis.com/v1beta/openai/` | Google's OpenAI-compatible endpoint |
| `vlm_api_key_env` | `GEMINI_API_KEY` | Name of the env var to read |
| `vlm_max_retries` | `5` | Each retry re-issues the call with the same instruction + frozen scene snapshot |
| `vlm_retry_base_sec` | `1.0` | Backoff base; sleep = `base * 2^attempt`, capped at 30 s |
| `min_detections_before_plan` | `3` | Logs a warning when the detection dict has fewer labels, but still plans with what's cached |
| `instruction_topic` | `/instruction/text` | Rename if you want two planners on the same graph |

**Outputs:**

| Topic | Type | Contents |
|-------|------|----------|
| `/planner/pick_place_plan` | `geometry_msgs/PoseArray` | frame `map`; poses alternate as `[pick_0, place_0, pick_1, place_1, …]`. `position = (x, y, z)`; picks have identity orientation, places carry the region's anchor yaw as a quaternion. Empty array on failure. |
| `/planner/plan_visualization` | `visualization_msgs/MarkerArray` | pick→place arrows for RViz |

Side channel: every prompt + raw response is appended to
`/tmp/rearrangement_vlm_log.jsonl` — essential for diagnosing prompt drift
or LLM-side errors without re-running the robot.

**Standalone bring-up.** Two options, depending on whether you want real
detections or canned ones:

**(A) With a real detector upstream** — start the detector first (see
§3a), then:

```bash
export GEMINI_API_KEY=...   # or source ~/.gemini_env
ros2 run exploration_rearrangement task_planner_node --ros-args \
    -p regions_yaml:=$(ros2 pkg prefix --share exploration_rearrangement)/config/regions.yaml \
    -p vlm_model:=gemini-2.5-flash

# Another terminal — one instruction triggers one plan:
ros2 topic pub --once /instruction/text std_msgs/String \
    "{data: 'move the white bottle to region B'}"
ros2 topic echo /planner/pick_place_plan --once
```

**(B) With `fake_planner_inputs` — no camera, no SLAM, no TF.** This is
the recommended loop for debugging prompt quality, JSON parsing, or
`filter_actionable` edge cases. The fixture publishes all three inputs the
planner needs (`/detector/objects`, `/map`, and the static
`map→base_link` TF) from `config/planner_test_scenarios.yaml`:

```bash
# Terminal 1 — the planner itself
export GEMINI_API_KEY=...
ros2 run exploration_rearrangement task_planner_node --ros-args \
    -p regions_yaml:=$(ros2 pkg prefix --share exploration_rearrangement)/config/regions.yaml

# Terminal 2 — fake the upstream inputs (pick any scenario from the YAML)
ros2 run exploration_rearrangement fake_planner_inputs --ros-args \
    -p scenario:=quadrants_mixed

# Terminal 3 — send an instruction (the YAML lists sample ones per scenario)
ros2 topic pub --once /instruction/text std_msgs/String \
    "{data: 'move the white bottle to region C, the green cup to region A, and the blue cup to region B'}"

# Terminal 4 — observe the plan + the JSONL log
ros2 topic echo /planner/pick_place_plan --once
tail -f /tmp/rearrangement_vlm_log.jsonl
```

Available scenarios in the fixture:

| Scenario | What it tests |
|----------|---------------|
| `quadrants_mixed` | Happy path — three objects across three regions |
| `already_sorted` | Objects already in the instructed regions → expect empty plan (`filter_actionable` drops them) |
| `one_missing` | Only 2 of 3 labels visible → exercises the `min_detections_before_plan` warn path |
| `cluster_single_region` | All objects clustered in one region; instruction forces them to spread |
| `off_map_object` | One object outside every region polygon → exercises `current_region=None` handling |

Add your own by editing
`src/exploration_rearrangement/config/planner_test_scenarios.yaml` and
rebuilding (`colcon build --packages-select exploration_rearrangement
--symlink-install`).

---

## 4. Pitfalls specific to detection + planner

1. **Exported-model / `objects.yaml` prompt-index mismatch.** Exported
   artifacts (`.engine` / `.onnx` / `_openvino_model`) bake the YOLOE
   text prompts at export time. If you later edit `config/objects.yaml`
   without re-running `ros2 run exploration_rearrangement set_up_yolo_e`,
   the runtime class indices will point at the old names — a green cup
   will confidently announce itself as `blue_cup`. Re-export, or switch
   back to the `.pt` weights which are re-prompted at startup.

2. **VLM API key.** The planner reads `$GEMINI_API_KEY` on first use
   (env var name is the `vlm_api_key_env` parameter). If the variable is
   missing — or the SDK can't reach the endpoint — the node raises
   `VLMPlanError` after 5 exponential-backoff retries and publishes an
   empty `/planner/pick_place_plan`. Set it before launching.

3. **YOLOE prompt / lighting behavior.** The text prompts default to
   short generic phrases; if your lighting or object finish confuses the
   model, add synonyms in `config/objects.yaml` (§2a) and use
   `/detector/debug_image` in `rqt_image_view` to confirm detections
   visually.

4. **Detection cache has no TTL — by design.** The head detector keeps
   the last-seen pose per label indefinitely. In a static scene, walking
   away and coming back re-associates the new detection with the cached
   pose (same name + within `merge_dist_m`). If you run in a dynamic
   scene, call `/detector/clear` between episodes or drop `merge_dist_m`.
