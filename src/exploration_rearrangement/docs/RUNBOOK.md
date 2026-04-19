# Stretch 3 Rearrangement Runbook

End-to-end procedure for running the exploration + rearrangement system on a
real Hello Robot Stretch 3. Sim instructions are at the bottom.

---

## 0. One-time prerequisites

On the **robot computer** (the on-board NUC/Jetson running ROS):

```bash
# Stretch's stock workspace + drivers should already be installed:
#   ~/ament_ws/install/stretch_core            (stretch_driver, stretch_calibration)
#   ~/ament_ws/install/stretch_nav2            (Nav2 launch wrappers, tuned costmaps)
#   /opt/ros/jazzy/share/{slam_toolbox, nav2_*, realsense2_camera, rplidar_ros, ...}

# Pull this package next to it:
cd ~/Homework/16-762/project/stretch_rearrangement_ws
colcon build --packages-select exploration_rearrangement --symlink-install
source install/setup.bash
```

Verify:

```bash
ros2 pkg executables exploration_rearrangement
# → exploration_rearrangement exploration_node
#   exploration_rearrangement object_detector_node        (head D435i, always-on)
#   exploration_rearrangement fine_object_detector_node   (gripper D405, Bool-gated)
#   exploration_rearrangement region_manager_node
#   exploration_rearrangement task_planner_node           (instruction-driven VLM)
#   exploration_rearrangement task_executor_node
#   exploration_rearrangement manipulation_node
#   exploration_rearrangement head_scan_node
#   exploration_rearrangement fake_sim_node
#   exploration_rearrangement set_up_yolo_e
```

The planner is Gemini-only (no greedy fallback). Export the key once:

```bash
export GEMINI_API_KEY="..."   # put in ~/.bashrc so all terminals inherit it
```

---

## 1. Tune the manipulation constants for your tabletops

Open `src/exploration_rearrangement/exploration_rearrangement/manipulation_node.py`
and confirm three numbers match your physical setup before driving the robot:

| Constant | Default | What it must match |
|----------|---------|--------------------|
| `pick_height_m`  | 0.75 | Table-top height (m) where pick objects sit |
| `place_height_m` | 0.78 | Drop-off surface height (m) — slightly above pick |
| `arm_extend_m`   | 0.30 | How far to telescope. Must be **less than** `pick_standoff_m` (0.55 in `task_executor_node.py`), so the gripper actually reaches past the standoff |

Also at the top of that file:

```python
OPEN_GRIPPER = 0.25     # rad — verify on YOUR gripper
CLOSED_GRIPPER = -0.10  # rad — closed but not stalling on small objects
```

After editing, `colcon build --symlink-install` (with `--symlink-install`,
Python edits don't require rebuild for the next launch, but a rebuild won't
hurt).

---

## 2. First-time mapping & calibration

Do this once per physical environment. Output: `map.yaml` + `map.pgm` you can
reload with `map_server`, plus tuned `regions.yaml` and `objects.yaml`.

### 2a. Bring up drivers (terminal 1, on robot)

```bash
ros2 launch stretch_core stretch_driver.launch.py mode:=navigation
```

Wait until you see `Stretch driver ready`.

### 2b. Bring up sensors (terminal 2)

```bash
ros2 launch stretch_core d435i_high_resolution.launch.py
ros2 launch stretch_core rplidar.launch.py
```

Quick sanity check before launching the rest:

```bash
ros2 topic hz /scan                                  # ~10 Hz
ros2 topic hz /camera/color/image_raw                # ~30 Hz (BEST_EFFORT)
ros2 topic hz /camera/aligned_depth_to_color/image_raw
```

### 2c. Run mapping + exploration (terminal 3)

```bash
ros2 launch exploration_rearrangement mapping.launch.py
```

This starts SLAM Toolbox (online_async), Nav2, the head-scan node, the
exploration node (auto-enabled), the object detector, and RViz with the
project's display config.

In RViz, watch the OccupancyGrid `/map` fill in. The exploration node will
publish frontier goals to Nav2 automatically. Drive in via teleop if you want
to seed initial coverage:

```bash
# terminal 4 — teleop is optional during mapping
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

When the map looks complete, save it:

```bash
# terminal 5
ros2 run nav2_map_server map_saver_cli -f ~/maps/lab_map
# produces ~/maps/lab_map.yaml and ~/maps/lab_map.pgm
```

### 2d. Annotate region polygons in `regions.yaml`

Open RViz, hover the mouse over the map, and read the (x, y) coordinates from
the bottom status bar. Edit
`src/exploration_rearrangement/config/regions.yaml`:

- Each region needs a CCW polygon `[[x0,y0], [x1,y1], ...]` in **map** frame.
- `place_anchor: [x, y, yaw]` is where the robot stands (and faces) when
  placing into that region — pick a spot reachable by Nav2 with the gripper
  facing inward.
- The defaults assume a quadrant layout near the origin; replace them with
  real coordinates from your map.

Rebuild after editing the config:

```bash
colcon build --packages-select exploration_rearrangement --symlink-install
```

### 2e. Tune YOLOE prompts in `objects.yaml`

While the detector is running (`mapping.launch.py` already started it), look at:

```bash
ros2 run rqt_image_view rqt_image_view /detector/debug_image
```

If an object is missed or confused with something else, edit
`config/objects.yaml` and adjust the `prompts` list for that object. Multiple
synonyms are allowed and all get mapped back to the same `name` (e.g.
`["blue bottle", "blue water bottle"]`). Prefer short concrete phrases;
avoid attributes the vision backbone wasn't trained on.

If you want to switch model size or backend (e.g. exported OpenVINO for
CPU-only Stretch 3), pass `yolo_model:=...` to `bringup.launch.py` or set
the `model_path` parameter directly.

After editing, restart only the detector node:

```bash
# Ctrl-C the launch and re-launch — symlink-install picks up YAML edits.
# Prompts baked into exported artifacts (.engine / .onnx / _openvino_model)
# require re-running `ros2 run exploration_rearrangement set_up_yolo_e`.
```

### 2f. `tasks.yaml` is no longer read by the planner

The VLM planner is **instruction-driven** — at run time it consumes
`/instruction/text` (a natural-language string you publish) plus the live
detection dict, and lets Gemini decide the object→region assignment. The
static `tasks.yaml` file is still present for the legacy executor state
machine but the planner ignores it. Skip this step.

---

## 3. Each-run rearrangement procedure

Do this every time you want the robot to perform a rearrangement.

### 3a. Drivers + sensors (terminals 1 + 2 — same as §2a/§2b)

```bash
# terminal 1
ros2 launch stretch_core stretch_driver.launch.py mode:=navigation
# terminal 2
ros2 launch stretch_core d435i_high_resolution.launch.py
ros2 launch stretch_core rplidar.launch.py
```

### 3b. Localization on the saved map (terminal 3)

You have two options. Pick one.

**Option A — SLAM Toolbox in localization mode** (preferred — handles minor map
changes gracefully):

```bash
ros2 launch slam_toolbox localization_launch.py \
    slam_params_file:=$(ros2 pkg prefix exploration_rearrangement)/share/exploration_rearrangement/config/slam_params.yaml \
    map_file_name:=$HOME/maps/lab_map
```

**Option B — `map_server` + AMCL** (lighter, but you must give an initial pose):

```bash
ros2 run nav2_map_server map_server --ros-args \
    -p yaml_filename:=$HOME/maps/lab_map.yaml -p use_sim_time:=false
ros2 run nav2_amcl amcl --ros-args -p use_sim_time:=false
ros2 run nav2_lifecycle_manager lifecycle_manager \
    --ros-args -p autostart:=true -p node_names:='[map_server, amcl]'
```

If you used Option B, after the next step (RViz comes up with `bringup.launch.py`),
click **2D Pose Estimate** in the RViz toolbar and click+drag at the robot's
true location and heading on the map. Without this AMCL stays at (0,0,0) and
Nav2 will plan nonsense paths.

### 3c. Bring up the rearrangement stack (terminal 4)

```bash
ros2 launch exploration_rearrangement bringup.launch.py \
    run_slam:=false start_on_launch:=false \
    yolo_model:=yoloe-11s-seg.pt \
    vlm_model:=gemini-2.5-flash \
    vlm_api_key_env:=GEMINI_API_KEY \
    instruction_topic:=/instruction/text
```

`run_slam:=false` because §3b is already providing localization. Keep
`start_on_launch:=false` so you can verify everything is healthy before
anything moves.

This launch brings up, in one process group:

| Node | Role |
|------|------|
| `object_detector_node` | Head D435i detector — always-on, publishes `/detector/objects` + `/detector/bboxes_3d` + markers |
| `fine_object_detector_node` | Gripper D405 detector — **idle** until `/fine_detector/activate` gets `True` |
| `region_manager_node` | Serves the `regions.yaml` polygons + place anchors |
| `task_planner_node` | Subscribes `/instruction/text`, publishes `/planner/pick_place_plan` (PoseArray) |
| `task_executor_node`, `manipulation_node`, `head_scan_node`, `exploration_node` | State machine + arm + head sweeps + frontier drive |

Wait until you see all of:

- `object_detector_node` → `ready | mode=robot ... output_frame=map`
- `fine_object_detector_node` → `fine_object_detector_node ready ... activate on /fine_detector/activate`
- `task_planner_node` → `Received instruction:` (nothing until you publish one — that's fine)
- Nav2 `Creating bond timer...` lines stop printing

### 3d. Sanity checks before triggering

```bash
# 1. Frames
ros2 run tf2_ros tf2_echo map base_link
# Should print a stable transform — if it spins forever, localization is broken.

# 2. Head detector — /detector/objects populates as the head scan runs
ros2 topic echo /detector/objects --once
# vision_msgs/Detection3DArray; bbox.size is non-zero once bbox estimation fires.

# 3. Fine detector — should be ALIVE but silent (no Detection3DArray on /fine_detector/objects yet)
ros2 topic hz /fine_detector/objects
# Expect 'no new messages' until you activate it.

# 4. Nav2 healthy
ros2 action list | grep navigate_to_pose
# → /navigate_to_pose

# 5. Manipulation servers up
ros2 action list | grep manipulation
# → /manipulation/pick, /manipulation/place
ros2 service list | grep manipulation
# → /manipulation/stow
```

If anything fails, fix it before triggering.

### 3e. Trigger a run via a natural-language instruction (terminal 5)

The planner node listens on `/instruction/text`. Each message triggers one
plan attempt — snapshot current detections → classify into regions → ask
Gemini → publish a `geometry_msgs/PoseArray` on `/planner/pick_place_plan`
with alternating pick / place poses.

```bash
# Example — move all three objects to different regions
ros2 topic pub --once /instruction/text std_msgs/String \
    "{data: 'move the white bottle to region B, the blue cup to region C, and the green cup to region A'}"

# Watch the plan that comes out
ros2 topic echo /planner/pick_place_plan --once
# PoseArray; first pose = pick_0, second = place_0, third = pick_1, fourth = place_1, ...
```

The plan can also be kicked into the legacy executor state machine (the
executor still supports `/executor/start` for the older tasks.yaml flow):

```bash
ros2 service call /executor/start std_srvs/srv/Trigger
```

When the executor reaches the manipulation phase it sends `True` on
`/fine_detector/activate`, at which point `/fine_detector/objects` starts
streaming the refined gripper-camera pose in `base_link`. After the pick
completes it sends `False` to release CPU.

Watch the state machine progress in terminal 4:

```
State: IDLE → HEAD_SCAN
State: HEAD_SCAN → EXPLORE       # arm auto-stows here, then head sweeps
State: EXPLORE → WAIT_OBJECTS
State: WAIT_OBJECTS → PLAN
Planned 3 tasks: white_bottle→C, green_cup→A, blue_cup→D
State: PLAN → NAV_TO_PICK
State: NAV_TO_PICK → PICK
State: PICK → NAV_TO_PLACE
State: NAV_TO_PLACE → PLACE
State: PLACE → NAV_TO_PICK       # next task
...
State: PLACE → DONE
Metrics written to /tmp/rearrangement_metrics.json
```

### 3f. Abort / reset

```bash
# stop everything mid-run:
ros2 service call /executor/abort std_srvs/srv/Trigger

# clear the detector's accumulated object map between runs:
ros2 service call /detector/clear std_srvs/srv/Trigger

# stop / start exploration manually:
ros2 service call /exploration/stop std_srvs/srv/Trigger
ros2 service call /exploration/start std_srvs/srv/Trigger
```

### 3g. Inspect results

```bash
cat /tmp/rearrangement_metrics.json
```

Key fields: `success`, `pick_attempts/successes`, `place_attempts/successes`,
`task_results[]`, and timestamps you can subtract to get exploration time vs
total time.

---

## 3h. Standalone / debug startup for the three core nodes

Each of the three can be run in isolation (bench-side development) instead
of through `bringup.launch.py`. The contracts below are what each node
**expects on its inputs** and **guarantees on its outputs**. The full launch
wires them up correctly — read this section only if you're running one by
hand, or debugging one when the others are faulty.

Data flow at a glance:

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

### `object_detector_node` — head-mounted, always-on

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

### `fine_object_detector_node` — gripper-mounted, gated

**Purpose.** Idle until `/fine_detector/activate` receives `True`. While
active, every D405 RGB-D pair runs through YOLOE, the top-confidence
detection per known class is back-projected to 3D, transformed into
`base_link`, and published — giving the manipulation node a refined,
continuously-refreshed pose for whichever object is in the wrist's FOV.
Goes silent again on `False` (releases CPU during navigation).

This node shares `_parse_yolo_tracks` with `object_detector_node` — they
are identical in the way they rasterize YOLOE's segmentation masks at
`orig_shape` via `cv2.fillPoly`, so the letterbox-misalignment fix applies
to both.

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

### `task_planner_node` — instruction-driven VLM planner

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
above), then:

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

## 4. Pitfalls / things that have bitten us before

These are *not* fixed in code — you must verify each per environment.

1. **Pick/place heights vs your table.** `manipulation_node.py` defaults to
   `pick_height_m=0.75`, `place_height_m=0.78`. If your tabletop is at
   0.55 m, the gripper will swing through air; if at 0.95 m it will hit the
   underside. Measure once per setup.

2. **`arm_extend_m` < `pick_standoff_m`.** The executor parks the robot
   `pick_standoff_m=0.55 m` away from the object centroid. The arm only
   extends `arm_extend_m=0.30 m`. So the gripper closes ~0.25 m short of the
   object. For real picks, either bump `arm_extend_m` (in
   `manipulation_node.py`) or drop `pick_standoff_m` (in
   `task_executor_node.py`) so they match.

3. **Gripper open/closed values.** `OPEN_GRIPPER=0.25`, `CLOSED_GRIPPER=-0.10`
   in `manipulation_node.py`. Verify on YOUR gripper — too negative will
   stall the motor on small objects; not negative enough won't grip.

4. **Localization init.** With AMCL you MUST set initial pose via RViz "2D
   Pose Estimate" or the costmap stays empty and Nav2 returns
   `goal_aborted` immediately.

5. **VLM API key.** The planner reads `$GEMINI_API_KEY` on first use (name
   is the `vlm_api_key_env` parameter). If the variable is missing — or
   the SDK can't reach the endpoint — the node raises `VLMPlanError` after
   5 exponential-backoff retries and publishes an empty
   `/planner/pick_place_plan`. Set it before launching.

6. **YOLOE prompt/lighting behavior.** The text prompts default to generic
   phrases; if your lighting or object finish confuses the model, add
   synonyms in `config/objects.yaml` (see §2e). `/detector/debug_image`
   overlays boxes + track ids — use it to confirm detections.

7. **Stretch's two control modes.** The driver must be in `mode:=navigation`
   for Nav2; `manipulation_node` switches to `position` mode briefly during
   pick/place and back to `navigation` after. If you see joint commands
   silently dropped, check `ros2 service call /switch_to_position_mode` and
   `/switch_to_navigation_mode` are responding.

---

## 5. Sim verification (no robot needed)

For verifying the planner / executor / detector pipeline without hardware:

```bash
GEMINI_API_KEY="..." ros2 launch exploration_rearrangement sim.launch.py \
    start_on_launch:=true
```

The `fake_sim_node` synthesizes TF, an OccupancyGrid, a top-down RGB-D image
of three colored objects, the Nav2 action, and the manipulation actions. The
state machine self-triggers and writes
`/tmp/rearrangement_metrics.json` on completion. A successful run looks like:

```json
{
  "backend": "vlm",
  "success": true,
  "pick_attempts": 3,  "pick_successes": 3,
  "place_attempts": 3, "place_successes": 3
}
```

To drive the planner with a specific instruction in sim, launch with
`start_on_launch:=false` and publish once on `/instruction/text` yourself
(see §3e).

---

## 6. Terminal layout cheat-sheet

| Terminal | Real robot | Sim |
|----------|-----------|-----|
| 1 | `stretch_driver.launch.py mode:=navigation` | — |
| 2 | `d435i_high_resolution.launch.py` + `rplidar.launch.py` | — |
| 3 | localization (slam_toolbox or map_server+amcl) | — |
| 4 | `bringup.launch.py run_slam:=false` | `sim.launch.py start_on_launch:=true` |
| 5 | `ros2 topic pub /instruction/text std_msgs/String ...` (then `/executor/start` if using the state machine) | — (sim auto-starts) |
