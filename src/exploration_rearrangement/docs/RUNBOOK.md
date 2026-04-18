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
#   exploration_rearrangement object_detector_node
#   exploration_rearrangement region_manager_node
#   exploration_rearrangement task_planner_node
#   exploration_rearrangement task_executor_node
#   exploration_rearrangement manipulation_node
#   exploration_rearrangement head_scan_node
#   exploration_rearrangement fake_sim_node
```

If you plan to use the VLM planner backend:

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

### 2f. Edit `tasks.yaml` for the goal assignment

```yaml
assignments:
  white_bottle: C
  green_cup:    A
  blue_cup:     D
```

Labels MUST match `objects.yaml`. Region letters must match `regions.yaml`.

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
# Greedy planner (fast, no API):
ros2 launch exploration_rearrangement bringup.launch.py \
    planner_backend:=greedy run_slam:=false start_on_launch:=false

# OR VLM planner (needs GEMINI_API_KEY in env):
ros2 launch exploration_rearrangement bringup.launch.py \
    planner_backend:=vlm run_slam:=false start_on_launch:=false
```

`run_slam:=false` because §3b is already providing localization. Keep
`start_on_launch:=false` so you can verify everything is healthy before the
state machine begins.

Wait until you see all of:

- `TaskExecutorNode ready (backend=...)`
- `ObjectDetectorNode ready.`
- `Loaded N color specs: [...]`
- Nav2 `Creating bond timer...` lines stop printing

### 3d. Sanity checks before triggering

```bash
# 1. Frames
ros2 run tf2_ros tf2_echo map base_link
# Should print a stable transform — if it spins forever, localization is broken.

# 2. Detector
ros2 topic echo /detected_objects --once
# Should print a MarkerArray (probably empty until head moves).

# 3. Nav2 healthy
ros2 action list | grep navigate_to_pose
# → /navigate_to_pose

# 4. Manipulation servers up
ros2 action list | grep manipulation
# → /manipulation/pick, /manipulation/place
ros2 service list | grep manipulation
# → /manipulation/stow
```

If anything fails, fix it before triggering.

### 3e. Trigger the run (terminal 5)

```bash
ros2 service call /executor/start std_srvs/srv/Trigger
```

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

5. **VLM API key.** `planner_backend:=vlm` reads `$GEMINI_API_KEY`. If unset,
   the VLM planner falls back to a "no plan" output, which the executor
   reports as "Nothing to do." Set it before launching.

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
ros2 launch exploration_rearrangement sim.launch.py \
    planner_backend:=greedy start_on_launch:=true
```

The `fake_sim_node` synthesizes TF, an OccupancyGrid, a top-down RGB-D image
of three colored objects, the Nav2 action, and the manipulation actions. The
state machine self-triggers and writes
`/tmp/rearrangement_metrics.json` on completion. A successful run looks like:

```json
{
  "backend": "greedy",
  "success": true,
  "pick_attempts": 3,  "pick_successes": 3,
  "place_attempts": 3, "place_successes": 3
}
```

For the VLM backend in sim:

```bash
GEMINI_API_KEY="..." ros2 launch exploration_rearrangement sim.launch.py \
    planner_backend:=vlm start_on_launch:=true
```

---

## 6. Terminal layout cheat-sheet

| Terminal | Real robot | Sim |
|----------|-----------|-----|
| 1 | `stretch_driver.launch.py mode:=navigation` | — |
| 2 | `d435i_high_resolution.launch.py` + `rplidar.launch.py` | — |
| 3 | localization (slam_toolbox or map_server+amcl) | — |
| 4 | `bringup.launch.py planner_backend:=... run_slam:=false` | `sim.launch.py start_on_launch:=true` |
| 5 | `ros2 service call /executor/start ...` | — (sim auto-starts) |
