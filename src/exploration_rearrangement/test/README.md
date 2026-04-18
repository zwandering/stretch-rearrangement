# Test Suite

This directory contains two categories of tests for the `exploration_rearrangement` package:

1. **Pytest unit tests** — offline, no ROS runtime needed (`conftest.py`, `test_utils.py`, `test_planners.py`, `test_integration.py`)
2. **Robot integration tests** — require ROS nodes running (`test_head_scan.py`, `test_navigation.py`, etc.)

---

## Part A: Pytest Unit Tests

These tests verify core algorithms and data structures **without** launching any ROS nodes.
Run them with:

```bash
cd src/exploration_rearrangement
python -m pytest test/ -v
```

### conftest.py

Shared pytest configuration. Adds the package root to `sys.path` so that
`exploration_rearrangement.*` modules are importable outside of a colcon workspace.

### test_utils.py — Utility Function Tests (15 tests)

Tests the three utility libraries: frontier extraction, color segmentation, and region polygon helpers.

**Frontier extraction** (`frontier_utils.py`):
- `test_extract_frontiers_finds_boundary` — Creates a 10×10 grid with a free region surrounded by unknown cells. Verifies that `extract_frontiers()` returns at least one frontier cluster with nonzero total cells.
- `test_extract_frontiers_empty_when_fully_known` — A fully free (all-zero) grid should produce no frontiers.
- `test_grid_world_roundtrip` — Converts grid indices to world coordinates and back via `grid_to_world()` / `world_to_grid()`. Verifies lossless roundtrip with a non-trivial origin and resolution.
- `test_score_frontier_prefers_closer_larger` — A frontier closer to the robot should have a lower (better) score than a far frontier of the same size.

**Color segmentation** (`color_segmentation.py`):
- `test_segment_color_detects_red` — Feeds a synthetic 64×64 image with a red square. Verifies `segment_color()` detects it with correct center and sufficient area.
- `test_segment_color_returns_none_on_blank` — A blank (all-black) image should produce no detection.
- `test_segment_all_multiple_specs` — Image with both a red and a blue square. `segment_all()` should detect both labels.
- `test_pixel_to_camera_reasonable` — A uniform 1m depth image at the principal point should map to (0, 0, 1.0) in camera coordinates.
- `test_pixel_to_camera_rejects_zero_depth` — Zero-depth pixels should return `None` (invalid reading).
- `test_annotate_runs` — `annotate()` should return an image of the same shape without errors.

**Region polygon helpers** (`region_manager_node.py`):
- `test_point_in_polygon_inside` — Point (1,1) should be inside a 2×2 square at origin.
- `test_point_in_polygon_outside` — Points at x=−0.1 and x=3.0 should be outside.
- `test_point_in_polygon_concave` — Tests an L-shaped concave polygon: (0.5, 2.5) is inside, (2.0, 2.0) is outside the concavity.
- `test_polygon_centroid_square` — Centroid of a 2×2 square should be (1.0, 1.0).

### test_planners.py — Planner Backend Tests (10 tests)

Tests the `GreedyPlanner`, `VLMPlanner`, and shared utilities (`filter_actionable`, `euclidean`).

**filter_actionable**:
- `test_filter_actionable_skips_objects_already_placed` — An object whose `current_region` matches its goal should be excluded. Only misplaced objects remain.
- `test_filter_actionable_drops_unassigned_and_unknown_region` — Objects with no matching assignment, or assignments targeting a nonexistent region, should both be filtered out.

**GreedyPlanner**:
- `test_greedy_plan_length_and_order` — 3 misplaced objects → plan has 3 tasks with contiguous `order_index` [0, 1, 2], and each task's `place_xy` matches the target region's `place_anchor`.
- `test_greedy_first_pick_is_nearest_to_robot` — Greedy should pick the nearest object first. With robot at origin, object at (0.1, 0.1) should precede one at (1.9, 1.9).
- `test_greedy_empty_when_everything_placed` — If all objects are already in their goal regions, the plan should be empty.

**VLMPlanner**:
- `test_vlm_falls_back_to_greedy_when_no_api_key` — With no `GEMINI_API_KEY` env var, VLMPlanner should silently fall back to GreedyPlanner and still produce a valid plan.
- `test_vlm_parses_valid_json` — Injects a fake OpenAI client that returns valid JSON. Verifies VLMPlanner correctly parses the task ordering from the response.
- `test_vlm_auto_appends_missing_objects` — If the VLM only returns a subset of objects, VLMPlanner should auto-append the missing ones (via greedy fallback) so no object is skipped.
- `test_vlm_falls_back_on_invalid_json` — If the VLM returns non-JSON garbage, VLMPlanner should fall back to GreedyPlanner rather than crashing.

**Utility**:
- `test_euclidean_symmetry` — `euclidean((0,0), (3,4))` should be 5.0, and the function should be symmetric.

### test_integration.py — End-to-End Planner Integration (5 tests)

Tests both planner backends against the **real YAML config files** (`config/regions.yaml`, `config/tasks.yaml`), verifying that the config → planner pipeline works end-to-end.

- `test_real_yaml_loads_cleanly` — Loads `regions.yaml` and `tasks.yaml`, verifies 4 regions (A/B/C/D) with valid polygons and anchors, and 3 task assignments (blue_bottle, red_box, yellow_cup).
- `test_greedy_against_real_configs` — Simulates a scene with 3 misplaced objects, runs GreedyPlanner with real configs. Verifies 3 tasks produced, all routed to the correct goal region with correct `place_xy`, and contiguous ordering.
- `test_vlm_backend_wraps_greedy_when_offline` — With no API key, VLMPlanner still produces a valid 3-task plan using greedy fallback.
- `test_skip_objects_already_in_goal_region` — Places blue_bottle already in region C (its goal). Plan should only contain 2 tasks for the remaining objects.
- `test_greedy_plan_is_deterministic` — Running GreedyPlanner twice with identical input produces the same task ordering.

---

## Part B: Robot Integration Tests

## Robot Integration Test Running Order

Test subfunctions incrementally — earlier tests validate dependencies of later ones:

| # | Script | Validates | Dependencies |
|---|--------|-----------|--------------|
| 1 | `test_head_scan.py` | Head scanning | stretch_driver or fake_sim |
| 2 | `test_navigation.py` | Nav2 navigation | stretch_driver + Nav2 or fake_sim |
| 3 | `test_object_detector.py` | Object detection | Camera + TF |
| 4 | `test_region_manager.py` | Region management | No extra dependencies |
| 5 | `test_manipulation.py` | Pick / place | stretch_driver or fake_sim |
| 6 | `test_exploration.py` | Autonomous exploration | SLAM + Nav2 or fake_sim + exploration_node |
| 7 | `test_task_planner.py` | Task planning | detector + planner nodes |
| 8 | `test_task_executor.py` | State machine | All nodes |
| 9 | `test_e2e_sim.py` | End-to-end | sim.launch.py |

---

## Quick Start (Simulation)

```bash
# Build
cd your_ws && colcon build --symlink-install && source install/setup.bash

# Terminal 1 — start simulator
ros2 run exploration_rearrangement fake_sim_node

# Terminal 2 — run individual tests
python3 test/test_head_scan.py
python3 test/test_navigation.py
# ...
```

---

## Test Descriptions

### 1. test_head_scan.py — HeadScanNode

**Target**: `/head/start_scan`, `/head/stop_scan`, `/head/scan_once` (Trigger services)

**Description**: HeadScanNode controls the Stretch 3 head pan/tilt for periodic scanning
to enlarge camera FOV for object detection. It sends `joint_head_pan` and `joint_head_tilt`
targets via the `FollowJointTrajectory` action.

**Prerequisites**:
- Simulation: `ros2 run exploration_rearrangement fake_sim_node` + `ros2 run exploration_rearrangement head_scan_node`
- Real robot: `ros2 launch stretch_core stretch_driver.launch.py` + `ros2 run exploration_rearrangement head_scan_node`

**Input / Output**:
- Input: Trigger service request (no args)
- Output: Trigger response (`success: bool`, `message: str`)
- Side effect: head pan/tilt moves to different angles

**Tests**:
1. Call `/head/start_scan` → expect `success=True`, head starts periodic movement
2. Wait several seconds, call `/head/stop_scan` → expect `success=True`, head stops
3. Call `/head/scan_once` → expect `success=True` after a full sweep
4. Rapid start/stop toggling → expect no crashes or state corruption
5. Two consecutive `scan_once` calls → expect both succeed

---

### 2. test_navigation.py — Nav2 NavigateToPose

**Target**: `/navigate_to_pose` (NavigateToPose action)

**Description**: Tests robot navigation. Both ExplorationNode and TaskExecutorNode rely on
Nav2's `/navigate_to_pose` action server. This script sends goals directly and verifies arrival.

**Prerequisites**:
- Simulation: `ros2 run exploration_rearrangement fake_sim_node`
- Real robot: `ros2 launch stretch_core stretch_driver.launch.py` + Nav2 launch

**Input / Output**:
- Input: `NavigateToPose.Goal` — target PoseStamped (x, y, yaw) in map frame
- Output: `NavigateToPose.Result` — whether navigation succeeded
- Side effect: robot moves to goal

**Tests**:
1. Navigate to (1.0, 0.0) → short forward
2. Navigate to (0.0, 0.0) → return to origin
3. Navigate to (−1.0, 1.0) → diagonal + rotation
4. Navigate to (2.0, 2.0) → long distance (region A)
5. Send and cancel a goal → verify cancellation works
6. Send two consecutive goals → verify second is reached

---

### 3. test_object_detector.py — ObjectDetectorNode

**Target**: `/detected_objects`, `/detector/debug_image`, `/detector/clear`

**Description**: ObjectDetectorNode subscribes to RGB-D camera topics, performs HSV color segmentation
to detect target objects (blue_bottle, red_box, yellow_cup), then transforms pixel coordinates
to 3D map-frame positions via depth + TF. Results are published as `MarkerArray`.

**Prerequisites**:
- Simulation: `fake_sim_node` + `object_detector_node --ros-args -p objects_yaml:=<path>/config/objects.yaml`
- Real robot: `stretch_driver` + `object_detector_node`

**Input / Output**:
- Input: `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/camera/color/camera_info`
- Output: `/detected_objects` (MarkerArray), `/detector/debug_image` (Image)

**Tests**:
1. Verify camera topics are publishing data
2. Wait and verify `/detected_objects` has CUBE markers
3. Verify detected labels ⊆ {blue_bottle, red_box, yellow_cup}
4. Verify object positions are within map bounds (±5 m)
5. Verify `/detector/debug_image` is being published
6. Call `/detector/clear` → verify markers cleared
7. Wait for re-detection after clear

---

### 4. test_region_manager.py — RegionManagerNode

**Target**: `/regions/visualization`, `/regions/reload`, `point_in_polygon`, `polygon_centroid`

**Description**: RegionManagerNode manages semantic regions (A/B/C/D) as map-frame polygons
with `place_anchor` points. It provides RViz visualization, a reload service, and programmatic
APIs (`which_region`, `place_pose`, `pick_approach_pose`).

**Prerequisites**:
- `ros2 run exploration_rearrangement region_manager_node --ros-args -p regions_yaml:=<path>/config/regions.yaml`
- No fake_sim_node needed (no TF / Nav2 dependency)

**Input / Output**:
- Input: `regions.yaml` config file
- Output: `/regions/visualization` (MarkerArray)

**Tests**:
1. Verify `/regions/visualization` publishes markers
2. Verify markers contain 4 regions (A/B/C/D)
3. Call `/regions/reload` → expect success
4. Pure-Python: `point_in_polygon` — points inside correct regions
5. Pure-Python: `point_in_polygon` — points outside correct regions
6. Pure-Python: origin (0,0) boundary case (vertex of all 4 regions)
7. Pure-Python: `polygon_centroid` correctness
8. Grid-sample [−3,3]×[−3,3] to verify full coverage

---

### 5. test_manipulation.py — ManipulationNode (pick / place)

**Target**: `/manipulation/pick`, `/manipulation/place` (FollowJointTrajectory actions), `/manipulation/stow` (Trigger)

**Description**: ManipulationNode wraps Stretch 3 pick/place action sequences. It controls
`joint_lift`, `wrist_extension`, `gripper`, and head joints via `/stretch_controller/follow_joint_trajectory`.

Pick sequence: look forward → open gripper → lower → extend → close gripper → lift → retract → stow
Place sequence: lift → extend → lower → open gripper → retract → stow

**Prerequisites**:
- Simulation (recommended first): `ros2 run exploration_rearrangement fake_sim_node`
  (fake_sim_node has built-in pick/place servers; no separate manipulation_node needed)
- Real robot: `stretch_driver` + `manipulation_node`

**Input / Output**:
- Input: `FollowJointTrajectory.Goal` (sentinel), `Trigger.Request`
- Output: `FollowJointTrajectory.Result` (`error_code=0` success, non-zero failure), `Trigger.Response`

**Tests**:
1. Stow → verify arm retracts to safe position
2. Navigate near object → pick → expect `error_code=0`
3. Navigate to target region → place → expect `error_code=0`
4. Two consecutive stow calls → verify idempotency
5. Pick far from objects → expect failure (`error_code≠0`)
6. Place with nothing held → expect failure
7. Full flow: nav → pick → nav → place (end-to-end pick-and-place)

---

### 6. test_exploration.py — ExplorationNode

**Target**: `/exploration/start`, `/exploration/stop` (Trigger), `/exploration/status`, `/exploration/frontiers`

**Description**: ExplorationNode uses frontier-based exploration — extracts boundaries between
known and unknown map regions, selects optimal frontier, and sends Nav2 navigation goals.
Publishes "done" when no frontiers remain.

**Prerequisites**:
- Simulation: `fake_sim_node` + `exploration_node`
- Real robot: `stretch_driver` + SLAM + Nav2 + `exploration_node`

**Input / Output**:
- Input: `/map` (OccupancyGrid from SLAM), TF `map→base_link`
- Output: `/exploration/frontiers` (MarkerArray), `/exploration/status` ("navigating" / "done")
- Side effect: sends NavigateToPose goals to Nav2

**Note**: In simulation, the map is mostly known (walls only), so frontiers may be few and
exploration finishes quickly. Real SLAM produces more frontiers.

**Tests**:
1. Call `/exploration/start` → expect `success=True`
2. After starting, verify status or frontier markers are published
3. Call `/exploration/stop` → expect `success=True`
4. After stop, verify no new "navigating" status messages
5. Rapid start/stop toggling (3 rounds) → no crashes
6. On a mostly-known map, verify exploration completes quickly ("done")

---

### 7. test_task_planner.py — TaskPlannerNode

**Target**: `/planner/compute` (Trigger), `/planner/plan_visualization`

**Description**: TaskPlannerNode subscribes to `/detected_objects`, and when `/planner/compute`
is called it: (1) reads detected object positions, (2) gets robot pose from TF, (3) determines
each object's region via `point_in_polygon`, (4) calls the planner backend (greedy/VLM) to
generate `List[PickPlaceTask]`, (5) publishes plan visualization.

**Prerequisites**:
- Simulation: `fake_sim_node` + `object_detector_node` + `task_planner_node`

**Input / Output**:
- Input: `/detected_objects` (MarkerArray), TF, `tasks.yaml` + `regions.yaml`
- Output: `Trigger.Response` with plan summary, `/planner/plan_visualization` (MarkerArray)

**Tests**:
1. Wait for detections, call `/planner/compute` → expect `success=True`
2. Parse response message, verify it mentions detected object names
3. Verify `/planner/plan_visualization` has LINE_STRIP markers
4. Call compute again → verify deterministic result (greedy)
5. Clear detections then compute → verify empty plan
6. Pure-Python: GreedyPlanner with different scenarios
7. Pure-Python: `filter_actionable` correctly filters objects

---

### 8. test_task_executor.py — TaskExecutorNode

**Target**: `/executor/start`, `/executor/abort` (Trigger), `/executor/state`

**Description**: TaskExecutorNode is the top-level state machine:

```
IDLE → HEAD_SCAN → EXPLORE → WAIT_OBJECTS → PLAN →
  (NAV_TO_PICK → PICK → NAV_TO_PLACE → PLACE) × N → DONE
```

It orchestrates ExplorationNode, HeadScanNode, ManipulationNode, and Nav2 via service and action clients.

**Prerequisites**:
- Simulation: `ros2 launch exploration_rearrangement sim.launch.py start_on_launch:=false`
  or start all nodes individually

**Input / Output**:
- Input: Trigger requests (start/abort), `/detected_objects`, TF, Nav2, Manipulation actions
- Output: `/executor/state` (String), `/tmp/rearrangement_metrics.json`

**Tests**:
1. Verify initial state is IDLE
2. Call `/executor/start` → expect `success=True`
3. Monitor state transitions: IDLE → HEAD_SCAN → EXPLORE → ...
4. Call `/executor/abort` → verify state becomes FAILED
5. Attempt start after abort → verify it has no effect (not in IDLE)
6. Check metrics file (from a prior full run)

---

### 9. test_e2e_sim.py — End-to-End Simulation

**Target**: Full system pipeline via `sim.launch.py`

**Description**: Passive monitor that subscribes to `/executor/state` and `/detected_objects`,
watches the entire IDLE → explore → detect → plan → pick/place → DONE pipeline, and verifies
the final metrics file.

**Prerequisites**:
```bash
ros2 launch exploration_rearrangement sim.launch.py start_on_launch:=true planner_backend:=greedy
```

**Input / Output**:
- Input: None (pure listener)
- Output: Test report based on `/executor/state` and metrics file

**Tests**:
1. System enters HEAD_SCAN / EXPLORE
2. At least 3 objects detected
3. System enters PLAN state
4. Executes NAV_TO_PICK / PICK / NAV_TO_PLACE / PLACE
5. Reaches DONE state
6. Metrics file reports success
7. Total time within reasonable bounds (< 180s in simulation)
