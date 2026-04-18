# Exploration Rearrangement — Node Interfaces & Integration Guide

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Data Flow](#data-flow)
3. [Node Interfaces](#node-interfaces)
   - [ExplorationNode](#1-explorationnode)
   - [ObjectDetectorNode](#2-objectdetectornode)
   - [RegionManagerNode](#3-regionmanagernode)
   - [HeadScanNode](#4-headscannode)
   - [ManipulationNode](#5-manipulationnode)
   - [TaskPlannerNode](#6-taskplannernode)
   - [TaskExecutorNode](#7-taskexecutornode)
   - [FakeSimNode](#8-fakesimnode)
4. [Planner Backends](#planner-backends)
5. [Utility Libraries](#utility-libraries)
6. [Configuration Files](#configuration-files)
7. [Integration Workflow](#integration-workflow)
8. [Testing Guide](#testing-guide)

---

## System Architecture

```
                     ┌─────────────────────┐
                     │   TaskExecutorNode   │   (state-machine orchestrator)
                     │  /executor/start     │
                     │  /executor/abort     │
                     └──┬────┬────┬────┬───┘
                        │    │    │    │
           ┌────────────┘    │    │    └────────────┐
           ▼                 ▼    ▼                  ▼
  ┌────────────────┐ ┌──────────────┐  ┌──────────────────┐
  │ ExplorationNode│ │ HeadScanNode │  │ ManipulationNode │
  │ (frontier nav) │ │ (head sweep) │  │ (pick/place)     │
  └───────┬────────┘ └──────────────┘  └──────────────────┘
          │                                      │
          ▼                                      ▼
  ┌────────────────┐                ┌────────────────────┐
  │     Nav2       │                │  stretch_controller │
  │ /navigate_to_  │                │ /follow_joint_      │
  │  pose          │                │  trajectory          │
  └────────────────┘                └────────────────────┘
          ▲
          │
  ┌───────┴─────────┐    ┌──────────────────┐
  │ ObjectDetector  │───▶│ TaskPlannerNode  │
  │ (HSV + depth)   │    │ (greedy/VLM)     │
  └─────────────────┘    └──────────────────┘
          │                       │
          ▼                       ▼
  ┌─────────────────┐    ┌──────────────────┐
  │ RegionManager   │    │ /planner/compute │
  │ (region mgmt)   │    │ → generate plan  │
  └─────────────────┘    └──────────────────┘
```

## Data Flow

```
 /map (OccupancyGrid)  ──────────────────────────▶  ExplorationNode
                                                        │
 /camera/color/image_raw ┐                              │ /navigate_to_pose
 /camera/aligned_depth   ├──▶ ObjectDetectorNode        ▼
 /camera/color/info      ┘        │                   Nav2
                                  │ /detected_objects
                                  ▼
 TaskPlannerNode  ◀────────  MarkerArray (CUBE markers, ns=label)
       │                          │
       │ /planner/compute         │
       │ → PickPlaceTask[]        ▼
       ▼                     TaskExecutorNode
 /planner/plan_visualization      │
                                  ├──▶ /navigate_to_pose  (Nav2)
                                  ├──▶ /manipulation/pick  (ManipulationNode)
                                  ├──▶ /manipulation/place (ManipulationNode)
                                  ├──▶ /exploration/start  (ExplorationNode)
                                  ├──▶ /exploration/stop   (ExplorationNode)
                                  ├──▶ /head/start_scan    (HeadScanNode)
                                  ├──▶ /head/stop_scan     (HeadScanNode)
                                  └──▶ /manipulation/stow  (ManipulationNode)
```

---

## Node Interfaces

### 1. ExplorationNode

**File**: `exploration_node.py`
**Function**: Frontier-based autonomous exploration. Extracts boundaries between known and unknown map regions, selects the best frontier, and sends Nav2 navigation goals.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `map_frame` | string | `map` | Map coordinate frame |
| `base_frame` | string | `base_link` | Robot base frame |
| `min_cluster_size` | int | `8` | Minimum frontier cluster size (smaller clusters ignored) |
| `goal_tolerance_m` | float | `0.5` | Distance tolerance for reaching a frontier (m) |
| `alpha_dist` | float | `1.0` | Distance weight (higher prefers closer frontiers) |
| `beta_info` | float | `0.05` | Information weight (higher prefers larger frontiers) |
| `replan_period_s` | float | `3.0` | Re-planning period (s) |
| `goal_timeout_s` | float | `60.0` | Single goal timeout (s) |
| `enabled_on_start` | bool | `False` | Whether to start exploring automatically on launch |

#### Subscriptions

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `/map` | `OccupancyGrid` | RELIABLE, TRANSIENT_LOCAL | Map from SLAM |

#### Publishers

| Topic | Type | Description |
|-------|------|-------------|
| `/exploration/frontiers` | `MarkerArray` | Frontier visualization (SPHERE_LIST) |
| `/exploration/status` | `String` | `"navigating"` / `"done"` |

#### Services (Server)

| Service | Type | Description |
|---------|------|-------------|
| `/exploration/start` | `Trigger` | Start exploration |
| `/exploration/stop` | `Trigger` | Stop exploration |

#### Action Client

| Action | Type | Description |
|--------|------|-------------|
| `/navigate_to_pose` | `NavigateToPose` | Send frontier navigation goals |

#### TF Dependencies
- Reads `map` → `base_link` transform (robot pose)

---

### 2. ObjectDetectorNode

**File**: `object_detector_node.py`
**Function**: Detects target objects via HSV color segmentation, combines with depth image and TF to convert pixel coordinates to 3D positions in the map frame. Uses EMA smoothing to merge repeated detections.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `objects_yaml` | string | `""` | Path to HSV color config file |
| `rgb_topic` | string | `/camera/color/image_raw` | RGB image topic |
| `depth_topic` | string | `/camera/aligned_depth_to_color/image_raw` | Depth image topic |
| `info_topic` | string | `/camera/color/camera_info` | Camera intrinsics topic |
| `camera_frame` | string | `camera_color_optical_frame` | Camera optical frame |
| `map_frame` | string | `map` | Map coordinate frame |
| `merge_dist_m` | float | `0.3` | Merge distance for new vs. existing detections (m) |
| `ema_alpha` | float | `0.3` | EMA smoothing factor (higher biases toward newer values) |
| `publish_debug_image` | bool | `True` | Whether to publish annotated debug image |

#### Subscriptions

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `/camera/color/camera_info` | `CameraInfo` | sensor_data | Camera intrinsics |
| `/camera/color/image_raw` | `Image` | sensor_data | RGB image (synced) |
| `/camera/aligned_depth_to_color/image_raw` | `Image` | sensor_data | Depth image (synced) |

> RGB and Depth are synchronized with `ApproximateTimeSynchronizer`, slop=0.1s

#### Publishers

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/detected_objects` | `MarkerArray` | 1 Hz (timer) | **Core output** — CUBE markers (ns=label) + TEXT markers |
| `/detector/debug_image` | `Image` | At frame rate | Annotated BGR debug image |

#### `/detected_objects` MarkerArray Format

Each detected object produces two Markers:
- **CUBE** (type=1): `ns` = object label (e.g. `"blue_bottle"`), `pose` = 3D position in map frame
- **TEXT_VIEW_FACING**: Displays the label text

Other nodes filter by `marker.type == Marker.CUBE` and read `marker.ns` and `marker.pose` to obtain detection results.

#### Services (Server)

| Service | Type | Description |
|---------|------|-------------|
| `/detector/clear` | `Trigger` | Clear all detected objects |

#### TF Dependencies
- Reads `camera_color_optical_frame` → `map` transform

---

### 3. RegionManagerNode

**File**: `region_manager_node.py`
**Function**: Manages semantic regions on the map (polygons), provides region lookup and placement pose computation.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `regions_yaml` | string | `""` | Path to region definition file |
| `map_frame` | string | `map` | Map coordinate frame |

#### Publishers

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/regions/visualization` | `MarkerArray` | 2 Hz | Region boundaries (LINE_STRIP) + labels (TEXT) |

#### Services (Server)

| Service | Type | Description |
|---------|------|-------------|
| `/regions/reload` | `Trigger` | Reload regions.yaml |

#### Programmatic API (in-process calls)

```python
which_region(x: float, y: float) -> Optional[str]
```
Returns the region name containing point (x,y), or None.

```python
place_pose(region_name: str) -> Optional[PoseStamped]
```
Returns the placement navigation target for the region (from place_anchor).

```python
pick_approach_pose(target_xy, robot_xy, standoff_m=0.55) -> PoseStamped
```
Computes an approach pose offset by standoff_m from the target, facing toward it.

#### Standalone Helper Functions

```python
point_in_polygon(px, py, polygon) -> bool
polygon_centroid(polygon) -> Tuple[float, float]
```

These functions are directly imported by TaskPlannerNode and TaskExecutorNode.

---

### 4. HeadScanNode

**File**: `head_scan_node.py`
**Function**: Periodically sweeps the Stretch 3 head pan/tilt to enlarge camera field of coverage for object detection.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trajectory_action` | string | `/stretch_controller/follow_joint_trajectory` | Joint trajectory action |
| `period_s` | float | `8.0` | Scan period (s) |
| `enabled_on_start` | bool | `False` | Whether to start scanning on launch |
| `pan_waypoints` | float[] | `[-1.2, -0.6, 0.0, 0.6, 1.2]` | Pan angle waypoints (rad) |
| `tilt_angle` | float | `-0.55` | Fixed tilt angle (rad, negative = looking down) |

#### Services (Server)

| Service | Type | Description |
|---------|------|-------------|
| `/head/start_scan` | `Trigger` | Start periodic scanning |
| `/head/stop_scan` | `Trigger` | Stop scanning |
| `/head/scan_once` | `Trigger` | Execute one full sweep (blocking) |

#### Action Client

| Action | Type | Description |
|--------|------|-------------|
| `/stretch_controller/follow_joint_trajectory` | `FollowJointTrajectory` | Controls `joint_head_pan`, `joint_head_tilt` |

---

### 5. ManipulationNode

**File**: `manipulation_node.py`
**Function**: Wraps Stretch 3 pick/place action sequences.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trajectory_action` | string | `/stretch_controller/follow_joint_trajectory` | Joint control action |
| `switch_to_position_srv` | string | `/switch_to_position_mode` | Switch to position mode service |
| `switch_to_navigation_srv` | string | `/switch_to_navigation_mode` | Switch to navigation mode service |
| `stow_srv` | string | `/stow_the_robot` | Stow service |
| `pick_height_m` | float | `0.75` | Pick height (m) |
| `place_height_m` | float | `0.78` | Place height (m) |
| `arm_extend_m` | float | `0.30` | Arm extension length (m) |

#### Action Servers

| Action | Type | Description |
|--------|------|-------------|
| `/manipulation/pick` | `FollowJointTrajectory` | **Pick sequence**: open gripper → lower → extend → close → lift → retract → stow |
| `/manipulation/place` | `FollowJointTrajectory` | **Place sequence**: lift → extend → lower → open → retract → stow |

> Goal contents are ignored (used as sentinel); preset heights/distances from parameters are used.

**Result**:
- `error_code = 0`: Success
- `error_code != 0`: Failure (e.g. `PATH_TOLERANCE_VIOLATED`)

#### Services (Server)

| Service | Type | Description |
|---------|------|-------------|
| `/manipulation/stow` | `Trigger` | Retract arm to safe position |

#### Action Client / Service Client

| Target | Type | Description |
|--------|------|-------------|
| `/stretch_controller/follow_joint_trajectory` | `FollowJointTrajectory` (action) | Control individual joints |
| `/switch_to_position_mode` | `Trigger` (service) | Switch to position mode |
| `/switch_to_navigation_mode` | `Trigger` (service) | Switch to navigation mode |
| `/stow_the_robot` | `Trigger` (service) | stretch_driver stow |

#### Controlled Joints

| Joint Name | Description | Used in pick | Used in place |
|------------|-------------|--------------|---------------|
| `joint_lift` | Vertical lift (m) | ✓ | ✓ |
| `wrist_extension` | Arm extension (m) | ✓ | ✓ |
| `joint_wrist_yaw/pitch/roll` | Wrist joints (rad) | ✓ | |
| `joint_gripper_finger_left` | Gripper (rad, +open/−close) | ✓ | ✓ |
| `joint_head_pan/tilt` | Head joints (rad) | ✓ | |

---

### 6. TaskPlannerNode

**File**: `task_planner_node.py`
**Function**: Generates pick-and-place task plans based on detected objects and goal assignments.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `planner_backend` | string | `greedy` | Planning backend: `greedy` or `vlm` |
| `tasks_yaml` | string | `""` | Task assignment file |
| `regions_yaml` | string | `""` | Region definition file |
| `map_frame` | string | `map` | Map coordinate frame |
| `base_frame` | string | `base_link` | Robot base frame |
| `vlm_model` | string | `gemini-2.5-flash` | VLM model name |
| `vlm_base_url` | string | `https://...` | VLM API URL |
| `vlm_api_key_env` | string | `GEMINI_API_KEY` | API key environment variable name |
| `vlm_use_image` | bool | `True` | Whether to send image to VLM |
| `vlm_max_retries` | int | `2` | Maximum VLM call retries |

#### Subscriptions

| Topic | Type | Description |
|-------|------|-------------|
| `/detected_objects` | `MarkerArray` | From ObjectDetectorNode |
| `/detector/debug_image` | `Image` | Debug image (for VLM) |

#### Publishers

| Topic | Type | Description |
|-------|------|-------------|
| `/planner/plan_visualization` | `MarkerArray` | Plan visualization: pick→place lines + labels |

#### Services (Server)

| Service | Type | Description |
|---------|------|-------------|
| `/planner/compute` | `Trigger` | Execute planning. Response.message contains plan summary |

When called:
1. Gets robot pose from TF
2. Reads `latest_objects` (from /detected_objects subscription)
3. Determines each object's current region (point_in_polygon)
4. Calls planner backend to generate `List[PickPlaceTask]`
5. Publishes visualization, returns result

#### Programmatic API

```python
get_plan() -> List[PickPlaceTask]
```
Returns the result of the most recent compute call.

#### TF Dependencies
- Reads `map` → `base_link` transform

---

### 7. TaskExecutorNode

**File**: `task_executor_node.py`
**Function**: Top-level state machine orchestrating the entire explore → detect → plan → execute pipeline.

#### State Machine

```
IDLE → HEAD_SCAN → EXPLORE → WAIT_OBJECTS → PLAN →
  ┌→ NAV_TO_PICK → PICK → NAV_TO_PLACE → PLACE ─┐
  └──────────── loop until all tasks done ◀────────┘
                                        ↓
                                       DONE
```

- **IDLE**: Waiting for /executor/start call
- **HEAD_SCAN**: Calls stow + start_head_scan to prepare for scanning
- **EXPLORE**: Starts exploration, waits until enough objects detected or timeout
- **WAIT_OBJECTS**: After exploration ends, waits `wait_after_explore_s` seconds for detections to stabilize
- **PLAN**: Calls planner to generate plan
- **NAV_TO_PICK**: Navigates to the object's approach pose (standoff)
- **PICK**: Executes /manipulation/pick action
- **NAV_TO_PLACE**: Navigates to target region's place_anchor
- **PLACE**: Executes /manipulation/place action
- **DONE**: All tasks complete, writes metrics
- **FAILED**: Error or abort

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `planner_backend` | string | `greedy` | Planning backend |
| `tasks_yaml` | string | `""` | Task assignment file |
| `regions_yaml` | string | `""` | Region definition file |
| `map_frame` | string | `map` | Map coordinate frame |
| `base_frame` | string | `base_link` | Robot base frame |
| `explore_timeout_s` | float | `180.0` | Exploration timeout (s) |
| `min_objects_required` | int | `3` | Minimum detected objects before stopping exploration |
| `wait_after_explore_s` | float | `6.0` | Wait time after exploration (s) |
| `pick_standoff_m` | float | `0.55` | Pick approach distance (m) |
| `metrics_path` | string | `/tmp/rearrangement_metrics.json` | Metrics output file |
| `start_on_launch` | bool | `False` | Whether to start automatically on launch |

#### Subscriptions

| Topic | Type | Description |
|-------|------|-------------|
| `/detected_objects` | `MarkerArray` | From ObjectDetectorNode |

#### Publishers

| Topic | Type | Description |
|-------|------|-------------|
| `/executor/state` | `String` | Current state name (published on each transition) |

#### Services (Server)

| Service | Type | Description |
|---------|------|-------------|
| `/executor/start` | `Trigger` | Start the state machine (must be called in IDLE) |
| `/executor/abort` | `Trigger` | Abort, state → FAILED |

#### Action Clients

| Action | Type | Description |
|--------|------|-------------|
| `/navigate_to_pose` | `NavigateToPose` | Navigate to pick/place locations |
| `/manipulation/pick` | `FollowJointTrajectory` | Execute pick |
| `/manipulation/place` | `FollowJointTrajectory` | Execute place |

#### Service Clients

| Service | Type | Description |
|---------|------|-------------|
| `/exploration/start` | `Trigger` | Start exploration |
| `/exploration/stop` | `Trigger` | Stop exploration |
| `/head/start_scan` | `Trigger` | Start head scanning |
| `/head/stop_scan` | `Trigger` | Stop head scanning |
| `/manipulation/stow` | `Trigger` | Retract arm |

#### Metrics Output

`/tmp/rearrangement_metrics.json`:
```json
{
  "backend": "greedy",
  "t_start": 1712345678.0,
  "t_explore_done": 1712345770.3,
  "t_all_objects_seen": 1712345776.3,
  "first_detection": {"blue_bottle": 23.5, "red_box": 41.2, "yellow_cup": 55.9},
  "pick_attempts": 3, "pick_successes": 3,
  "place_attempts": 3, "place_successes": 3,
  "task_results": [
    {"object": "blue_bottle", "target": "C", "success": true, "ts": 1712345800.0}
  ],
  "t_end": 1712345900.0,
  "success": true
}
```

---

### 8. FakeSimNode

**File**: `sim/fake_sim_node.py`
**Function**: Lightweight simulator replacing Gazebo/stretch_driver/Nav2/SLAM for fast end-to-end testing.

#### Simulated Capabilities

| Capability | Description |
|------------|-------------|
| TF | `map→odom→base_link` (dynamic), `map→camera_color_optical_frame` (static) |
| `/map` | 8m×8m room with boundary walls, TRANSIENT_LOCAL |
| `/odom` | Odometry |
| `/joint_states` | Fake joint states |
| `/camera/*` | Bird's-eye virtual camera, renders colored circles for objects |
| `/navigate_to_pose` | Interpolates robot toward goal, ~1.5 m/s |
| `/manipulation/pick` | Picks nearest object (distance < 1.5m) |
| `/manipulation/place` | Drops object 0.4m in front of robot |
| Various Trigger services | `/head/*`, `/switch_to_*`, `/stow_the_robot` — all no-ops |

#### Initial Scene

| Object | Initial Position | Goal Region (tasks.yaml) |
|--------|-----------------|--------------------------|
| blue_bottle | (1.5, 1.5) Region A | C |
| red_box | (1.5, -1.5) Region C | A |
| yellow_cup | (2.5, 0.5) Region A | D |

---

## Planner Backends

### GreedyPlanner

**File**: `planners/greedy.py`

Greedy nearest-neighbor: at each step selects the task with minimum `(distance to object + object to placement)`. Deterministic, no external API dependency.

### VLMPlanner

**File**: `planners/vlm.py`

Calls Gemini API (via OpenAI SDK), sends a structured scene description + optional debug image, and has the VLM generate a JSON-format ordered plan. Automatically falls back to GreedyPlanner on API / JSON failure.

### Shared Data Structures

```python
@dataclass
class DetectedObject:
    label: str                           # object label
    pose_xy: Tuple[float, float]         # map coordinates
    current_region: Optional[str]        # current region
    z: float = 0.0

@dataclass
class RegionInfo:
    name: str
    polygon: List[Tuple[float, float]]   # polygon vertices
    place_anchor: Tuple[float, float, float]  # placement pose (x, y, yaw)

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
    goal_assignment: Dict[str, str]      # label → region
    robot_xy: Tuple[float, float]
    context_image_bgr: Optional[ndarray] = None
```

---

## Utility Libraries

### frontier_utils.py

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `extract_frontiers(grid, min_cluster_size)` | OccupancyGrid, int | List[Frontier] | Extract frontier clusters |
| `score_frontier(frontier, robot_xy, α, β)` | Frontier, (x,y), float, float | float | α×dist − β×size |
| `grid_to_world(grid, i, j)` | OccupancyGrid, int, int | (x, y) | Grid → world coordinates |
| `world_to_grid(grid, x, y)` | OccupancyGrid, float, float | (i, j) | World → grid coordinates |

### color_segmentation.py

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `segment_color(bgr, spec)` | ndarray, ColorSpec | Detection2D or None | Single-color segmentation |
| `segment_all(bgr, specs)` | ndarray, List[ColorSpec] | List[Detection2D] | Multi-color segmentation |
| `pixel_to_camera(depth, u, v, fx, fy, cx, cy)` | ndarray, ... | (x,y,z) or None | Pixel → camera 3D |
| `annotate(bgr, dets)` | ndarray, List[Detection2D] | ndarray | Draw annotation boxes |
| `load_color_specs(cfg)` | dict | List[ColorSpec] | Load HSV config |

### transform_utils.py

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `robot_pose_in_map(node, tf_buffer, ...)` | Node, Buffer | PoseStamped or None | Get robot pose in map |
| `transform_point_to_frame(tf_buffer, xyz, src, tgt)` | ... | (x,y,z) or None | Transform point to target frame |
| `yaw_to_quat(yaw)` | float | (x,y,z,w) | Yaw → quaternion |
| `quat_to_yaw(x,y,z,w)` | float×4 | float | Quaternion → yaw |

---

## Configuration Files

| File | Description |
|------|-------------|
| `config/objects.yaml` | HSV color ranges. H[0,180], S/V[0,255]. Red uses two ranges. |
| `config/regions.yaml` | Region polygons (CCW, map frame meters). Each region has polygon + place_anchor. |
| `config/tasks.yaml` | `assignments: {label: region}` goal assignment. |
| `config/slam_params.yaml` | SLAM Toolbox parameters |
| `config/nav2_params.yaml` | Nav2 parameters |

---

## Integration Workflow

### Simulation Mode (Quick Validation)

```bash
# Build
cd your_ws && colcon build --symlink-install && source install/setup.bash

# One-command launch (fake_sim + detector + planner + executor + exploration)
ros2 launch exploration_rearrangement sim.launch.py start_on_launch:=true

# Monitor state
ros2 topic echo /executor/state
# Check results
cat /tmp/rearrangement_metrics.json
```

### Real Robot Mode

```bash
# 1. stretch_driver
ros2 launch stretch_core stretch_driver.launch.py

# 2. SLAM + Nav2
ros2 launch exploration_rearrangement mapping.launch.py   # or bringup.launch.py

# 3. Our nodes
ros2 launch exploration_rearrangement rearrangement.launch.py planner_backend:=greedy

# 4. Start the task
ros2 service call /executor/start std_srvs/srv/Trigger
```

### Step-by-Step Debugging

Recommended order for incremental sub-function verification:

| Order | Test Script | Validates | Dependencies |
|-------|-------------|-----------|--------------|
| 1 | `test_head_scan.py` | Head scanning | stretch_driver or fake_sim |
| 2 | `test_navigation.py` | Nav2 navigation | stretch_driver+Nav2 or fake_sim |
| 3 | `test_object_detector.py` | Object detection | Camera + TF |
| 4 | `test_region_manager.py` | Region management | No extra dependencies |
| 5 | `test_manipulation.py` | Pick/place | stretch_driver or fake_sim |
| 6 | `test_exploration.py` | Autonomous exploration | SLAM+Nav2 or fake_sim + exploration_node |
| 7 | `test_task_planner.py` | Task planning | detector + planner nodes |
| 8 | `test_task_executor.py` | State machine | All nodes |
| 9 | `test_e2e_sim.py` | End-to-end | sim.launch.py |

---

## Testing Guide

### Run All pytest Unit Tests (No ROS Runtime Required)

```bash
cd src/exploration_rearrangement
python -m pytest test/ -v
```

### Run Robot Integration Test Scripts

```bash
# 1. Start simulation backend
ros2 run exploration_rearrangement fake_sim_node

# 2. Run sub-function tests (in another terminal)
python3 test/test_head_scan.py
python3 test/test_navigation.py
python3 test/test_manipulation.py
python3 test/test_object_detector.py    # also needs detector node running
python3 test/test_region_manager.py     # also needs region_manager node running
python3 test/test_exploration.py        # also needs exploration_node running
python3 test/test_task_planner.py       # needs detector + planner nodes
python3 test/test_task_executor.py      # needs full simulation stack

# 3. End-to-end test
ros2 launch exploration_rearrangement sim.launch.py start_on_launch:=true
# In another terminal:
python3 test/test_e2e_sim.py
```
