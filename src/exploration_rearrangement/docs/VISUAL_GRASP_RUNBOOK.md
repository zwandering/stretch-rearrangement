# Visual Grasp Runbook

Two-stage visually-guided pick: coarse approach with head camera, then fine
grasp with gripper camera.

```
 ┌─────────────────── Stage 1: COARSE APPROACH ───────────────────┐
 │                                                                 │
 │  D435i RGB-D (head)                                             │
 │        │                                                        │
 │        ▼                                                        │
 │  object_detector_node         visual_servo_arm_node             │
 │        │                              │                         │
 │        │  /detector/objects            │  subscribes             │
 │        │  (Detection3DArray, map) ────►│                         │
 │        │                              │                         │
 │        └──────────────────────────────┘                         │
 │                                                                 │
 │  Loop:                                                          │
 │    1. head detector publishes object 3D pose (map frame)        │
 │    2. visual_servo_arm transforms to base_link                  │
 │    3. IK target = object_pos + offset in gripper frame          │
 │       (default: 10 cm in front of gripper)                      │
 │    4. δ-step IK until gripper is near the offset goal           │
 │    5. reached = True → stop                                     │
 │                                                                 │
 │  Result: gripper is positioned near the object, object is       │
 │          now in the D405 gripper camera's field of view.        │
 └─────────────────────────────────────────────────────────────────┘

                            ↓ operator runs Stage 2

 ┌─────────────────── Stage 2: FINE GRASP ────────────────────────┐
 │                                                                 │
 │  D405 RGB-D (gripper)                                           │
 │        │                                                        │
 │        ▼                                                        │
 │  fine_object_detector_node        visual_grasp_node             │
 │        │                                  │                     │
 │        │  /fine_detector/objects           │  subscribes         │
 │        │  (Detection3DArray, base_link) ──►│                     │
 │        │                                  │                     │
 │        ▲                                  │  publishes           │
 │        │  /fine_detector/activate (Bool) ◄─│                     │
 │        │                                  │                     │
 │        └──────────────────────────────────┘                     │
 │                                                                 │
 │  Loop:                                                          │
 │    1. fine detector publishes refined pose in base_link         │
 │    2. visual_grasp IK-steps gripper toward object               │
 │    3. when distance < δ → close gripper, retract arm            │
 │    4. deactivate fine detector                                  │
 └─────────────────────────────────────────────────────────────────┘
```

---

## 0. Prerequisites

| What | How to verify |
|------|---------------|
| Stretch driver | `ros2 topic list \| grep /stretch/joint_states` |
| Head D435i camera | `ros2 topic hz /camera/color/image_raw` |
| Gripper D405 camera | `ros2 topic hz /gripper_camera/color/image_raw` |
| TF tree complete | `ros2 run tf2_tools view_frames` |
| Package built | `ros2 pkg executables exploration_rearrangement` |

### Start the robot

```bash
# Terminal 0a — robot driver
ros2 launch stretch_core stretch_driver.launch.py

# Terminal 0b — head D435i camera
ros2 launch stretch_core d435i_high_resolution.launch.py

# Terminal 0c — gripper D405 camera
ros2 launch stretch_core d405_basic.launch.py
```

If the dedicated camera launch files don't exist, use realsense2_camera
directly (see §8 Troubleshooting).

---

## 1. Build

```bash
cd ~/16762/jinyao/stretch-rearrangement
colcon build --packages-select exploration_rearrangement
source install/setup.bash
```

---

## 2. Start the head detector (used by Stage 1)

```bash
# Terminal 1 — head object detector (D435i, output in map frame)
source install/setup.bash
ros2 run exploration_rearrangement object_detector_node --ros-args \
    -p mode:=robot \
    -p model_path:=yoloe-11s-seg.pt \
    -p objects_yaml:=$(ros2 pkg prefix --share exploration_rearrangement)/config/objects.yaml
```

If you don't have SLAM running and don't need the map frame, override:

```bash
    -p output_frame:=camera_color_optical_frame
```

Confirm it detects objects:

```bash
ros2 topic echo /detector/objects --once
ros2 run rqt_image_view rqt_image_view /detector/debug_image
```

---

## 3. Start the fine detector (used by Stage 2)

```bash
# Terminal 2 — fine object detector (D405, output in base_link)
source install/setup.bash
ros2 run exploration_rearrangement fine_object_detector_node --ros-args \
    -p mode:=robot \
    -p model_path:=yoloe-11s-seg.pt \
    -p objects_yaml:=$(ros2 pkg prefix --share exploration_rearrangement)/config/objects.yaml
```

This node starts idle. It will be activated automatically by
`visual_grasp_node` in Stage 2.

---

## 4. Stage 1 — Coarse approach (`visual_servo_arm_node`)

Moves the gripper near the object using the head camera so the object
enters the D405's field of view.

### 4a. Set the target object name

Edit `visual_servo_arm_node.py` line 44:

```python
self.target_object_name = 'white_bottle'   # must match objects.yaml
```

Adjust the gripper-frame offset if needed (line 42, default 10 cm ahead):

```python
self.offset_in_gripper = np.array([0.1, 0.0, 0.0])
```

### 4b. Run

```bash
# Terminal 3 — coarse approach
source install/setup.bash
ros2 run exploration_rearrangement visual_servo_arm_node
```

### 4c. What to expect

```
At Ready Pose
Listening to /detector/objects for 'white_bottle'
```

The arm will IK-step toward the object. Each callback prints:

```
Solution Found
IK Config
     Base Rotation: ...
     ...
Distance to offset goal after move: 0.12 m
```

When the gripper reaches the offset goal:

```
Arm positioned — within delta of offset goal
```

The node stops. The object should now be visible in the D405 gripper camera.

**Verify before moving to Stage 2:**

```bash
ros2 run rqt_image_view rqt_image_view /gripper_camera/color/image_raw
# Confirm the target object is visible in the gripper camera feed.
```

---

## 5. Stage 2 — Fine grasp (`visual_grasp_node`)

Closes in on the object using the gripper camera and picks it up.

### 5a. Set the target object name

Edit `visual_grasp_node.py` line 45:

```python
self.target_object_name = 'white_bottle'   # same as Stage 1
```

### 5b. Run

```bash
# Terminal 4 — fine grasp (replaces Terminal 3 or run in a new one)
source install/setup.bash
ros2 run exploration_rearrangement visual_grasp_node
```

### 5c. What to expect

```
At Ready Pose
Fine detector activated — waiting for detections
```

Once the D405 detects the target:

```
Solution Found
IK Config ...
Distance to goal after move: 0.083 m
```

When within delta (0.06 m):

```
Within delta threshold — picking object
Pick complete
```

Gripper closes → arm retracts → fine detector deactivated.

---

## 6. Monitor / debug topics

```bash
# Stage 1 — head detector overlay
ros2 run rqt_image_view rqt_image_view /detector/debug_image

# Stage 2 — gripper detector overlay
ros2 run rqt_image_view rqt_image_view /fine_detector/debug_image

# 3D detections
ros2 topic echo /detector/objects          # head (map frame)
ros2 topic echo /fine_detector/objects      # gripper (base_link frame)

# Detection rates
ros2 topic hz /detector/objects
ros2 topic hz /fine_detector/objects

# RViz — 3D bbox visualization
# Add → MarkerArray → /detector/bboxes_3d_markers
# Add → MarkerArray → /fine_detector/bboxes_3d_markers

# Manually toggle fine detector
ros2 topic pub --once /fine_detector/activate std_msgs/Bool "{data: true}"
ros2 topic pub --once /fine_detector/activate std_msgs/Bool "{data: false}"
```

---

## 7. Tunable parameters

### `visual_servo_arm_node` (Stage 1)

| Variable | Default | Purpose |
|----------|---------|---------|
| `self.delta` | `0.03` | IK step size (m) and "reached" threshold |
| `self.offset_in_gripper` | `[0.1, 0, 0]` | Offset from object in gripper frame (m). `x=0.1` = 10 cm in front of gripper. Increase if the gripper bumps the object; decrease to get closer. |
| `self.target_object_name` | `None` | Which class from `objects.yaml` to track. **Must be set.** |

### `visual_grasp_node` (Stage 2)

| Variable | Default | Purpose |
|----------|---------|---------|
| `self.delta` | `0.06` | IK step size and "close enough to pick" threshold |
| `self.shift_x` | `0.04` | Grasp offset in base_link x. Compensates camera→grasp-point gap. |
| `self.shift_y` | `-0.03` | Grasp offset in base_link y |
| `self.shift_z` | `0.03` | Grasp offset in base_link z. `0.03` for bottles, `0.01` for cups. |
| `self.target_object_name` | `None` | Same class as Stage 1. **Must be set.** |

### Detector parameters (via `--ros-args -p`)

| Param | Default | Notes |
|-------|---------|-------|
| `conf_threshold` | `0.25` (head) / `0.30` (fine) | Raise if false positives |
| `ema_alpha` | `1.0` | `1.0` = no smoothing; lower for temporal filtering |
| `output_frame` | `map` (head) / `base_link` (fine) | Override for debug |

---

## 8. Troubleshooting

### Cameras not publishing

```bash
# Check what's available
ros2 topic list | grep camera

# If no camera topics, start manually:
# Head D435i
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=camera \
    enable_color:=true enable_depth:=true align_depth.enable:=true

# Gripper D405
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=gripper_camera \
    enable_color:=true enable_depth:=true align_depth.enable:=true
```

### Stage 1: no detections from head camera

| Check | Fix |
|-------|-----|
| `object_detector_node` not running | Start Terminal 1 |
| No camera feed | `ros2 topic hz /camera/color/image_raw` |
| Wrong `target_object_name` | Must match `objects.yaml` exactly |
| Object not in head camera FOV | Place object in front of the robot |
| TF `camera → map` missing | Run SLAM, or set `-p output_frame:=camera_color_optical_frame` |

### Stage 1: arm doesn't reach the object

- Object too far → move the robot closer first
- Reduce `self.delta` for smaller IK steps
- Check IK joint limits in `ik_ros_utils.py`

### Stage 2: object not in gripper camera after Stage 1

- Increase `self.offset_in_gripper[0]` in Stage 1 (approach from farther)
- Check gripper camera is actually running:
  `ros2 topic hz /gripper_camera/color/image_raw`

### Stage 2: IK fails or base moves too much

- Reduce `self.delta` for smaller steps
- Tighten base rotation/translation limits in `ik_ros_utils.py`

### Stage 2: gripper closes too early / misses object

- Adjust `shift_x/y/z` to compensate camera-to-grasp-point offset
- Reduce `self.delta` for more precise approach

---

## 9. Full terminal summary

| Terminal | Command | Stage |
|----------|---------|-------|
| 0a | `ros2 launch stretch_core stretch_driver.launch.py` | — |
| 0b | `ros2 launch stretch_core d435i_high_resolution.launch.py` | — |
| 0c | `ros2 launch stretch_core d405.launch.py` | — |
| 1 | `ros2 run exploration_rearrangement object_detector_node --ros-args -p mode:=robot -p model_path:=yoloe-11s-seg.pt -p objects_yaml:=...` | 1 |
| 2 | `ros2 run exploration_rearrangement fine_object_detector_node --ros-args -p mode:=robot -p model_path:=yoloe-11s-seg.pt -p objects_yaml:=...` | 2 |
| 3 | `ros2 run exploration_rearrangement visual_servo_arm_node` | 1 |
| 4 | `ros2 run exploration_rearrangement visual_grasp_node` | 2 |
| 5 (optional) | `ros2 run rqt_image_view rqt_image_view` | debug |
