# Visual Grasp Runbook

IK target-following + D405 fine detection → visually-guided pick.

```
  D405 RGB-D (gripper)
        │
        ▼
  fine_object_detector_node          target_following.py
        │                                    │
        │  /fine_detector/objects             │  subscribes
        │  (Detection3DArray, base_link)  ───►│
        │                                    │
        ▲                                    │  publishes
        │  /fine_detector/activate (Bool)  ◄──│
        │                                    │
        └────────────────────────────────────┘

  Loop:
    1. fine detector publishes object 3D pose in base_link
    2. target_following reads it, computes IK waypoint (δ step)
    3. move_to_configuration → arm moves one increment
    4. repeat until gripper–object distance < δ
    5. pick(): close gripper → retract arm → deactivate detector
```

---

## 0. Prerequisites

| What | How to verify |
|------|---------------|
| Stretch driver running | `ros2 topic list \| grep /stretch/joint_states` |
| D405 gripper camera publishing | `ros2 topic hz /gripper_camera/color/image_raw` |
| TF tree includes `gripper_camera_color_optical_frame → base_link` | `ros2 run tf2_tools view_frames` |
| Package built | `ros2 pkg executables exploration_rearrangement \| grep fine_object_detector_node` |

Start the Stretch driver first (all terminals need this):

```bash
# Terminal 0 — robot driver (if not already running)
ros2 launch stretch_core stretch_driver.launch.py
```

---

## 1. Build

```bash
cd ~/Homework/16-762/project/stretch_rearrangement_ws
colcon build --packages-select exploration_rearrangement --symlink-install
source install/setup.bash
```

---

## 2. Start the fine detector

The fine detector sits idle until it receives `Bool(True)` on
`/fine_detector/activate` — `target_following.py` sends this automatically,
so you only need to start the node.

```bash
# Terminal 1 — fine object detector (robot mode)
source install/setup.bash
ros2 run exploration_rearrangement fine_object_detector_node --ros-args \
    -p mode:=robot \
    -p model_path:=yoloe-11s-seg.pt \
    -p objects_yaml:=$(ros2 pkg prefix --share exploration_rearrangement)/config/objects.yaml
```

Confirm it starts without errors. It will print the class list and then sit
idle (`fine detector: ready …`).

### Debug mode (bench, single D435i, no robot)

If you don't have the Stretch but want to test detection logic with a
standalone D435i:

```bash
# start a D435i first
ros2 launch realsense2_camera rs_launch.py \
    enable_color:=true enable_depth:=true align_depth.enable:=true

# then run the fine detector in debug mode
ros2 run exploration_rearrangement fine_object_detector_node --ros-args \
    -p mode:=debug \
    -p model_path:=yoloe-11s-seg.pt
```

---

## 3. Start target following (with pick)

`target_following.py` is a HelloNode script in the `manipulation/`
directory. It needs `ik_ros_utils` and `ikpy` on its Python path.

### 3a. Set the target object name

Before running, edit `target_following.py` line 36 to set which object to
track. The name must match one of the classes in `config/objects.yaml`:

```python
self.target_object_name = 'white_bottle'   # or 'green_cup', 'blue_cup'
```

### 3b. Run

```bash
# Terminal 2 — target following + pick
source install/setup.bash
cd ~/Homework/16-762/project/stretch_rearrangement_ws/src/exploration_rearrangement/exploration_rearrangement/manipulation
python target_following.py
```

On launch it will:
1. `stow_the_robot()`
2. `move_to_ready_pose()` (READY\_POSE\_P2: lift 0.8 m, arm retracted,
   wrist neutral, head looking left-down)
3. Publish `Bool(True)` → `/fine_detector/activate` (wakes the fine
   detector)
4. Subscribe to `/fine_detector/objects` and start the IK tracking loop

### What to expect

```
At Ready Pose
Fine detector activated — waiting for detections
```

Once the D405 sees the target object, the console will print IK solutions
and distance updates every callback:

```
Solution Found
IK Config
     Base Rotation: 0.012
     Base Translation: 0.003
     Lift 0.78
     Arm 0.15
     Gripper Yaw: 0.01
     Gripper Pitch: -0.09
     Gripper Roll: 0.00
Distance to goal after move: 0.083 m
```

When the distance drops below `self.delta` (0.06 m):

```
Within delta threshold — picking object
Pick complete
```

The gripper closes, the arm retracts, and the fine detector is deactivated.

---

## 4. Monitor / debug topics

Open additional terminals to inspect what's happening:

```bash
# see the fine detector's 2D overlay (confirm YOLOE is detecting)
ros2 run rqt_image_view rqt_image_view /fine_detector/debug_image

# echo the 3D detections (xyz in base_link)
ros2 topic echo /fine_detector/objects

# check detection rate
ros2 topic hz /fine_detector/objects

# manually activate / deactivate the fine detector
ros2 topic pub --once /fine_detector/activate std_msgs/Bool "{data: true}"
ros2 topic pub --once /fine_detector/activate std_msgs/Bool "{data: false}"
```

---

## 5. Tunable parameters

All tuned in `target_following.py` `__init__`:

| Variable | Default | What it does |
|----------|---------|--------------|
| `self.delta` | `0.06` | IK step size (m) **and** "close enough to pick" threshold. Smaller = more precise but slower convergence. |
| `self.shift_x` | `0.04` | Grasp offset in base\_link x (forward). Compensates for the gap between the camera optical center and the physical grasp point. |
| `self.shift_y` | `-0.03` | Grasp offset in base\_link y (left). |
| `self.shift_z` | `0.03` | Grasp offset in base\_link z (up). Use `0.03` for bottles, `0.01` for cups. |
| `self.target_object_name` | `None` | Which class from `objects.yaml` to track. **Must be set before running.** |

Fine detector parameters are set via `--ros-args -p`:

| Param | Default | Notes |
|-------|---------|-------|
| `conf_threshold` | `0.30` | Raise if you get false positives at close range |
| `ema_alpha` | `1.0` | `1.0` = no smoothing; lower (e.g. `0.3`) for temporal smoothing |
| `clear_state_on_activate` | `true` | Forgets stale poses every time the detector is re-activated |

---

## 6. Troubleshooting

### "No detections" — script waits forever

| Check | Fix |
|-------|-----|
| Fine detector not running or crashed | Restart Terminal 1 |
| D405 not publishing | `ros2 topic hz /gripper_camera/color/image_raw` — if 0, check `stretch_driver` or USB |
| Target not in camera FOV | The ready pose points the arm forward; physically place the object in front of the gripper |
| Wrong `target_object_name` | Must exactly match a `name` in `objects.yaml` (`white_bottle`, `green_cup`, `blue_cup`) |
| Fine detector not activated | Check `ros2 topic echo /fine_detector/activate` — should show `data: true` |

### IK fails ("IKPy did not find a valid solution")

- The object is too far or at an unreachable angle. Move the robot closer
  to the object before starting.
- Reduce `self.delta` to take smaller steps (avoids large jumps the solver
  can't handle).
- Check the IK joint limits in `ik_ros_utils.py` — the virtual base
  rotation/translation bounds can be narrowed to prevent the solver from
  relying too heavily on base movement.

### Robot base moves too much

Per the code comments — three options:

1. **Reduce `self.delta`** so each step is smaller.
2. **Gate base motion** manually: in `ik_ros_utils.move_to_configuration`,
   only execute `rotate_mobile_base` / `translate_mobile_base` when the
   magnitude exceeds a threshold.
3. **Tighten IK base limits** in `ik_ros_utils.py` (the bounds on
   `joint_base_rotation` and `joint_base_translation` in the URDF or the
   chain).

### Gripper closes too early / too late

Adjust `self.delta`. Larger = triggers pick from farther away (less
precise). Smaller = needs to get very close before picking (may overshoot).

Also check `shift_x/y/z` — these offsets account for the physical distance
between the camera sensor and the actual grasp center. Wrong values mean
the gripper aims at the wrong spot.

---

## 7. Full terminal summary

| Terminal | Command |
|----------|---------|
| 0 | `ros2 launch stretch_core stretch_driver.launch.py` |
| 1 | `ros2 run exploration_rearrangement fine_object_detector_node --ros-args -p mode:=robot -p model_path:=yoloe-11s-seg.pt -p objects_yaml:=...` |
| 2 | `cd .../manipulation && python target_following.py` |
| 3 (optional) | `ros2 run rqt_image_view rqt_image_view /fine_detector/debug_image` |
