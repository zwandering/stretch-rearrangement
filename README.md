# Stretch 3 — Object Rearrangement

16-762 team project. The operator pre-builds a map and snapshots object
positions, manually annotates regions, then issues natural-language
rearrangement instructions. A Gemini-based VLM planner turns each
instruction into a pick/place sequence; the upstream `NavigationModule`
coordinator drives the Stretch with `stretch_nav2`, and the manipulation
node executes pick/place at each handoff.

## Architecture

```
/instruction/text ─► task_planner_node ─► /planner/pick_place_plan
                       │ (VLM)                  (PoseArray of pick0,
                       │                         place0, pick1, place1, …)
                       ▼
                    task_executor_node ──► /nav/goals + /nav/control_flag
                       │                         │
                       │                         ▼
                       │                   navigation_node ──► stretch_nav2 (Nav2 + AMCL)
                       │                         │
                       │  /nav/arrived_flag ◄────┘  (within 1 m of goal)
                       ▼
                    /manipulation/pick or /manipulation/place
```

## Three-stage workflow

### Stage 1 — Mapping + object snapshot (one-time, offline)

Stage 1 runs as **three separate terminals** so each subsystem can be
restarted independently while teleop-mapping. Do NOT bundle them into
one launch file, and do NOT separately start `stretch_core
stretch_driver.launch.py` — `stretch_nav2 offline_mapping.launch.py`
already brings the driver up.

```bash
# Terminal 1 — SLAM + Stretch driver (driver is included by this launch)
ros2 launch stretch_nav2 offline_mapping.launch.py

# Terminal 2 — RealSense camera(s) onboard
ros2 launch stretch_core d435i_low_resolution.launch.py

# Terminal 3 — YOLOE detector (latches each object's map-frame position)
ros2 run exploration_rearrangement set_up_yolo_e --format openvino
ros2 launch exploration_rearrangement mapping.launch.py objects_snapshot:=$HOME/maps/myroom_objects.yaml yolo_model:=$PWD/yoloe-11s-seg_openvino_model
```

Then teleop the robot. When the SLAM map looks complete and every
target object has been seen by the detector at least once:

```bash
ros2 service call /detector/snapshot std_srvs/srv/Trigger
ros2 run nav2_map_server map_saver_cli -f $HOME/maps/myroom
```

Outputs:
- `myroom.{pgm,yaml}` — saved occupancy map.
- `myroom_objects.yaml` — `{label: {x, y, z, conf}}` of every object the
  detector latched onto in the `map` frame.

### Stage 2 — Region annotation (manual)

Edit `src/exploration_rearrangement/config/regions.yaml`. Each entry is
a CCW polygon (vertices in `map` frame) plus a `place_anchor: [x, y, yaw]`
that the planner uses as the place pose. Cross-check vertex coordinates
by opening the saved map in RViz with `map_server` and using *Publish
Point* to read off `(x, y)` for each corner.

### Stage 3 — Run

```bash
# Terminal 1:
export GEMINI_API_KEY="xxx"
ros2 launch exploration_rearrangement bringup.launch.py map:=$HOME/maps/myroom.yaml objects_snapshot:=$HOME/maps/myroom_objects.yaml
# In RViz: click "2D Pose Estimate" to localize.

# Terminal 2:
ros2 service call /executor/start std_srvs/srv/Trigger
ros2 topic pub --once /instruction/text std_msgs/msg/String '{data: "put the orange cup to region B, put the yellow cup to region A."}'
(Optional) ros2 topic pub --once /nav/control_flag std_msgs/msg/String '{data: "proceed"}'
```

The detector keeps running during stage 3, so any object that's been
moved between snapshot and execution gets corrected when it re-enters
the camera FOV.

## Topic contract

| Topic                      | Type                                        | Direction                  |
| -------------------------- | ------------------------------------------- | -------------------------- |
| `/instruction/text`        | `std_msgs/String`                           | operator → planner         |
| `/planner/pick_place_plan` | `geometry_msgs/PoseArray`                   | planner → executor         |
| `/nav/goals`               | `geometry_msgs/PoseArray`                   | executor → nav coordinator |
| `/nav/control_flag`        | `std_msgs/String` (`"proceed"` / `"stop"`)  | executor → nav coordinator |
| `/nav/arrived_flag`        | `std_msgs/String` (`"arrived"`)             | nav coordinator → executor |
| `/manipulation/pick`       | `control_msgs/FollowJointTrajectory` action | executor → manipulation    |
| `/manipulation/place`      | `control_msgs/FollowJointTrajectory` action | executor → manipulation    |
| `/detector/objects`        | `vision_msgs/Detection3DArray`              | detector → planner         |
| `/detector/snapshot`       | `std_srvs/Trigger` service                  | operator → detector        |
| `/executor/start`          | `std_srvs/Trigger` service                  | operator → executor        |

The nav coordinator stops Nav2 when the base is within
`STOP_DISTANCE_M` metres of each goal (currently `0.15`) and publishes
`"arrived"`. Tune by editing the constant at the top of
`navigation_node.py`.

## Building

```bash
cd stretch_rearrangement_ws
colcon build --symlink-install --packages-select exploration_rearrangement
source install/setup.bash
```

Python deps (install in the same env that runs ROS 2):

```bash
pip install openai>=1.30.0 ultralytics>=8.3.0 numpy opencv-python pyyaml
```

External ROS 2 packages: `stretch_nav2`, `nav2_bringup`, `nav2_map_server`
(install via `apt install ros-${ROS_DISTRO}-...`).

## Further reading

- [`src/exploration_rearrangement/docs/VISUAL_GRASP_RUNBOOK.md`](src/exploration_rearrangement/docs/VISUAL_GRASP_RUNBOOK.md) — two-stage visual grasp: head-camera coarse approach + gripper-camera fine grasp.
- [`src/exploration_rearrangement/docs/navigation_module.md`](src/exploration_rearrangement/docs/navigation_module.md) — `NavigationModule` topic protocol (`/nav/goals`, `/nav/control_flag`, `/nav/arrived_flag`).
- [`src/exploration_rearrangement/docs/README.md`](src/exploration_rearrangement/docs/README.md) — index for the two docs above.

## Credits

The `navigation_node.py` and the sample `maps/asangium.{yaml,pgm}` are
vendored from
[AuWeerachai/MobileManipulation](https://github.com/AuWeerachai/MobileManipulation),
commit `e4b53e52`.
