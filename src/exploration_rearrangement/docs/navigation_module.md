# NavigationModule

My slice of the three-person final project. Given map-frame target points from
the perception teammate and "proceed"/"stop" control flags from the
manipulation teammate, this module drives the Stretch with Nav2 and hands off
when the robot is within 1 m of each goal.

## Files

- `navigation_node.py` - the coordinator node.
- `asangium.yaml`, `asangium.pgm` - saved map, copied from Lab4. Feed this to
  `stretch_nav2`'s launch file via `map:=...`.

## ROS interface

Three topics. Message types are standard, so no custom ROS package is needed.

| Direction          | Topic                | Type                        | Payload                           |
| ------------------ | -------------------- | --------------------------- | --------------------------------- |
| friend1 -> me      | `/nav/goals`         | `geometry_msgs/PoseArray`   | Up to 4 target poses in `map`. Only x/y are used. |
| friend2 -> me      | `/nav/control_flag`  | `std_msgs/String`           | `"proceed"` or `"stop"`           |
| me -> friend2      | `/nav/arrived_flag`  | `std_msgs/String`           | `"arrived"`                       |

### Friend 1 (perception) - what to publish

Publish a single `geometry_msgs/PoseArray` whose `poses[]` list contains each
target object's map-frame position. Orientation is ignored; the navigation
node computes a yaw that faces each goal at dispatch time.

Example (Python):

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray

class GoalsPublisher(Node):
    def __init__(self):
        super().__init__("perception_goals_publisher")
        self.pub = self.create_publisher(PoseArray, "/nav/goals", 10)
        self.timer = self.create_timer(1.0, self.send_once)

    def send_once(self):
        msg = PoseArray()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for (x, y, z) in [(1.0, 2.0, 0.0), (3.5, -1.2, 0.0),
                          (5.0,  0.5, 0.0), (2.0, -4.0, 0.0)]:
            p = Pose()
            p.position.x, p.position.y, p.position.z = x, y, z
            p.orientation.w = 1.0
            msg.poses.append(p)
        self.pub.publish(msg)
        self.destroy_timer(self.timer)  # publish once and stop
```

New goals overwrite the current array; if this happens mid-navigation the
current Nav2 task is cancelled and the coordinator reverts to `AWAITING_GO`
on index 0 of the new list.

### Friend 2 (manipulation) - what to publish

Publish `std_msgs/String` on `/nav/control_flag`:

- `"proceed"` - start navigating to the next goal. Publish this:
  - once after the first PoseArray is received (starts goal 0);
  - again each time your IK / grasp / place finishes (advances to goal i+1).
- `"stop"` - cancel any in-flight Nav2 task; next `"proceed"` retries the
  same goal index.

Subscribe to `/nav/arrived_flag`. When you see `"arrived"`, the robot is
within 1 m of the current goal, Nav2 has been cancelled, and it is safe to
run IK / manipulate. When you are done, publish `"proceed"` to continue.

## State machine

```
IDLE ── PoseArray ──> AWAITING_GO ── "proceed" ──> NAVIGATING
                                                      │
                                                      │ distance ≤ 1.0 m
                                                      │  (or Nav2 finishes early)
                                                      ▼
                          HANDOFF ◄── "proceed" ── (cancel Nav2 + publish "arrived",
                                                    goal_index += 1)
                               │
                               │ if goal_index == len(goals)
                               ▼
                             IDLE
```

`"stop"` from any state cancels Nav2 (if running) and drops us into
`AWAITING_GO` on the current index, so the next `"proceed"` retries.

## Assumptions

- Nav2 is already up against a saved map (the included `asangium.yaml` is the
  Lab4 map; substitute your own via the launch `map:=` arg).
- AMCL has been given an initial pose, e.g. by clicking "2D Pose Estimate" in
  RViz after the map loads. The coordinator does not call `setInitialPose`
  itself so perception can re-use its own localization if needed.
- The TF tree publishes `map -> base_link` once AMCL is localized.
- The stretch driver is running with `broadcast_odom_tf:=true` (otherwise
  Nav2 will never see the `odom` frame and everything will stall).

## Running it

Three terminals. Adjust the `map:=` path if your map lives elsewhere.

Terminal 1 - robot driver:
```bash
ros2 launch stretch_core stretch_driver.launch.py broadcast_odom_tf:=true
```

Terminal 2 - Nav2 + localization against the saved map:
```bash
ros2 launch stretch_nav2 navigation.launch.py \
    map:=$(pwd)/MobileManipulation/NavigationModule/asangium.yaml
```

Terminal 3 - the coordinator node:
```bash
python3 MobileManipulation/NavigationModule/navigation_node.py
```

Then either open RViz and click "2D Pose Estimate" to localize, or let your
teammates' setup handle it. Once the node prints
`NavigationCoordinator ready`, publish goals + control flags to exercise it.

## Quick manual test without friends

You can drive the whole flow from the command line:

```bash
# Publish 2 goals
ros2 topic pub --once /nav/goals geometry_msgs/msg/PoseArray \
  '{header: {frame_id: "map"}, poses: [
     {position: {x: 2.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}},
     {position: {x: 3.5, y: -1.5, z: 0.0}, orientation: {w: 1.0}}
   ]}'

# Listen for my arrival flag in another terminal
ros2 topic echo /nav/arrived_flag

# Start goal 0
ros2 topic pub --once /nav/control_flag std_msgs/msg/String '{data: "proceed"}'

# ...wait for "arrived"...

# Advance to goal 1
ros2 topic pub --once /nav/control_flag std_msgs/msg/String '{data: "proceed"}'
```

## Tuning

Edit the constants at the top of `navigation_node.py`:

- `STOP_DISTANCE_M` - how close to the goal before handing off (default `1.0`).
- `DISTANCE_CHECK_HZ` - how often we poll TF for distance-to-goal (default 5).
- `GOALS_TOPIC` / `CONTROL_TOPIC` / `ARRIVED_TOPIC` - topic names.
- `MAP_FRAME` / `BASE_FRAME` - TF frame names.

No external libraries are needed beyond what Labs 1-4 already use
(`rclpy`, `tf2_ros`, `geometry_msgs`, `std_msgs`, `stretch_nav2`).
