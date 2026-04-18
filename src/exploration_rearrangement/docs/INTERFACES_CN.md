# Exploration Rearrangement — 节点接口与集成文档

## 目录

1. [系统架构总览](#系统架构总览)
2. [数据流图](#数据流图)
3. [节点详细接口](#节点详细接口)
   - [ExplorationNode](#1-explorationnode)
   - [ObjectDetectorNode](#2-objectdetectornode)
   - [RegionManagerNode](#3-regionmanagernode)
   - [HeadScanNode](#4-headscannode)
   - [ManipulationNode](#5-manipulationnode)
   - [TaskPlannerNode](#6-taskplannernode)
   - [TaskExecutorNode](#7-taskexecutornode)
   - [FakeSimNode](#8-fakesimnode)
4. [Planner 后端](#planner-后端)
5. [工具函数库](#工具函数库)
6. [配置文件](#配置文件)
7. [集成流程](#集成流程)
8. [测试指南](#测试指南)

---

## 系统架构总览

```
                     ┌─────────────────────┐
                     │   TaskExecutorNode   │   (状态机编排器)
                     │  /executor/start     │
                     │  /executor/abort     │
                     └──┬────┬────┬────┬───┘
                        │    │    │    │
           ┌────────────┘    │    │    └────────────┐
           ▼                 ▼    ▼                  ▼
  ┌────────────────┐ ┌──────────────┐  ┌──────────────────┐
  │ ExplorationNode│ │ HeadScanNode │  │ ManipulationNode │
  │ (前沿探索)      │ │ (头部扫描)    │  │ (pick/place)     │
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
  │ (HSV+深度检测)   │    │ (greedy/VLM)     │
  └─────────────────┘    └──────────────────┘
          │                       │
          ▼                       ▼
  ┌─────────────────┐    ┌──────────────────┐
  │ RegionManager   │    │ /planner/compute │
  │ (区域管理)       │    │ → 生成 plan      │
  └─────────────────┘    └──────────────────┘
```

## 数据流图

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

## 节点详细接口

### 1. ExplorationNode

**文件**: `exploration_node.py`
**功能**: 基于前沿 (frontier) 的自主探索. 从地图中提取已知/未知边界, 选择最优前沿发送 Nav2 导航目标.

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `map_frame` | string | `map` | 地图坐标系 |
| `base_frame` | string | `base_link` | 机器人基座坐标系 |
| `min_cluster_size` | int | `8` | 前沿最小聚类大小 (小于此值忽略) |
| `goal_tolerance_m` | float | `0.5` | 到达前沿的距离容差 (m) |
| `alpha_dist` | float | `1.0` | 距离权重 (越大越偏好近前沿) |
| `beta_info` | float | `0.05` | 信息量权重 (越大越偏好大前沿) |
| `replan_period_s` | float | `3.0` | 重新规划周期 (s) |
| `goal_timeout_s` | float | `60.0` | 单个目标超时 (s) |
| `enabled_on_start` | bool | `False` | 启动时是否自动开始探索 |

#### 订阅

| 话题 | 类型 | QoS | 说明 |
|------|------|-----|------|
| `/map` | `OccupancyGrid` | RELIABLE, TRANSIENT_LOCAL | SLAM 输出的地图 |

#### 发布

| 话题 | 类型 | 说明 |
|------|------|------|
| `/exploration/frontiers` | `MarkerArray` | 前沿可视化 (SPHERE_LIST) |
| `/exploration/status` | `String` | `"navigating"` / `"done"` |

#### 服务 (Server)

| 服务 | 类型 | 说明 |
|------|------|------|
| `/exploration/start` | `Trigger` | 启动探索 |
| `/exploration/stop` | `Trigger` | 停止探索 |

#### Action Client

| Action | 类型 | 说明 |
|--------|------|------|
| `/navigate_to_pose` | `NavigateToPose` | 发送前沿导航目标 |

#### TF 依赖
- 读取 `map` → `base_link` 变换 (获取机器人位置)

---

### 2. ObjectDetectorNode

**文件**: `object_detector_node.py`
**功能**: 用 HSV 颜色分割检测目标物体, 结合深度图和 TF 将像素坐标转换为 map 坐标系的 3D 位置. 使用 EMA 平滑合并重复检测.

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `objects_yaml` | string | `""` | HSV 颜色配置文件路径 |
| `rgb_topic` | string | `/camera/color/image_raw` | RGB 图像话题 |
| `depth_topic` | string | `/camera/aligned_depth_to_color/image_raw` | 深度图话题 |
| `info_topic` | string | `/camera/color/camera_info` | 相机内参话题 |
| `camera_frame` | string | `camera_color_optical_frame` | 相机光学坐标系 |
| `map_frame` | string | `map` | 地图坐标系 |
| `merge_dist_m` | float | `0.3` | 新检测与已有位置的合并距离 (m) |
| `ema_alpha` | float | `0.3` | EMA 平滑系数 (越大越偏新值) |
| `publish_debug_image` | bool | `True` | 是否发布调试图像 |

#### 订阅

| 话题 | 类型 | QoS | 说明 |
|------|------|-----|------|
| `/camera/color/camera_info` | `CameraInfo` | sensor_data | 相机内参 |
| `/camera/color/image_raw` | `Image` | sensor_data | RGB 图像 (同步) |
| `/camera/aligned_depth_to_color/image_raw` | `Image` | sensor_data | 深度图 (同步) |

> RGB 和 Depth 使用 `ApproximateTimeSynchronizer` 同步, slop=0.1s

#### 发布

| 话题 | 类型 | 频率 | 说明 |
|------|------|------|------|
| `/detected_objects` | `MarkerArray` | 1 Hz (timer) | **核心输出** — CUBE markers (ns=标签) + TEXT markers |
| `/detector/debug_image` | `Image` | 随帧率 | 带标注框的 BGR 调试图像 |

#### `/detected_objects` MarkerArray 格式

每个检测到的物体产生两个 Marker:
- **CUBE** (type=1): `ns` = 物体标签 (如 `"blue_bottle"`), `pose` = map frame 3D 位置
- **TEXT_VIEW_FACING**: 显示标签文字

其他节点通过过滤 `marker.type == Marker.CUBE` 并读取 `marker.ns` 和 `marker.pose` 来获取检测结果.

#### 服务 (Server)

| 服务 | 类型 | 说明 |
|------|------|------|
| `/detector/clear` | `Trigger` | 清空所有已检测物体 |

#### TF 依赖
- 读取 `camera_color_optical_frame` → `map` 变换

---

### 3. RegionManagerNode

**文件**: `region_manager_node.py`
**功能**: 管理地图上的语义区域 (多边形), 提供区域查询和放置位姿计算.

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `regions_yaml` | string | `""` | 区域定义文件路径 |
| `map_frame` | string | `map` | 地图坐标系 |

#### 发布

| 话题 | 类型 | 频率 | 说明 |
|------|------|------|------|
| `/regions/visualization` | `MarkerArray` | 2 Hz | 区域边界 (LINE_STRIP) + 标签 (TEXT) |

#### 服务 (Server)

| 服务 | 类型 | 说明 |
|------|------|------|
| `/regions/reload` | `Trigger` | 重新加载 regions.yaml |

#### 编程 API (同进程调用)

```python
which_region(x: float, y: float) -> Optional[str]
```
返回点 (x,y) 所在区域名, 或 None.

```python
place_pose(region_name: str) -> Optional[PoseStamped]
```
返回该区域的放置导航目标 (来自 place_anchor).

```python
pick_approach_pose(target_xy, robot_xy, standoff_m=0.55) -> PoseStamped
```
计算从 robot 朝向 target 方向后退 standoff_m 的接近位姿.

#### 独立辅助函数

```python
point_in_polygon(px, py, polygon) -> bool
polygon_centroid(polygon) -> Tuple[float, float]
```

这两个函数被 TaskPlannerNode 和 TaskExecutorNode 直接导入使用.

---

### 4. HeadScanNode

**文件**: `head_scan_node.py`
**功能**: 周期性地移动 Stretch 3 头部云台, 扩大相机视野覆盖以辅助物体检测.

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `trajectory_action` | string | `/stretch_controller/follow_joint_trajectory` | 关节轨迹 action |
| `period_s` | float | `8.0` | 扫描周期 (s) |
| `enabled_on_start` | bool | `False` | 启动时是否自动扫描 |
| `pan_waypoints` | float[] | `[-1.2, -0.6, 0.0, 0.6, 1.2]` | Pan 角度路径点 (rad) |
| `tilt_angle` | float | `-0.55` | 固定 tilt 角度 (rad, 负=向下看) |

#### 服务 (Server)

| 服务 | 类型 | 说明 |
|------|------|------|
| `/head/start_scan` | `Trigger` | 开始周期性扫描 |
| `/head/stop_scan` | `Trigger` | 停止扫描 |
| `/head/scan_once` | `Trigger` | 执行一次完整扫描 (阻塞) |

#### Action Client

| Action | 类型 | 说明 |
|--------|------|------|
| `/stretch_controller/follow_joint_trajectory` | `FollowJointTrajectory` | 控制 `joint_head_pan`, `joint_head_tilt` |

---

### 5. ManipulationNode

**文件**: `manipulation_node.py`
**功能**: 封装 Stretch 3 的抓取/放置动作序列.

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `trajectory_action` | string | `/stretch_controller/follow_joint_trajectory` | 关节控制 action |
| `switch_to_position_srv` | string | `/switch_to_position_mode` | 切换到位置模式服务 |
| `switch_to_navigation_srv` | string | `/switch_to_navigation_mode` | 切换到导航模式服务 |
| `stow_srv` | string | `/stow_the_robot` | Stow 服务 |
| `pick_height_m` | float | `0.75` | 抓取高度 (m) |
| `place_height_m` | float | `0.78` | 放置高度 (m) |
| `arm_extend_m` | float | `0.30` | 手臂伸展长度 (m) |

#### Action Server

| Action | 类型 | 说明 |
|--------|------|------|
| `/manipulation/pick` | `FollowJointTrajectory` | **抓取序列**: open gripper → lower → extend → close → lift → retract → stow |
| `/manipulation/place` | `FollowJointTrajectory` | **放置序列**: lift → extend → lower → open → retract → stow |

> Goal 内容被忽略 (作为 sentinel), 使用参数中的预设高度/距离.

**Result**:
- `error_code = 0`: 成功
- `error_code != 0`: 失败 (如 `PATH_TOLERANCE_VIOLATED`)

#### 服务 (Server)

| 服务 | 类型 | 说明 |
|------|------|------|
| `/manipulation/stow` | `Trigger` | 收回手臂到安全位置 |

#### Action Client / Service Client

| 目标 | 类型 | 说明 |
|------|------|------|
| `/stretch_controller/follow_joint_trajectory` | `FollowJointTrajectory` (action) | 控制各关节 |
| `/switch_to_position_mode` | `Trigger` (service) | 切换到位置模式 |
| `/switch_to_navigation_mode` | `Trigger` (service) | 切换到导航模式 |
| `/stow_the_robot` | `Trigger` (service) | stretch_driver 的 stow |

#### 操作的关节

| 关节名 | 说明 | pick 使用 | place 使用 |
|--------|------|-----------|------------|
| `joint_lift` | 垂直升降 (m) | ✓ | ✓ |
| `wrist_extension` | 手臂伸缩 (m) | ✓ | ✓ |
| `joint_wrist_yaw/pitch/roll` | 手腕关节 (rad) | ✓ | |
| `joint_gripper_finger_left` | 夹爪 (rad, +开/-关) | ✓ | ✓ |
| `joint_head_pan/tilt` | 头部关节 (rad) | ✓ | |

---

### 6. TaskPlannerNode

**文件**: `task_planner_node.py`
**功能**: 根据检测到的物体和目标分配, 生成 pick-and-place 任务计划.

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `planner_backend` | string | `greedy` | 规划后端: `greedy` 或 `vlm` |
| `tasks_yaml` | string | `""` | 任务分配文件 |
| `regions_yaml` | string | `""` | 区域定义文件 |
| `map_frame` | string | `map` | 地图坐标系 |
| `base_frame` | string | `base_link` | 机器人基座坐标系 |
| `vlm_model` | string | `gemini-2.5-flash` | VLM 模型名 |
| `vlm_base_url` | string | `https://...` | VLM API 地址 |
| `vlm_api_key_env` | string | `GEMINI_API_KEY` | API key 环境变量名 |
| `vlm_use_image` | bool | `True` | 是否给 VLM 发送图像 |
| `vlm_max_retries` | int | `2` | VLM 调用最大重试次数 |

#### 订阅

| 话题 | 类型 | 说明 |
|------|------|------|
| `/detected_objects` | `MarkerArray` | 来自 ObjectDetectorNode |
| `/detector/debug_image` | `Image` | 调试图像 (给 VLM 用) |

#### 发布

| 话题 | 类型 | 说明 |
|------|------|------|
| `/planner/plan_visualization` | `MarkerArray` | 计划可视化: pick→place 连线 + 标签 |

#### 服务 (Server)

| 服务 | 类型 | 说明 |
|------|------|------|
| `/planner/compute` | `Trigger` | 执行规划. Response.message 包含计划摘要 |

调用时:
1. 获取 TF 中的机器人位置
2. 读取 `latest_objects` (来自 /detected_objects 订阅)
3. 确定每个物体所在区域 (point_in_polygon)
4. 调用 planner backend 生成 `List[PickPlaceTask]`
5. 发布可视化, 返回结果

#### 编程 API

```python
get_plan() -> List[PickPlaceTask]
```
返回最近一次 compute 的结果.

#### TF 依赖
- 读取 `map` → `base_link` 变换

---

### 7. TaskExecutorNode

**文件**: `task_executor_node.py`
**功能**: 顶层状态机, 编排整个探索→检测→规划→执行流程.

#### 状态机

```
IDLE → HEAD_SCAN → EXPLORE → WAIT_OBJECTS → PLAN →
  ┌→ NAV_TO_PICK → PICK → NAV_TO_PLACE → PLACE ─┐
  └──────────── 循环直到所有任务完成 ◀─────────────┘
                                        ↓
                                       DONE
```

- **IDLE**: 等待 /executor/start 调用
- **HEAD_SCAN**: 调用 stow + start_head_scan, 准备扫描
- **EXPLORE**: 启动 exploration, 等待检测到足够物体或超时
- **WAIT_OBJECTS**: 探索结束后等待 `wait_after_explore_s` 秒让检测稳定
- **PLAN**: 调用 planner 生成计划
- **NAV_TO_PICK**: 导航到物体的接近位姿 (standoff)
- **PICK**: 执行 /manipulation/pick action
- **NAV_TO_PLACE**: 导航到目标区域的 place_anchor
- **PLACE**: 执行 /manipulation/place action
- **DONE**: 所有任务完成, 写入 metrics
- **FAILED**: 出错或 abort

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `planner_backend` | string | `greedy` | 规划后端 |
| `tasks_yaml` | string | `""` | 任务分配文件 |
| `regions_yaml` | string | `""` | 区域定义文件 |
| `map_frame` | string | `map` | 地图坐标系 |
| `base_frame` | string | `base_link` | 机器人基座坐标系 |
| `explore_timeout_s` | float | `180.0` | 探索超时 (s) |
| `min_objects_required` | int | `3` | 探索阶段最少检测物体数 |
| `wait_after_explore_s` | float | `6.0` | 探索后等待时间 (s) |
| `pick_standoff_m` | float | `0.55` | 抓取接近距离 (m) |
| `metrics_path` | string | `/tmp/rearrangement_metrics.json` | 指标输出文件 |
| `start_on_launch` | bool | `False` | 启动时是否自动开始 |

#### 订阅

| 话题 | 类型 | 说明 |
|------|------|------|
| `/detected_objects` | `MarkerArray` | 来自 ObjectDetectorNode |

#### 发布

| 话题 | 类型 | 说明 |
|------|------|------|
| `/executor/state` | `String` | 当前状态名 (每次转换时发布) |

#### 服务 (Server)

| 服务 | 类型 | 说明 |
|------|------|------|
| `/executor/start` | `Trigger` | 启动状态机 (必须在 IDLE 时调用) |
| `/executor/abort` | `Trigger` | 中止, 状态→FAILED |

#### Action Clients

| Action | 类型 | 说明 |
|--------|------|------|
| `/navigate_to_pose` | `NavigateToPose` | 导航到 pick/place 位置 |
| `/manipulation/pick` | `FollowJointTrajectory` | 执行抓取 |
| `/manipulation/place` | `FollowJointTrajectory` | 执行放置 |

#### Service Clients

| 服务 | 类型 | 说明 |
|------|------|------|
| `/exploration/start` | `Trigger` | 启动探索 |
| `/exploration/stop` | `Trigger` | 停止探索 |
| `/head/start_scan` | `Trigger` | 启动头部扫描 |
| `/head/stop_scan` | `Trigger` | 停止头部扫描 |
| `/manipulation/stow` | `Trigger` | 收回手臂 |

#### Metrics 输出

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

**文件**: `sim/fake_sim_node.py`
**功能**: 轻量级仿真, 替代 Gazebo/stretch_driver/Nav2/SLAM, 用于快速端到端测试.

#### 提供的模拟功能

| 功能 | 说明 |
|------|------|
| TF | `map→odom→base_link` (动态), `map→camera_color_optical_frame` (静态) |
| `/map` | 8m×8m 带边界墙的空房间, TRANSIENT_LOCAL |
| `/odom` | 里程计 |
| `/joint_states` | 假关节状态 |
| `/camera/*` | 鸟瞰虚拟相机, 渲染彩色圆形代表物体 |
| `/navigate_to_pose` | 插值移动到目标, ~1.5 m/s |
| `/manipulation/pick` | 拾取最近物体 (距离<1.5m) |
| `/manipulation/place` | 在机器人前方 0.4m 放下物体 |
| 各种 Trigger 服务 | `/head/*`, `/switch_to_*`, `/stow_the_robot` — 全部 no-op |

#### 初始场景

| 物体 | 初始位置 | 目标区域 (tasks.yaml) |
|------|----------|----------------------|
| blue_bottle | (1.5, 1.5) 区域 A | C |
| red_box | (1.5, -1.5) 区域 C | A |
| yellow_cup | (2.5, 0.5) 区域 A | D |

---

## Planner 后端

### GreedyPlanner

**文件**: `planners/greedy.py`

贪心最近邻: 每次选择 `(到物体距离 + 物体到放置点距离)` 最小的任务.
确定性, 不依赖外部 API.

### VLMPlanner

**文件**: `planners/vlm.py`

调用 Gemini API (通过 OpenAI SDK), 发送结构化场景描述 + 可选的调试图像,
让 VLM 生成 JSON 格式的排序计划. 失败时自动 fallback 到 GreedyPlanner.

### 共享数据结构

```python
@dataclass
class DetectedObject:
    label: str                           # 物体标签
    pose_xy: Tuple[float, float]         # map 坐标
    current_region: Optional[str]        # 当前所在区域
    z: float = 0.0

@dataclass
class RegionInfo:
    name: str
    polygon: List[Tuple[float, float]]   # 多边形顶点
    place_anchor: Tuple[float, float, float]  # 放置位姿 (x, y, yaw)

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

## 工具函数库

### frontier_utils.py

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `extract_frontiers(grid, min_cluster_size)` | OccupancyGrid, int | List[Frontier] | 提取前沿聚类 |
| `score_frontier(frontier, robot_xy, α, β)` | Frontier, (x,y), float, float | float | α×dist − β×size |
| `grid_to_world(grid, i, j)` | OccupancyGrid, int, int | (x, y) | 栅格→世界坐标 |
| `world_to_grid(grid, x, y)` | OccupancyGrid, float, float | (i, j) | 世界→栅格坐标 |

### color_segmentation.py

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `segment_color(bgr, spec)` | ndarray, ColorSpec | Detection2D or None | 单色分割 |
| `segment_all(bgr, specs)` | ndarray, List[ColorSpec] | List[Detection2D] | 多色分割 |
| `pixel_to_camera(depth, u, v, fx, fy, cx, cy)` | ndarray, ... | (x,y,z) or None | 像素→相机 3D |
| `annotate(bgr, dets)` | ndarray, List[Detection2D] | ndarray | 画标注框 |
| `load_color_specs(cfg)` | dict | List[ColorSpec] | 加载 HSV 配置 |

### transform_utils.py

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `robot_pose_in_map(node, tf_buffer, ...)` | Node, Buffer | PoseStamped or None | 获取机器人 map 位姿 |
| `transform_point_to_frame(tf_buffer, xyz, src, tgt)` | ... | (x,y,z) or None | 点变换到目标坐标系 |
| `yaw_to_quat(yaw)` | float | (x,y,z,w) | Yaw→四元数 |
| `quat_to_yaw(x,y,z,w)` | float×4 | float | 四元数→Yaw |

---

## 配置文件

| 文件 | 说明 |
|------|------|
| `config/objects.yaml` | HSV 颜色范围. H[0,180], S/V[0,255]. 红色用两段范围. |
| `config/regions.yaml` | 区域多边形 (CCW, map frame 米). 每个区域含 polygon + place_anchor. |
| `config/tasks.yaml` | `assignments: {label: region}` 目标分配. |
| `config/slam_params.yaml` | SLAM Toolbox 参数 |
| `config/nav2_params.yaml` | Nav2 参数 |

---

## 集成流程

### 仿真模式 (快速验证)

```bash
# 编译
cd your_ws && colcon build --symlink-install && source install/setup.bash

# 一键启动全部 (fake_sim + detector + planner + executor + exploration)
ros2 launch exploration_rearrangement sim.launch.py start_on_launch:=true

# 监控状态
ros2 topic echo /executor/state
# 查看结果
cat /tmp/rearrangement_metrics.json
```

### 真机模式

```bash
# 1. stretch_driver
ros2 launch stretch_core stretch_driver.launch.py

# 2. SLAM + Nav2
ros2 launch exploration_rearrangement mapping.launch.py   # 或 bringup.launch.py

# 3. 我们的节点
ros2 launch exploration_rearrangement rearrangement.launch.py planner_backend:=greedy

# 4. 开始任务
ros2 service call /executor/start std_srvs/srv/Trigger
```

### 分步调试

建议按以下顺序逐步验证子功能:

| 顺序 | 测试脚本 | 验证内容 | 依赖 |
|------|----------|----------|------|
| 1 | `test/test_head_scan.py` | 头部扫描 | stretch_driver 或 fake_sim |
| 2 | `test/test_navigation.py` | Nav2 导航 | stretch_driver+Nav2 或 fake_sim |
| 3 | `test/test_object_detector.py` | 物体检测 | 相机 + TF |
| 4 | `test/test_region_manager.py` | 区域管理 | 无额外依赖 |
| 5 | `test/test_manipulation.py` | 抓取/放置 | stretch_driver 或 fake_sim |
| 6 | `test/test_exploration.py` | 自主探索 | SLAM+Nav2 或 fake_sim + exploration_node |
| 7 | `test/test_task_planner.py` | 任务规划 | detector + planner 节点 |
| 8 | `test/test_task_executor.py` | 状态机 | 全部节点 |
| 9 | `test/test_e2e_sim.py` | 端到端 | sim.launch.py |

---

## 测试指南

### 运行所有 pytest 单元测试 (无需 ROS 运行环境)

```bash
cd src/exploration_rearrangement
python -m pytest test/ -v
```

### 运行机器人集成测试脚本

```bash
# 1. 启动仿真后台
ros2 run exploration_rearrangement fake_sim_node

# 2. 运行各子功能测试 (在另一终端)
python3 test/test_head_scan.py
python3 test/test_navigation.py
python3 test/test_manipulation.py
python3 test/test_object_detector.py    # 需要同时运行 detector node
python3 test/test_region_manager.py     # 需要同时运行 region_manager node
python3 test/test_exploration.py        # 需要同时运行 exploration_node
python3 test/test_task_planner.py       # 需要 detector + planner nodes
python3 test/test_task_executor.py      # 需要完整仿真栈

# 3. 端到端测试
ros2 launch exploration_rearrangement sim.launch.py start_on_launch:=true
# 在另一终端:
python3 test/test_e2e_sim.py
```
