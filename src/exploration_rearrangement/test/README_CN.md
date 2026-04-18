# 测试套件

本目录包含 `exploration_rearrangement` 包的两类测试：

1. **Pytest 单元测试** — 离线运行，不需要 ROS 环境（`conftest.py`、`test_utils.py`、`test_planners.py`、`test_integration.py`）
2. **机器人集成测试** — 需要 ROS 节点运行（`test_head_scan.py`、`test_navigation.py` 等）

---

## 第一部分：Pytest 单元测试

这些测试验证核心算法和数据结构，**无需**启动任何 ROS 节点。
运行方式：

```bash
cd src/exploration_rearrangement
python -m pytest test/ -v
```

### conftest.py

共享 pytest 配置文件。将包根目录添加到 `sys.path`，使 `exploration_rearrangement.*`
模块在 colcon 工作空间外也可以导入。

### test_utils.py — 工具函数测试（15 个测试）

测试三个工具库：前沿提取、颜色分割和区域多边形辅助函数。

**前沿提取**（`frontier_utils.py`）：
- `test_extract_frontiers_finds_boundary` — 创建 10×10 栅格，中心有空闲区域被未知单元包围。验证 `extract_frontiers()` 返回至少一个前沿簇且总单元数非零。
- `test_extract_frontiers_empty_when_fully_known` — 全空闲（全零）栅格不应产生前沿。
- `test_grid_world_roundtrip` — 通过 `grid_to_world()` / `world_to_grid()` 在栅格索引和世界坐标间转换。验证非平凡原点和分辨率下的无损往返。
- `test_score_frontier_prefers_closer_larger` — 距机器人更近的前沿应有更低（更优）的分数。

**颜色分割**（`color_segmentation.py`）：
- `test_segment_color_detects_red` — 输入合成 64×64 含红色方块图像。验证 `segment_color()` 检测到正确中心和足够面积。
- `test_segment_color_returns_none_on_blank` — 全黑图像不应产生检测结果。
- `test_segment_all_multiple_specs` — 含红色和蓝色方块的图像。`segment_all()` 应检测到两个标签。
- `test_pixel_to_camera_reasonable` — 均匀 1m 深度图在主点处应映射为相机坐标 (0, 0, 1.0)。
- `test_pixel_to_camera_rejects_zero_depth` — 零深度像素应返回 `None`（无效读数）。
- `test_annotate_runs` — `annotate()` 应返回相同形状的图像且无报错。

**区域多边形辅助函数**（`region_manager_node.py`）：
- `test_point_in_polygon_inside` — 点 (1,1) 应在原点处 2×2 正方形内部。
- `test_point_in_polygon_outside` — x=−0.1 和 x=3.0 处的点应在外部。
- `test_point_in_polygon_concave` — 测试 L 形凹多边形：(0.5, 2.5) 在内部，(2.0, 2.0) 在凹陷外部。
- `test_polygon_centroid_square` — 2×2 正方形的质心应为 (1.0, 1.0)。

### test_planners.py — 规划器后端测试（10 个测试）

测试 `GreedyPlanner`、`VLMPlanner` 及共享工具（`filter_actionable`、`euclidean`）。

**filter_actionable**：
- `test_filter_actionable_skips_objects_already_placed` — `current_region` 与目标匹配的物体应被排除。仅保留错误放置的物体。
- `test_filter_actionable_drops_unassigned_and_unknown_region` — 无匹配分配的物体或分配到不存在区域的物体都应被过滤。

**GreedyPlanner**：
- `test_greedy_plan_length_and_order` — 3 个错误放置物体 → 计划有 3 个任务，`order_index` 连续为 [0, 1, 2]，每个任务的 `place_xy` 与目标区域的 `place_anchor` 匹配。
- `test_greedy_first_pick_is_nearest_to_robot` — 贪心算法应先选最近的物体。机器人在原点时，(0.1, 0.1) 处物体应排在 (1.9, 1.9) 之前。
- `test_greedy_empty_when_everything_placed` — 所有物体已在目标区域时，计划应为空。

**VLMPlanner**：
- `test_vlm_falls_back_to_greedy_when_no_api_key` — 无 `GEMINI_API_KEY` 环境变量时，VLMPlanner 应静默回退到 GreedyPlanner 并产生有效计划。
- `test_vlm_parses_valid_json` — 注入返回有效 JSON 的假 OpenAI 客户端。验证 VLMPlanner 正确解析响应中的任务排序。
- `test_vlm_auto_appends_missing_objects` — 若 VLM 只返回部分物体，VLMPlanner 应自动补充遗漏物体（通过贪心回退），确保不遗漏任何物体。
- `test_vlm_falls_back_on_invalid_json` — 若 VLM 返回非 JSON 内容，VLMPlanner 应回退到 GreedyPlanner 而非崩溃。

**工具函数**：
- `test_euclidean_symmetry` — `euclidean((0,0), (3,4))` 应为 5.0，且函数应满足对称性。

### test_integration.py — 端到端规划器集成测试（5 个测试）

使用**真实 YAML 配置文件**（`config/regions.yaml`、`config/tasks.yaml`）测试两个规划器后端，验证配置 → 规划器管线的端到端工作流。

- `test_real_yaml_loads_cleanly` — 加载 `regions.yaml` 和 `tasks.yaml`，验证 4 个区域（A/B/C/D）具有有效多边形和锚点，以及 3 个任务分配（blue_bottle、red_box、yellow_cup）。
- `test_greedy_against_real_configs` — 模拟 3 个错误放置物体的场景，使用真实配置运行 GreedyPlanner。验证产生 3 个任务，全部路由到正确目标区域、正确 `place_xy`、连续排序。
- `test_vlm_backend_wraps_greedy_when_offline` — 无 API key 时，VLMPlanner 仍通过贪心回退产生有效的 3 任务计划。
- `test_skip_objects_already_in_goal_region` — 将 blue_bottle 放在区域 C（其目标）。计划应仅包含其余 2 个物体的任务。
- `test_greedy_plan_is_deterministic` — 对相同输入运行 GreedyPlanner 两次，应产生相同任务排序。

---

## 第二部分：机器人集成测试

## 机器人集成测试顺序

按顺序递增测试子功能——前面的测试验证后面测试的依赖项：

| # | 脚本 | 验证内容 | 依赖 |
|---|------|----------|------|
| 1 | `test_head_scan.py` | 头部扫描 | stretch_driver 或 fake_sim |
| 2 | `test_navigation.py` | Nav2 导航 | stretch_driver + Nav2 或 fake_sim |
| 3 | `test_object_detector.py` | 物体检测 | 相机 + TF |
| 4 | `test_region_manager.py` | 区域管理 | 无额外依赖 |
| 5 | `test_manipulation.py` | 抓取/放置 | stretch_driver 或 fake_sim |
| 6 | `test_exploration.py` | 自主探索 | SLAM + Nav2 或 fake_sim + exploration_node |
| 7 | `test_task_planner.py` | 任务规划 | 检测器 + 规划器节点 |
| 8 | `test_task_executor.py` | 状态机 | 全部节点 |
| 9 | `test_e2e_sim.py` | 端到端 | sim.launch.py |

---

## 快速开始（仿真模式）

```bash
# 编译
cd your_ws && colcon build --symlink-install && source install/setup.bash

# 终端 1 — 启动仿真器
ros2 run exploration_rearrangement fake_sim_node

# 终端 2 — 运行各个测试
python3 test/test_head_scan.py
python3 test/test_navigation.py
# ...
```

---

## 测试详细说明

### 1. test_head_scan.py — HeadScanNode

**测试对象**: `/head/start_scan`、`/head/stop_scan`、`/head/scan_once`（Trigger 服务）

**功能说明**:
HeadScanNode 控制 Stretch 3 机器人的头部云台进行周期性扫描，以扩大相机视野覆盖范围，辅助物体检测。它通过 FollowJointTrajectory action 向 stretch_controller 发送 joint_head_pan 和 joint_head_tilt 关节目标。

**前置条件**:
- 仿真模式: `ros2 run exploration_rearrangement fake_sim_node` + `ros2 run exploration_rearrangement head_scan_node`
- 真机模式: `ros2 launch stretch_core stretch_driver.launch.py` + `ros2 run exploration_rearrangement head_scan_node`

**输入/输出**:
- 输入: Trigger 服务请求（无参数）
- 输出: Trigger 服务响应（success: bool, message: str）
- 副作用: 头部云台移动到不同 pan/tilt 角度

**测试列表**:
1. 调用 `/head/start_scan` → 验证 success=True，头部开始周期性移动
2. 等待若干秒后调用 `/head/stop_scan` → 验证 success=True，头部停止
3. 调用 `/head/scan_once` → 验证完整扫描一轮后返回 success=True
4. 反复 start/stop 切换 → 验证不会崩溃或状态异常
5. 连续两次 scan_once → 验证不会因状态残留失败

---

### 2. test_navigation.py — Nav2 NavigateToPose

**测试对象**: `/navigate_to_pose`（NavigateToPose action）

**功能说明**:
测试机器人导航功能。ExplorationNode 和 TaskExecutorNode 都依赖 Nav2 的 /navigate_to_pose action server 来执行移动。此脚本直接发送导航目标，验证机器人能正确到达。

**前置条件**:
- 仿真模式: `ros2 run exploration_rearrangement fake_sim_node`
- 真机模式: `ros2 launch stretch_core stretch_driver.launch.py` + Nav2 launch

**输入/输出**:
- 输入: NavigateToPose.Goal — 目标 PoseStamped (x, y, yaw) in map frame
- 输出: NavigateToPose.Result — 导航是否成功完成
- 副作用: 机器人移动到目标位置

**测试列表**:
1. 导航到 (1.0, 0.0) → 验证正前方短距离导航
2. 导航到 (0.0, 0.0) → 验证回到原点
3. 导航到 (−1.0, 1.0) → 验证斜方向导航 + 朝向
4. 导航到 (2.0, 2.0) → 验证较远距离导航（区域 A）
5. 发送取消导航请求 → 验证目标可以被取消
6. 连续发送两个目标 → 验证第二个目标是否覆盖第一个

---

### 3. test_object_detector.py — ObjectDetectorNode

**测试对象**: `/detected_objects`、`/detector/debug_image`、`/detector/clear`

**功能说明**:
ObjectDetectorNode 订阅 RGB-D 相机话题，用 HSV 颜色分割检测目标物体（blue_bottle, red_box, yellow_cup），然后通过深度图 + TF 将像素坐标转换为 map 坐标系下的 3D 位置。检测结果以 MarkerArray 发布到 /detected_objects，调试图像发布到 /detector/debug_image。

**前置条件**:
- 仿真模式: `fake_sim_node` + `object_detector_node --ros-args -p objects_yaml:=<path>/config/objects.yaml`
- 真机模式: `stretch_driver` + `object_detector_node`

**输入/输出**:
- 输入: `/camera/color/image_raw`、`/camera/aligned_depth_to_color/image_raw`、`/camera/color/camera_info`
- 输出: `/detected_objects`（MarkerArray）、`/detector/debug_image`（Image）

**测试列表**:
1. 检查相机话题是否有数据发布
2. 等待并检查 /detected_objects 是否有 Marker 发布
3. 验证检测到的物体标签是否在预期集合内
4. 验证物体位置是否在合理范围内（地图边界之内）
5. 检查 /detector/debug_image 是否正常发布
6. 调用 /detector/clear 清空检测 → 验证 markers 被清空
7. 等待重新检测 → 验证清空后能恢复检测

---

### 4. test_region_manager.py — RegionManagerNode

**测试对象**: `/regions/visualization`、`/regions/reload`、`point_in_polygon`、`polygon_centroid`

**功能说明**:
RegionManagerNode 管理地图上的语义区域（A/B/C/D），每个区域定义为 map frame 中的多边形，并带有 place_anchor（放置物体时的导航目标点）。它提供 RViz 可视化、重新加载服务和编程 API（which_region、place_pose、pick_approach_pose）。

**前置条件**:
- `ros2 run exploration_rearrangement region_manager_node --ros-args -p regions_yaml:=<path>/config/regions.yaml`
- 此测试不需要 fake_sim_node（不依赖 TF/导航）

**输入/输出**:
- 输入: regions.yaml 配置文件
- 输出: `/regions/visualization`（MarkerArray）

**测试列表**:
1. 验证 /regions/visualization 有 markers 发布
2. 验证 markers 包含 4 个区域（A/B/C/D）
3. 调用 /regions/reload → 验证成功
4. Python 测试 point_in_polygon：点在区域内
5. Python 测试 point_in_polygon：点在区域外
6. Python 测试 point_in_polygon：边界情况（原点在多个区域交界）
7. Python 测试 polygon_centroid 正确性
8. 网格采样 [−3,3]×[−3,3] 验证区域完整覆盖

---

### 5. test_manipulation.py — ManipulationNode（抓取/放置）

**测试对象**: `/manipulation/pick`、`/manipulation/place`（FollowJointTrajectory actions）、`/manipulation/stow`（Trigger）

**功能说明**:
ManipulationNode 封装了 Stretch 3 的抓取和放置动作。它通过 /stretch_controller/follow_joint_trajectory action 来控制关节（lift, wrist_extension, gripper 等）。

Pick 序列: 看向前方 → 打开夹爪 → 下降 → 伸臂 → 合拢夹爪 → 抬升 → 缩臂 → stow
Place 序列: 抬升 → 伸臂 → 下降 → 打开夹爪 → 缩臂 → stow

**前置条件**:
- 仿真模式（推荐先用仿真验证）: `ros2 run exploration_rearrangement fake_sim_node`（fake_sim_node 自带 pick/place 服务，不需要 manipulation_node）
- 真机模式: `stretch_driver` + `manipulation_node`

**输入/输出**:
- 输入: FollowJointTrajectory.Goal（sentinel）、Trigger.Request
- 输出: FollowJointTrajectory.Result（error_code=0 成功，非 0 失败）、Trigger.Response

**测试列表**:
1. stow → 验证机器人收回手臂到安全位置
2. 导航到物体附近 → pick → 验证抓取动作执行
3. 导航到目标区域 → place → 验证放置动作执行
4. 连续 stow 两次 → 验证幂等性
5. 远离物体时 pick → 验证失败处理（仿真中返回 error_code≠0）
6. 无物体持有时 place → 验证失败处理
7. 完整流程: 导航→pick→导航→place — 端到端抓放验证

---

### 6. test_exploration.py — ExplorationNode

**测试对象**: `/exploration/start`、`/exploration/stop`（Trigger）、`/exploration/status`、`/exploration/frontiers`

**功能说明**:
ExplorationNode 使用前沿（frontier）探索算法，从 /map 中提取已知区域与未知区域的边界，选择最优前沿点发送给 Nav2 导航。探索完成（无前沿剩余）后发布 "done" 状态。

**前置条件**:
- 仿真模式: `fake_sim_node` + `exploration_node`
- 真机模式: `stretch_driver` + SLAM + Nav2 + `exploration_node`

**输入/输出**:
- 输入: `/map`（OccupancyGrid，来自 SLAM）、TF map→base_link
- 输出: `/exploration/frontiers`（MarkerArray）、`/exploration/status`（"navigating" / "done"）
- 副作用: 向 Nav2 发送 NavigateToPose 目标

**注意**: fake_sim_node 提供的地图几乎全是已知区域（边界上有少量 unknown），所以前沿可能很少或很快耗尽。这是仿真环境的正常行为。在真机 SLAM 场景下前沿会更多。

**测试列表**:
1. 验证 /exploration/start 服务可用并返回 success=True
2. 启动后验证 /exploration/status 或 /exploration/frontiers 有输出
3. 验证 /exploration/stop 能正常停止
4. stop 后验证不再发送新的导航目标
5. 反复 start/stop 切换不崩溃
6. 在完全已知地图上，验证 exploration 快速完成并发布 "done"

---

### 7. test_task_planner.py — TaskPlannerNode

**测试对象**: `/planner/compute`（Trigger）、`/planner/plan_visualization`

**功能说明**:
TaskPlannerNode 订阅 /detected_objects（来自 ObjectDetectorNode），并提供 /planner/compute 服务。调用该服务时，它将：(1) 获取当前检测到的物体位置 (2) 获取机器人当前位置（TF）(3) 根据 tasks.yaml 中的目标分配，调用 planner backend（greedy/vlm）生成计划 (4) 发布计划可视化 (5) 返回计划摘要。

**前置条件**:
- 仿真模式: `fake_sim_node` + `object_detector_node` + `task_planner_node`

**输入/输出**:
- 输入: `/detected_objects`（MarkerArray）、TF、`tasks.yaml` + `regions.yaml`
- 输出: Trigger.Response（success + 计划摘要字符串）、`/planner/plan_visualization`（MarkerArray）

**测试列表**:
1. 等待物体检测就绪，调用 /planner/compute → 验证 success=True
2. 解析返回消息，验证包含所有需要移动的物体
3. 验证 /planner/plan_visualization 有 markers 发布
4. 再次调用 compute → 验证计划一致性（greedy 应确定性）
5. 无检测物体时调用 compute → 验证空计划或失败
6. Python 纯逻辑测试: GreedyPlanner 不同场景
7. Python 纯逻辑测试: filter_actionable 正确过滤

---

### 8. test_task_executor.py — TaskExecutorNode

**测试对象**: `/executor/start`、`/executor/abort`（Trigger）、`/executor/state`

**功能说明**:
TaskExecutorNode 是整个系统的状态机，按以下流程编排所有子功能：

```
IDLE → HEAD_SCAN → EXPLORE → WAIT_OBJECTS → PLAN →
  (NAV_TO_PICK → PICK → NAV_TO_PLACE → PLACE) × N → DONE
```

它通过 service 和 action 客户端协调 ExplorationNode、HeadScanNode、ManipulationNode 和 Nav2。

**前置条件**:
- 仿真模式: `ros2 launch exploration_rearrangement sim.launch.py start_on_launch:=false`
  或分别启动所有节点

**输入/输出**:
- 输入: Trigger 请求（start/abort）、/detected_objects、TF、Nav2、Manipulation actions
- 输出: `/executor/state`（String）、`/tmp/rearrangement_metrics.json`

**测试列表**:
1. 验证 executor 初始状态为 IDLE
2. 调用 /executor/start → 验证 success=True
3. 监听状态转换: IDLE → HEAD_SCAN → EXPLORE → ...
4. 调用 /executor/abort → 验证状态变为 FAILED
5. abort 后尝试 start → 验证无效（不在 IDLE 状态）
6. 检查 metrics 文件（需要先完成一次完整运行）

---

### 9. test_e2e_sim.py — 端到端仿真测试

**测试对象**: 通过 `sim.launch.py` 运行的完整系统流水线

**功能说明**:
被动监控脚本，订阅 `/executor/state` 和 `/detected_objects`，观察整个 IDLE → 探索 → 检测 → 规划 → 抓取/放置 → DONE 的流程，并在最终验证 metrics 文件。

**前置条件**:
```bash
ros2 launch exploration_rearrangement sim.launch.py start_on_launch:=true planner_backend:=greedy
```

**输入/输出**:
- 输入: 无（纯监听模式）
- 输出: 测试报告（基于 /executor/state 和 metrics 文件）

**测试列表**:
1. 系统是否进入 HEAD_SCAN / EXPLORE
2. 是否检测到 3 个物体
3. 是否进入 PLAN 状态
4. 是否执行 NAV_TO_PICK / PICK / NAV_TO_PLACE / PLACE
5. 最终是否到达 DONE 状态
6. metrics 文件中的成功率
7. 总耗时是否在合理范围内（仿真中 < 180s）
