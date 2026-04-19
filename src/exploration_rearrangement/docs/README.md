# `exploration_rearrangement` package docs

The canonical entry point for build / mapping / annotation / runbook is
the workspace-root [`README.md`](../../../README.md). This directory
holds two focused references:

| Doc | Scope |
|---|---|
| [`VISUAL_GRASP_RUNBOOK.md`](VISUAL_GRASP_RUNBOOK.md) | Two-stage visually-guided pick (head-camera coarse approach with `visual_servo_arm_node`, then gripper-camera fine grasp with `visual_grasp_node` + `fine_object_detector_node`). |
| [`navigation_module.md`](navigation_module.md) | Vendored upstream README for the `NavigationModule` coordinator (`navigation_node.py`). Documents the `/nav/goals` + `/nav/control_flag` + `/nav/arrived_flag` topic protocol. |

Older `RUNBOOK.md`, `INTEGRATION_RUNBOOK.md`, `INTERFACES.md`, and
`INTERFACES_CN.md` were removed when the Nav2 driver was replaced with
the vendored `NavigationModule` and the executor was rewritten as a
plan/nav/manipulation glue — the nodes those docs described
(`exploration_node`, `head_scan_node`, `fake_sim_node`,
`fake_planner_inputs`, the old HEAD_SCAN/EXPLORE/WAIT_OBJECTS state
machine) no longer exist. The current behaviour is documented inline
in the source files plus the root README.
