"""Unit tests for frontier extraction, depth projection, and region polygon helpers."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exploration_rearrangement.utils.frontier_utils import (  # noqa: E402
    extract_frontiers, grid_to_world, world_to_grid, score_frontier,
)
from exploration_rearrangement.utils.depth_projection import (  # noqa: E402
    pixel_to_camera,
)
from exploration_rearrangement.region_manager_node import (  # noqa: E402
    point_in_polygon, polygon_centroid,
)


# -------- Frontier -----------------------------------------------------

def _make_grid(data_2d: np.ndarray, resolution: float = 0.1,
               origin_x: float = 0.0, origin_y: float = 0.0):
    from nav_msgs.msg import OccupancyGrid
    grid = OccupancyGrid()
    grid.info.resolution = float(resolution)
    grid.info.width = int(data_2d.shape[1])
    grid.info.height = int(data_2d.shape[0])
    grid.info.origin.position.x = float(origin_x)
    grid.info.origin.position.y = float(origin_y)
    grid.data = [int(v) for v in data_2d.flatten().tolist()]
    return grid


def test_extract_frontiers_finds_boundary():
    arr = np.full((10, 10), -1, dtype=np.int16)
    arr[3:7, 3:7] = 0  # free region in the middle
    grid = _make_grid(arr)
    frontiers = extract_frontiers(grid, min_cluster_size=2)
    assert len(frontiers) >= 1
    total_cells = sum(f.size for f in frontiers)
    assert total_cells > 0


def test_extract_frontiers_empty_when_fully_known():
    arr = np.zeros((5, 5), dtype=np.int16)  # all free
    grid = _make_grid(arr)
    assert extract_frontiers(grid) == []


def test_grid_world_roundtrip():
    arr = np.zeros((5, 5), dtype=np.int16)
    grid = _make_grid(arr, resolution=0.5, origin_x=1.0, origin_y=-2.0)
    wx, wy = grid_to_world(grid, 2, 3)
    i, j = world_to_grid(grid, wx, wy)
    assert (i, j) == (2, 3)


def test_score_frontier_prefers_closer_larger():
    from exploration_rearrangement.utils.frontier_utils import Frontier
    far = Frontier((10.0, 10.0), 10, [])
    near = Frontier((1.0, 1.0), 10, [])
    assert score_frontier(near, (0, 0)) < score_frontier(far, (0, 0))


# -------- Depth projection --------------------------------------------

def test_pixel_to_camera_reasonable():
    depth = np.full((100, 100), 1000, dtype=np.uint16)  # 1 m
    xyz = pixel_to_camera(depth, u=50, v=50, fx=500, fy=500, cx=50, cy=50)
    assert xyz is not None
    x, y, z = xyz
    assert abs(z - 1.0) < 1e-3
    assert abs(x) < 1e-3 and abs(y) < 1e-3


def test_pixel_to_camera_rejects_zero_depth():
    depth = np.zeros((50, 50), dtype=np.uint16)
    assert pixel_to_camera(depth, 25, 25, 500, 500, 25, 25) is None


def test_pixel_to_camera_offset_pixel():
    depth = np.full((100, 100), 2000, dtype=np.uint16)  # 2 m uniform
    xyz = pixel_to_camera(depth, u=60, v=50, fx=500, fy=500, cx=50, cy=50)
    assert xyz is not None
    x, _, z = xyz
    # 10 px off-axis at z=2m with fx=500 → x = 10*2/500 = 0.04 m
    assert abs(z - 2.0) < 1e-3
    assert abs(x - 0.04) < 1e-3


# -------- Region polygon ----------------------------------------------

def test_point_in_polygon_inside():
    square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert point_in_polygon(1.0, 1.0, square)


def test_point_in_polygon_outside():
    square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert not point_in_polygon(-0.1, 1.0, square)
    assert not point_in_polygon(3.0, 1.0, square)


def test_point_in_polygon_concave():
    L = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]
    assert point_in_polygon(0.5, 2.5, L)
    assert not point_in_polygon(2.0, 2.0, L)


def test_polygon_centroid_square():
    cx, cy = polygon_centroid([(0, 0), (2, 0), (2, 2), (0, 2)])
    assert pytest.approx(cx) == 1.0
    assert pytest.approx(cy) == 1.0


# -------- 3D detection dedup ------------------------------------------

from exploration_rearrangement.object_detector_node import _dedup_candidates_3d  # noqa: E402


def test_dedup_drops_lower_conf_when_within_threshold():
    # Two different-class candidates within 0.1 m — weaker one must be dropped.
    a = (0.9, 'blue_cup', None, (1.0, 0.0, 0.5))
    b = (0.6, 'green_cup', None, (1.05, 0.02, 0.5))
    kept = _dedup_candidates_3d([a, b], dedup_dist=0.25)
    assert [k[1] for k in kept] == ['blue_cup']


def test_dedup_keeps_both_when_far_apart():
    a = (0.8, 'blue_cup', None, (1.0, 0.0, 0.5))
    b = (0.7, 'green_cup', None, (2.0, 2.0, 0.5))
    kept = _dedup_candidates_3d([a, b], dedup_dist=0.25)
    assert sorted(k[1] for k in kept) == ['blue_cup', 'green_cup']


def test_dedup_is_order_independent():
    a = (0.4, 'green_cup', None, (1.0, 0.0, 0.5))
    b = (0.9, 'blue_cup',  None, (1.1, 0.0, 0.5))
    assert [k[1] for k in _dedup_candidates_3d([a, b], dedup_dist=0.25)] == ['blue_cup']
    assert [k[1] for k in _dedup_candidates_3d([b, a], dedup_dist=0.25)] == ['blue_cup']


def test_dedup_zero_threshold_keeps_everything():
    a = (0.9, 'blue_cup', None, (1.0, 0.0, 0.5))
    b = (0.6, 'green_cup', None, (1.0, 0.0, 0.5))
    kept = _dedup_candidates_3d([a, b], dedup_dist=0.0)
    assert len(kept) == 2


# -------- Fine detector helpers ---------------------------------------

from exploration_rearrangement.fine_object_detector_node import (  # noqa: E402
    _ema_update,
)


def test_ema_update_initializes_on_none_prev():
    out = _ema_update(None, (1.0, 2.0, 3.0), alpha=0.3)
    assert out == (1.0, 2.0, 3.0)


def test_ema_update_blends_with_alpha():
    prev = (0.0, 0.0, 0.0)
    out = _ema_update(prev, (1.0, 1.0, 1.0), alpha=0.25)
    assert out == (0.25, 0.25, 0.25)


def test_ema_update_alpha_one_takes_new_value():
    out = _ema_update((5.0, 5.0, 5.0), (1.0, 2.0, 3.0), alpha=1.0)
    assert out == pytest.approx((1.0, 2.0, 3.0))


def test_ema_update_teleport_replaces_when_jump_exceeds_threshold():
    prev = (0.0, 0.0, 0.0)
    # jump of sqrt(2) ~ 1.41 m in xy, threshold 0.5 m → replace
    out = _ema_update(prev, (1.0, 1.0, 0.5), alpha=0.3, teleport_dist=0.5)
    assert out == (1.0, 1.0, 0.5)


def test_ema_update_teleport_smooths_when_jump_under_threshold():
    prev = (0.0, 0.0, 0.0)
    out = _ema_update(prev, (0.1, 0.0, 0.0), alpha=0.5, teleport_dist=1.0)
    assert out == pytest.approx((0.05, 0.0, 0.0))
