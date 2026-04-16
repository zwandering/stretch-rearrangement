"""Unit tests for frontier extraction, color segmentation, and region polygon helpers."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exploration_rearrangement.utils.frontier_utils import (  # noqa: E402
    extract_frontiers, grid_to_world, world_to_grid, score_frontier,
)
from exploration_rearrangement.utils.color_segmentation import (  # noqa: E402
    ColorSpec, segment_color, segment_all, pixel_to_camera, annotate,
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


# -------- Color segmentation ------------------------------------------

def _red_square_image(size=64):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[20:44, 20:44] = (0, 0, 200)  # BGR red
    return img


def test_segment_color_detects_red():
    img = _red_square_image()
    spec = ColorSpec(
        name='red_box',
        hsv_low=(0, 120, 70), hsv_high=(10, 255, 255),
        hsv_low_2=(170, 120, 70), hsv_high_2=(180, 255, 255),
        min_area_px=100,
    )
    det = segment_color(img, spec)
    assert det is not None
    cx, cy = det.center_px
    assert 28 <= cx <= 36 and 28 <= cy <= 36
    assert det.area_px >= 400


def test_segment_color_returns_none_on_blank():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    spec = ColorSpec(
        name='red_box', hsv_low=(0, 120, 70), hsv_high=(10, 255, 255),
        min_area_px=50,
    )
    assert segment_color(img, spec) is None


def test_segment_all_multiple_specs():
    img = np.zeros((64, 128, 3), dtype=np.uint8)
    img[20:44, 10:34] = (0, 0, 200)     # red
    img[20:44, 70:94] = (200, 0, 0)     # blue
    specs = [
        ColorSpec('red', (0, 120, 70), (10, 255, 255),
                  (170, 120, 70), (180, 255, 255), min_area_px=100),
        ColorSpec('blue', (100, 120, 60), (130, 255, 255), min_area_px=100),
    ]
    dets = segment_all(img, specs)
    assert {d.label for d in dets} == {'red', 'blue'}


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


def test_annotate_runs():
    img = _red_square_image()
    spec = ColorSpec('red_box', (0, 120, 70), (10, 255, 255),
                     (170, 120, 70), (180, 255, 255), min_area_px=100)
    dets = segment_all(img, [spec])
    out = annotate(img, dets)
    assert out.shape == img.shape


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
