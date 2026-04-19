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
    _erode_bool_mask, aabb_iou_3d, estimate_bbox_3d,
    estimate_bbox_3d_from_mask, pixel_to_camera,
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


# -------- 3D bbox estimation ------------------------------------------

def test_estimate_bbox_3d_uniform_depth():
    # 20x20 px bbox at z=1m with fx=fy=500 → 0.04 m x 0.04 m; z-size clamped
    # to min_size_m because uniform depth has no spread.
    depth = np.full((100, 100), 1000, dtype=np.uint16)  # 1 m
    res = estimate_bbox_3d(depth, x=40, y=40, w=20, h=20,
                           fx=500, fy=500, cx=50, cy=50)
    assert res is not None
    (cx, cy, cz), (dx, dy, dz) = res
    assert abs(cz - 1.0) < 1e-3
    assert abs(dx - 0.04) < 1e-3
    assert abs(dy - 0.04) < 1e-3
    assert dz >= 0.02  # min_size_m clamp


def test_estimate_bbox_3d_returns_none_on_empty_depth():
    depth = np.zeros((50, 50), dtype=np.uint16)
    assert estimate_bbox_3d(depth, 10, 10, 10, 10, 500, 500, 25, 25) is None


def test_estimate_bbox_3d_size_scales_linearly_with_depth():
    depth_1m = np.full((100, 100), 1000, dtype=np.uint16)
    depth_2m = np.full((100, 100), 2000, dtype=np.uint16)
    _, s1 = estimate_bbox_3d(depth_1m, 40, 40, 20, 20, 500, 500, 50, 50)
    _, s2 = estimate_bbox_3d(depth_2m, 40, 40, 20, 20, 500, 500, 50, 50)
    assert abs(s2[0] - 2.0 * s1[0]) < 1e-3
    assert abs(s2[1] - 2.0 * s1[1]) < 1e-3


def test_estimate_bbox_3d_depth_spread_drives_z_extent():
    # Gradient depth from 0.8 m (top) to 1.2 m (bottom) inside the bbox.
    depth = np.zeros((100, 100), dtype=np.uint16)
    for row in range(100):
        depth[row, :] = int(800 + (row / 99.0) * 400)  # 800 → 1200 mm
    res = estimate_bbox_3d(depth, x=40, y=10, w=20, h=80,
                           fx=500, fy=500, cx=50, cy=50)
    assert res is not None
    _, (_, _, dz) = res
    # 5–95 percentile span of a linear 0.8–1.2 gradient ≈ 0.36 m.
    assert 0.25 < dz < 0.45


def test_estimate_bbox_3d_from_mask_excludes_background():
    # 100x100 scene: background at 3 m, a 20x20 foreground patch at 1 m.
    # Rectangular bbox estimator would see BOTH depths (dz ≈ 2 m).
    # Mask estimator should see only the 1 m pixels (dz ≈ min_size).
    # erode_px=0 keeps the full 20x20 silhouette so size checks are precise.
    depth = np.full((100, 100), 3000, dtype=np.uint16)
    depth[40:60, 40:60] = 1000
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    res = estimate_bbox_3d_from_mask(
        depth, mask, fx=500, fy=500, cx=50, cy=50, erode_px=0,
    )
    assert res is not None
    (cx, cy, cz), (dx, dy, dz) = res
    assert abs(cz - 1.0) < 1e-3
    # 20 px at z=1m, fx=500 → 0.04 m
    assert abs(dx - 0.04) < 5e-3
    assert abs(dy - 0.04) < 5e-3
    # Flat patch — z-span should clamp near min_size_m, not 2 m.
    assert dz < 0.05


def test_estimate_bbox_3d_from_mask_returns_none_on_empty_mask():
    depth = np.full((50, 50), 1000, dtype=np.uint16)
    mask = np.zeros((50, 50), dtype=bool)
    assert estimate_bbox_3d_from_mask(depth, mask, 500, 500, 25, 25) is None


def test_estimate_bbox_3d_from_mask_resizes_downscaled_mask():
    # ultralytics delivers masks at a downscaled resolution; the helper
    # nearest-neighbor-resizes to the depth resolution before sampling.
    depth = np.full((100, 100), 1000, dtype=np.uint16)
    small = np.zeros((25, 25), dtype=bool)
    small[10:15, 10:15] = True  # covers roughly (40..60, 40..60) after upscale
    res = estimate_bbox_3d_from_mask(depth, small, 500, 500, 50, 50)
    assert res is not None
    (_, _, cz), _ = res
    assert abs(cz - 1.0) < 1e-3


def test_estimate_bbox_3d_from_mask_rejects_depth_shadow_bleed():
    # Mask grabs a foreground patch at 1 m but leaks 5% of pixels onto
    # a 5 m "shadow" — percentile clip should keep the 1 m reading.
    depth = np.full((100, 100), 1000, dtype=np.uint16)
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    # Simulate bleed onto background pixels (still inside mask).
    depth[40:42, 40:42] = 5000
    res = estimate_bbox_3d_from_mask(depth, mask, 500, 500, 50, 50)
    assert res is not None
    (_, _, cz), (_, _, dz) = res
    # Center stays on the foreground plane; z-extent stays small.
    assert abs(cz - 1.0) < 0.1
    assert dz < 0.3


def test_estimate_bbox_3d_from_mask_erosion_removes_boundary_bleed():
    # Mask covers (40..60, 40..60). Background is at 3 m, foreground at 1 m.
    # The mask border ring has leaked onto the 3 m background (1-pixel halo);
    # interior is correct foreground. With erode_px=2 the halo is peeled off
    # and the center should sit on the 1 m plane instead of being pulled out.
    depth = np.full((100, 100), 3000, dtype=np.uint16)
    depth[42:58, 42:58] = 1000          # interior 1 m plane
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True            # mask overreaches by 2 px on each side
    with_erode = estimate_bbox_3d_from_mask(
        depth, mask, 500, 500, 50, 50, erode_px=2,
    )
    no_erode = estimate_bbox_3d_from_mask(
        depth, mask, 500, 500, 50, 50, erode_px=0,
    )
    assert with_erode is not None and no_erode is not None
    (_, _, cz_e), _ = with_erode
    (_, _, cz_n), _ = no_erode
    # After erosion only interior-1m pixels survive, so center is ~1.0 m.
    assert abs(cz_e - 1.0) < 0.05
    # Without erosion the boundary ring pulls the center off the foreground
    # plane; require a clearly larger z than the eroded case.
    assert cz_n > cz_e + 0.05


def test_estimate_bbox_3d_from_mask_erosion_preserves_small_objects():
    # A mask only 5 px wide would vanish if eroded 2 px — verify the function
    # keeps the raw mask in that case instead of returning None.
    depth = np.full((50, 50), 1000, dtype=np.uint16)
    mask = np.zeros((50, 50), dtype=bool)
    mask[22:28, 22:28] = True  # 6x6 — eroding 2 px leaves only 2x2=4 < 8
    res = estimate_bbox_3d_from_mask(
        depth, mask, 500, 500, 25, 25, erode_px=2,
    )
    assert res is not None
    (_, _, cz), _ = res
    assert abs(cz - 1.0) < 1e-3


# -------- Boolean mask erosion helper ---------------------------------

def test_erode_bool_mask_shrinks_10x10_by_2():
    m = np.zeros((20, 20), dtype=bool)
    m[5:15, 5:15] = True  # 10x10 True patch
    out = _erode_bool_mask(m, iterations=2)
    # After 2-px erosion a 10x10 square becomes a 6x6 square at (7:13, 7:13).
    assert out.sum() == 6 * 6
    assert out[7:13, 7:13].all()
    assert not out[6, 7]  # border row peeled
    assert not out[13, 7]


def test_erode_bool_mask_empties_3x3_by_2():
    m = np.zeros((10, 10), dtype=bool)
    m[4:7, 4:7] = True   # 3x3
    out = _erode_bool_mask(m, iterations=2)
    assert out.sum() == 0


def test_erode_bool_mask_zero_iterations_is_identity():
    m = np.zeros((10, 10), dtype=bool)
    m[3:8, 3:8] = True
    out = _erode_bool_mask(m, iterations=0)
    assert np.array_equal(out, m)


# -------- AABB 3D IoU --------------------------------------------------

def test_aabb_iou_identical_boxes_is_one():
    a = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert aabb_iou_3d(a[0], a[1], a[0], a[1]) == pytest.approx(1.0)


def test_aabb_iou_no_overlap_is_zero():
    a = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b = ((5.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert aabb_iou_3d(a[0], a[1], b[0], b[1]) == 0.0


def test_aabb_iou_touching_faces_is_zero():
    # Two unit boxes sharing only a face (zero-volume intersection).
    a = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b = ((1.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert aabb_iou_3d(a[0], a[1], b[0], b[1]) == 0.0


def test_aabb_iou_half_overlap():
    # Two unit cubes offset by 0.5 along x only.
    # inter = 0.5 * 1 * 1 = 0.5; union = 1 + 1 - 0.5 = 1.5; IoU = 1/3.
    a = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b = ((0.5, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert aabb_iou_3d(a[0], a[1], b[0], b[1]) == pytest.approx(1.0 / 3.0)


def test_aabb_iou_fully_enclosed():
    # Small box fully inside a large one — IoU = vol(small) / vol(large).
    big = ((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))      # vol 8
    small = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))    # vol 1
    assert aabb_iou_3d(big[0], big[1], small[0], small[1]) == pytest.approx(1.0 / 8.0)


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

from exploration_rearrangement.object_detector_node import (  # noqa: E402
    _bbox3d_line_list_points, _dedup_candidates_iou_3d,
)


def _cand(conf, name, center, size):
    """(conf, name, det_placeholder, cam_xyz, bbox) — bbox = (center, size)."""
    return (conf, name, None, center, (center, size))


def test_dedup_iou_drops_lower_conf_when_bboxes_overlap():
    # Two different-class candidates with the same AABB — weaker one dropped.
    a = _cand(0.9, 'blue_cup',  (1.0, 0.0, 0.5), (0.1, 0.1, 0.1))
    b = _cand(0.6, 'green_cup', (1.0, 0.0, 0.5), (0.1, 0.1, 0.1))
    kept = _dedup_candidates_iou_3d([a, b], iou_threshold=0.3)
    assert [k[1] for k in kept] == ['blue_cup']


def test_dedup_iou_keeps_both_when_bboxes_disjoint():
    a = _cand(0.8, 'blue_cup',  (1.0, 0.0, 0.5), (0.1, 0.1, 0.1))
    b = _cand(0.7, 'green_cup', (2.0, 2.0, 0.5), (0.1, 0.1, 0.1))
    kept = _dedup_candidates_iou_3d([a, b], iou_threshold=0.3)
    assert sorted(k[1] for k in kept) == ['blue_cup', 'green_cup']


def test_dedup_iou_is_order_independent():
    a = _cand(0.4, 'green_cup', (1.0, 0.0, 0.5), (0.1, 0.1, 0.1))
    b = _cand(0.9, 'blue_cup',  (1.0, 0.0, 0.5), (0.1, 0.1, 0.1))
    assert [k[1] for k in _dedup_candidates_iou_3d([a, b], 0.3)] == ['blue_cup']
    assert [k[1] for k in _dedup_candidates_iou_3d([b, a], 0.3)] == ['blue_cup']


def test_dedup_iou_threshold_one_keeps_everything():
    a = _cand(0.9, 'blue_cup',  (1.0, 0.0, 0.5), (0.1, 0.1, 0.1))
    b = _cand(0.6, 'green_cup', (1.0, 0.0, 0.5), (0.1, 0.1, 0.1))
    kept = _dedup_candidates_iou_3d([a, b], iou_threshold=1.0)
    assert len(kept) == 2


def test_dedup_iou_keeps_bboxless_candidates():
    # A candidate without a bbox has no geometry to compare — keep it.
    a = _cand(0.9, 'blue_cup', (1.0, 0.0, 0.5), (0.1, 0.1, 0.1))
    b = (0.6, 'green_cup', None, (5.0, 5.0, 0.5), None)
    kept = _dedup_candidates_iou_3d([a, b], iou_threshold=0.3)
    assert sorted(k[1] for k in kept) == ['blue_cup', 'green_cup']


# -------- BBox wireframe points ---------------------------------------

def test_bbox3d_line_list_points_shape():
    pts = _bbox3d_line_list_points((0.2, 0.4, 0.6))
    # 12 edges × 2 endpoints = 24 points.
    assert len(pts) == 24


def test_bbox3d_line_list_points_corner_extents():
    pts = _bbox3d_line_list_points((0.2, 0.4, 0.6))
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    assert min(xs) == pytest.approx(-0.1) and max(xs) == pytest.approx(0.1)
    assert min(ys) == pytest.approx(-0.2) and max(ys) == pytest.approx(0.2)
    assert min(zs) == pytest.approx(-0.3) and max(zs) == pytest.approx(0.3)


def test_bbox3d_line_list_points_only_uses_eight_corners():
    pts = _bbox3d_line_list_points((1.0, 1.0, 1.0))
    unique = {(round(p.x, 6), round(p.y, 6), round(p.z, 6)) for p in pts}
    assert len(unique) == 8


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
