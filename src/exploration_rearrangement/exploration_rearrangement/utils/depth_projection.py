"""Pinhole back-projection from a depth image to 3D camera-frame points."""

from typing import Optional, Tuple

import numpy as np


def pixel_to_camera(
    depth_img: np.ndarray,
    u: int,
    v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    window: int = 5,
    depth_scale: float = 1e-3,
) -> Optional[Tuple[float, float, float]]:
    """Back-project pixel (u,v) through a depth image to (X,Y,Z) in camera frame.

    Uses the median depth in a small window around (u,v) to reject speckle noise
    and pixels that landed on a depth shadow. ``depth_scale`` converts integer
    depth units to metres (RealSense 16UC1 is millimetres → 1e-3).
    """
    h, w = depth_img.shape
    u0 = max(u - window, 0)
    u1 = min(u + window + 1, w)
    v0 = max(v - window, 0)
    v1 = min(v + window + 1, h)
    patch = depth_img[v0:v1, u0:u1].astype(np.float32)
    valid = patch[(patch > 0) & np.isfinite(patch)]
    if valid.size == 0:
        return None
    z_raw = float(np.median(valid))
    z = z_raw * depth_scale if depth_img.dtype != np.float32 else z_raw
    if z <= 0.05 or z > 5.0:
        return None
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z


def estimate_bbox_3d(
    depth_img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 1e-3,
    min_depth_m: float = 0.05,
    max_depth_m: float = 5.0,
    min_size_m: float = 0.02,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Estimate a rough axis-aligned 3D bbox (in camera frame) from a 2D bbox.

    Returns ``((cx,cy,cz), (dx,dy,dz))`` — center and metric size — or ``None``
    if the 2D region has insufficient valid depth. Image-plane extent is
    scaled by the median in-bbox depth to get x/y size; z size comes from the
    5–95 percentile span of in-bbox depths, which rejects the sparse shadows
    that sit at the silhouette boundary. Rectangular — includes background
    pixels; prefer ``estimate_bbox_3d_from_mask`` when a seg mask is available.
    """
    H, W = depth_img.shape[:2]
    x0 = max(int(x), 0)
    x1 = min(int(x + w), W)
    y0 = max(int(y), 0)
    y1 = min(int(y + h), H)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = depth_img[y0:y1, x0:x1].astype(np.float32)
    if depth_img.dtype != np.float32:
        patch = patch * depth_scale
    mask = (patch > min_depth_m) & (patch < max_depth_m) & np.isfinite(patch)
    valid = patch[mask]
    if valid.size < 4:
        return None
    z_med = float(np.median(valid))
    z_lo = float(np.quantile(valid, 0.05))
    z_hi = float(np.quantile(valid, 0.95))
    px_w = x1 - x0
    px_h = y1 - y0
    dx = max(px_w * z_med / fx, min_size_m)
    dy = max(px_h * z_med / fy, min_size_m)
    dz = max(z_hi - z_lo, min_size_m)
    cu = 0.5 * (x0 + x1)
    cv = 0.5 * (y0 + y1)
    cx_m = (cu - cx) * z_med / fx
    cy_m = (cv - cy) * z_med / fy
    return (cx_m, cy_m, z_med), (dx, dy, dz)


def _resize_bool_mask(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """Nearest-neighbor resize of a 2D boolean/numeric mask to ``(H, W)``.
    Used to match ultralytics' downscaled mask output to the depth image
    resolution without pulling in cv2 here.
    """
    if mask.shape == (H, W):
        return mask.astype(bool)
    mh, mw = mask.shape[:2]
    row_idx = (np.arange(H) * mh / H).astype(np.int32)
    col_idx = (np.arange(W) * mw / W).astype(np.int32)
    return mask[row_idx[:, None], col_idx[None, :]].astype(bool)


def _erode_bool_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Morphological erosion with a 3x3 square structuring element.

    Shrinks each True region inward by ``iterations`` pixels. Pure-numpy
    shift-and-AND — avoids a scipy.ndimage dependency. Image borders are
    implicitly False (padding), so edge-adjacent True pixels are peeled off
    just like any other boundary pixel.
    """
    if iterations <= 0:
        return mask.astype(bool)
    m = mask.astype(bool)
    for _ in range(int(iterations)):
        padded = np.zeros((m.shape[0] + 2, m.shape[1] + 2), dtype=bool)
        padded[1:-1, 1:-1] = m
        m = (
            padded[0:-2, 0:-2] & padded[0:-2, 1:-1] & padded[0:-2, 2:] &
            padded[1:-1, 0:-2] & padded[1:-1, 1:-1] & padded[1:-1, 2:] &
            padded[2:,   0:-2] & padded[2:,   1:-1] & padded[2:,   2:]
        )
    return m


def estimate_bbox_3d_from_mask(
    depth_img: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 1e-3,
    min_depth_m: float = 0.05,
    max_depth_m: float = 5.0,
    min_size_m: float = 0.02,
    pct_lo: float = 2.5,
    pct_hi: float = 97.5,
    erode_px: int = 2,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Back-project the masked pixels to 3D points in camera frame, then fit
    an axis-aligned 3D bbox to that point cloud.

    Much tighter than the rectangular variant because background (table,
    floor, wall) pixels are excluded by the silhouette. Uses percentile
    clipping on both depth and each projected axis to reject:
      - mask edge-bleed pixels that straddle foreground/background depth,
      - depth sensor speckle on the silhouette boundary.
    """
    H, W = depth_img.shape[:2]
    m = _resize_bool_mask(mask, H, W)
    if erode_px > 0:
        eroded = _erode_bool_mask(m, erode_px)
        # Small/thin objects would vanish — keep the original mask in that case.
        if np.count_nonzero(eroded) >= 8:
            m = eroded
    vs, us = np.where(m)
    if us.size < 8:
        return None
    z_raw = depth_img[vs, us].astype(np.float32)
    if depth_img.dtype != np.float32:
        z_raw = z_raw * depth_scale
    ok = (z_raw > min_depth_m) & (z_raw < max_depth_m) & np.isfinite(z_raw)
    if np.count_nonzero(ok) < 8:
        return None
    z = z_raw[ok]
    u = us[ok].astype(np.float32)
    v = vs[ok].astype(np.float32)
    # First-pass depth percentile clip — kills mask-edge background bleed.
    z_lo_d, z_hi_d = np.percentile(z, [pct_lo, pct_hi])
    keep = (z >= z_lo_d) & (z <= z_hi_d)
    if np.count_nonzero(keep) >= 8:
        z = z[keep]; u = u[keep]; v = v[keep]
    # Back-project the surviving pixels into camera-frame 3D.
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    # Per-axis percentile for a robust AABB (hard min/max bleeds outliers).
    x_lo, x_hi = np.percentile(x, [pct_lo, pct_hi])
    y_lo, y_hi = np.percentile(y, [pct_lo, pct_hi])
    z_lo2, z_hi2 = np.percentile(z, [pct_lo, pct_hi])
    dx = max(float(x_hi - x_lo), min_size_m)
    dy = max(float(y_hi - y_lo), min_size_m)
    dz = max(float(z_hi2 - z_lo2), min_size_m)
    return (
        (float(0.5 * (x_lo + x_hi)),
         float(0.5 * (y_lo + y_hi)),
         float(0.5 * (z_lo2 + z_hi2))),
        (dx, dy, dz),
    )


def aabb_iou_3d(
    center_a: Tuple[float, float, float],
    size_a: Tuple[float, float, float],
    center_b: Tuple[float, float, float],
    size_b: Tuple[float, float, float],
) -> float:
    """IoU of two axis-aligned 3D bboxes given center+size. Returns 0.0 if
    either box has non-positive volume or no overlap."""
    a_min = [center_a[i] - 0.5 * size_a[i] for i in range(3)]
    a_max = [center_a[i] + 0.5 * size_a[i] for i in range(3)]
    b_min = [center_b[i] - 0.5 * size_b[i] for i in range(3)]
    b_max = [center_b[i] + 0.5 * size_b[i] for i in range(3)]
    inter = 1.0
    for i in range(3):
        lo = max(a_min[i], b_min[i])
        hi = min(a_max[i], b_max[i])
        d = hi - lo
        if d <= 0.0:
            return 0.0
        inter *= d
    vol_a = float(size_a[0] * size_a[1] * size_a[2])
    vol_b = float(size_b[0] * size_b[1] * size_b[2])
    union = vol_a + vol_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)
