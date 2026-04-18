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
