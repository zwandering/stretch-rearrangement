"""HSV color segmentation + RGB-D 3D localization."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ColorSpec:
    name: str
    hsv_low: Tuple[int, int, int]
    hsv_high: Tuple[int, int, int]
    hsv_low_2: Optional[Tuple[int, int, int]] = None
    hsv_high_2: Optional[Tuple[int, int, int]] = None
    min_area_px: int = 400


@dataclass
class Detection2D:
    label: str
    center_px: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    area_px: int
    mask: np.ndarray


def segment_color(bgr: np.ndarray, spec: ColorSpec) -> Optional[Detection2D]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(spec.hsv_low), np.array(spec.hsv_high))
    if spec.hsv_low_2 is not None and spec.hsv_high_2 is not None:
        mask2 = cv2.inRange(
            hsv, np.array(spec.hsv_low_2), np.array(spec.hsv_high_2)
        )
        mask = cv2.bitwise_or(mask, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(c))
    if area < spec.min_area_px:
        return None
    x, y, w, h = cv2.boundingRect(c)
    M = cv2.moments(c)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return Detection2D(spec.name, (cx, cy), (x, y, w, h), area, mask)


def segment_all(bgr: np.ndarray, specs: List[ColorSpec]) -> List[Detection2D]:
    results = []
    for spec in specs:
        det = segment_color(bgr, spec)
        if det is not None:
            results.append(det)
    return results


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


def annotate(bgr: np.ndarray, dets: List[Detection2D]) -> np.ndarray:
    out = bgr.copy()
    for d in dets:
        x, y, w, h = d.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, d.center_px, 4, (0, 0, 255), -1)
        cv2.putText(
            out, f"{d.label} ({d.area_px}px)",
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
    return out


def load_color_specs(cfg: Dict) -> List[ColorSpec]:
    specs = []
    for entry in cfg.get('objects', []):
        specs.append(ColorSpec(
            name=entry['name'],
            hsv_low=tuple(entry['hsv_low']),
            hsv_high=tuple(entry['hsv_high']),
            hsv_low_2=tuple(entry['hsv_low_2']) if 'hsv_low_2' in entry else None,
            hsv_high_2=tuple(entry['hsv_high_2']) if 'hsv_high_2' in entry else None,
            min_area_px=int(entry.get('min_area_px', 400)),
        ))
    return specs
