"""Frontier extraction from a nav_msgs/OccupancyGrid."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from nav_msgs.msg import OccupancyGrid

FREE = 0
UNKNOWN = -1
OCCUPIED = 100


@dataclass
class Frontier:
    centroid_world: Tuple[float, float]
    size: int
    cells: List[Tuple[int, int]]


def occ_grid_to_array(grid: OccupancyGrid) -> np.ndarray:
    w = grid.info.width
    h = grid.info.height
    return np.asarray(grid.data, dtype=np.int16).reshape(h, w)


def grid_to_world(grid: OccupancyGrid, i: int, j: int) -> Tuple[float, float]:
    res = grid.info.resolution
    ox = grid.info.origin.position.x
    oy = grid.info.origin.position.y
    return ox + (j + 0.5) * res, oy + (i + 0.5) * res


def world_to_grid(grid: OccupancyGrid, x: float, y: float) -> Tuple[int, int]:
    res = grid.info.resolution
    ox = grid.info.origin.position.x
    oy = grid.info.origin.position.y
    j = int(np.floor((x - ox) / res))
    i = int(np.floor((y - oy) / res))
    return i, j


def extract_frontiers(
    grid: OccupancyGrid,
    min_cluster_size: int = 5,
    free_thresh: int = 50,
) -> List[Frontier]:
    data = occ_grid_to_array(grid)
    h, w = data.shape

    free_mask = (data >= 0) & (data < free_thresh)
    unknown_mask = data == UNKNOWN

    frontier_mask = np.zeros_like(free_mask, dtype=bool)
    frontier_mask[1:-1, 1:-1] = free_mask[1:-1, 1:-1] & (
        unknown_mask[0:-2, 1:-1]
        | unknown_mask[2:, 1:-1]
        | unknown_mask[1:-1, 0:-2]
        | unknown_mask[1:-1, 2:]
    )

    clusters = _flood_fill_clusters(frontier_mask)
    frontiers: List[Frontier] = []
    for cells in clusters:
        if len(cells) < min_cluster_size:
            continue
        ci = float(np.mean([c[0] for c in cells]))
        cj = float(np.mean([c[1] for c in cells]))
        wx, wy = grid_to_world(grid, int(ci), int(cj))
        frontiers.append(Frontier((wx, wy), len(cells), cells))
    return frontiers


def _flood_fill_clusters(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    clusters: List[List[Tuple[int, int]]] = []
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not visited[i, j]:
                stack = [(i, j)]
                cells: List[Tuple[int, int]] = []
                while stack:
                    ci, cj = stack.pop()
                    if ci < 0 or cj < 0 or ci >= h or cj >= w:
                        continue
                    if visited[ci, cj] or not mask[ci, cj]:
                        continue
                    visited[ci, cj] = True
                    cells.append((ci, cj))
                    stack.extend([
                        (ci + 1, cj), (ci - 1, cj),
                        (ci, cj + 1), (ci, cj - 1),
                        (ci + 1, cj + 1), (ci - 1, cj - 1),
                        (ci + 1, cj - 1), (ci - 1, cj + 1),
                    ])
                clusters.append(cells)
    return clusters


def score_frontier(
    frontier: Frontier,
    robot_xy: Tuple[float, float],
    alpha_dist: float = 1.0,
    beta_info: float = 0.05,
) -> float:
    dx = frontier.centroid_world[0] - robot_xy[0]
    dy = frontier.centroid_world[1] - robot_xy[1]
    dist = float(np.hypot(dx, dy))
    return alpha_dist * dist - beta_info * float(frontier.size)
