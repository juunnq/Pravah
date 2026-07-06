"""World geometry: walls, obstacles, and distance computations."""

from dataclasses import dataclass, field
import numpy as np
from sim.core.helpers import safe_normalize


@dataclass
class Wall:
    """A line-segment wall defined by start and end points.

    Attributes:
        start: Start point, shape (2,).
        end: End point, shape (2,).
    """

    start: np.ndarray
    end: np.ndarray


@dataclass
class Obstacle:
    """A polygonal obstacle defined by its vertices.

    Attributes:
        vertices: Ordered vertices, shape (M, 2).
    """

    vertices: np.ndarray


@dataclass
class World:
    """Simulation world containing walls and obstacles.

    Attributes:
        walls: List of Wall segments.
        obstacles: List of Obstacle polygons.
    """

    walls: list[Wall]
    obstacles: list[Obstacle] = field(default_factory=list)


def point_to_segment_distance(
    point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute distance from a point to a line segment.

    Args:
        point: Query point, shape (2,).
        seg_start: Segment start, shape (2,).
        seg_end: Segment end, shape (2,).

    Returns:
        Tuple of (distance, closest_point, normal_toward_point).
    """
    ab = seg_end - seg_start
    ap = point - seg_start
    ab_sq = np.dot(ab, ab)

    # Degenerate segment (start == end)
    if ab_sq < 1e-12:
        diff = point - seg_start
        dist = float(np.linalg.norm(diff))
        if dist < 1e-8:
            normal = np.array([0.0, 1.0])
        else:
            normal = diff / dist
        return dist, seg_start.copy(), normal

    t = np.clip(np.dot(ap, ab) / ab_sq, 0.0, 1.0)
    closest = seg_start + t * ab
    diff = point - closest
    dist = float(np.linalg.norm(diff))

    if dist < 1e-8:
        # Point is on segment — normal is perpendicular to segment
        perp = np.array([-ab[1], ab[0]])
        normal = safe_normalize(perp)
    else:
        normal = diff / dist

    return dist, closest, normal


def agents_to_walls(
    positions: np.ndarray, walls: list[Wall]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute distances and normals from all agents to all walls.

    Vectorized over agents, loops over walls.

    Args:
        positions: Agent positions, shape (N, 2).
        walls: List of W Wall objects.

    Returns:
        Tuple of (distances (N, W), normals (N, W, 2)).
    """
    n = len(positions)
    w = len(walls)
    distances = np.zeros((n, w))
    normals = np.zeros((n, w, 2))

    for j, wall in enumerate(walls):
        ab = wall.end - wall.start
        ab_sq = np.dot(ab, ab)

        if ab_sq < 1e-12:
            # Degenerate wall
            diff = positions - wall.start[None, :]
            dist = np.linalg.norm(diff, axis=1)
            safe_dist = np.maximum(dist, 1e-8)
            nrm = diff / safe_dist[:, None]
            distances[:, j] = dist
            normals[:, j] = nrm
            continue

        ap = positions - wall.start[None, :]
        t = np.clip(np.dot(ap, ab) / ab_sq, 0.0, 1.0)
        closest = wall.start[None, :] + t[:, None] * ab[None, :]
        diff = positions - closest
        dist = np.linalg.norm(diff, axis=1)

        # Normal: diff / dist, with fallback to segment perpendicular
        safe_dist = np.maximum(dist, 1e-8)
        nrm = diff / safe_dist[:, None]

        # For agents very close to segment, use perpendicular
        on_seg = dist < 1e-8
        if np.any(on_seg):
            perp = safe_normalize(np.array([-ab[1], ab[0]]))
            nrm[on_seg] = perp

        distances[:, j] = dist
        normals[:, j] = nrm

    return distances, normals
