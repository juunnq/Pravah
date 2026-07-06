"""Core helper functions: force validation, normalization, speed clamping."""

import numpy as np
import warnings


def check_forces(forces: np.ndarray, name: str) -> np.ndarray:
    """Validate force array. MUST be called as last line of every force function.

    Args:
        forces: Force array of shape (N, 2).
        name: Human-readable name for error messages.

    Returns:
        Validated (and possibly clamped) force array.

    Raises:
        ValueError: If NaN values are detected.
    """
    if np.any(np.isnan(forces)):
        bad = np.unique(np.where(np.isnan(forces))[0])
        raise ValueError(f"NaN in {name} forces at agents {bad[:5].tolist()}")
    mask = np.abs(forces) > 1e6
    if np.any(mask):
        bad = np.unique(np.where(mask)[0])
        warnings.warn(f"Clamped extreme {name} forces at agents {bad[:5].tolist()}")
        forces = np.clip(forces, -1e6, 1e6)
    return np.array(forces)


def safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize vector(s). Returns zero for sub-epsilon magnitudes.

    Args:
        v: Vector of shape (2,) or array of shape (N, 2).
        eps: Minimum magnitude threshold.

    Returns:
        Unit vector(s) of same shape, or zeros where magnitude < eps.
    """
    if v.ndim == 1:
        m = np.linalg.norm(v)
        return v / m if m > eps else np.zeros_like(v)
    mag = np.linalg.norm(v, axis=1, keepdims=True)
    safe_mag = np.maximum(mag, eps)
    return np.where(mag > eps, v / safe_mag, 0.0)


def clamp_speed(velocities: np.ndarray, max_speeds: np.ndarray) -> np.ndarray:
    """Per-agent speed clamping.

    Args:
        velocities: Velocity array of shape (N, 2).
        max_speeds: Maximum speed per agent, shape (N,).

    Returns:
        Clamped velocity array of shape (N, 2).
    """
    speeds = np.linalg.norm(velocities, axis=1)
    scale = np.where(speeds > max_speeds, max_speeds / np.maximum(speeds, 1e-8), 1.0)
    return velocities * scale[:, None]
