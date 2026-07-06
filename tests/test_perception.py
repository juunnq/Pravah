"""Tests for the perception layer (Phase A1/A2).

Homography tests are pure math (no model, no downloads). The density-model
test is skipped unless the CSRNet weights are already in the local HF cache
(never downloads during tests).
"""

import numpy as np
import pytest

from sim.perception import HomographyCalibrator


def _identityish_calibrator(cell_m=1.0):
    """Camera that sees a 20x10 m ground area as a 200x100 px image (10 px/m)."""
    image_pts = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=float)
    world_pts = np.array([[0, 0], [20, 0], [20, 10], [0, 10]], dtype=float)
    return HomographyCalibrator(image_pts, world_pts, (0.0, 20.0, 0.0, 10.0),
                                cell_m=cell_m)


# 1. A blob of known mass at a known image spot lands in the right world cell.
def test_homography_blob_lands_correctly():
    cal = _identityish_calibrator()
    dmap = np.zeros((100, 200))  # image-space density (H, W)
    dmap[50, 150] = 5.0          # 5 people at px (150, 50) -> world (15, 5)
    grid = cal.density_to_grid(dmap, frame_shape=(100, 200))
    assert grid.shape == (10, 20)
    iy, ix = np.unravel_index(np.argmax(grid), grid.shape)
    assert (ix, iy) == (15, 5)


# 2. Count preservation through resize + warp.
def test_homography_count_preserved():
    cal = _identityish_calibrator()
    rng = np.random.default_rng(0)
    dmap = rng.random((25, 50)) * 0.2   # low-res model output (1/4 scale)
    total = dmap.sum()
    grid = cal.density_to_grid(dmap, frame_shape=(100, 200))
    assert grid.sum() * cal.cell_m ** 2 == pytest.approx(total, rel=1e-3)


# 3. Perspective case: oblique camera still maps corners correctly.
def test_homography_oblique():
    # Trapezoid image of the same 20x10 ground plane (far edge compressed).
    image_pts = np.array([[40, 90], [160, 90], [120, 20], [80, 20]], dtype=float)
    world_pts = np.array([[0, 0], [20, 0], [20, 10], [0, 10]], dtype=float)
    cal = HomographyCalibrator(image_pts, world_pts, (0.0, 20.0, 0.0, 10.0))
    dmap = np.zeros((100, 200))
    dmap[90, 40] = 3.0   # near-left corner -> world (0, 0) cell
    grid = cal.density_to_grid(dmap, frame_shape=(100, 200))
    iy, ix = np.unravel_index(np.argmax(grid), grid.shape)
    assert (ix, iy) == (0, 0)
    assert grid.sum() == pytest.approx(3.0, rel=1e-3)


# 4. Fewer than 4 points rejected.
def test_homography_needs_four_points():
    with pytest.raises(ValueError):
        HomographyCalibrator(np.zeros((3, 2)), np.zeros((3, 2)), (0, 1, 0, 1))


# 5. Density model (only if weights already cached — never downloads in tests).
def _weights_cached() -> bool:
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download("rootstrap-org/crowd-counting", "weights.pth",
                        local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _weights_cached(), reason="CSRNet weights not cached")
def test_density_model_smoke():
    from sim.perception import CrowdDensityModel
    m = CrowdDensityModel()
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)  # blank gray
    d = m.estimate(frame)
    assert d.ndim == 2
    assert d.min() >= 0.0
    assert float(d.sum()) < 20.0  # near-zero people in a blank frame


# 6. VideoCCTVProvider drives the detector end-to-end with a fake model —
#    the full camera-side seam, no downloads, no CNN.
def test_video_provider_detector_end_to_end():
    from sim.detector import ThresholdDetector
    from sim.providers import VideoCCTVProvider

    class FakeModel:
        """Puts 6 'people' at image px (150, 50) -> world (15, 5)."""
        def estimate(self, frame_bgr):
            d = np.zeros((100, 200))
            d[50, 150] = 6.0
            return d

    cal = _identityish_calibrator()
    prov = VideoCCTVProvider(FakeModel(), cal)
    det = ThresholdDetector((10.0, 20.0, 0.0, 10.0), prov)

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    for t in range(3):
        r = det.update(prov.state_from_frame(float(t), frame))
    assert r.zone_peak == pytest.approx(6.0, rel=1e-2)
    assert r.band == "critical"
    assert r.crush  # 3 consecutive ticks >= 5.5 -> latched
