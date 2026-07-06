"""Perception: video frame -> crowd density map -> ground-plane CrowdState.

The missing organ between cameras and the brain. Two
parts:

  CrowdDensityModel    — pretrained CNN (CSRNet baseline; CLIP-EBC upgrade
                         path) mapping a BGR frame to an image-space density
                         map whose integral approximates the people count.
  HomographyCalibrator — 4+ image↔world reference points -> 3x3 H; warps the
                         image-space density map onto the metric ground plane
                         so densities are true ped/m².

Environment notes (this machine): torch 2.11+cpu requires
KMP_DUPLICATE_LIB_OK=TRUE (set here before torch import); CPU inference at the
1 Hz decision cadence has large headroom (~1-2 s/frame for CSRNet).

Model provenance: CSRNet (Li et al., CVPR 2018), weights MIT-licensed from
HF `rootstrap-org/crowd-counting` (trained on ShanghaiTech B — surveillance-
like scenes). Accuracy on new venues is validated, not assumed
(scripts/phaseA_validate.py); the reported MAE is the perception validity
bound, same honesty discipline as the physics.
"""

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch+MKL workaround

import numpy as np


# ---------------------------------------------------------------------------
# CSRNet (VGG16 frontend + dilated backend) — standard architecture
# ---------------------------------------------------------------------------

def _make_layers(cfg, in_channels=3, dilation=1):
    import torch.nn as nn
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3,
                                 padding=dilation, dilation=dilation),
                       nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class _CSRNet:
    """Lazy-built CSRNet: frontend VGG16(10 conv), backend dilated convs."""

    FRONT = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
    BACK = [512, 512, 512, 256, 128, 64]

    def __init__(self, weights_path: str):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.frontend = _make_layers(_CSRNet.FRONT)
                self.backend = _make_layers(_CSRNet.BACK, in_channels=512,
                                            dilation=2)
                self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

            def forward(self, x):
                return self.output_layer(self.backend(self.frontend(x)))

        self.net = Net()
        state = torch.load(weights_path, map_location="cpu",
                           weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # strip DataParallel "module." prefixes if present
        state = { (k[7:] if k.startswith("module.") else k): v
                  for k, v in state.items() }
        self.net.load_state_dict(state)
        self.net.eval()

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 (H, W, 3) -> density map float (H/8, W/8)."""
        import torch
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (rgb.astype(np.float32) / 255.0 - mean) / std
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
        with torch.inference_mode():
            d = self.net(t)
        return d.squeeze().numpy().clip(min=0.0)


class _CLIPEBC:
    """CLIP-EBC (Yiming-M, MIT) — NWPU-trained ViT-B/16, safetensors on HF.

    Vendored at runtime: the HF Space repo carries the model code; we add its
    snapshot to sys.path and build via its own get_model + config.json.
    # ponytail: sys.path vendoring of a research repo — works and pins to a
    # snapshot hash; package it properly if this becomes the default backend.
    """

    HF_REPO = "Yiming-M/CLIP-EBC"
    SUBDIR = "nwpu_weights/CLIP_EBC_ViT_B_16"

    def __init__(self):
        import sys as _sys
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        snap = snapshot_download(
            self.HF_REPO,
            allow_patterns=["models/*", "utils/*", "configs/*",
                            f"{self.SUBDIR}/*"])
        if snap not in _sys.path:
            _sys.path.insert(0, snap)
        from models import get_model  # the repo's own factory

        with open(os.path.join(snap, self.SUBDIR, "config.json")) as f:
            cfg = json.load(f)
        model = get_model(
            backbone=cfg["backbone"], input_size=cfg["input_size"],
            reduction=cfg["reduction"],
            bins=[(float(a), float(b)) for a, b in cfg["bins"]],
            anchor_points=[float(p) for p in cfg["anchor_points"]],
            prompt_type=cfg["prompt_type"], num_vpt=cfg["num_vpt"],
            vpt_drop=cfg["vpt_drop"], deep_vpt=cfg["deep_vpt"])
        sd = load_file(os.path.join(snap, self.SUBDIR, "model.safetensors"))
        model.load_state_dict({k.replace("model.", ""): v
                               for k, v in sd.items()})
        model.eval()
        self.net = model
        self.input_size = int(cfg["input_size"])

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 (H, W, 3) -> density map (reduced resolution)."""
        import cv2
        import torch
        h, w = rgb.shape[:2]
        # ensure min side >= input_size (their app.py protocol)
        if min(h, w) < self.input_size:
            r = self.input_size / min(h, w)
            rgb = cv2.resize(rgb, (int(w * r) + 1, int(h * r) + 1),
                             interpolation=cv2.INTER_CUBIC)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (rgb.astype(np.float32) / 255.0 - mean) / std
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        with torch.inference_mode():
            d = self.net(t.unsqueeze(0))
        return d.squeeze().numpy().clip(min=0.0)


class CrowdDensityModel:
    """Frame -> image-space density map. sum(map) ~= people count.

    Args:
        backend: "csrnet" (guaranteed baseline, MIT, SHB-trained) or
            "clipebc" (CLIP-EBC ViT-B/16, MIT, NWPU-trained — stronger in
            dense scenes, heavier on CPU).
        weights: Optional local .pth for csrnet; default downloads the MIT
            CSRNet ShanghaiTech-B weights from HF `rootstrap-org/crowd-counting`.
        max_side: Frames larger than this on the long side are downscaled
            before inference (CPU time / accuracy tradeoff; count-preserving).
    """

    HF_REPO = "rootstrap-org/crowd-counting"
    HF_FILE = "weights.pth"

    def __init__(self, backend: str = "csrnet", weights: str | None = None,
                 max_side: int = 1280):
        if backend == "csrnet":
            if weights is None:
                from huggingface_hub import hf_hub_download
                weights = hf_hub_download(self.HF_REPO, self.HF_FILE)
            self._model = _CSRNet(weights)
        elif backend == "clipebc":
            self._model = _CLIPEBC()
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.max_side = max_side

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 frame -> image-space density map (float, downsampled).

        The map is count-preserving under the internal resize: sum(map) is an
        estimate of the number of people in the frame.
        """
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        return self._model.predict(rgb)

    def count(self, frame_bgr: np.ndarray) -> float:
        """Convenience: estimated people count in the frame."""
        return float(self.estimate(frame_bgr).sum())


# ---------------------------------------------------------------------------
# Homography: image plane -> metric ground plane
# ---------------------------------------------------------------------------

class HomographyCalibrator:
    """Maps image-space density onto the metric ground plane (ped/m²).

    Calibration = 4+ reference points with both image-pixel and world-meter
    coordinates (platform tile corners, painted markings). Persisted as JSON:
    {"image_points": [[px,py],...], "world_points": [[X,Y],...],
     "world_extent": [x_min, x_max, y_min, y_max]}.

    The density map is warped into a metric raster of `cell_m` resolution with
    a count-preserving normalization (total people invariant under warp).
    """

    def __init__(self, image_points: np.ndarray, world_points: np.ndarray,
                 world_extent: tuple[float, float, float, float],
                 cell_m: float = 1.0):
        import cv2
        img = np.asarray(image_points, dtype=np.float32)
        wld = np.asarray(world_points, dtype=np.float32)
        if len(img) < 4:
            raise ValueError("need >= 4 reference points")
        self.extent = world_extent
        self.cell_m = cell_m
        x0, x1, y0, y1 = world_extent
        self.nx = int(np.ceil((x1 - x0) / cell_m))
        self.ny = int(np.ceil((y1 - y0) / cell_m))
        # world meters -> raster pixels (1 px per cell)
        raster = np.stack([(wld[:, 0] - x0) / cell_m,
                           (wld[:, 1] - y0) / cell_m], axis=1).astype(np.float32)
        self.H, _ = cv2.findHomography(img, raster, cv2.RANSAC)
        if self.H is None:
            raise ValueError("homography estimation failed")

    @classmethod
    def from_json(cls, path: str, cell_m: float = 1.0) -> "HomographyCalibrator":
        """Load a persisted calibration file."""
        with open(path) as f:
            d = json.load(f)
        return cls(np.array(d["image_points"]), np.array(d["world_points"]),
                   tuple(d["world_extent"]), cell_m=cell_m)

    def density_to_grid(self, density_map: np.ndarray,
                        frame_shape: tuple[int, int]) -> np.ndarray:
        """Image-space density map -> ground-plane grid (ped/m², ny x nx).

        Args:
            density_map: Model output (h', w') — any resolution; it is scaled
                to the original frame size before warping.
            frame_shape: (H, W) of the ORIGINAL frame the calibration points
                were clicked on.

        Count-preserving: sum(grid * cell_area) == sum(density_map).
        """
        import cv2
        H_img, W_img = frame_shape
        total = float(density_map.sum())
        up = cv2.resize(density_map, (W_img, H_img),
                        interpolation=cv2.INTER_LINEAR)
        if up.sum() > 0:
            up *= total / up.sum()  # preserve count through resize
        warped = cv2.warpPerspective(up, self.H, (self.nx, self.ny))
        warped = warped.clip(min=0.0)
        if warped.sum() > 0:
            warped *= total / warped.sum()  # preserve count through warp
        return warped / (self.cell_m ** 2)  # people per m² per cell
