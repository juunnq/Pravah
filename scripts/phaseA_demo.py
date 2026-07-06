"""Phase A3: end-to-end alarm pipeline on REAL footage.

frame sequence/video -> CrowdDensityModel -> HomographyCalibrator ->
VideoCCTVProvider -> ThresholdDetector + forecaster -> annotated video +
event log. Optionally compares per-frame counts against ground truth (Mall
dataset format) — the model-transfer bound on THIS footage.

Usage (Mall dataset):
  python scripts/phaseA_demo.py --frames <mall>/frames --calib onboard/examples/mall_calib.json \
      --gt <mall>/mall_gt.mat --limit 300 --out results/phaseA/mall_demo.mp4

Notes:
  - The example Mall calibration is [ASSUMPTION]-tagged (no surveyed reference
    points exist for that camera); RELATIVE density evolution and alarm logic
    are meaningful, absolute ped/m² carries the tag. Real deployments use
    surveyed points via scripts/phaseA_calibrate.py.
  - Thresholds are injected (detector defaults are the railway bands); tune
    per venue during onboarding.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BANDS = {"clear": (63, 185, 80), "watch": (34, 153, 210),
         "amber": (62, 136, 240), "critical": (73, 81, 248)}  # BGR
RANK = {"clear": 0, "watch": 1, "amber": 2, "critical": 3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="dir of sequential .jpg frames")
    ap.add_argument("--calib", required=True, help="calibration JSON")
    ap.add_argument("--gt", help="optional mall_gt.mat for count comparison")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default=os.path.join("results", "phaseA", "demo.mp4"))
    ap.add_argument("--zone", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"),
                    help="watch zone in world meters (default: central band)")
    args = ap.parse_args()

    import cv2
    from sim.detector import ThresholdDetector
    from sim.forecaster import RateOfRiseForecaster
    from sim.perception import CrowdDensityModel, HomographyCalibrator
    from sim.providers import VideoCCTVProvider

    frames = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))[:args.limit]
    if not frames:
        sys.exit(f"no frames in {args.frames}")

    cal = HomographyCalibrator.from_json(args.calib)
    with open(args.calib) as f:
        calib_note = json.load(f).get("note", "")
    model = CrowdDensityModel()
    prov = VideoCCTVProvider(model, cal)
    x0, x1, y0, y1 = cal.extent
    zone = tuple(args.zone) if args.zone else (
        x0 + (x1 - x0) * 0.2, x1 - (x1 - x0) * 0.2, y0, y1)
    det = ThresholdDetector(zone, prov)
    fc = RateOfRiseForecaster(zone, prov, horizon=30.0, window=30)

    gt_counts = None
    if args.gt:
        from scipy.io import loadmat
        gt_counts = loadmat(args.gt)["count"].ravel()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    first = cv2.imread(frames[0])
    H, W = first.shape[:2]
    panel_w = 220
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 8,
                         (W + panel_w, H))

    latch = ("clear", None)
    rows = []
    t0 = time.time()
    for i, path in enumerate(frames):
        frame = cv2.imread(path)
        state = prov.state_from_frame(float(i), frame)
        r = det.update(state)
        f = fc.update(state)
        pred_count = float(state.density_grid.sum() * cal.cell_m ** 2)
        if RANK[r.band] > RANK[latch[0]]:
            latch = (r.band, i)

        # ---- annotate: original frame + side panel with ground-plane grid
        canvas = np.zeros((H, W + panel_w, 3), dtype=np.uint8)
        canvas[:, :W] = frame
        g = state.density_grid
        gh = int(panel_w * g.shape[0] / g.shape[1])
        heat = cv2.applyColorMap(
            (np.clip(g / 5.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        canvas[40:40 + gh, W:W + panel_w] = cv2.resize(
            heat[::-1], (panel_w, gh), interpolation=cv2.INTER_NEAREST)
        band, since = latch
        color = BANDS[band]
        cv2.rectangle(canvas, (0, 0), (W + panel_w, 28), (20, 20, 20), -1)
        label = (f"t={i}s  count~{pred_count:.0f}"
                 + (f"  GT={gt_counts[i]}" if gt_counts is not None
                    and i < len(gt_counts) else "")
                 + f"  peak={r.zone_peak:.1f}/m2  [{band.upper()}"
                 + (f" since {since}s]" if since is not None else "]"))
        cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)
        cv2.putText(canvas, "ground-plane density", (W + 6, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, f"forecast+30s: {f.zone_peak_pred:.1f}",
                    (W + 6, 60 + gh), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 200, 100), 1)
        vw.write(canvas)

        rows.append({"t": i, "pred_count": round(pred_count, 1),
                     "gt": int(gt_counts[i]) if gt_counts is not None
                     and i < len(gt_counts) else None,
                     "zone_peak": r.zone_peak, "band": r.band,
                     "forecast_peak": f.zone_peak_pred})
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[{i + 1}/{len(frames)}] {el:.0f}s", flush=True)

    vw.release()
    df = pd.DataFrame(rows)
    csv_path = os.path.splitext(args.out)[0] + "_events.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {args.out} + {csv_path}")
    if gt_counts is not None:
        m = df.dropna(subset=["gt"])
        # NB: m["gt"], never m.gt — the column name collides with DataFrame.gt()
        mae = (m["pred_count"] - m["gt"]).abs().mean()
        bias = (m["pred_count"] - m["gt"]).mean()
        print(f"COUNT TRANSFER BOUND on this footage: n={len(m)} "
              f"MAE={mae:.2f} bias={bias:+.2f} "
              f"(GT mean {m['gt'].mean():.1f})")
    if calib_note:
        print(f"calibration note: {calib_note}")


if __name__ == "__main__":
    main()
