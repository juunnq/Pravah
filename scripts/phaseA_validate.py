"""Phase A1 validation: CSRNet counts vs ShanghaiTech ground truth.

Runs the perception model over ShanghaiTech Part B test (surveillance-like
street scenes, 316 images, head-point annotations) and reports MAE/RMSE
overall and per count band. This number is the PERCEPTION VALIDITY BOUND —
published CSRNet MAE on SHB is ~10.6; large deviation means an integration
bug (stop the line), moderate deviation is the honest measured bound.

Usage:
  python scripts/phaseA_validate.py --data <path-to>/ShanghaiTech/part_B/test_data
  [--limit N]   # quick pass on first N images
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.viz.style import save_figure, set_style


def gt_count(mat_path: str) -> int:
    """Head count from a ShanghaiTech GT .mat (standard nested structure)."""
    from scipy.io import loadmat
    m = loadmat(mat_path)
    return int(m["image_info"][0, 0][0, 0][0].shape[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help=".../part_B/test_data (contains images/ and ground_truth/)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backend", default="csrnet", choices=["csrnet", "clipebc"])
    args = ap.parse_args()

    import cv2
    from sim.perception import CrowdDensityModel

    images = sorted(glob.glob(os.path.join(args.data, "images", "*.jpg")))
    if args.limit:
        images = images[:args.limit]
    if not images:
        sys.exit(f"no images under {args.data}/images")

    model = CrowdDensityModel(backend=args.backend)
    rows = []
    t0 = time.time()
    for i, img_path in enumerate(images):
        name = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(args.data, "ground_truth", f"GT_{name}.mat")
        if not os.path.exists(gt_path):
            continue
        frame = cv2.imread(img_path)
        pred = float(model.estimate(frame).sum())
        true = gt_count(gt_path)
        rows.append({"image": name, "true": true, "pred": round(pred, 1),
                     "abs_err": round(abs(pred - true), 1)})
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[{i + 1}/{len(images)}] {el:.0f}s "
                  f"running MAE={np.mean([r['abs_err'] for r in rows]):.2f}",
                  flush=True)

    df = pd.DataFrame(rows)
    mae = df.abs_err.mean()
    rmse = float(np.sqrt(((df.pred - df.true) ** 2).mean()))
    print(f"\nOVERALL: n={len(df)}  MAE={mae:.2f}  RMSE={rmse:.2f}  "
          f"(published CSRNet SHB MAE ~10.6)")
    bands = [(0, 50), (50, 100), (100, 200), (200, 10 ** 9)]
    for lo, hi in bands:
        m = df[(df.true >= lo) & (df.true < hi)]
        if len(m):
            print(f"  band {lo}-{hi if hi < 1e9 else '+'}: n={len(m)} "
                  f"MAE={m.abs_err.mean():.2f} "
                  f"rel={100 * (m.abs_err / m.true.clip(lower=1)).mean():.1f}%")

    outdir = os.path.join("results", "phaseA")
    os.makedirs(outdir, exist_ok=True)
    suffix = "" if args.backend == "csrnet" else f"_{args.backend}"
    df.to_csv(os.path.join(outdir, f"perception_eval{suffix}.csv"), index=False)

    set_style()
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.scatter(df.true, df.pred, s=10, alpha=0.5, color="#0072b2")
    lim = (0, max(df.true.max(), df.pred.max()) * 1.05)
    ax.plot(lim, lim, ls="--", lw=1, color="gray")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("ground truth count"); ax.set_ylabel("CSRNet estimate")
    ax.set_title(f"ShanghaiTech B test [{args.backend}]: MAE {mae:.1f} "
                 f"(n={len(df)})", fontsize=9)
    print(save_figure(fig, f"fig_perception_scatter{suffix}"))
    print(os.path.join(outdir, f"perception_eval{suffix}.csv"))


if __name__ == "__main__":
    main()
