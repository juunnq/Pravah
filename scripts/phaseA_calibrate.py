"""Camera calibration tool: reference points -> homography JSON.

Onboarding procedure (per camera):
  1. Grab one frame from the camera (`--frame img.jpg --grab-only` shows it
     with a pixel grid so you can read off coordinates).
  2. Identify >=4 ground-plane reference points visible in the frame whose
     real-world positions you know (platform tile corners, painted markings,
     pillar bases). More points = more robust (RANSAC).
  3. Write them into a points file (JSON):
       {"image_points": [[px,py], ...], "world_points": [[X,Y], ...],
        "world_extent": [x_min, x_max, y_min, y_max]}
  4. Run this tool to validate + persist the calibration, and (optionally)
     render a verification overlay: the world grid re-projected onto the
     frame. If the projected grid does not lie flat on the floor in the
     overlay, the points are wrong — fix them before trusting any density.

Usage:
  python scripts/phaseA_calibrate.py --points cam1_points.json --out cam1_calib.json [--frame frame.jpg]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True, help="points JSON (see docstring)")
    ap.add_argument("--out", required=True, help="calibration JSON to write")
    ap.add_argument("--frame", help="optional frame image for the verification overlay")
    ap.add_argument("--cell-m", type=float, default=1.0)
    args = ap.parse_args()

    with open(args.points) as f:
        d = json.load(f)

    from sim.perception import HomographyCalibrator
    cal = HomographyCalibrator(np.array(d["image_points"]),
                               np.array(d["world_points"]),
                               tuple(d["world_extent"]), cell_m=args.cell_m)

    # Round-trip check: each reference image point -> world -> compare.
    import cv2
    img_pts = np.array(d["image_points"], dtype=np.float32).reshape(-1, 1, 2)
    raster = cv2.perspectiveTransform(img_pts, cal.H).reshape(-1, 2)
    world = raster * cal.cell_m + np.array([d["world_extent"][0],
                                            d["world_extent"][2]])
    err = np.linalg.norm(world - np.array(d["world_points"]), axis=1)
    print(f"reprojection error per point (m): {np.round(err, 3).tolist()}")
    print(f"mean {err.mean():.3f} m, max {err.max():.3f} m "
          f"({'OK' if err.max() < 0.5 else 'SUSPECT — check your points'})")

    with open(args.out, "w") as f:
        json.dump({**d, "cell_m": args.cell_m}, f, indent=1)
    print(f"wrote {args.out}")

    if args.frame:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        frame = cv2.cvtColor(cv2.imread(args.frame), cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(frame)
        # re-project world grid lines onto the image (inverse homography)
        Hinv = np.linalg.inv(cal.H)
        x0, x1, y0, y1 = d["world_extent"]
        for wx in np.arange(x0, x1 + 0.01, 2.0):
            pts = np.array([[[(wx - x0) / cal.cell_m, (wy - y0) / cal.cell_m]]
                            for wy in np.linspace(y0, y1, 20)], dtype=np.float32)
            proj = cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)
            ax.plot(proj[:, 0], proj[:, 1], color="cyan", lw=0.7, alpha=0.7)
        for wy in np.arange(y0, y1 + 0.01, 2.0):
            pts = np.array([[[(wx - x0) / cal.cell_m, (wy - y0) / cal.cell_m]]
                            for wx in np.linspace(x0, x1, 20)], dtype=np.float32)
            proj = cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)
            ax.plot(proj[:, 0], proj[:, 1], color="cyan", lw=0.7, alpha=0.7)
        ip = np.array(d["image_points"])
        ax.scatter(ip[:, 0], ip[:, 1], color="red", s=40, zorder=5,
                   label="reference points")
        ax.legend()
        ax.set_title("calibration overlay — grid must lie flat on the floor")
        out_img = os.path.splitext(args.out)[0] + "_overlay.png"
        fig.savefig(out_img, dpi=110, bbox_inches="tight")
        print(f"wrote {out_img}")


if __name__ == "__main__":
    main()
