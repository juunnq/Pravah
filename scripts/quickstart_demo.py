"""Quickstart: generate a small demo dataset and launch the dashboard.

For fresh clones (simulation artifacts are gitignored). Runs a fast C2
burst-vs-phased pair with the fallen-blockage event (~2-4 min total on a
laptop CPU), persists 1-Hz trajectories where the dashboard looks for them,
then serves the control room.

Usage:  python scripts/quickstart_demo.py          # generate + serve
        python scripts/quickstart_demo.py --no-serve
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "blockage", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "phase3_blockage_screen.py"))
blockage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(blockage)

from sim.scenarios.railway_release import ReleaseSchedule

OUTDIR = os.path.join("results", "quickstart")
SCHEDULES = {
    "burst": ReleaseSchedule(((0.0, 50), (3.0, 50), (6.0, 50))),
    "phased": ReleaseSchedule(((0.0, 30), (20.0, 30), (40.0, 30),
                               (60.0, 30), (80.0, 30))),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args()

    for arm, sched in SCHEDULES.items():
        tag = f"blockfallen_{arm}_v2.0_tb10.0_C2_seed42_traj.parquet"
        if os.path.exists(os.path.join(OUTDIR, tag)):
            print(f"{arm}: already generated", flush=True)
            continue
        print(f"generating {arm} demo run (~1-2 min)...", flush=True)
        r = blockage.run_blockage(
            sched, 2.0, 150, 10.0, seed=42, W=3.66, config="C2",
            max_time=150.0, overrides=blockage.SURGE_DESIRE_OVERRIDES,
            mode="fallen", traj_every=100, outdir=OUTDIR,
        )
        print(f"  {arm}: pinned={r['max_pinned_upstream']} "
              f"grid_peak={r['grid_peak']}", flush=True)

    if not args.no_serve:
        subprocess.run([sys.executable,
                        os.path.join("dashboard", "serve.py")])


if __name__ == "__main__":
    main()
