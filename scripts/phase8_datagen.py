"""Phase 8 data generation: parameter sweep -> surrogate training corpus.

Sweeps release schedule x throat width x crowd state x blockage timing under
C2 (the validated screening proxy; ~40x cheaper than C4), plus a small C4
anchor subset for calibration. Each run appends one labeled row to
results/phase8/sweep.csv and persists a 1-Hz trajectory parquet (the Step-4
forecaster corpus).

Crash-safe and resumable: rows are appended per completed run; existing keys
are skipped on restart. Shardable for parallelism: --part K --parts N runs
every N-th combo starting at K.

Labels per run (from run_blockage): max_pinned_upstream (crowd-at-risk),
exposure_area_sec_3/_4, grid_peak, grid_t_4.0 (watch onset), final_active.

Usage:
  python scripts/phase8_datagen.py --part 0 --parts 2 --seeds 42 43   # shard A
  python scripts/phase8_datagen.py --part 1 --parts 2 --seeds 42 43   # shard B
  python scripts/phase8_datagen.py --anchors                          # C4 subset
"""

import argparse
import csv
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.scenarios.railway_release import ReleaseSchedule

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "blockage", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "phase3_blockage_screen.py"))
blockage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(blockage)

OUTDIR = os.path.join("results", "phase8")
CSV_PATH = os.path.join(OUTDIR, "sweep.csv")
NO_BLOCK = 10_000.0  # t_block beyond max_time = no blockage event

# Release schedules, all sum to N=150 (equal-N discipline).
SCHEDULES: dict[str, ReleaseSchedule] = {
    "burst": ReleaseSchedule(((0.0, 50), (3.0, 50), (6.0, 50))),
    "phased10": ReleaseSchedule(((0.0, 30), (10.0, 30), (20.0, 30),
                                 (30.0, 30), (40.0, 30))),
    "phased20": ReleaseSchedule(((0.0, 30), (20.0, 30), (40.0, 30),
                                 (60.0, 30), (80.0, 30))),
    "phased40": ReleaseSchedule(((0.0, 50), (40.0, 50), (80.0, 50))),
    "trickle": ReleaseSchedule(tuple((10.0 * k, 15) for k in range(10))),
}
assert all(s.total == 150 for s in SCHEDULES.values())

WIDTHS = [1.5, 2.4, 3.66, 6.0]
CROWDS = ["calm", "surge"]          # calm: v0=1.34, calibrated Weidmann;
                                    # surge: v0=2.0 + governor released
T_BLOCKS = [8.0, 15.0, NO_BLOCK]
# Expansion grid (2026-07-07): denser event-timing coverage — the watch-onset
# target failed to learn at n=100 crossings; more blockage cells fix the
# label scarcity. Used with --expand.
T_BLOCKS_EXPANDED = [5.0, 8.0, 11.0, 15.0, 20.0, 30.0, NO_BLOCK]

FIELDS = ["schedule", "W", "crowd", "t_block", "config", "seed",
          "max_pinned_upstream", "exposure_area_sec_3", "exposure_area_sec_4",
          "grid_peak", "grid_t_4.0", "t_blockage_formed", "final_active",
          "n_total", "wall_s"]


def key_of(row: dict) -> tuple:
    return (row["schedule"], float(row["W"]), row["crowd"],
            float(row["t_block"]), row["config"], int(row["seed"]))


def done_keys() -> set:
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH, newline="") as f:
        return {key_of(r) for r in csv.DictReader(f)}


def run_one(sched_name: str, W: float, crowd: str, t_block: float,
            config: str, seed: int) -> dict:
    v0 = 1.34 if crowd == "calm" else 2.0
    overrides = None if crowd == "calm" else blockage.SURGE_DESIRE_OVERRIDES
    r = blockage.run_blockage(
        SCHEDULES[sched_name], v0, 150, t_block, seed=seed, W=W,
        config=config, max_time=150.0, overrides=overrides, mode="fallen",
        traj_every=100, outdir=OUTDIR,
        tag_prefix=f"{sched_name}-{crowd}",  # unambiguous parquet identity
    )
    return {
        "schedule": sched_name, "W": W, "crowd": crowd, "t_block": t_block,
        "config": config, "seed": seed,
        **{k: r[k] for k in ["max_pinned_upstream", "exposure_area_sec_3",
                             "exposure_area_sec_4", "grid_peak", "grid_t_4.0",
                             "t_blockage_formed", "final_active", "n_total",
                             "wall_s"]},
    }


def append_row(row: dict) -> None:
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--anchors", action="store_true",
                    help="run the C4 anchor subset instead of the C2 grid")
    ap.add_argument("--expand", action="store_true",
                    help="use the denser event-timing grid (T_BLOCKS_EXPANDED)")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    if args.anchors:
        # C4 anchors: one seed, surge crowd, blockage @ 8 s, all schedules x
        # two widths -> 10 runs for C2->C4 calibration.
        combos = [(s, W, "surge", 8.0, "C4", 42)
                  for s in SCHEDULES for W in (3.66, 6.0)]
    else:
        tblocks = T_BLOCKS_EXPANDED if args.expand else T_BLOCKS
        grid = list(itertools.product(SCHEDULES, WIDTHS, CROWDS, tblocks))
        combos = [(s, W, c, tb, "C2", seed)
                  for seed in args.seeds for (s, W, c, tb) in grid]
    combos = combos[args.part::args.parts]  # sharding applies to anchors too

    done = done_keys()
    todo = [c for c in combos
            if (c[0], float(c[1]), c[2], float(c[3]), c[4], int(c[5])) not in done]
    print(f"{len(combos)} combos in shard, {len(combos) - len(todo)} done, "
          f"{len(todo)} to run", flush=True)

    for i, (s, W, c, tb, cfg, seed) in enumerate(todo):
        row = run_one(s, W, c, tb, cfg, seed)
        append_row(row)
        print(f"[{i + 1}/{len(todo)}] {s} W={W} {c} tb={tb} {cfg} seed={seed}: "
              f"pinned={row['max_pinned_upstream']} "
              f"a3={row['exposure_area_sec_3']} ({row['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
