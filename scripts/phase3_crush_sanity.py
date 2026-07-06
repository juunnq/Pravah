"""Phase 3: crush-sanity experiment — does the baseline surge reach the lethal
regime at the FOB throat, and does the phased release avoid it?

Runs the two frozen release schedules under C4 with calibrated
Weidmann and Voronoi steering density, logs positions, then post-hoc samples the
FOB zone at the 1 s detector cadence (§4.2):

  - peak FOB density  = max per-agent Voronoi density among agents in the FOB box
    (§4.3), plus an occupancy cross-check (agents-in-box / box area);
  - first crossings of 4.0 (ORCA-fade onset, lead-time anchor) and 5.5 (rho_crit);
  - warning lead time = gap between the 4.0 and 5.5 crossings (§4.4 metric 2);
  - crush verdict: density >= 5.5 sustained for >= SUSTAIN_TICKS consecutive
    ticks (operationalization of §4.4 metric 3's "sustained interval" —
    an analysis choice, since the spec leaves "sustained" undefined).

This is an EXPERIMENT script (Phase 3), not the Phase-4 detector component.
Writes per-tick CSVs to results/phase3/ (gitignored) and prints a summary.

Usage:
  python scripts/phase3_crush_sanity.py [--pilot] [--seeds 42 43 44]
"""

import argparse
import csv
import os
import sys
import time as walltime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.density.voronoi import VoronoiDensityEstimator
from sim.scenarios.railway_platform import RailwayPlatformScenario
from sim.scenarios.railway_release import (
    BASELINE_SCHEDULE,
    INTERVENTION_SCHEDULE,
    run_surge,
)

RHO_FADE = 4.0    # ORCA-fade onset / lead-time anchor [repo: params.yaml:6]
RHO_CRIT = 5.5    # crush threshold [repo: params.yaml:6]
SUSTAIN_TICKS = 3  # consecutive 1s ticks >= RHO_CRIT to call "crush" [analysis choice]
CADENCE_STEPS = 100  # 1.0 s at dt=0.01 (frozen decision cadence)
VOR_CAP = 100.0   # VoronoiDensityEstimator max_density cap — flag ticks at cap


def analyze(sim, scenario) -> tuple[list[dict], dict]:
    """Post-hoc FOB-zone density analysis at 1 s cadence from the position log.

    Args:
        sim: Completed Simulation with log_positions data.
        scenario: The scenario (for fob_zone and domain_polygon).

    Returns:
        (per-tick rows, summary dict).
    """
    x0, x1, y0, y1 = scenario.fob_zone()
    box_area = (x1 - x0) * (y1 - y0)
    estimator = VoronoiDensityEstimator(domain=scenario.domain_polygon())

    rows = []
    for k, (t, idx, pos, _vel) in enumerate(sim._position_log):
        if k % CADENCE_STEPS != 0:
            continue
        dens = estimator.estimate(pos)
        in_box = (
            (pos[:, 0] >= x0) & (pos[:, 0] <= x1)
            & (pos[:, 1] >= y0) & (pos[:, 1] <= y1)
        )
        n_box = int(np.sum(in_box))
        peak = float(np.max(dens[in_box])) if n_box else 0.0
        occ = n_box / box_area
        rows.append({
            "t": round(t, 2), "n_active": len(idx), "n_fob": n_box,
            "peak_voronoi": round(peak, 3), "occupancy": round(occ, 3),
            "at_cap": peak >= VOR_CAP - 1.0,
        })

    # Crossings on the peak-Voronoi series (spec's headline signal).
    t_fade = next((r["t"] for r in rows if r["peak_voronoi"] >= RHO_FADE), None)
    t_crit = next((r["t"] for r in rows if r["peak_voronoi"] >= RHO_CRIT), None)
    # Sustained-crush verdict: >= RHO_CRIT for SUSTAIN_TICKS consecutive ticks.
    crush, t_crush, streak = False, None, 0
    for r in rows:
        streak = streak + 1 if r["peak_voronoi"] >= RHO_CRIT else 0
        if streak >= SUSTAIN_TICKS:
            crush, t_crush = True, r["t"]
            break
    # Occupancy cross-check crossings (robust to Voronoi cell artifacts).
    t_crit_occ = next((r["t"] for r in rows if r["occupancy"] >= RHO_CRIT), None)

    summary = {
        "peak_voronoi_max": max((r["peak_voronoi"] for r in rows), default=0.0),
        "occupancy_max": max((r["occupancy"] for r in rows), default=0.0),
        "t_cross_4.0": t_fade,
        "t_cross_5.5": t_crit,
        "t_cross_5.5_occupancy": t_crit_occ,
        "lead_time_s": (t_crit - t_fade) if (t_fade is not None and t_crit is not None) else None,
        "crush_sustained": crush,
        "t_crush_sustained": t_crush,
        "ticks_at_voronoi_cap": sum(1 for r in rows if r["at_cap"]),
        "final_n": sim.state.n, "final_active": sim.state.n_active,
        "sim_time_end": round(sim.time, 1),
    }
    return rows, summary


def run_condition(name: str, schedule, seed: int, max_time: float,
                  outdir: str, throat_width: float = 3.66,
                  speed_mean: float = 1.34) -> dict:
    """Run one (schedule, seed) condition under C4 and analyze it."""
    scenario = RailwayPlatformScenario(throat_width=throat_width,
                                       speed_mean=speed_mean)
    est = VoronoiDensityEstimator(domain=scenario.domain_polygon())
    t0 = walltime.time()
    sim = run_surge(
        scenario, schedule, config="C4", seed=seed,
        density_estimator=est, max_time=max_time,
        max_steps=int(max_time / 0.01) + 10, log_positions=True,
    )
    wall = walltime.time() - t0
    rows, summary = analyze(sim, scenario)
    summary.update({"condition": name, "seed": seed, "wall_s": round(wall, 1)})

    os.makedirs(outdir, exist_ok=True)
    # Persist full trajectories: re-analysis (taper diagnostics, plots) and
    # future ML training data (the physics-as-teacher path) both need them.
    sim.write_logs(trajectory_path=os.path.join(
        outdir, f"{name}_seed{seed}_traj.parquet"))
    path = os.path.join(outdir, f"{name}_seed{seed}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="short cost-calibration run (baseline, 10 s sim)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--baseline-time", type=float, default=120.0)
    ap.add_argument("--intervention-time", type=float, default=150.0)
    ap.add_argument("--condition", choices=["baseline", "intervention", "both"],
                    default="both")
    ap.add_argument("--throat-width", type=float, default=3.66,
                    help="FOB throat width W (m); 3.66=legacy, 1.4-2.8=Elphinstone-"
                         "audited usable widths, 6.0=CAG safe target")
    ap.add_argument("--speed-mean", type=float, default=1.34,
                    help="mean desired speed (m/s); 1.34=FZJ calm walking, "
                         "1.8-2.5=hurried surge crowd [ASSUMPTION, Helbing 2000]")
    args = ap.parse_args()
    outdir = os.path.join("results", "phase3")

    if args.pilot:
        s = run_condition("pilot_baseline", BASELINE_SCHEDULE, 42, 10.0, outdir)
        print(f"PILOT: 10 s sim (1000 steps, 150 agents by step 600) "
              f"took {s['wall_s']} s wall -> "
              f"{1000 / s['wall_s']:.0f} steps/s")
        print(f"  extrapolated 120 s baseline: ~{120 * s['wall_s'] / 10 / 60:.1f} min")
        print(f"  peak so far: voronoi={s['peak_voronoi_max']}, occ={s['occupancy_max']}")
        return

    conditions = [
        ("baseline", BASELINE_SCHEDULE, args.baseline_time),
        ("intervention", INTERVENTION_SCHEDULE, args.intervention_time),
    ]
    if args.condition != "both":
        conditions = [c for c in conditions if c[0] == args.condition]

    wtag = "" if args.throat_width == 3.66 else f"_w{args.throat_width}"
    vtag = "" if args.speed_mean == 1.34 else f"_v{args.speed_mean}"
    for seed in args.seeds:
        for name, sched, mt in conditions:
            s = run_condition(name + wtag + vtag, sched, seed, mt, outdir,
                              throat_width=args.throat_width,
                              speed_mean=args.speed_mean)
            print(f"[{name} seed={seed}] wall={s['wall_s']}s "
                  f"peak_vor={s['peak_voronoi_max']} occ_max={s['occupancy_max']} "
                  f"t4.0={s['t_cross_4.0']} t5.5={s['t_cross_5.5']} "
                  f"lead={s['lead_time_s']} crush={s['crush_sustained']} "
                  f"(t={s['t_crush_sustained']}) cap_ticks={s['ticks_at_voronoi_cap']} "
                  f"end={s['sim_time_end']}s active={s['final_active']}")


if __name__ == "__main__":
    main()
