"""Phase 3c: cheap C2 screening of pressure levers (v0, N, W) for crush onset.

C4 runs cost ~15 min each; ORCA is faded out (w_o -> 0) above rho=4, so for
SCREENING which lever combinations approach the crush band we use C2 (SFM+TTC,
no ORCA LP, ~40x faster). Any configuration that crosses is then CONFIRMED
under C4 before being reported as a result.

# ponytail: C2 screening biases slightly denser than C4 (no ORCA yielding in
# the approach phase); acceptable for screening, which is why C4 confirmation
# is mandatory before any number is quoted.

Screening uses a one-shot release of all N at t=0 (the frozen 3-batch baseline
spreads over 6 s; the difference is negligible for peak density). Trajectories
are analyzed in memory and NOT persisted (screening throughput).

Usage:
  python scripts/phase3_screen_levers.py            # full grid
  python scripts/phase3_screen_levers.py --quick    # coarse corners first
"""

import argparse
import csv
import itertools
import os
import sys
import time as walltime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.density.voronoi import VoronoiDensityEstimator
from sim.scenarios.railway_platform import RailwayPlatformScenario
from sim.scenarios.railway_release import ReleaseSchedule, run_surge

RHO_CRIT = 5.5
CADENCE = 100  # sample every 100 steps = 1 s


PROXIMITY_BOX = (12.0, 19.5, 3.0, 17.0)  # crowd converged at FOB approach
                                          # [ASSUMPTION - surge convergence]


def screen_one(W: float, v0: float, n: int, seed: int = 42,
               max_time: float = 120.0, config: str = "C2",
               spawn_area: tuple | None = None,
               overrides: dict | None = None) -> dict:
    """Run one lever combination and return zone-peak summary."""
    scenario = RailwayPlatformScenario(throat_width=W, speed_mean=v0,
                                       spawn_area=spawn_area)
    est = VoronoiDensityEstimator(domain=scenario.domain_polygon())
    t0 = walltime.time()
    sim = run_surge(
        scenario, ReleaseSchedule(((0.0, n),)), config=config, seed=seed,
        density_estimator=est, max_time=max_time,
        max_steps=int(max_time / 0.01) + 10, log_positions=True,
        param_overrides=overrides,
    )
    wall = walltime.time() - t0

    peak = {"hold": 0.0, "taper": 0.0, "throat": 0.0}
    t_at = {}
    t_cross = None
    for k, (t, _idx, pos, _vel) in enumerate(sim._position_log):
        if k % CADENCE != 0 or len(pos) < 4:
            continue
        d = est.estimate(pos)
        x = pos[:, 0]
        ind = x <= 52.0  # exclude out-of-domain agents (Voronoi cap artifacts)
        d, x = d[ind], x[ind]
        for z, m in (("hold", x < 20), ("taper", (x >= 20) & (x < 24)),
                     ("throat", (x >= 24) & (x <= 32))):
            if m.any():
                v = float(d[m].max())
                if v > peak[z]:
                    peak[z], t_at[z] = v, t
        if t_cross is None and max(peak["taper"], peak["throat"]) >= RHO_CRIT:
            t_cross = t

    return {
        "W": W, "v0": v0, "n": n, "config": config, "seed": seed,
        "spawn": "proximity" if spawn_area else "frozen",
        "peak_hold": round(peak["hold"], 2),
        "peak_taper": round(peak["taper"], 2),
        "peak_throat": round(peak["throat"], 2),
        "t_taper_peak": round(t_at.get("taper", -1), 0),
        "t_cross_5.5": t_cross,
        "crossed": t_cross is not None,
        "wall_s": round(wall, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="corners only: (W,v0,N) extremes")
    ap.add_argument("--proximity", action="store_true",
                    help="spawn crowd converged at the FOB approach "
                         "(PROXIMITY_BOX) instead of the frozen mid-concourse box")
    ap.add_argument("--out", default="screen_levers.csv")
    args = ap.parse_args()

    spawn = PROXIMITY_BOX if args.proximity else None
    if args.proximity:
        # mildest-first: find the LOWEST-pressure proximity combo that crosses
        combos = list(itertools.product([3.66, 1.5], [2.0, 2.5], [150, 200]))
    elif args.quick:
        combos = [(1.5, 3.0, 200), (1.5, 2.0, 200), (3.66, 3.0, 200),
                  (1.5, 3.0, 150)]
    else:
        combos = list(itertools.product([3.66, 2.4, 1.5], [2.0, 2.5, 3.0],
                                        [150, 200]))

    outpath = os.path.join("results", "phase3", args.out)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    rows = []
    for W, v0, n in combos:
        r = screen_one(W, v0, n, spawn_area=spawn)
        rows.append(r)
        print(f"[{r['spawn']}] W={W} v0={v0} N={n}: taper={r['peak_taper']} "
              f"throat={r['peak_throat']} crossed={r['crossed']} "
              f"({r['wall_s']}s)", flush=True)

    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {outpath}")
    crossers = [r for r in rows if r["crossed"]]
    print(f"crossers: {len(crossers)}/{len(rows)}")
    for r in crossers:
        print(f"  W={r['W']} v0={r['v0']} N={r['n']} -> "
              f"taper {r['peak_taper']} at t={r['t_taper_peak']}")


if __name__ == "__main__":
    main()
