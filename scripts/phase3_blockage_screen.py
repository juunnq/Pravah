"""Phase 3d: blockage-event screening — the true crush aetiology.

The motivating events (NDLS Feb-2025, Elphinstone 2017) tipped when flow
STOPPED while inflow continued (fall on stairs / halted crowd), not because a
throat was merely narrow. This script inserts a gate (two wall segments
closing the throat at x=GATE_X) at t=T_BLOCK into an otherwise frozen-spec
scenario and measures whether accumulation crosses the crush band.

No physics constants are touched; no engine code is modified. The gate is a
pure geometry event appended to world.walls by this script's own step loop.
# ponytail: the step loop duplicates run_surge's ~20 lines because run_surge
# owns a closed loop; unify if a third variant ever appears.

Usage:
  python scripts/phase3_blockage_screen.py [--v0 1.34] [--n 150] [--t-block 25]
"""

import argparse
import os
import sys
import time as walltime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.core.simulation import Simulation
from sim.core.world import Wall
from sim.density.voronoi import VoronoiDensityEstimator
from sim.scenarios.railway_platform import RailwayPlatformScenario
from sim.scenarios.railway_release import (
    ReleaseSchedule,
    RAILWAY_PARAM_OVERRIDES,
    _batch_seeds,
)

RHO_CRIT = 5.5
GATE_X = 28.0  # mid-throat blockage plane


SURGE_DESIRE_OVERRIDES = {
    # [ASSUMPTION - surge state] Weidmann governor released: Weidmann (1993)
    # encodes calm-walking compliance (slow down in density); crush rear ranks
    # cannot perceive front density and keep pressing (Fruin's "craze").
    # gamma=50 ~= Helbing-2000 constant-desire panic model; rho_max=10 = motion
    # ceases at the empirical fatal-pack density (~8-11 ped/m2 in real crushes).
    "weidmann_gamma": 50.0,
    "weidmann_rho_max": 10.0,
}


N_FALLEN = 6  # bodies in the fall pile (~3 m of a 3.66 m throat; edge gaps real)


DANGER_ZONE = (20.0, 32.0, 3.0, 17.0)  # taper+throat: exposure/pinned metrics
PIN_X = (20.0, 28.0)  # "pressure column": upstream of the pile plane


def run_blockage(schedule: ReleaseSchedule, v0: float, n_total: int,
                 t_block: float, seed: int = 42, W: float = 3.66,
                 config: str = "C2", max_time: float = 150.0,
                 reopen_after: float | None = None,
                 overrides: dict | None = None,
                 mode: str = "wall",
                 traj_every: int = 1,
                 outdir: str = os.path.join("results", "phase3"),
                 tag_prefix: str | None = None) -> dict:
    """Run a surge with a mid-throat gate closing at t_block; zone-peak summary.

    Args:
        schedule: Release schedule (first batch = initial population).
        v0: Mean desired speed (1.34 calm / ~2.0 hurried).
        n_total: Expected total agents (schedule.total; asserted).
        t_block: Gate-closure time (s).
        seed: Master seed.
        W: Throat width (m).
        config: Steering config for screening ("C2") or confirmation ("C4").
        max_time: Simulation horizon (s).
        reopen_after: If set, remove the gate this many seconds after closure.

    Returns:
        Summary dict with zone peaks, crossing times, crush verdict.
    """
    assert schedule.total == n_total
    scenario = RailwayPlatformScenario(throat_width=W, speed_mean=v0)
    est = VoronoiDensityEstimator(domain=scenario.domain_polygon())

    batches = schedule.batches
    scenario.n_agents = batches[0][1]
    merged = dict(RAILWAY_PARAM_OVERRIDES)
    if overrides:
        merged.update(overrides)
    sim = Simulation.from_scenario(
        scenario, config, seed=seed, density_estimator=est,
        param_overrides=merged,
    )
    sim.log_positions = True
    dt = sim.params.get("dt", 0.01)
    release_steps = schedule.release_steps(dt)
    seeds = _batch_seeds(seed, len(batches))
    released = {0}

    half = scenario.throat_width / 2.0
    # Thick barrier: 8 wall lines 0.05 m apart (0.35 m band > agent radius, so
    # an agent inside the band overlaps >=5 lines at once). Sparser gates creep-
    # leak under sustained surge chain-pressure (soft force walls: observed 9
    # agents through a 4-line/0.1 m gate; ~65 through a single line).
    gate = [Wall(np.array([GATE_X + k * 0.05, 10.0 - half]),
                 np.array([GATE_X + k * 0.05, 10.0 + half])) for k in range(8)]
    block_step = int(round(t_block / dt))
    reopen_step: int | None = None  # set relative to the ACTUAL close step
    close_step_actual: int | None = None
    t_gate_closed: float | None = None
    gate_on = False
    fallen_ids: np.ndarray | None = None  # mode="fallen": pinned agent ids
    fallen_pos: np.ndarray | None = None

    t0 = walltime.time()
    max_steps = int(max_time / dt) + 10
    while sim.step_count < max_steps and sim.time < max_time:
        if scenario.is_complete(sim.state, sim.time):
            break
        for bi in range(1, len(batches)):
            if bi not in released and sim.step_count >= release_steps[bi]:
                sim.inject_agents(batches[bi][1], seed=seeds[bi],
                                  spawn_area=scenario.spawn_area,
                                  goal=scenario.goal, avoid_overlap=True,
                                  speed_mean=v0)
                released.add(bi)
        if mode == "wall":
            if not gate_on and sim.step_count >= block_step:
                # Close only when the gate band is momentarily clear — a barrier
                # materializing on top of agents embeds them in the walls
                # (observed: cap-level density artifacts + agents expelled
                # through). Caveat: at surge flow the band may never clear.
                act = sim.state.active
                xs = sim.state.positions[act, 0]
                band_clear = not np.any((xs > GATE_X - 0.35) & (xs < GATE_X + 0.7))
                if band_clear:
                    sim.world.walls.extend(gate)
                    gate_on = True
                    t_gate_closed = sim.time
                    close_step_actual = sim.step_count
                    if reopen_after is not None:
                        reopen_step = close_step_actual + int(round(reopen_after / dt))
            if gate_on and reopen_step is not None and sim.step_count >= reopen_step:
                for g in gate:
                    sim.world.walls.remove(g)
                gate_on = False
        else:  # mode == "fallen": the blockage IS the crowd (NDLS/Elphinstone)
            if fallen_ids is None and sim.step_count >= block_step:
                act = sim.state.active_indices
                xs = sim.state.positions[act, 0]
                cand = act[(xs > 25.0) & (xs < 31.0)]
                if len(cand) >= N_FALLEN:
                    order = np.argsort(np.abs(sim.state.positions[cand, 0] - GATE_X))
                    fallen_ids = cand[order[:N_FALLEN]]
                    fallen_pos = sim.state.positions[fallen_ids].copy()
                    t_gate_closed = sim.time
                    close_step_actual = sim.step_count
                    if reopen_after is not None:
                        reopen_step = close_step_actual + int(round(reopen_after / dt))
            if (fallen_ids is not None and reopen_step is not None
                    and sim.step_count >= reopen_step):
                fallen_ids = None  # pile recovers; stop pinning
        sim.step()
        if fallen_ids is not None:
            # Pin fallen bodies in place: they remain contact obstacles.
            sim.state.positions[fallen_ids] = fallen_pos
            sim.state.velocities[fallen_ids] = 0.0
    wall = walltime.time() - t0

    # Persist trajectories (Phase-3 lesson: re-analysis is free, re-running isn't).
    os.makedirs(outdir, exist_ok=True)
    wtag = "" if W == 3.66 else f"_w{W}"  # default width keeps legacy filenames
    # tag_prefix (e.g. the actual schedule name) prevents filename collisions:
    # the burst/phased fallback is ambiguous across schedule variants (observed:
    # 240 sweep runs -> only 96 surviving parquets, last-writer-wins).
    stem = tag_prefix or ("burst" if len(batches) <= 3 else "phased")
    tag = (f"block{mode}_{stem}_v{v0}"
           f"_tb{t_block}_{config}{wtag}_seed{seed}")
    traj_path = os.path.join(outdir, f"{tag}_traj.parquet")
    if traj_every > 1:
        # Persist a downsampled log (e.g. 1 Hz at traj_every=100) — 100x
        # smaller, sufficient for detector-cadence analysis and ML windows.
        # In-memory analyses below still use the full-rate log.
        full_log = sim._position_log
        sim._position_log = full_log[::traj_every]
        sim.write_logs(trajectory_path=traj_path)
        sim._position_log = full_log
    else:
        sim.write_logs(trajectory_path=traj_path)

    # Gate-integrity check: count agents that CROSSED the gate plane while the
    # gate was closed (position log carries ids; compare consecutive samples).
    leaked_ids: set[int] = set()
    prev: dict[int, float] = {}
    for k, (t, idx, pos, _vel) in enumerate(sim._position_log):
        if k % 100 != 0:
            continue
        step = k + 1
        closed = (close_step_actual is not None
                  and step >= close_step_actual
                  and (reopen_step is None or step < reopen_step))
        cur = {int(a): float(p[0]) for a, p in zip(idx, pos)}
        if closed:
            for a, x_now in cur.items():
                x_prev = prev.get(a)
                if x_prev is not None and x_prev <= GATE_X < x_now:
                    leaked_ids.add(a)
        prev = cur
    leaked = len(leaked_ids)

    # Grid-based zone analysis (CANONICAL: the deployment/CCTV measure; Voronoi
    # is unreliable above ~4-5 ped/m2 in packed confined geometry — observed
    # 48-98 readings at contact packing that grid counts read as 3-5).
    from sim.providers import SimulationStateProvider
    prov = SimulationStateProvider(scenario, cell_size=1.0)
    giy, gix = prov.zone_cells(scenario.fob_zone())
    diy, dix = prov.zone_cells(DANGER_ZONE)
    grid_peak, g4, g55, gstreak, g_crush = 0.0, None, None, 0, None
    area_sec_3 = area_sec_4 = 0  # exposure: danger-zone cells >=3/>=4, summed over 1 s ticks
    max_pinned = 0               # crowd-at-risk: peak count upstream of the pile plane
    for k, (t, _idx, pos, _vel) in enumerate(sim._position_log):
        if k % 100 != 0 or len(pos) < 4:
            continue
        grid = prov.sample(t, pos).density_grid
        zone = grid[giy, gix]
        zmax = float(zone.max()) if zone.size else 0.0
        grid_peak = max(grid_peak, zmax)
        if g4 is None and zmax >= 4.0:
            g4 = t
        if g55 is None and zmax >= RHO_CRIT:
            g55 = t
        gstreak = gstreak + 1 if zmax >= RHO_CRIT else 0
        if gstreak >= 3 and g_crush is None:
            g_crush = t
        danger = grid[diy, dix]
        area_sec_3 += int((danger >= 3.0).sum())
        area_sec_4 += int((danger >= 4.0).sum())
        pinned = int(np.sum((pos[:, 0] >= PIN_X[0]) & (pos[:, 0] <= PIN_X[1])))
        max_pinned = max(max_pinned, pinned)

    # Voronoi zone analysis (SUPPLEMENTARY) at 1 s cadence; out-of-domain
    # agents and cap-level readings (>=99: degenerate cells) excluded.
    peak = {"hold": 0.0, "taper": 0.0, "throat": 0.0}
    t_at = {}
    t4, t55, t_crush, streak = None, None, None, 0
    cap_readings = 0
    for k, (t, _idx, pos, _vel) in enumerate(sim._position_log):
        if k % 100 != 0 or len(pos) < 4:
            continue
        d = est.estimate(pos)
        x = pos[:, 0]
        ind = x <= 52.0
        d, x = d[ind], x[ind]
        cap = d >= 99.0
        cap_readings += int(np.sum(cap))
        d, x = d[~cap], x[~cap]
        tick_max = 0.0
        for z, m in (("hold", x < 20), ("taper", (x >= 20) & (x < 24)),
                     ("throat", (x >= 24) & (x <= 32))):
            if m.any():
                v = float(d[m].max())
                if v > peak[z]:
                    peak[z], t_at[z] = v, t
                if z in ("taper", "throat"):
                    tick_max = max(tick_max, v)
        if t4 is None and tick_max >= 4.0:
            t4 = t
        if t55 is None and tick_max >= RHO_CRIT:
            t55 = t
        streak = streak + 1 if tick_max >= RHO_CRIT else 0
        if streak >= 3 and t_crush is None:
            t_crush = t

    return {
        "schedule": "burst" if len(batches) <= 3 else "phased",
        "v0": v0, "W": W, "config": config, "seed": seed,
        "t_block": t_block, "reopen_after": reopen_after,
        "peak_hold": round(peak["hold"], 2),
        "peak_taper": round(peak["taper"], 2),
        "peak_throat": round(peak["throat"], 2),
        "t_taper_peak": round(t_at.get("taper", -1), 0),
        "t_throat_peak": round(t_at.get("throat", -1), 0),
        "t_cross_4.0": t4, "t_cross_5.5": t55,
        "lead_time_s": (t55 - t4) if (t4 is not None and t55 is not None) else None,
        "crush_sustained": t_crush is not None, "t_crush": t_crush,
        # CANONICAL deployment-measure (1 m2 grid) metrics:
        "grid_peak": round(grid_peak, 2),
        "grid_t_4.0": g4, "grid_t_5.5": g55, "grid_crush_t": g_crush,
        # Phase-8 label metrics (danger zone = taper+throat):
        "max_pinned_upstream": max_pinned,
        "exposure_area_sec_3": area_sec_3,
        "exposure_area_sec_4": area_sec_4,
        "mode": mode, "t_blockage_formed": t_gate_closed,
        "passed_blockage_while_active": leaked,  # wall: leak; fallen: squeezed past (real)
        "cap_readings_excluded": cap_readings,
        "final_active": sim.state.n_active, "n_total": sim.state.n,
        "wall_s": round(wall, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", type=float, default=1.34)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--t-block", type=float, default=25.0)
    ap.add_argument("--config", default="C2")
    ap.add_argument("--schedule", choices=["burst", "phased"], default="burst")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-time", type=float, default=150.0)
    ap.add_argument("--surge-desire", action="store_true",
                    help="apply SURGE_DESIRE_OVERRIDES (Weidmann governor "
                         "released per Helbing-2000 panic desire; pack limit 10)")
    ap.add_argument("--throat-width", type=float, default=3.66,
                    help="FOB throat width W (m); 6.0 = CAG mandated minimum")
    ap.add_argument("--mode", choices=["wall", "fallen"], default="wall",
                    help="wall = shutter gate (needs a clear band); fallen = "
                         "pin the N_FALLEN agents nearest mid-throat (the "
                         "NDLS/Elphinstone mechanism: the crowd becomes the "
                         "obstruction)")
    args = ap.parse_args()

    # Frozen release schedules; equal-N (150) in both.
    sched = (ReleaseSchedule(((0.0, 50), (3.0, 50), (6.0, 50)))
             if args.schedule == "burst" else
             ReleaseSchedule(((0.0, 30), (20.0, 30), (40.0, 30),
                              (60.0, 30), (80.0, 30))))
    r = run_blockage(sched, args.v0, args.n, args.t_block, seed=args.seed,
                     W=args.throat_width,
                     config=args.config, max_time=args.max_time,
                     overrides=SURGE_DESIRE_OVERRIDES if args.surge_desire else None,
                     mode=args.mode)
    r["surge_desire"] = args.surge_desire
    print({k: v for k, v in r.items()}, flush=True)


if __name__ == "__main__":
    main()
