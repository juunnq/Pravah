"""Phase B: closed-loop conditional autonomy (L3) — the agent prevents the crush.

Scenario: phased release (0,10,20,30,40 s x 30 agents), fall in the throat at
t=10 s, surge crowd. Without intervention the remaining batches walk into the
pile (~65 pinned, Phase-8 data). With the agent ON, the system watches its own
detector+forecaster each second and autonomously HOLDS pending releases when
the watch band is (or is forecast to be) crossed, RESUMING once the zone
clears — the machine-actuatable gate-metering intervention.

This is the L3 demonstration: autonomous perception -> decision -> physical
action -> measured consequence, with abstention and a full audit log.

# ponytail: third copy of the blockage step-loop (run_blockage owns a closed
# loop; the agent needs in-loop control). Unify the three into one controllable
# runner — flagged debt, do it when a fourth variant appears.

Usage:
  python scripts/phaseB_closedloop.py --seeds 42 43 44 45            # agent ON
  python scripts/phaseB_closedloop.py --seeds 42 --no-agent          # control
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_spec = importlib.util.spec_from_file_location(
    "blockage", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "phase3_blockage_screen.py"))
blockage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(blockage)

from sim.agent import SafetyAgent
from sim.core.simulation import Simulation
from sim.detector import ThresholdDetector
from sim.forecaster import RateOfRiseForecaster, ZoneForecaster
from sim.providers import SimulationStateProvider
from sim.scenarios.railway_platform import RailwayPlatformScenario
from sim.scenarios.railway_release import (RAILWAY_PARAM_OVERRIDES,
                                           ReleaseSchedule, _batch_seeds)

SCHEDULE = ReleaseSchedule(((0.0, 30), (10.0, 30), (20.0, 30),
                            (30.0, 30), (40.0, 30)))
ZONE = blockage.DANGER_ZONE
T_BLOCK = 10.0


def run(seed: int, agent_on: bool, max_time: float = 150.0,
        config: str = "C2",
        act_threshold: float = 4.0, clear_threshold: float = 2.5) -> dict:
    """One closed-loop run; returns metrics + the agent's action log."""
    scenario = RailwayPlatformScenario(speed_mean=2.0)
    scenario.n_agents = SCHEDULE.batches[0][1]
    overrides = dict(RAILWAY_PARAM_OVERRIDES)
    overrides.update(blockage.SURGE_DESIRE_OVERRIDES)
    sim = Simulation.from_scenario(scenario, config, seed=seed,
                                   param_overrides=overrides)
    sim.log_positions = True
    dt = sim.params.get("dt", 0.01)

    prov = SimulationStateProvider(scenario, cell_size=1.0)
    det = ThresholdDetector(ZONE, prov)
    # Trained forecaster (saturation-aware — the baseline's linear ramp
    # overshoots into the validity ceiling during surges; run 1 proved it).
    fpath = os.path.join("results", "phase9", "forecaster.joblib")
    if os.path.exists(fpath):
        import joblib
        blob = joblib.load(fpath)
        fc = ZoneForecaster(ZONE, prov, horizon=float(blob["horizon"]),
                            window=int(blob["window"]))
        fc.model = blob["model"]
    else:
        fc = RateOfRiseForecaster(ZONE, prov, horizon=30.0, window=30)
    agent = SafetyAgent(act_threshold=act_threshold,
                        clear_threshold=clear_threshold)

    release_steps = {i: int(round(t / dt))
                     for i, (t, _) in enumerate(SCHEDULE.batches)}
    seeds = _batch_seeds(seed, len(SCHEDULE.batches))
    released = {0}
    # Gate metering: a resumed backlog is re-released one batch per
    # BATCH_SPACING, never dumped at once (run 2 proved a queue flush
    # recreates the burst the agent exists to prevent).
    BATCH_SPACING = int(round(10.0 / dt))
    last_release_step = 0

    fallen_ids = None
    fallen_pos = None
    block_step = int(round(T_BLOCK / dt))
    max_steps = int(max_time / dt) + 10

    while sim.step_count < max_steps and sim.time < max_time:
        step = sim.step_count
        # --- release control (the actuator the agent commands): at most ONE
        #     batch per BATCH_SPACING, in order — a metered gate, never a flush.
        pending = [bi for bi in range(1, len(SCHEDULE.batches))
                   if bi not in released]
        if pending:
            bi = pending[0]
            due = step >= release_steps[bi]
            gap_ok = (step - last_release_step) >= BATCH_SPACING
            if due and gap_ok and (not agent_on or not agent.holding):
                sim.inject_agents(SCHEDULE.batches[bi][1], seed=seeds[bi],
                                  spawn_area=scenario.spawn_area,
                                  goal=scenario.goal, avoid_overlap=True,
                                  speed_mean=2.0)
                released.add(bi)
                last_release_step = step
        # --- fallen-pile blockage (identical mechanism to phase 3)
        if fallen_ids is None and step >= block_step:
            act = sim.state.active_indices
            xs = sim.state.positions[act, 0]
            cand = act[(xs > 25.0) & (xs < 31.0)]
            if len(cand) >= blockage.N_FALLEN:
                order = np.argsort(np.abs(
                    sim.state.positions[cand, 0] - blockage.GATE_X))
                fallen_ids = cand[order[:blockage.N_FALLEN]]
                fallen_pos = sim.state.positions[fallen_ids].copy()
        sim.step()
        if fallen_ids is not None:
            sim.state.positions[fallen_ids] = fallen_pos
            sim.state.velocities[fallen_ids] = 0.0
        # --- the agent's decision tick (1 Hz)
        if step % 100 == 0:
            act = sim.state.active_indices
            pos = sim.state.positions[act]
            state = prov.sample(sim.time, pos)
            r = det.update(state)
            f = fc.update(state)
            if agent_on:
                agent.decide(r, f)

    # metrics (same definitions as phase 3/8)
    diy, dix = prov.zone_cells(ZONE)
    max_pinned = a3 = 0
    for k, (t, _idx, pos, _v) in enumerate(sim._position_log):
        if k % 100 != 0 or len(pos) < 4:
            continue
        grid = prov.sample(t, pos).density_grid
        a3 += int((grid[diy, dix] >= 3.0).sum())
        max_pinned = max(max_pinned, int(np.sum(
            (pos[:, 0] >= 20.0) & (pos[:, 0] <= 28.0))))
    return {
        "seed": seed, "agent": agent_on,
        "max_pinned": max_pinned, "exposure_a3": a3,
        "released_batches": len(released), "total_agents": sim.state.n,
        "final_active": sim.state.n_active,
        "actions": [(round(d.timestamp), d.action, d.reason)
                    for d in agent.actions_taken] if agent_on else [],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--no-agent", action="store_true")
    args = ap.parse_args()
    for seed in args.seeds:
        r = run(seed, agent_on=not args.no_agent)
        print({k: v for k, v in r.items() if k != "actions"}, flush=True)
        for a in r["actions"]:
            print("   ", a, flush=True)


if __name__ == "__main__":
    main()
