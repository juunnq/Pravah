"""Phase C seed: the SAME brain on a SECOND domain (room evacuation).

The generality claim ("geometry, agents, and danger definitions are supplied
to the engine, never baked in") is earned domain by domain. This SEED — not
yet a study — runs the engine's classic evacuation-crush scenario
(CrushRoomScenario: packed room, one narrow exit) through the UNCHANGED
product stack: provider -> detector -> forecaster -> SafetyAgent (advisory
only; there is no release gate in this domain). Zero changes to any brain
component; the venue supplies its own polygon and danger zone.

What "success" means for a seed: the stack runs, the readings are physically
sane for the new venue, the alarm ladder fires appropriately at the exit
crush, and every advisory is logged. A full second-domain STUDY (n seeds,
domain-specific danger metrics, validation data) is future work.

Usage: python scripts/phaseC_evacuation_seed.py [--n 200] [--max-time 90]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.agent import SafetyAgent
from sim.core.simulation import Simulation
from sim.detector import ThresholdDetector
from sim.forecaster import RateOfRiseForecaster
from sim.providers import SimulationStateProvider
from sim.scenarios.crush_room import CrushRoomScenario


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-time", type=float, default=90.0)
    args = ap.parse_args()

    W, H = 5.0, 5.0
    scenario = CrushRoomScenario(n_agents=args.n, width=W, height=H,
                                 exit_width=0.6)
    # The venue supplies its own walkable polygon (domain-neutral seam).
    room = np.array([[0, 0], [W + 1.5, 0], [W + 1.5, H], [0, H]], dtype=float)
    exit_zone = (W - 1.5, W + 0.5, H / 2 - 1.5, H / 2 + 1.5)  # exit approach

    sim = Simulation.from_scenario(scenario, "C2", seed=42)
    prov = SimulationStateProvider(scenario, cell_size=0.5, domain=room)
    det = ThresholdDetector(exit_zone, prov)
    fc = RateOfRiseForecaster(exit_zone, prov, horizon=15.0, window=15)
    agent = SafetyAgent()

    dt = sim.params.get("dt", 0.01)
    max_steps = int(args.max_time / dt)
    n0 = sim.state.n_active
    while sim.step_count < max_steps and sim.state.n_active > 0:
        sim.step()
        if sim.step_count % 100 == 0:
            act = sim.state.active_indices
            state = prov.sample(sim.time, sim.state.positions[act])
            r = det.update(state)
            f = fc.update(state)
            agent.decide(r, f)

    evacuated = n0 - sim.state.n_active
    print(f"DOMAIN 2 (room evacuation, {args.n} agents, 0.6 m exit):")
    print(f"  evacuated {evacuated}/{n0} in {sim.time:.0f}s "
          f"({'complete' if sim.state.n_active == 0 else 'ongoing at horizon'})")
    peaks = [d.reading.zone_peak for d in agent.log]
    print(f"  exit-zone peak density: max {max(peaks):.1f} ped/m² "
          f"(0.5 m grid), median {sorted(peaks)[len(peaks)//2]:.1f}")
    print(f"  agent calls ({len(agent.actions_taken)}):")
    for d in agent.actions_taken[:8]:
        print(f"    t={d.timestamp:5.0f}s {d.action}: {d.reason}")
    print("  NOTE: advisory-only in this domain (no release gate exists); "
          "the identical brain components ran unchanged.")


if __name__ == "__main__":
    main()
