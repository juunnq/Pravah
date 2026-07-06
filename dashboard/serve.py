"""Operator dashboard — local web app (stdlib only, no framework).

Serves a single-page control-room view: density grid, latched alarm ladder,
pressure-column trend, forecast, and the sweep-derived playbook. Data is
produced by the REAL product components (SimulationStateProvider ->
ThresholdDetector -> trained ZoneForecaster) replaying persisted trajectories
— the same code path a live camera would drive via VideoCCTVProvider.

Usage:
  python dashboard/serve.py                 # replay burst vs phased (seed 42)
  python dashboard/serve.py --port 8750
Then open http://localhost:8750

# ponytail: stdlib http.server, no Flask — a downloadable system should not
# need a web framework to show a page; upgrade path exists if routes grow.
"""

import argparse
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.agent import SafetyAgent
from sim.detector import ThresholdDetector
from sim.forecaster import RateOfRiseForecaster, ZoneForecaster
from sim.providers import SimulationStateProvider
from sim.scenarios.railway_platform import RailwayPlatformScenario

ZONE = (20.0, 32.0, 3.0, 17.0)  # taper+throat danger zone
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Preferred: the full-physics (C4) demo pair; fallback: quickstart-generated
# C2 pair (fresh clones — see scripts/quickstart_demo.py).
ARM_CANDIDATES = {
    "burst": [
        os.path.join("results", "phase3",
                     "blockfallen_burst_v2.0_tb10.0_C4_seed42_traj.parquet"),
        os.path.join("results", "quickstart",
                     "blockfallen_burst_v2.0_tb10.0_C2_seed42_traj.parquet"),
    ],
    "phased": [
        os.path.join("results", "phase3",
                     "blockfallen_phased_v2.0_tb10.0_C4_seed42_traj.parquet"),
        os.path.join("results", "quickstart",
                     "blockfallen_phased_v2.0_tb10.0_C2_seed42_traj.parquet"),
    ],
}


def arm_path(arm: str) -> str:
    for p in ARM_CANDIDATES[arm]:
        if os.path.exists(p):
            return p
    raise SystemExit(
        f"no trajectory for '{arm}' — run scripts/quickstart_demo.py first")


def load_forecaster(prov) -> ZoneForecaster | RateOfRiseForecaster:
    """Trained forecaster if available, else the rate-of-rise baseline."""
    path = os.path.join("results", "phase9", "forecaster.joblib")
    if os.path.exists(path):
        import joblib
        blob = joblib.load(path)
        fc = ZoneForecaster(ZONE, prov, horizon=float(blob["horizon"]),
                            window=int(blob["window"]))
        fc.model = blob["model"]
        return fc
    return RateOfRiseForecaster(ZONE, prov, horizon=30.0, window=30)


def build_replay(arm: str, prov, scenario) -> dict:
    """Replay one trajectory through detector + forecaster -> JSON-able dict."""
    df = pd.read_parquet(arm_path(arm))
    groups = [(t, g) for t, g in df.groupby("t", sort=True)]
    step = 100 if len(groups) > 2000 else 1
    frames = groups[::step]

    det = ThresholdDetector(ZONE, prov)
    fc = load_forecaster(prov)
    agent = SafetyAgent()

    ticks = []
    for t, g in frames:
        pos = g[["x", "y"]].to_numpy()
        state = prov.sample(float(t), pos)
        r = det.update(state)
        f = fc.update(state)
        d = agent.decide(r, f)
        pinned = int(((g["x"] >= 20) & (g["x"] <= 28)).sum())
        tick = {
            "t": round(float(t), 1),
            "grid": np.round(state.density_grid, 1).tolist(),
            "agents": np.round(pos, 1).tolist(),
            "peak": r.zone_peak,
            "band": r.band,
            "crush": r.crush,
            "pinned": pinned,
            "forecast_peak": f.zone_peak_pred,
        }
        if d.action != "NONE":
            tick["call"] = {"action": d.action, "instruction": d.instruction}
        ticks.append(tick)
    return {
        "arm": arm,
        "zone": ZONE,
        "extent": [0, 52, 0, 20],
        "walls": [[w.start.tolist(), w.end.tolist()]
                  for w in scenario.build(seed=0)[0].walls],
        "ticks": ticks,
    }


def build_playbook() -> list[dict]:
    """Ranked release policies from the Phase-8 sweep (surge + blockage cells)."""
    path = os.path.join("results", "phase8", "sweep.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    m = df[(df.crowd == "surge") & (df.t_block < 9999) & (df.config == "C2")]
    rank = (m.groupby(["schedule", "W"]).max_pinned_upstream.mean()
            .reset_index().sort_values("max_pinned_upstream"))
    return [{"schedule": r.schedule, "W": float(r.W),
             "crowd_at_risk": round(float(r.max_pinned_upstream))}
            for r in rank.itertuples()]


class Handler(SimpleHTTPRequestHandler):
    replays: dict = {}
    playbook: list = []

    def do_GET(self):
        if self.path.startswith("/api/replay/"):
            arm = self.path.rsplit("/", 1)[-1]
            if arm in self.replays:
                self._json(self.replays[arm])
            else:
                self.send_error(404, f"unknown arm {arm}")
        elif self.path == "/api/playbook":
            self._json(self.playbook)
        else:
            super().do_GET()

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8750)
    args = ap.parse_args()

    scenario = RailwayPlatformScenario()
    prov = SimulationStateProvider(scenario, cell_size=1.0)
    print("building replays from persisted trajectories...", flush=True)
    for arm in ARM_CANDIDATES:
        Handler.replays[arm] = build_replay(arm, prov, scenario)
        print(f"  {arm}: {len(Handler.replays[arm]['ticks'])} ticks", flush=True)
    Handler.playbook = build_playbook()
    print(f"  playbook: {len(Handler.playbook)} entries", flush=True)

    os.chdir(STATIC)
    srv = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard: http://localhost:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
