"""Self-firing post-sweep queue: waits for the expanded sweep, then runs
every compute-dependent upgrade unattended, logging to results/queue_report.md.

Queue (each step isolated — one failure never kills the rest):
  1. Retrain the surrogate on the full sweep (phase8_train) + conformal
     intervals with empirical coverage.
  2. Retrain the forecaster on the enlarged corpus (phase9_train).
  3. C4 closed-loop confirmation of the L3 result (2 paired seeds).
  4. Agent threshold grid (act x clear, 2 seeds, C2) -> tuned operating point.

Launch (survives the launching session; runs until done):
  python scripts/after_sweep_queue.py          # waits, then runs
  python scripts/after_sweep_queue.py --now    # skip waiting (sweep done)
"""

import argparse
import importlib.util
import io
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SWEEP_CSV = os.path.join("results", "phase8", "sweep.csv")
TARGET_ROWS = 851  # 840 C2 grid + 10 C4 anchors + header
REPORT = os.path.join("results", "queue_report.md")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def rows() -> int:
    try:
        with open(SWEEP_CSV) as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def wait_for_sweep() -> None:
    log(f"waiting for sweep: {rows()}/{TARGET_ROWS} rows")
    stall_count, last = 0, rows()
    while rows() < TARGET_ROWS:
        time.sleep(600)
        cur = rows()
        if cur == last:
            stall_count += 1
            if stall_count >= 12:  # 2 h without progress: shards died
                log(f"SWEEP STALLED at {cur} rows for 2 h — proceeding with "
                    f"what exists (queue is re-runnable after a sweep restart)")
                return
        else:
            stall_count = 0
        last = cur
    log(f"sweep complete: {rows()} rows")


def step(name: str, fn) -> None:
    log(f"--- {name} ---")
    try:
        fn()
        log(f"{name}: OK")
    except Exception as e:  # isolate failures
        log(f"{name}: FAILED — {type(e).__name__}: {e}")


def retrain_surrogate() -> None:
    r = subprocess.run([sys.executable, "scripts/phase8_train.py"],
                       capture_output=True, text=True, timeout=3600)
    log("phase8_train tail:\n" + "\n".join(r.stdout.splitlines()[-6:]))

    # Conformal intervals on the retrained surrogate (crowd-at-risk).
    import numpy as np
    import pandas as pd
    spec = importlib.util.spec_from_file_location(
        "tr", os.path.join("scripts", "phase8_train.py"))
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sim.conformal import SplitConformal

    df = pd.read_csv(SWEEP_CSV)
    df = df[df.config == "C2"].copy()
    combo = (df.schedule + "|" + df.W.astype(str) + "|" + df.crowd
             + "|" + df.t_block.astype(str))
    X, y = tr.featurize(df), df["max_pinned_upstream"]
    _, _, _, oof = tr.cv_mae(
        lambda: HistGradientBoostingRegressor(random_state=0), X, y, combo)
    res = np.abs(oof - y.to_numpy())
    half = len(res) // 2
    sc = SplitConformal(alpha=0.1).fit(res[:half])
    cov = SplitConformal.empirical_coverage(
        y.to_numpy()[half:], oof[half:], sc.q)
    log(f"conformal (alpha=0.1): q={sc.q:.1f} people, "
        f"empirical coverage {cov:.3f} (target >= 0.90)")


def retrain_forecaster() -> None:
    r = subprocess.run([sys.executable, "scripts/phase9_train.py"],
                       capture_output=True, text=True, timeout=7200)
    log("phase9_train tail:\n" + "\n".join(r.stdout.splitlines()[-4:]))


def c4_confirmation() -> None:
    spec = importlib.util.spec_from_file_location(
        "cl", os.path.join("scripts", "phaseB_closedloop.py"))
    cl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    for seed in (42, 43):
        on = cl.run(seed, agent_on=True, config="C4")
        off = cl.run(seed, agent_on=False, config="C4")
        log(f"C4 closed-loop seed {seed}: pinned ON {on['max_pinned']} "
            f"vs OFF {off['max_pinned']} | exposure {on['exposure_a3']} "
            f"vs {off['exposure_a3']}")


def threshold_grid() -> None:
    spec = importlib.util.spec_from_file_location(
        "cl", os.path.join("scripts", "phaseB_closedloop.py"))
    cl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    best = None
    for act in (3.5, 4.0, 4.5):
        for clear in (2.0, 2.5, 3.0):
            pinned, released = [], []
            for seed in (42, 43):
                r = cl.run(seed, agent_on=True, config="C2",
                           act_threshold=act, clear_threshold=clear)
                pinned.append(r["max_pinned"])
                released.append(r["released_batches"])
            score = sum(pinned) / len(pinned)
            log(f"grid act={act} clear={clear}: mean pinned {score:.1f}, "
                f"batches released {released}")
            if best is None or score < best[0]:
                best = (score, act, clear)
    log(f"BEST operating point: act={best[1]} clear={best[2]} "
        f"(mean pinned {best[0]:.1f}) — throughput tradeoff per released "
        f"counts above; operator picks the final knob")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    log("QUEUE START")
    if not args.now:
        wait_for_sweep()
    step("1/4 retrain surrogate + conformal", retrain_surrogate)
    step("2/4 retrain forecaster", retrain_forecaster)
    step("3/4 C4 closed-loop confirmation", c4_confirmation)
    step("4/4 agent threshold grid", threshold_grid)
    log("QUEUE COMPLETE — see results/queue_report.md; next: update "
        "project reports from these numbers (follow-up task)")


if __name__ == "__main__":
    main()
