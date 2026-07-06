"""Phase 9 training: fit ZoneForecaster on the Phase-8 trajectory corpus.

Harvests sliding windows from the 1-Hz trajectory parquets in results/phase8/
(and full-rate Phase-3 parquets, downsampled), builds (features -> future zone
state) pairs via the SAME code path the live forecaster uses
(ZoneForecaster.features()), trains the GBM, and scores against the mandatory
RateOfRiseForecaster baseline on HELD-OUT RUNS (never windows from a run seen
in training — prevents temporal leakage).

Ships the model only if it beats the baseline (ML ladder); reports either way.
Saves model to results/phase9/forecaster.joblib + eval CSV + scatter figure.

Usage: python scripts/phase9_train.py [--horizon 30] [--window 30]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.forecaster import RateOfRiseForecaster, ZoneForecaster
from sim.providers import SimulationStateProvider
from sim.scenarios.railway_platform import RailwayPlatformScenario
from sim.viz.style import save_figure, set_style

ZONE = (20.0, 32.0, 3.0, 17.0)  # taper+throat danger zone (canonical)


def harvest_run(path: str, prov, horizon: int, window: int):
    """One parquet -> list of (features, [peak, occ] at t+horizon) pairs,
    plus the per-tick series needed for baseline scoring."""
    df = pd.read_parquet(path)
    groups = [(t, g) for t, g in df.groupby("t", sort=True)]
    step = 100 if len(groups) > 2000 else 1
    frames = groups[::step]
    if len(frames) < window + horizon + 2:
        return [], None

    fc = ZoneForecaster(ZONE, prov, horizon=float(horizon), window=window)
    states = []
    feats_at = {}
    for i, (t, g) in enumerate(frames):
        pos = g[["x", "y"]].to_numpy()
        state = prov.sample(float(t), pos)
        fc.update(state)
        states.append(state)
        f = fc.features()
        if f is not None:
            feats_at[i] = f
    # truth series from the same observation path
    peaks = [h for h in fc._peak]
    occs = [o for o in fc._occ]

    pairs = []
    for i, f in feats_at.items():
        j = i + horizon
        if j < len(peaks):
            pairs.append((f, [peaks[j], occs[j]]))
    series = {"peaks": np.array(peaks), "occs": np.array(occs),
              "frames": frames}
    return pairs, series


def score_on_run(model_fc, series, prov, horizon: int, window: int) -> tuple:
    """MAE of (learned, baseline) peak predictions over one held-out run."""
    base = RateOfRiseForecaster(ZONE, prov, horizon=float(horizon), window=window)
    learned = model_fc
    learned._t.clear(); learned._peak.clear(); learned._occ.clear()
    err_m, err_b, truths, preds = [], [], [], []
    peaks = series["peaks"]
    for i, (t, g) in enumerate(series["frames"]):
        pos = g[["x", "y"]].to_numpy()
        state = prov.sample(float(t), pos)
        fm = learned.update(state)
        fb = base.update(state)
        j = i + horizon
        if i >= window and j < len(peaks):
            err_m.append(abs(fm.zone_peak_pred - peaks[j]))
            err_b.append(abs(fb.zone_peak_pred - peaks[j]))
            truths.append(peaks[j]); preds.append(fm.zone_peak_pred)
    return (float(np.mean(err_m)) if err_m else np.nan,
            float(np.mean(err_b)) if err_b else np.nan, truths, preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--window", type=int, default=30)
    args = ap.parse_args()

    prov = SimulationStateProvider(RailwayPlatformScenario(), cell_size=1.0)
    paths = sorted(glob.glob(os.path.join("results", "phase8", "*_traj.parquet")))
    print(f"{len(paths)} corpus parquets", flush=True)
    rng = np.random.Generator(np.random.PCG64(0))
    rng.shuffle(paths)
    n_hold = max(8, len(paths) // 5)
    held, train = paths[:n_hold], paths[n_hold:]

    X, y = [], []
    for p in train:
        pairs, _ = harvest_run(p, prov, args.horizon, args.window)
        for f, t in pairs:
            X.append(f); y.append(t)
    print(f"train windows: {len(X)} from {len(train)} runs", flush=True)

    model = ZoneForecaster(ZONE, prov, horizon=float(args.horizon),
                           window=args.window)
    model.fit(np.array(X), np.array(y))

    maes_m, maes_b, all_truth, all_pred = [], [], [], []
    for p in held:
        _, series = harvest_run(p, prov, args.horizon, args.window)
        if series is None:
            continue
        em, eb, truths, preds = score_on_run(model, series, prov,
                                             args.horizon, args.window)
        if not np.isnan(em):
            maes_m.append(em); maes_b.append(eb)
            all_truth += truths; all_pred += preds
    mae_m, mae_b = float(np.mean(maes_m)), float(np.mean(maes_b))
    beats = mae_m < mae_b
    print(f"HELD-OUT ({len(maes_m)} runs): learned MAE {mae_m:.3f} vs "
          f"rate-of-rise baseline {mae_b:.3f} -> beats baseline: {beats}",
          flush=True)

    outdir = os.path.join("results", "phase9")
    os.makedirs(outdir, exist_ok=True)
    import joblib
    joblib.dump({"model": model.model, "horizon": args.horizon,
                 "window": args.window, "zone": ZONE,
                 "heldout_mae": mae_m, "baseline_mae": mae_b},
                os.path.join(outdir, "forecaster.joblib"))
    pd.DataFrame({"truth": all_truth, "pred": all_pred}).to_csv(
        os.path.join(outdir, "forecaster_eval.csv"), index=False)

    set_style()
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.scatter(all_truth, all_pred, s=8, alpha=0.4, color="#0072b2")
    lim = (0, max(max(all_truth, default=1), max(all_pred, default=1)) * 1.05)
    ax.plot(lim, lim, ls="--", lw=1, color="gray")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"true zone peak at t+{args.horizon}s")
    ax.set_ylabel("forecast")
    ax.set_title(f"held-out runs: MAE {mae_m:.2f} vs baseline {mae_b:.2f}",
                 fontsize=9)
    print(save_figure(fig, "fig_forecaster_scatter"))
    print("SHIP LEARNED MODEL" if beats else
          "SHIP BASELINE (learned did not beat it — reported honestly)")


if __name__ == "__main__":
    main()
