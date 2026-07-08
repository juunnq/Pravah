"""Comparative figures for the README, from measured artifacts only.

Sources:
- policy arms: per-seed pinned counts, C4 physics, seeds 42-45 (results/phase3)
- surrogate:   results/phase8/surrogate_eval.csv (bootstrap CIs, GroupKFold)
- forecaster:  results/phase9/forecaster.joblib bundle (held-out runs)
- perception:  Mall-footage held-out count error, before/after per-camera
               scalar calibration (results/phaseA)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORANGE, BLUE, GREEN, GREY = "#D55E00", "#0072B2", "#009E73", "#767676"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white",
                     "svg.fonttype": "none"})

import os
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------- Figure 1: release-policy arms (the physics result) --------
arms = [
    ("burst release\nlegacy 3.66 m bridge", [113, 115, 114, 106], ORANGE),
    ("burst release\nwidened 6.0 m bridge", [93, 95, 83, 93], GREY),
    ("phased release\nlegacy 3.66 m bridge", [34, 49, 52, 52], BLUE),
]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for i, (label, vals, c) in enumerate(arms):
    med = np.median(vals)
    ax.bar(i, med, width=0.58, color=c, alpha=0.28, edgecolor=c, linewidth=1.6, zorder=2)
    jit = np.linspace(-0.10, 0.10, len(vals))
    ax.scatter(i + jit, vals, color=c, s=42, zorder=3, label=None)
    ax.text(i, med * 0.5, f"median {med:.0f}", ha="center", fontsize=10.5,
            color=c, fontweight="bold")
meds = [np.median(v) for _, v, _ in arms]
ax.annotate(f"engineering fix: −{100*(1-meds[1]/meds[0]):.0f}%",
            xy=(0.72, meds[1] + 1), xytext=(0.18, 127), fontsize=10, color=GREY,
            arrowprops=dict(arrowstyle="->", color=GREY, lw=1.1))
ax.annotate(f"operational fix: −{100*(1-meds[2]/meds[0]):.0f}%",
            xy=(1.72, meds[2] + 1), xytext=(1.28, 127), fontsize=10, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))
ax.set_xticks(range(3)); ax.set_xticklabels([a[0] for a in arms], fontsize=10.5)
ax.set_ylabel("people pinned in the crush pressure column")
ax.set_ylim(0, 140)
ax.set_title("Same fall, same 150 people — release policy vs. bridge widening\n"
             "(full C4 physics, 4 seeds each, dots = individual seeds; zero overlap)",
             fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/comparative_policy.svg"); fig.savefig(f"{OUT}/comparative_policy.png", dpi=200)

# ---------------- Figure 2: learned components vs their baselines ----------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))

# (a) surrogate vs baselines — surrogate_eval.csv, target = max_pinned_upstream
ax = axes[0]
names = ["predict\nthe mean", "linear\nregression", "GBM\nsurrogate"]
mae = [31.31, 14.96, 7.86]
ci = [(30.25, 32.39), (14.24, 15.66), (7.34, 8.41)]
colors = [GREY, GREY, GREEN]
err = np.array([[m - lo for m, (lo, hi) in zip(mae, ci)],
                [hi - m for m, (lo, hi) in zip(mae, ci)]])
ax.bar(names, mae, color=colors, alpha=0.8, width=0.6, yerr=err, capsize=4,
       error_kw=dict(lw=1.2, ecolor="black"))
for i, m in enumerate(mae):
    ax.text(i, m + 2.2, f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("held-out MAE (people at risk)")
ax.set_ylim(0, 40)
ax.set_title("Risk surrogate vs. baselines\n(840 runs, GroupKFold, bootstrap 95% CI)",
             fontsize=10.5)

# (b) forecaster vs trend extrapolation — forecaster.joblib bundle
ax = axes[1]
names = ["rate-of-rise\nextrapolation", "trained zone\nforecaster"]
mae = [1.063, 0.421]
ax.bar(names, mae, color=[GREY, GREEN], alpha=0.8, width=0.5)
for i, m in enumerate(mae):
    ax.text(i, m + 0.03, f"{m:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.annotate("−60%", xy=(1, 0.49), xytext=(0.42, 0.90), fontsize=11,
            color=GREEN, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
ax.set_ylabel("held-out MAE (ped/m², 30 s ahead)")
ax.set_ylim(0, 1.25)
ax.set_title("Density forecaster vs. trend baseline\n(held-out simulation runs)",
             fontsize=10.5)

# (c) perception calibration — Mall footage, held-out frames
ax = axes[2]
x = np.arange(2); w = 0.36
uncal = [35.9, 7.8]; cal = [8.9, 6.7]
ax.bar(x - w/2, uncal, w, color=GREY, alpha=0.8, label="off-the-shelf")
ax.bar(x + w/2, cal, w, color=BLUE, alpha=0.85, label="+ per-camera scalar")
for xi, (u, c) in enumerate(zip(uncal, cal)):
    ax.text(xi - w/2, u + 0.8, f"{u:.1f}%", ha="center", fontsize=9.5)
    ax.text(xi + w/2, c + 0.8, f"{c:.1f}%", ha="center", fontsize=9.5)
ax.set_xticks(x); ax.set_xticklabels(["CSRNet", "CLIP-EBC"])
ax.set_ylabel("held-out count error (%)")
ax.set_ylim(0, 42)
ax.legend(frameon=False, fontsize=9.5)
ax.set_title("Perception transfer to real footage\n(Mall dataset, calibration fit on 100 frames)",
             fontsize=10.5)

fig.tight_layout()
fig.savefig(f"{OUT}/comparative_models.svg"); fig.savefig(f"{OUT}/comparative_models.png", dpi=200)
print("figures written")
