"""Phase 6: demo figures + animation from persisted Phase-3/5 trajectories.

Renders (to figures/, gitignored — deliver files directly):
  1. fig_heatmap_pair.pdf   — burst vs phased density grids at fall+60 s
  2. fig_alarm_timeline.pdf — zone density vs time, threshold bands, detector
                              alarm markers (ThresholdDetector replayed over
                              provider grids — the product loop end-to-end)
  3. fig_crowd_at_risk.pdf  — pinned-upstream per seed per arm (+ CAG arm when
                              its parquets exist)
  4. fig_trajectories_pair.pdf — per-agent paths (own renderer: the legacy
                              sim/viz/trajectories.py assumes constant agent
                              count, which injection breaks)
  5. anim_pair.gif          — side-by-side burst vs phased, agents colored by
                              local grid density, live alarm banner

All inputs are results/phase3/*.parquet — NO new simulation.
Usage: python scripts/phase6_figures.py [--seed 42] [--no-anim]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from sim.detector import ThresholdDetector
from sim.providers import SimulationStateProvider
from sim.scenarios.railway_platform import RailwayPlatformScenario
from sim.viz.style import save_figure, set_style

RESULTS = os.path.join("results", "phase3")
DANGER_ZONE = (20.0, 32.0, 3.0, 17.0)   # taper+throat (matches phase-3/5 labels)
T_BLOCK = 10.0
SEEDS = [42, 43, 44, 45]
# Okabe-Ito colorblind-safe
C_BURST, C_PHASED, C_CAG = "#d55e00", "#0072b2", "#009e73"
ARM_LABEL = {"burst": "Burst release (baseline)",
             "phased": "Phased release (intervention)",
             "cag": "Burst @ CAG 6.0 m width"}


def traj_path(arm: str, seed: int, W: float = 3.66) -> str:
    wtag = "" if W == 3.66 else f"_w{W}"
    return os.path.join(
        RESULTS, f"blockfallen_{arm}_v2.0_tb{T_BLOCK}_C4{wtag}_seed{seed}_traj.parquet")


def load_ticks(path: str) -> list[tuple[float, pd.DataFrame]]:
    """1-Hz (tick, frame) list from a trajectory parquet (any log rate).

    Single groupby pass (deterministic; repeated boolean filtering over 15k
    ticks intermittently returned near-empty frames under concurrent load).
    Near-empty frames are reported loudly, never silently dropped.
    """
    df = pd.read_parquet(path)
    groups = [(t, g) for t, g in df.groupby("t", sort=True)]
    step = 100 if len(groups) > 2000 else 1  # full-rate vs already-1-Hz logs
    frames = groups[::step]
    sizes = [len(g) for _, g in frames]
    med = np.median(sizes) if sizes else 0
    bad = [(t, n) for (t, _), n in zip(frames, sizes) if n < 0.2 * med]
    if bad:
        print(f"WARNING: {len(bad)} near-empty frames in {os.path.basename(path)}: "
              f"{bad[:5]}")
    return frames


def find_pile(frames) -> np.ndarray:
    """Locate the fallen pile: the 6 least-moving agents inside the throat
    window after the fall. Returns positions (6, 2)."""
    rows = []
    for t, g in frames:
        if t < 15:
            continue
        rows.append(g[["agent_id", "x", "y"]])
    allr = pd.concat(rows)
    stats = allr.groupby("agent_id").agg(sx=("x", "std"), mx=("x", "mean"),
                                         my=("y", "mean"), sy=("y", "std"))
    cand = stats[(stats.mx > 25) & (stats.mx < 31)]
    pile = cand.nsmallest(6, "sx")
    return pile[["mx", "my"]].to_numpy()


def draw_walls(ax, scenario) -> None:
    world, _ = scenario.build(seed=0)
    for w in world.walls:
        ax.plot([w.start[0], w.end[0]], [w.start[1], w.end[1]],
                color="black", lw=1.5, zorder=3)


def replay_detector(frames, prov, scenario):
    """Replay ThresholdDetector over provider grids; return (t[], peak[], det)."""
    det = ThresholdDetector(DANGER_ZONE, prov)
    ts, peaks = [], []
    for t, g in frames:
        pos = g[["x", "y"]].to_numpy()
        det.update(prov.sample(t, pos))
        ts.append(t)
        peaks.append(det.history[-1].zone_peak)
    return np.array(ts), np.array(peaks), det


def agent_cell_density(pos, prov) -> np.ndarray:
    """Per-agent local density = its grid-cell value (for scatter coloring)."""
    state = prov.sample(0.0, pos)
    x0, y0 = state.origin
    cs = state.cell_size
    d = np.zeros(len(pos))
    H, W_ = state.density_grid.shape
    for i, p in enumerate(pos):
        ix, iy = int((p[0] - x0) / cs), int((p[1] - y0) / cs)
        if 0 <= ix < W_ and 0 <= iy < H:
            d[i] = state.density_grid[iy, ix]
    return d


def fig_heatmap_pair(frames_by_arm, prov, scenario, t_show: float) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for ax, arm in zip(axes, ["burst", "phased"]):
        frames = frames_by_arm[arm]
        t, g = min(frames, key=lambda fr: abs(fr[0] - t_show))
        pos = g[["x", "y"]].to_numpy()
        grid = prov.sample(t, pos).density_grid
        im = ax.imshow(grid, origin="lower", extent=(0, grid.shape[1],
                       0, grid.shape[0]), cmap="YlOrRd", vmin=0, vmax=5,
                       aspect="equal")
        draw_walls(ax, scenario)
        pile = find_pile(frames)
        ax.scatter(pile[:, 0], pile[:, 1], marker="x", s=48, color="black",
                   zorder=4, label="fallen pile")
        n_col = int(((g["x"] >= 20) & (g["x"] <= 28)).sum())
        ax.set_title(f"{ARM_LABEL[arm]} — t = {t:.0f} s "
                     f"({n_col} in pressure column)", fontsize=10)
        ax.set_ylabel("y (m)")
        ax.legend(loc="lower right", fontsize=8)
    axes[1].set_xlabel("x (m)")
    fig.colorbar(im, ax=axes, label="density (ped/m²)", shrink=0.85)
    return save_figure(fig, "fig_heatmap_pair")


def fig_alarm_timeline(frames_by_arm, prov, scenario) -> str:
    """Two panels: (top) pressure-column occupancy — the seed-robust
    discriminator; (bottom) zone peak with threshold bands — alarm semantics.
    Single-cell peak alone does NOT discriminate packed crowds (Phase-3
    measurement ruling), hence the top panel carries the story."""
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True,
                                   height_ratios=[3, 2])
    for arm, color in (("burst", C_BURST), ("phased", C_PHASED)):
        frames = frames_by_arm[arm]
        ts, peaks, det = replay_detector(frames, prov, scenario)
        pinned = np.array([int(((g["x"] >= 20) & (g["x"] <= 28)).sum())
                           for _, g in frames])
        ax0.plot(ts, pinned, color=color, lw=2.0, label=ARM_LABEL[arm])
        ax1.plot(ts, peaks, color=color, lw=1.4)
        idx = np.argmax(peaks >= 4.0) if np.any(peaks >= 4.0) else None
        if idx is not None:
            ax1.scatter(ts[idx], peaks[idx], marker="o", s=55, color=color,
                        zorder=4, edgecolor="black")
            dy = 10 if arm == "burst" else -16  # stagger to avoid overlap
            ax1.annotate(f"watch @ {ts[idx]:.0f}s", (ts[idx], peaks[idx]),
                         xytext=(8, dy), textcoords="offset points",
                         fontsize=8, color=color)
    for ax in (ax0, ax1):
        ax.axvline(T_BLOCK, color="gray", ls=":", lw=1.2)
    ax0.text(T_BLOCK + 1.5, 8, "fall occurs", fontsize=8, color="gray")
    ax0.set_ylabel("people in pressure column\n(x ∈ [20, 28] m)")
    ax0.legend(loc="center right", fontsize=9)
    ax0.set_title("Same fall, same crowd — release policy sets who is exposed")
    ax1.axhspan(4.0, 5.0, color="#f0e442", alpha=0.25)
    ax1.axhspan(5.0, 5.5, color="#e69f00", alpha=0.25)
    ax1.axhline(5.5, color="#b30000", ls="--", lw=1.0)
    ax1.text(1, 4.25, "watch ≥4.0", fontsize=7)
    ax1.text(1, 5.1, "amber ≥5.0", fontsize=7)
    ax1.set_ylim(0, 6.0)
    ax1.set_ylabel("zone peak\n(ped/m², 1 m² grid)")
    ax1.set_xlabel("time (s)")
    return save_figure(fig, "fig_alarm_timeline")


def fig_crowd_at_risk() -> str:
    arms = [("burst", 3.66, C_BURST), ("phased", 3.66, C_PHASED),
            ("burst", 6.0, C_CAG)]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ticklabels = []
    for i, (arm, W, color) in enumerate(arms):
        vals = []
        for seed in SEEDS:
            path = traj_path(arm, seed, W)
            if not os.path.exists(path):
                continue
            frames = load_ticks(path)
            pinned = max(int(((g["x"] >= 20) & (g["x"] <= 28)).sum())
                         for _, g in frames)
            vals.append(pinned)
        if not vals:
            continue
        x = np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, s=55, color=color, zorder=3, edgecolor="black")
        ax.hlines(np.median(vals), i - 0.25, i + 0.25, color=color, lw=2.5)
        key = "cag" if W == 6.0 else arm
        ticklabels.append(f"{ARM_LABEL[key]}\n(median {np.median(vals):.0f})")
    ax.set_xticks(range(len(ticklabels)))
    ax.set_xticklabels(ticklabels, fontsize=8)
    ax.set_ylabel("people in the pressure column (max)")
    ax.set_title("Crowd-at-risk when the fall lands — per seed")
    ax.set_ylim(0, 130)
    return save_figure(fig, "fig_crowd_at_risk")


def fig_trajectories_pair(frames_by_arm, scenario) -> str:
    """Per-agent paths, clipped at the first domain exit. Agents ejected
    through walls under crush-level contact forces are a documented
    model-validity artifact; their count is stated in the
    title and their post-ejection ballistic tails are clipped for legibility —
    never silently removed."""
    from shapely.geometry import Point, Polygon
    domain = Polygon(scenario.domain_polygon())

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for ax, arm in zip(axes, ["burst", "phased"]):
        frames = frames_by_arm[arm]
        df = pd.concat(g.assign(t=t) for t, g in frames)
        tmax = df["t"].max()
        ejected = 0
        for _aid, gr in df.groupby("agent_id"):
            pts = gr[["x", "y"]].to_numpy()
            tvals = gr["t"].to_numpy()
            inside = np.array([domain.contains(Point(p)) for p in pts])
            # Permanent ejection = ends outside AND exit was through a wall
            # (not the legitimate open far edge at x>=52).
            if not inside[-1] and pts[-1, 0] < 51.5:
                ejected += 1
            if len(pts) < 2:
                continue
            p = pts.reshape(-1, 1, 2)
            segs = np.concatenate([p[:-1], p[1:]], axis=1)
            keep = inside[:-1] & inside[1:]  # draw in-domain segments only
            if not np.any(keep):
                continue
            lc = LineCollection(segs[keep], cmap="viridis", lw=0.5, alpha=0.6)
            lc.set_array((tvals[:-1] / tmax)[keep])
            ax.add_collection(lc)
        draw_walls(ax, scenario)
        pile = find_pile(frames)
        ax.scatter(pile[:, 0], pile[:, 1], marker="x", s=48, color="red", zorder=4)
        ax.set_xlim(-1, 54)
        ax.set_ylim(-1, 21)
        ax.set_aspect("equal")
        note = (f" — {ejected} ejected through walls under crush load "
                f"(model artifact, paths clipped)") if ejected else ""
        ax.set_title(ARM_LABEL[arm] + note, fontsize=9)
        ax.set_ylabel("y (m)")
    axes[1].set_xlabel("x (m)")
    return save_figure(fig, "fig_trajectories_pair")


def anim_pair(frames_by_arm, prov, scenario, out="figures/anim_pair.gif") -> str:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
    n_frames = min(len(frames_by_arm["burst"]), len(frames_by_arm["phased"]), 121)
    dets = {arm: ThresholdDetector(DANGER_ZONE, prov) for arm in frames_by_arm}
    scats, banners = {}, {}
    for ax, arm in zip(axes, ["burst", "phased"]):
        draw_walls(ax, scenario)
        ax.set_xlim(-1, 54)
        ax.set_ylim(-1, 21)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        scats[arm] = ax.scatter([], [], s=14, c=[], cmap="YlOrRd",
                                vmin=0, vmax=5, edgecolor="none")
        banners[arm] = ax.set_title("", fontsize=10)

    # Operator-facing alarm banner LATCHES at the highest band reached
    # (an alarm that flickers off during single-tick dips is useless);
    # the underlying detector reading is instantaneous except crush.
    RANK = {"clear": 0, "watch": 1, "amber": 2, "critical": 3}
    latched = {arm: ("clear", None) for arm in frames_by_arm}

    def update(k):
        artists = []
        for arm in ["burst", "phased"]:
            t, g = frames_by_arm[arm][k]
            pos = g[["x", "y"]].to_numpy()
            state = prov.sample(t, pos)
            r = dets[arm].update(state)
            hi_band, hi_t = latched[arm]
            if RANK[r.band] > RANK[hi_band]:
                latched[arm] = (r.band, t)
                hi_band, hi_t = r.band, t
            scats[arm].set_offsets(pos)
            scats[arm].set_array(agent_cell_density(pos, prov))
            color = {"clear": "green", "watch": "#b38600",
                     "amber": "#cc6600", "critical": "red"}[hi_band]
            since = f" since {hi_t:.0f}s" if hi_t is not None else ""
            banners[arm].set_text(f"{ARM_LABEL[arm]}   t={t:5.0f}s   "
                                  f"peak={r.zone_peak:.1f}   "
                                  f"[{hi_band.upper()}{since}]")
            banners[arm].set_color(color)
            artists += [scats[arm], banners[arm]]
        return artists

    ani = FuncAnimation(fig, update, frames=n_frames, blit=False)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ani.save(out, writer=PillowWriter(fps=8), dpi=90)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="exemplar seed for heatmap/timeline/trajectories/anim")
    ap.add_argument("--no-anim", action="store_true")
    args = ap.parse_args()

    set_style()
    scenario = RailwayPlatformScenario()
    prov = SimulationStateProvider(scenario, cell_size=1.0)
    frames_by_arm = {arm: load_ticks(traj_path(arm, args.seed))
                     for arm in ["burst", "phased"]}

    print(fig_heatmap_pair(frames_by_arm, prov, scenario, t_show=T_BLOCK + 60))
    print(fig_alarm_timeline(frames_by_arm, prov, scenario))
    print(fig_crowd_at_risk())
    print(fig_trajectories_pair(frames_by_arm, scenario))
    if not args.no_anim:
        print(anim_pair(frames_by_arm, prov, scenario))


if __name__ == "__main__":
    main()
