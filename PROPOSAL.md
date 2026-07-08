# Proposal Dossier — Crowd-Safety Early-Warning Pilot

> For a railway/venue operator. Every claim below is reproducible from this
> repository; the pilot ask is at the end. Prepared 2026-07-06.

## The problem

Crowd crushes kill the same way every time: local density crosses a fatal
threshold minutes before anyone with authority perceives it (New Delhi FOB
2025, Elphinstone 2017, Itaewon 2022). Every existing safety layer sits on
the wrong side of that interval — CCTV observes, protocols react, inquiries
explain. This system closes the interval: it reads crowd density from the
CCTV a station already owns, forecasts danger before it is visible, and
recommends the specific operational action that prevents it.

## What exists today (all in this repo, all tested — 165 automated tests)

**1. A peer-reviewed physics core.** The simulation engine underlying this
system was published at SIMULTECH 2026, calibrated against 4,776 real
pedestrian trajectories (Forschungszentrum Jülich), with its
out-of-distribution error explicitly measured (26–43% flow underestimation —
a conservative, false-alarm-direction bias).

**2. Validated hazard findings** (n=4 seeds, zero overlap):
reconstructing the real disaster mechanism (a fall blocking a crowded
foot-overbridge during a surge):

| Configuration | People in the crush pressure column |
|---|---|
| Burst release, legacy 3.66 m FOB | **106–115** |
| Burst release, CAG-mandated 6.0 m FOB | 83–95 |
| **Phased release, legacy FOB** | **34–52** |

→ *The operational fix (phased release: −56%) outperforms the engineering
fix (bridge widening: −18%) threefold — and they compound.* You cannot always
prevent the fall; release policy decides how many people are standing in its
blast radius.

**3. A millisecond decision brain**: a surrogate model trained
on 840 physics simulations answers "how dangerous is this configuration?" in
~1 ms (crowd-at-risk MAE 7.9 people on configurations it never saw), and a
forecaster predicts zone density 30 s ahead (cuts the error of trend
extrapolation by 60% on held-out runs). The full C4-physics calibration confirmed
the training labels within ~6 people with the policy ranking preserved 10/10.

**4. Working perception**: pretrained crowd-density CNN
(MIT-licensed) running on CPU at the 1 Hz decision cadence, validated against
316 human-annotated images — measured bound: MAE 22.6, with a known
dense-scene undercount (~15–20%) and a per-camera calibration procedure that
corrects it during onboarding. Upgrade path to a stronger model is specified.

**5. The architecture that makes a pilot drop-in** — the provider seam: the
detector and forecaster consume abstract density grids and provably cannot
tell simulation from camera (automated tests enforce this). A camera pilot is
therefore a *configuration* exercise: calibrate the homography (4+ floor
points), set the zones, run.

**6. An operator dashboard** — local web app: live density view, latched
alarm ladder (watch 4.0 / amber 5.0 / critical 5.5 ped/m², Fruin-anchored),
30 s forecast, and the physics-ranked intervention playbook.

## Honest limitations (stated up front, not discovered later)

- Physics calibrated on European walking data; per-site Indian calibration is
  the pilot's first deliverable, not a precondition.
- The simulation cannot emergently exceed ~5 ped/m² (documented model bound);
  absolute crush-band detection (5.5+) relies on the camera side, where real
  crowds do reach 8–10 ped/m².
- Perception currently undercounts dense scenes (alarms-late direction) —
  corrected per-camera during onboarding calibration; model upgrade specified.
- Surge-state crowd parameters are literature-grounded assumptions
  (Helbing 2000), tagged as such in every result.
- All alarm thresholds are configurable per venue; defaults are
  literature-anchored (Fruin LoS), not site-certified.

## The pilot ask (shadow mode — zero operational authority, zero hardware)

1. **Access**: recorded footage from 2–4 existing platform/FOB cameras
   (historical rush-hour segments suffice to start; live RTSP later).
2. **We deliver, per camera**: calibration, per-site accuracy report
   (validated against manual counts), then a shadow-mode dossier — every
   alarm the system *would* have raised across N weeks, verified against
   what actually happened. Ordinary rush hours provide thousands of
   validation events; no incident is needed to prove the forecaster.
3. **Gate to any operational use**: miss rate ≈ 0 at an acceptable
   false-alarm rate over the shadow period, reviewed jointly. Until then the
   system recommends nothing to anyone — it only accumulates evidence.
4. **Cost to the operator**: camera access + a point of contact. The stack
   runs on a single commodity PC per station (CPU-only, demonstrated).

## Reproduce everything

```
pytest tests/ -q                      # 165 tests
python dashboard/serve.py             # the control room, on simulated replay
python scripts/phaseA_demo.py ...     # the pipeline on real public footage
```
Every number above is regenerable from the scripts in `scripts/`.
