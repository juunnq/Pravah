# Pravah — Predictive Crowd-Safety Intelligence

**Predict where and when a crowd turns dangerous — minutes before it is
visible — and know which intervention prevents it.**

Built from a peer-reviewed pedestrian-physics engine (SIMULTECH 2026,
calibrated on 4,776 real trajectories), distilled into a millisecond-fast ML
brain, behind a clean seam that any CCTV camera can plug into. Motivated by
the New Delhi foot-overbridge crush (Feb 2025) and every disaster that shares
its mechanism.

## The headline result (full physics, n=4 seeds, zero overlap)

Reconstructing the real disaster mechanism — a fall blocking a crowded
foot-overbridge mid-surge:

| Same fall, same 150 people | In the crush pressure column |
|---|---|
| Burst release, legacy 3.66 m bridge | **106–115 people** |
| Burst release, mandated 6.0 m bridge | 83–95 |
| **Phased release, legacy bridge** | **34–52** |

*You can't always prevent the fall. Release policy decides how many people
are standing in its blast radius — and a wider bridge alone won't save them.*

## Quickstart (fresh clone, CPU-only)

```bash
pip install -e .                      # core (numpy/scipy/shapely/sklearn...)
pytest tests/ -q                      # 165 tests, ~5-10 min
python scripts/quickstart_demo.py     # ~3 min of simulation, then opens the
                                      # control room at http://localhost:8750
```

The dashboard shows both release policies side by side: live density grid,
latched alarm ladder (watch/amber/critical), pressure-column trend, a 30 s
forecast, and the physics-ranked intervention playbook.

### Run it on real footage

```bash
pip install -e .[perception]          # + opencv, torch, huggingface_hub
python scripts/phaseA_calibrate.py --points cam.json --out calib.json --frame f.jpg
python scripts/phaseA_demo.py --frames <dir> --calib calib.json --out demo.mp4
```

Perception uses an MIT-licensed pretrained crowd-density CNN (CPU, ~3 s/frame
— ample at the 1 Hz decision cadence). Measured accuracy bound on 316
ground-truth images: MAE 22.6 with a known dense-scene undercount;
per-camera calibration corrects it during onboarding.

## Architecture — the seam is the point

```
   SIMULATION (offline oracle)          LIVE CAMERA (deployment)
   physics engine -> trajectories       frame -> density CNN -> homography
              \                              /
               ->  CrowdState density grid <-        (providers.py)
                          |
        ThresholdDetector + ZoneForecaster + playbook   (the brain)
                          |
                 operator dashboard / alarms
```

The brain consumes abstract density grids and **provably cannot tell
simulation from camera** (automated tests enforce it). That is what makes a
venue pilot a configuration exercise, not an integration project — and what
lets the slow, validated physics train the fast, deployable forecaster
(surrogate: crowd-at-risk MAE 11.3 on unseen configurations; forecaster:
halves the error of trend extrapolation).

## Honesty, by construction

Every result ships with its validity bounds: the physics' emergent density
ceiling and OOD bias, the perception model's measured undercount and its
direction (alarms-late), the surge-state assumptions (all `[ASSUMPTION]`-
tagged), and per-seed distributions rather than favorable subsets. See
`PROPOSAL.md` for the full evidence summary.

## Repository layout

```
sim/                the engine + the brain (core, steering, density,
                    scenarios, providers, detector, forecaster, perception)
scripts/            experiments, training, validation, demos (phase-stamped)
dashboard/          the control room (stdlib server + one HTML page)
onboard/            camera calibration examples & procedure
tests/              165 tests — the contract
PROPOSAL.md         the evidence summary, every number regenerable
```

## Data & licensing notes

This repo ships **no third-party footage or datasets**. All project code is
MIT-licensed (see `LICENSE`), matching the dependency stack. Peer-reviewed
core: Gang & Veluri, SIMULTECH 2026.
