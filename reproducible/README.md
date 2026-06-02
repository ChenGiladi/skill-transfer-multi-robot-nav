# Reproducible study: hybrid RRT + Behavior-Cloning multi-robot navigation

Deterministic, seeded code behind the manuscript *"Robot-to-Robot Skill Transfer for
Decentralized Multi-Robot Navigation via Hybrid RRT and Behavior Cloning in Grid-Based
Industrial Environments."* Every number and figure regenerates exactly. Requires
Python 3 with `numpy`, `scipy`, `matplotlib`, `pandas`, `torch`.

## What is faithful
The simulators reuse the **original student code verbatim** — the trained Behavior-Cloning
network (`../trained_model_NN.pth`), the RRT planner, the offline route datasets
(`../dataset/`), and the per-step navigation logic in `../Functions_code.py` /
`../Functions_code_RRT.py`. Only the original multiprocessing/shared-memory harness
(which did not scale) is replaced, by a single-process deterministic round-robin loop.

## Two things we measure
1. **Skill transfer (the validated thesis result).** The same hybrid framework on Map2
   with a **Map2-naive** route library (the `original` dataset, generated on Map1) vs a
   **Map2-adapted** library (the `alternative` dataset). Each library is filtered against
   its *native* map (Map1 vs Map2) and then deployed on Map2 — exactly as the original
   pipeline did (`Main_iter.py`: "filter routes on map1 and after that choose map2").
   The map-naive library keeps routes that cut through Map2 obstacles; the adapted one
   does not. Result: adaptation cuts collisions 37–85% and failures 43–69% across fleet
   sizes (all p<10⁻⁵).
2. **Collision-history sharing (the thesis's future-work idea, evaluated properly).** A
   subset `p` of robots avoid historically collision-prone cells via a shared map.
   Episodes run to a realistic task throughput (`steps=2000` ≈ 250 completed tasks at 5
   robots) so the shared database fills and the test has power. Result: a limited benefit
   (full sharing ~13% on Map1, p=0.02; selective and Map2 not significant) — the earlier
   single-run "halving" does not reproduce within the capable framework. (At the old
   `steps=300` ≈ 37 tasks the database stayed nearly empty and no effect was detectable —
   an under-powered artifact we report explicitly.)

## Quick start
```bash
python3 run_experiments_rrtbc.py --reps 15      # -> results_rrtbc.csv (skill transfer + sharing)
python3 run_experiments_rrt_online.py --reps 15 # -> results_rrt_online.csv (online-RRT baseline)
python3 run_experiments_rrtbc.py --throughput   # -> results_throughput.csv (throughput-dependence sweep)
python3 analyze_rrtbc.py                         # stats + LaTeX tables + performance figures
python3 make_heatmaps_rrtbc.py                   # spatial/schematic figures + route-diagnostics table
```
Figures land in `../../figures/`, generated LaTeX tables in `../../tables/`.

## Reproducibility (verified)
The simulators are fully deterministic given the per-episode seed: re-running any episode
reproduces the stored `tasks_completed`, `fails`, `collisions`, `coll_obstacle`,
`coll_robot` and `rrt_calls` exactly. This was verified two ways — (i) per-episode
re-runs across every experiment family (skill transfer on both libraries, sharing
ablation/sweep/scalability, and the online-RRT baseline) reproduce the committed CSV rows
exactly, and (ii) a full re-run of the campaign matches the committed
`results_*.csv` on every column except `wall_s`.

Notes for exact reproduction:
- **`wall_s` is hardware-dependent** (wall-clock runtime) and is the only column that
  varies between machines; all result columns reproduce exactly.
- **Online-RRT iteration cap.** The online baseline must run with `RRT_MAX_ITER=3000`
  (Table 1). Both `run_experiments_rrt_online.py` and `sim_rrt_online.py` set this by
  default, so the baseline reproduces whether you run the campaign or call
  `run_episode_rrt` directly. (The hybrid is insensitive to the cap: its RRT reconnects
  terminate in far fewer iterations.)
- Tested environment: Python 3.12, `numpy` 2.3, `scipy` 1.16, `pandas` 2.3, `torch` 2.9.
  Required assets (all released): `../trained_model_NN.pth`, `../dataset/*.csv`,
  `../env/{map1,map2}.txt`.

## Reproduction manifest
Which script and raw input produce each manuscript output:

| Output | Script | Raw input | Produces |
|---|---|---|---|
| Skill-transfer table | `analyze_rrtbc.py` | `results_rrtbc.csv`, `results_rrt_online.csv` | means, bootstrap CIs, MWU *p*-values, Cliff's δ |
| Skill-transfer figure | `analyze_rrtbc.py` | `results_rrtbc.csv`, `results_rrt_online.csv` | collisions/task and failure-rate curves with 95% CI bands |
| Collision heatmaps | `make_heatmaps_rrtbc.py` | seeded episodes (15 seeds) | per-cell collision heatmaps (per 100 completed tasks) |
| Collision-composition figure + table | `analyze_rrtbc.py` | `results_rrtbc.csv`, `results_rrt_online.csv` | static-obstacle vs robot–robot split, all fleet sizes |
| Planning-cost figure + table | `analyze_rrtbc.py` | `results_rrtbc.csv`, `results_rrt_online.csv` | runtime (mean ± SD) and RRT calls/task |
| Sharing statistics table | `analyze_rrtbc.py` | `results_rrtbc.csv` | nominal and Holm-corrected *p*-values |
| Sharing scalability figure | `analyze_rrtbc.py` | `results_rrtbc.csv` | throughput and collision rate vs fleet size (Map1) |
| Access-fraction sweep figure | `analyze_rrtbc.py` | `results_rrtbc.csv` | collision rate vs sharing fraction (Map1) |
| Throughput-dependence figure | `analyze_rrtbc.py` | `results_throughput.csv` | sharing effect vs episode length |
| Environments / framework / route-mismatch figures + route-diagnostics table | `make_heatmaps_rrtbc.py` | map layouts + route datasets | schematics and library size/coverage/overlap |

The implementation-settings and experimental-design tables are hand-specified (they list
fixed hyperparameters and the experiment matrix) and are not produced by a script.

Runtime numbers in the planning-cost table are hardware-sensitive (measured on a single
pinned CPU thread; the analysis script records the CPU model and Python version in the
table caption).

## Files
| File | Purpose |
|---|---|
| `sim_rrtbc.py` | Faithful hybrid RRT+BC simulator (single-process seeded loop) + optional collision-history sharing. |
| `sim_rrt_online.py` | Faithful online-RRT baseline (pure RRT each route; no library, no BC, no sharing). |
| `scenario.py` | Builds scenarios from the original code; filters each route library against its **native** map (the skill-transfer fix). |
| `run_experiments_rrtbc.py` | Skill-transfer (Map2, N=2..10) + sharing (ablation/sweep/scalability) campaign → `results_rrtbc.csv`. |
| `run_experiments_rrt_online.py` | Online-RRT baseline over the same fleet sizes → `results_rrt_online.csv`. |
| `analyze_rrtbc.py` | Statistics (Mann–Whitney U, Spearman) + figures (skill transfer, sharing sweep, scalability). |
| `make_heatmaps_rrtbc.py` | Per-cell collision heatmaps on Map2 (naive vs adapted hybrid vs online RRT). |
| `results_rrtbc.csv` / `results_rrt_online.csv` | Raw per-episode results. |
| `stats_summary_rrtbc.md` | Human-readable statistics. |

## Key finding
Navigation quality is governed by the match between the expert route library and the
deployment environment: an environment-adapted library transfers the learned skill and
sharply reduces collisions and failures. Communication-free collision-history sharing
adds only a limited benefit within this capable framework, and requires adequate task
throughput to be measurable at all.
