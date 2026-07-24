# Reproducible study: hybrid RRT + Behavior-Cloning multi-robot navigation

Deterministic, seeded code behind the article *"Environment-Specific Route-Library
Adaptation for Decentralized Multi-Robot Navigation via Hybrid RRT and Behavior Cloning in
Grid-Based Industrial Environments."* Every number and figure regenerates exactly. Requires
Python 3.12 with `numpy`, `scipy`, `matplotlib`, `pandas`, `torch`.

The released campaign uses **30 seeds per condition** across **eleven experiment families**.

## What is faithful
The simulators reuse the **original student code verbatim** — the trained Behavior-Cloning
network (`../model/trained_model_NN.pth`), the RRT planner, the offline route datasets
(`../data/`), and the per-step navigation logic in `../src/Functions_code.py` /
`../src/Functions_code_RRT.py`. Only the original multiprocessing/shared-memory harness
(which did not scale) is replaced, by a single-process deterministic round-robin loop.

## What we measure

**1. Route-library adaptation (the central result).** The same hybrid framework on Map2 with
a **Map2-naive** route library (the `original` dataset, generated on Map1) vs a
**Map2-adapted** library (the `alternative` dataset). Each library is filtered against its
*native* map and then deployed on Map2 — exactly as the original pipeline did
(`Main_iter.py`: "filter routes on map1 and after that choose map2"). The naive library keeps
routes that cut through Map2 obstacles; the adapted one does not. Result: adaptation cuts
collisions **38–85%** and task failures **43–70%** across fleet sizes (all *p* < 10⁻⁵,
Cliff's δ = 1.00), and **replicates on Map3** (65–96% collision reduction).

**2. Collision-history sharing (the earlier future-work idea, evaluated properly).** A subset
`q_share` of robots avoid historically collision-prone cells via a shared map. Episodes run
to a realistic task throughput (`steps=2000` ≈ 245 completed tasks at 5 robots) so the shared
database fills and the test has power. Result: only a **limited, layout-dependent** benefit —
on Map1, −9.3% under selective (*p* = 0.024) and −7.7% under full participation
(*p* = 0.047), **neither surviving Holm correction** (0.097 and 0.14); Map2 not significant;
the access-fraction sweep is flat (Spearman ρ = −0.06). The earlier single-run "halving" does
not reproduce. Below ~1000 steps the effect is undetectable — the database stays nearly empty.

**3. Baselines that bound the effect (added in revision).** Offline library remedies
(filter 43–86%, repair 39–86%, regenerate 47–89%) bracket the adapted library; a non-learning
scripted connector performs at least as well as the trained BC network; prioritized planning
over a shared reservation table dominates every communication-free method (≈2× tasks,
0.02–0.14 collisions per completed task, failure rate ≤ 0.01); and capping the online-RRT
planning budget collapses runtime (922 → 88 → 53 s at ten robots) with little change in
collisions per task (2.73 / 2.65 / 2.35) but a steeply rising failure rate (0.12 → 0.22 → 0.60).

## Quick start
```bash
# analysis only — the released results_*.csv are included
python3 analyze_rrtbc.py                         # main stats + LaTeX tables + performance figures
python3 analyze_revision.py                      # revision stats + tables
python3 make_heatmaps_rrtbc.py                   # spatial/schematic figures + route-diagnostics table

# full re-run (optional; seeded, so numbers match apart from wall-clock)
python3 run_experiments_rrtbc.py --reps 30       # -> results_rrtbc.csv (adaptation + sharing)
python3 run_experiments_rrt_online.py --reps 30  # -> results_rrt_online.csv (online-RRT baseline)
python3 run_experiments_rrtbc.py --throughput    # -> results_throughput.csv (throughput sweep)
python3 gen_map3.py                              # -> ../maps/map3.txt, ../data/map3_data.csv
python3 lib_variants.py                          # -> filtered / repaired / regenerated libraries
python3 run_experiments_revision.py --reps 30    # -> results_revision.csv (the five revision families)
```
`campaign_all.sh`, `post_campaign.sh` and `post_revision.sh` chain these steps; the matching
`.log` files record the campaign that produced the released CSVs. Figures land in
`../figures/`, generated LaTeX tables in `../tables/`.

## Reproducibility (verified)
The simulators are fully deterministic given the per-episode seed: re-running any episode
reproduces the stored `tasks_completed`, `fails`, `collisions`, `coll_obstacle`,
`coll_robot` and `rrt_calls` exactly. This was verified two ways — (i) per-episode re-runs
across every experiment family reproduce the committed CSV rows exactly, and (ii) a full
re-run of the campaign matches the committed `results_*.csv` on every column except `wall_s`.

Notes for exact reproduction:
- **`wall_s` is hardware-dependent** (wall-clock runtime) and is the only column that varies
  between machines; all result columns reproduce exactly.
- **Online-RRT iteration cap.** The online baseline must run with `RRT_MAX_ITER=3000`
  (Table 1); the reduced-budget arms use 1000 and 300. Both `run_experiments_rrt_online.py`
  and `sim_rrt_online.py` set the default, so the baseline reproduces either way. (The hybrid
  is insensitive to the cap: its RRT reconnects terminate in far fewer iterations.)
- Tested environment: Python 3.12, `numpy` 2.3, `scipy` 1.16, `pandas` 2.3, `torch` 2.9.
  Required assets (all released): `../model/trained_model_NN.pth`, `../data/*.csv`,
  `../maps/{map1,map2,map3}.txt`.

## Reproduction manifest
Which script and raw input produce each article output:

| Output | Script | Raw input | Produces |
|---|---|---|---|
| Route-library adaptation table + figure | `analyze_rrtbc.py` | `results_rrtbc.csv`, `results_rrt_online.csv` | means, bootstrap CIs, MWU/permutation *p*, Cliff's δ; collisions/task and failure-rate curves |
| Collision heatmaps | `make_heatmaps_rrtbc.py` | seeded episodes (30 seeds) | per-cell blocked-contact density (per 100 completed tasks) |
| Collision-composition figure + table | `analyze_rrtbc.py` | `results_rrtbc.csv`, `results_rrt_online.csv` | static-obstacle vs robot–robot split, all fleet sizes |
| Planning-cost figure + table | `analyze_rrtbc.py` | `results_rrtbc.csv`, `results_rrt_online.csv` | runtime (mean ± SD) and RRT calls/task |
| Sharing statistics table | `analyze_rrtbc.py` | `results_rrtbc.csv` | nominal, permutation and Holm-corrected *p*-values |
| Sharing scalability / sweep / throughput figures | `analyze_rrtbc.py` | `results_rrtbc.csv`, `results_throughput.csv` | rate vs fleet size, vs access fraction, vs episode length |
| Library-remedy table | `analyze_revision.py` | `results_revision.csv` | filter / repair / regenerate vs naive and adapted |
| Alternative-planner + connector table | `analyze_revision.py` | `results_revision.csv` | scripted connector, prioritized planning, online-RRT caps |
| Compute-budget frontier figure | `analyze_revision.py` | `results_revision.csv`, `results_rrtbc.csv` | runtime vs collisions/task across planning budgets |
| Map3 transfer table | `analyze_revision.py` | `results_revision.csv` | Map3-naive vs Map3-adapted, all fleet sizes |
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
| `sim_prioritized.py` | Prioritized planning: space–time A* over a shared reservation table (not communication-free). |
| `scenario.py` | Builds scenarios from the original code; filters each route library against its **native** map. |
| `lib_variants.py` | Offline library remedies: filter, repair, regenerate invalid transferred routes. |
| `gen_map3.py` | Generates the Map3 corridor layout and its adapted route library. |
| `run_experiments_rrtbc.py` | Adaptation (Map2, N=2..10) + sharing campaign → `results_rrtbc.csv`. |
| `run_experiments_rrt_online.py` | Online-RRT baseline over the same fleet sizes → `results_rrt_online.csv`. |
| `run_experiments_revision.py` | Remedies, scripted connector, prioritized planning, budget caps, Map3 → `results_revision.csv`. |
| `analyze_rrtbc.py` | Statistics (Mann–Whitney U, permutation, Holm, Spearman) + main figures/tables. |
| `analyze_revision.py` | Statistics and tables for the five revision experiment families. |
| `make_heatmaps_rrtbc.py` | Blocked-contact heatmaps on Map2 + schematic figures + route diagnostics. |
| `results_*.csv` | Raw per-episode results (30 seeds). |
| `stats_summary_rrtbc.md`, `stats_summary_revision.md` | Human-readable statistics. |

## Key finding
Navigation quality is governed by the match between the expert route library and the
deployment environment: regenerating the library on the deployment map transfers the learned
navigation and sharply reduces collisions and failures, and the effect replicates on a third,
structurally different layout. The learned connector is replaceable — a scripted greedy
controller does as well — so the environment-specific knowledge lives in the route library.
Communication-free collision-history sharing adds only a limited, layout-dependent benefit
that does not survive multiple-comparison correction.
