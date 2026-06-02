# Experiment campaign — porting + feasibility notes (2026-05-25)

Goal: run the new experiments the master-review/peer-review require (repeated trials +
statistics, selective-vs-full-vs-no sharing ablation, access-fraction sweep, Map2
replication) to make the manuscript publication-ready.

## What was fixed to make the code run on Linux (all in `Main_iter.py`, `Functions_code.py`, `agent_navigation.py`)
- Replaced hardcoded `C:\Users\...` paths for maps, datasets, and the trained NN model with
  paths relative to the code directory (`env/`, `dataset/`, `trained_model_NN.pth`).
- `import os` added where missing; `torch.load(..., map_location="cpu")`.
- Guarded the unused `from torchmetrics import ConfusionMatrix` (training-only; not installed).
- Made the RRT iteration cap configurable: `RRT_MAX_ITER` env var (**default 8000 — unchanged**).
- Added a per-task step cap (`MAX_TASK_STEPS=400`) so a non-converging task is counted as
  failed instead of hanging the iteration-limited loop.
- Parameterized the runner: `python3 Main_iter.py --agents N --iters M`; `main()` now returns a
  metrics dict; added a progress heartbeat and child-process exception reporting.

Single-agent simulation now **runs correctly** (e.g., ~5 easy tasks in ~5 s).

## Feasibility finding — multi-agent runs are not tractable in this environment
Measured throughput (iteration-limited, this machine):
| Agents | Result |
|---|---|
| 1 | works; ~1 task/s on easy tasks, but multi-minute stalls on RRT-heavy tasks |
| 2 | **1 completed task in 130 s** (then stalled on the next task) |
| 5 | **0 completed tasks in 100 s** |

Root causes (architectural, not a quick bug):
1. A global `multiprocessing.Lock` serializes every navigation step across agents.
2. Per step, each agent rebuilds the obstacle map (`Update2`) treating all other agents as
   obstacles, and frequently invokes an **8000-iteration pure-Python RRT** when off-route.
3. `multiprocessing.Manager` proxy dicts add a socket round-trip per shared-state access.

The Major-Revision experiments are **all multi-agent** (selective sharing, scalability,
online-RRT comparison). At ~1 task / 2 min for 2 agents, a statistical campaign (hundreds of
tasks × 3 sharing conditions × ~10 repeats × several fleet sizes) would take **weeks**.

## Additional gap
The **selective collision-history sharing mechanism is not implemented** in this released code
(the global collision grid is accumulated but never read for route selection). It would have to
be implemented before the headline ablation could be run.

## Options to actually obtain the experimental evidence
1. **Run on the original setup/hardware** (where the thesis numbers were produced) and provide
   the result CSVs for integration. Most faithful; preserves the published method exactly.
2. **Re-engineer the simulator for speed** (single-process scheduler, capped RRT). Tractable but
   it **changes the concurrency model and absolute numbers**, so results would differ from the
   thesis and must be re-validated/owned by the PI.
3. **Submit with honest preliminary framing** (current draft) and run the experiments later;
   expect a major-revision request, as the peer-review predicted.
