#!/usr/bin/env python3
"""
Faithful RRT + Behavior-Cloning experiment campaign -> results_rrtbc.csv.

Reuses the original RRT planner, the trained BC network, and the route datasets;
runs every episode in a deterministic single-process loop, parallelized across cores.

Two families of experiments:
  * SKILL TRANSFER (the thesis's validated result): the same hybrid framework on Map2
    with a Map2-naive route library ("original", generated on Map1) vs a Map2-adapted
    library ("alternative"). Reproduces the environment-mismatch / data-adaptation effect.
  * COLLISION-HISTORY SHARING (the thesis's future-work idea, evaluated properly): a
    subset p of robots avoid historically collision-prone cells via a shared map.
    Run to a high task throughput (steps=2000 ~ 250 completed tasks) so the shared
    database actually fills and the comparison has statistical power.

Episodes run long enough that collisions are reported BOTH absolutely and per completed
task downstream (see analyze_rrtbc.py). Set RRT_MAX_ITER to bound the online RRT.
"""
import os
# pin each worker to one thread (avoid torch/BLAS oversubscription across the Pool)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("RRT_MAX_ITER", "8000")   # must be set before importing Functions_code
os.environ.setdefault("MPLBACKEND", "Agg")
import csv, time, argparse
import torch
torch.set_num_threads(1)
from multiprocessing import Pool, cpu_count
from sim_rrtbc import run_episode_rrtbc

STEPS = 2000   # ~250 completed tasks at 5 agents: enough for the shared DB to fill and
               # for collision-rate comparisons to have power (the old STEPS=300 gave ~37
               # tasks -- the DB stayed nearly empty and the test was badly under-powered).


def build_configs(R):
    cfgs, seeds = [], list(range(R))
    native = {"map1": "original", "map2": "alternative"}

    # --- T0 SKILL TRANSFER (Map2): Map2-naive "original" vs Map2-adapted "alternative" ---
    for s in seeds:
        for n in (2, 4, 6, 8, 10):
            for ds in ("original", "alternative"):
                cfgs.append(dict(exp="T0_skill_transfer_map2", map_id="map2", dataset=ds,
                                 n_agents=n, share_fraction=0.0, steps=STEPS, seed=s))

    # --- S1 sharing ablation -- Map1 (native original) and Map2 (native alternative), 5 agents ---
    for s in seeds:
        for mp in ("map1", "map2"):
            for sh in (0.0, 0.6, 1.0):
                cfgs.append(dict(exp=f"S1_ablation_{mp}", map_id=mp, dataset=native[mp],
                                 n_agents=5, share_fraction=sh, steps=STEPS, seed=s))

    # --- S2 access-fraction sweep -- Map1, 5 agents ---
    for s in seeds:
        for sh in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            cfgs.append(dict(exp="S2_sweep_map1", map_id="map1", dataset="original",
                             n_agents=5, share_fraction=sh, steps=STEPS, seed=s))

    # --- S3 sharing scalability -- Map1, agents 2..10, none vs selective(0.6) ---
    for s in seeds:
        for n in (2, 4, 6, 8, 10):
            for sh in (0.0, 0.6):
                cfgs.append(dict(exp="S3_scalability_map1", map_id="map1", dataset="original",
                                 n_agents=n, share_fraction=sh, steps=STEPS, seed=s))
    return cfgs


# Task-budget horizons used to show WHY collision-history sharing only helps once enough
# tasks have populated the shared map (Fig. 8): short budgets keep the database nearly empty.
THROUGHPUT_HORIZONS = (300, 600, 1000, 1500, 2000)


def build_throughput_configs(R):
    """S4 throughput-dependence of sharing: Map1, 5 robots, p in {0,0.6,1.0} swept over
    increasing task budgets. Same seeds/conditions as S1 at the longest horizon, so the
    long-horizon slice reproduces the S1 ablation."""
    cfgs, seeds = [], list(range(R))
    for s in seeds:
        for steps in THROUGHPUT_HORIZONS:
            for sh in (0.0, 0.6, 1.0):
                cfgs.append(dict(exp="S4_throughput_map1", map_id="map1", dataset="original",
                                 n_agents=5, share_fraction=sh, steps=steps, seed=s))
    return cfgs


def worker(cfg):
    r = run_episode_rrtbc(map_id=cfg["map_id"], dataset=cfg["dataset"],
                          n_agents=cfg["n_agents"], steps=cfg["steps"],
                          seed=cfg["seed"], share_fraction=cfg["share_fraction"])
    r["exp"] = cfg["exp"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--procs", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--throughput", action="store_true",
                    help="run the S4 throughput-dependence campaign (Fig. 8) instead")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    if a.throughput:
        cfgs = build_throughput_configs(a.reps)
        out = a.out or os.path.join(here, "results_throughput.csv")
        tag = "throughput"
    else:
        cfgs = build_configs(a.reps)
        out = a.out or os.path.join(here, "results_rrtbc.csv")
        tag = "main"
    print(f"RRT+BC episodes [{tag}]: {len(cfgs)} | reps={a.reps} | procs={a.procs} | "
          f"RRT_MAX_ITER={os.environ['RRT_MAX_ITER']}", flush=True)
    t0 = time.time(); rows = []
    with Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(worker, cfgs, chunksize=1), 1):
            rows.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(cfgs)} ({time.time()-t0:.0f}s)", flush=True)
    cols = ["exp", "map", "dataset", "n_agents", "share_fraction", "n_access",
            "steps", "seed", "tasks_completed", "fails", "collisions",
            "coll_obstacle", "coll_robot", "rrt_calls", "wall_s"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"DONE: {len(rows)} rows -> {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
