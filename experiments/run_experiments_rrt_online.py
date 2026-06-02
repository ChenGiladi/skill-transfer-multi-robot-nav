#!/usr/bin/env python3
"""
Online-RRT baseline campaign -> results_rrt_online.csv.
The thesis's third comparison method (pure online RRT, no library, no BC), run over the
same Map2 fleet-size sweep as the hybrid skill-transfer experiment so the three methods
(Map2-naive 'original' hybrid, Map2-adapted 'alternative' hybrid, online RRT) line up.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# Bound the online RRT so it fails fast under congestion (the unbounded 15000-iter
# default makes dense episodes intractable); 3000 keeps failure rates low and feasible.
os.environ.setdefault("RRT_MAX_ITER", "3000")
os.environ.setdefault("MPLBACKEND", "Agg")
import csv, time, argparse
import torch
torch.set_num_threads(1)
from multiprocessing import Pool, cpu_count
from sim_rrt_online import run_episode_rrt

STEPS = 2000


def build_configs(R):
    cfgs = []
    for s in range(R):
        for mp in ("map1", "map2"):
            for n in (2, 4, 6, 8, 10):
                cfgs.append(dict(map_id=mp, n_agents=n, steps=STEPS, seed=s))
    return cfgs


def worker(cfg):
    return run_episode_rrt(map_id=cfg["map_id"], n_agents=cfg["n_agents"],
                           steps=cfg["steps"], seed=cfg["seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--procs", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_rrt_online.csv"))
    a = ap.parse_args()
    cfgs = build_configs(a.reps)
    print(f"online-RRT episodes: {len(cfgs)} | reps={a.reps} | procs={a.procs} | STEPS={STEPS}", flush=True)
    t0 = time.time(); rows = []
    with Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(worker, cfgs, chunksize=1), 1):
            rows.append(r)
            if i % 20 == 0:
                print(f"  {i}/{len(cfgs)} ({time.time()-t0:.0f}s)", flush=True)
    cols = ["map", "dataset", "n_agents", "share_fraction", "n_access",
            "steps", "seed", "tasks_completed", "fails", "collisions",
            "coll_obstacle", "coll_robot", "rrt_calls", "wall_s"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"DONE: {len(rows)} rows -> {a.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
