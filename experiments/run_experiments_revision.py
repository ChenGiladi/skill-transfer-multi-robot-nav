#!/usr/bin/env python3
"""
Revision-1 experiment campaign -> results_revision.csv.

New conditions requested by the reviewers of robotics-4389295:
  R1  Library-variant baselines on Map2 (Reviewer B c.3): the Map2-naive library
      after offline FILTER / REPAIR / REGEN transforms (lib_variants.py).
  R2  Non-learning connector ablation (Reviewer B c.8): the hybrid with the trained
      BC connector replaced by a scripted greedy controller, Map2, naive+adapted.
  R3  Prioritized planning with space-time A* (Reviewer A c.6): mainstream MAPF
      baseline (CBS family), Map2.
  R4  Compute-budget frontier for online RRT (Reviewer B c.7): the online baseline
      rerun at reduced iteration caps {1000, 300} (paper cap: 3000), Map2.
  R5  Third environment (Reviewer B c.4): Map3 corridor/bottleneck layout; transfer
      experiment Map3-naive (Map1 library) vs Map3-adapted (Map3 library).

All conditions: 30 seeds, fleet sizes {2,4,6,8,10}, 2000 scheduler steps.
The 'method' column disambiguates; rrt_calls holds planner calls for every method.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("RRT_MAX_ITER", "8000")
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
import csv, time, argparse
import torch
torch.set_num_threads(1)
from multiprocessing import Pool, cpu_count

STEPS = 2000
FLEETS = (2, 4, 6, 8, 10)


def build_configs(R):
    seeds = list(range(R))
    cfgs = []
    for s in seeds:
        for n in FLEETS:
            # R1 library variants (deployment map2, naive library transformed offline)
            for v in ("filter", "repair", "regen"):
                cfgs.append(dict(kind="libvariant", exp="R1_libvariants_map2", map_id="map2",
                                 dataset="original", variant=v, n_agents=n, seed=s))
            # R2 scripted-connector ablation
            for ds in ("original", "alternative"):
                cfgs.append(dict(kind="scripted", exp="R2_scripted_map2", map_id="map2",
                                 dataset=ds, n_agents=n, seed=s))
            # R3 prioritized planning
            cfgs.append(dict(kind="prioritized", exp="R3_prioritized_map2", map_id="map2",
                             n_agents=n, seed=s))
            # R4 online RRT at reduced caps
            for cap in (1000, 300):
                cfgs.append(dict(kind="online_cap", exp=f"R4_online_cap{cap}_map2",
                                 map_id="map2", rrt_cap=cap, n_agents=n, seed=s))
            # R5 Map3 transfer
            for ds in ("original", "map3lib"):
                cfgs.append(dict(kind="hybrid", exp="R5_transfer_map3", map_id="map3",
                                 dataset=ds, n_agents=n, seed=s))
    return cfgs


def worker(cfg):
    kind = cfg["kind"]
    if kind == "prioritized":
        from sim_prioritized import run_episode_prioritized
        r = run_episode_prioritized(map_id=cfg["map_id"], n_agents=cfg["n_agents"],
                                    steps=STEPS, seed=cfg["seed"])
        r["method"] = "prioritized"
    elif kind == "online_cap":
        import Functions_code_RRT as FR
        from sim_rrt_online import run_episode_rrt
        # RRT.__init__ evaluates its max_iter default from the environment at class
        # definition (import) time, so rebind the default explicitly per episode --
        # pool workers are reused across configs with different caps.
        d = list(FR.RRT.__init__.__defaults__)
        d[3] = int(cfg["rrt_cap"])                     # (expand_dis, path_res, goal_rate, max_iter, ...)
        FR.RRT.__init__.__defaults__ = tuple(d)
        r = run_episode_rrt(map_id=cfg["map_id"], n_agents=cfg["n_agents"],
                            steps=STEPS, seed=cfg["seed"])
        r["method"] = f"online_cap{cfg['rrt_cap']}"
    elif kind == "libvariant":
        from lib_variants import get_variant_routes
        from sim_rrtbc import run_episode_rrtbc
        routes = get_variant_routes(cfg["map_id"], cfg["dataset"], cfg["variant"])
        r = run_episode_rrtbc(map_id=cfg["map_id"], dataset=cfg["dataset"],
                              n_agents=cfg["n_agents"], steps=STEPS, seed=cfg["seed"],
                              routes_override=routes)
        r["method"] = f"lib_{cfg['variant']}"
    elif kind == "scripted":
        from sim_rrtbc import run_episode_rrtbc
        r = run_episode_rrtbc(map_id=cfg["map_id"], dataset=cfg["dataset"],
                              n_agents=cfg["n_agents"], steps=STEPS, seed=cfg["seed"],
                              policy="scripted")
        r["method"] = f"scripted_{cfg['dataset']}"
    else:  # hybrid (Map3 transfer)
        from sim_rrtbc import run_episode_rrtbc
        r = run_episode_rrtbc(map_id=cfg["map_id"], dataset=cfg["dataset"],
                              n_agents=cfg["n_agents"], steps=STEPS, seed=cfg["seed"])
        r["method"] = f"hybrid_{cfg['dataset']}"
    r["exp"] = cfg["exp"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--procs", type=int, default=max(1, cpu_count() - 2))
    ap.add_argument("--only", default=None, help="comma list of exp prefixes to run (e.g. R1,R5)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    cfgs = build_configs(a.reps)
    if a.only:
        pref = tuple(p.strip() for p in a.only.split(","))
        cfgs = [c for c in cfgs if c["exp"].startswith(pref)]
    out = a.out or os.path.join(here, "results_revision.csv")
    print(f"Revision episodes: {len(cfgs)} | reps={a.reps} | procs={a.procs}", flush=True)
    t0 = time.time(); rows = []
    with Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(worker, cfgs, chunksize=1), 1):
            rows.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(cfgs)} ({time.time()-t0:.0f}s)", flush=True)
    cols = ["exp", "method", "map", "dataset", "n_agents", "share_fraction", "n_access",
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
