#!/usr/bin/env python3
"""
Faithful ONLINE-RRT baseline (the thesis's third comparison method).
Reuses the original RRT-only navigation verbatim (Functions_code_RRT: Apply_RRT +
agent_navigation_step), where every route is generated online by RRT rather than
selected from a pre-computed library. Same deterministic single-process seeded loop
and same collision metric as sim_rrtbc.py, so the two are directly comparable.
No behavior-cloning, no route library, no collision-history sharing.
"""
import os, sys, time, json, argparse, random
import numpy as np

# Canonical online-RRT iteration cap (Table 1). Set here so a direct call to
# run_episode_rrt reproduces the released baseline even outside
# run_experiments_rrt_online.py (which also sets it); without it, Functions_code_RRT
# defaults to 15000 and the online baseline does not reproduce. setdefault keeps any
# caller-supplied value.
os.environ.setdefault("RRT_MAX_ITER", "3000")

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_CODE, "src"))
sys.path.insert(0, _HERE)
import Functions_code_RRT as FR
from scenario import _BR   # the obstacle layouts extracted from Main_iter.py

MAX_TASK_STEPS = 300


def _build_maps(map_id):
    mat = np.loadtxt(os.path.join(_CODE, "maps", f"{map_id}.txt"), delimiter="\t")
    black = _BR[map_id]
    _, up_inflated, up_black = FR.Update1(mat, black, [[0, 0]])
    return up_inflated, mat, black


def run_episode_rrt(map_id="map1", n_agents=5, steps=2000, seed=0, return_grid=False):
    random.seed(seed); np.random.seed(seed)
    up_inflated, orig_scaled, orig_black = _build_maps(map_id)
    routes = []   # online RRT ignores the library (nav_help/replan call Apply_RRT)
    _, _, static_black = FR.Update1(orig_scaled, orig_black, [[0, 0]])  # agent-free obstacles
    coll_type = [0, 0]            # [static-obstacle, robot-robot]
    FR.RRT_CALL_COUNT = 0
    _t0 = time.perf_counter()

    starts, ends = FR.generate_start_and_end(up_inflated, orig_black, num_agents=n_agents, min_distance=4)
    pos = [list(s) for s in starts]
    goals = [list(e) for e in ends]
    best_route = [[] for _ in range(n_agents)]
    k_value = [0] * n_agents
    path_taken = [[pos[i]] for i in range(n_agents)]
    agrid = [np.zeros_like(up_inflated) for _ in range(n_agents)]
    step_count = [0] * n_agents
    total_grid = np.zeros_like(up_inflated, dtype=float)

    tasks = 0; fails = 0; collisions = 0

    def new_task(i):
        s, e = FR.generate_start_and_end(up_inflated, orig_black, num_agents=1, min_distance=4)
        goals[i] = list(e[0]); pos[i] = list(s[0])
        best_route[i] = []; k_value[i] = 0; path_taken[i] = [pos[i]]
        agrid[i] = np.zeros_like(up_inflated); step_count[i] = 0

    for t in range(steps):
        for i in range(n_agents):
            obstacles = [pos[j] for j in range(n_agents) if j != i] or [[1, 1]]
            up_scaled, up_inf, up_black = FR.Update2(orig_scaled, orig_black, obstacles)
            try:
                (cc, cf, success, pt, br, kv, sf, cr) = FR.agent_navigation_step(
                    agent_id=i, routes=routes, best_route=best_route[i], end_point=goals[i],
                    current_position=pos[i], up_black_ranges=up_black, k_value=k_value[i],
                    agent_collision_grid=agrid[i], up_inflated_matrix=up_inf,
                    iteration_collision_grid=np.zeros_like(up_inf), path_taken=path_taken[i],
                    up_scaled_matrix=up_scaled,
                    static_black_ranges=static_black, coll_type_counter=coll_type)
            except Exception:
                fails += 1; new_task(i); continue
            collisions += cc
            best_route[i] = br; k_value[i] = kv; path_taken[i] = pt
            if pt:
                pos[i] = list(np.array(pt[-1]).tolist())
            step_count[i] += 1
            if success:
                if cf:
                    fails += 1
                else:
                    tasks += 1; total_grid += agrid[i]
                new_task(i)
            elif step_count[i] > MAX_TASK_STEPS:
                fails += 1; new_task(i)

    res = dict(map=map_id, dataset="rrt_online", n_agents=n_agents, share_fraction=0.0,
               steps=steps, seed=seed, n_access=0,
               tasks_completed=int(tasks), fails=int(fails), collisions=int(collisions),
               coll_obstacle=int(coll_type[0]), coll_robot=int(coll_type[1]),
               rrt_calls=int(FR.RRT_CALL_COUNT), wall_s=round(time.perf_counter() - _t0, 2))
    if return_grid:
        res["collision_grid"] = total_grid; res["obstacles"] = up_inflated
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="map1")
    ap.add_argument("--agents", type=int, default=5); ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    r = run_episode_rrt(a.map, a.agents, a.steps, a.seed)
    r["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(r))
