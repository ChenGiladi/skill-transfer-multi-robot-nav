#!/usr/bin/env python3
"""
FAITHFUL RRT + Behavior-Cloning multi-robot simulator (reproducible).
Reuses the ORIGINAL student navigation verbatim -- the trained BC network, the RRT
planner, route reuse and probabilistic replanning (Functions_code.agent_navigation_step)
-- but runs the agents in a single, deterministic, seeded process (round-robin),
replacing the original multiprocessing/Manager/global-lock harness that did not scale.
Optional collision-history sharing is layered on via Functions_code.Get_best_routes
(a subset of robots consult the shared collision grid). Set RRT_MAX_ITER to bound RRT.
"""
import os, sys, time, json, argparse, random
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
import Functions_code as F
from scenario import build_scenario

MAX_TASK_STEPS = 300  # safeguard against non-converging tasks (counts as failed)

# ---- Revision-1 additions -----------------------------------------------------------
# (i) routes_override: deploy a transformed route library (see lib_variants.py) while
#     keeping every other component byte-identical.
# (ii) policy="scripted": replace the trained BC connector with a simple non-learning
#      greedy route-following controller (Reviewer B, comment 8 ablation). The scripted
#      connector honors the exact Apply_BC contract: called to bridge the robot onto a
#      selected route start 2-7 cells away, returns (success, [positions after start]),
#      terminates within 1.5 cells of the target, and gives up after 100 steps.
_APPLY_BC_ORIG = F.Apply_BC


def _scripted_connector(model, start_point, goal_point, up_scaled_matrix, up_black_ranges):
    pos = [float(start_point[0]), float(start_point[1])]
    positions = [list(pos)]
    for _ in range(100):
        dx, dy = goal_point[0] - pos[0], goal_point[1] - pos[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= 1.5:
            return True, positions[1:]
        ux, uy = dx / dist, dy / dist
        moved = False
        # try the direct unit step, then deterministic +/-45 and +/-90 degree deviations
        for ca, sa in ((1.0, 0.0), (0.7071, 0.7071), (0.7071, -0.7071), (0.0, 1.0), (0.0, -1.0)):
            nx = pos[0] + ux * ca - uy * sa
            ny = pos[1] + ux * sa + uy * ca
            if not (0 <= nx <= 49 and 0 <= ny <= 49):
                continue
            if F.is_point_inside_black_ranges(nx, ny, up_black_ranges):
                continue
            pos = [nx, ny]
            positions.append(list(pos))
            moved = True
            break
        if not moved:
            return False, positions[1:]
    return False, positions[1:]


def _set_policy(policy):
    F.Apply_BC = _APPLY_BC_ORIG if policy == "bc" else _scripted_connector
# --------------------------------------------------------------------------------------


def run_episode_rrtbc(map_id="map1", dataset="original", n_agents=5, steps=600,
                      seed=0, share_fraction=0.0, return_grid=False, return_history=False,
                      routes_override=None, policy="bc"):
    random.seed(seed); np.random.seed(seed)
    _set_policy(policy)
    routes, up_inflated, orig_scaled, orig_black = build_scenario(map_id, dataset)
    if routes_override is not None:
        routes = routes_override
    R, C = up_inflated.shape
    # static (agent-free) obstacle ranges for the deployment map -> lets us classify each
    # collision as static-obstacle vs robot-robot without changing the navigation logic.
    _, _, static_black = F.Update1(orig_scaled, orig_black, [[0, 0]])
    coll_type = [0, 0]            # [static-obstacle, robot-robot]
    F.RRT_CALL_COUNT = 0          # reset opt-in planner-call counter
    _t0 = time.perf_counter()
    history = []                  # (tasks_completed, populated cells in shared grid)

    starts, ends = F.generate_start_and_end(up_inflated, orig_black, num_agents=n_agents, min_distance=4)
    pos = [list(s) for s in starts]
    goals = [list(e) for e in ends]
    best_route = [[] for _ in range(n_agents)]
    k_value = [0] * n_agents
    path_taken = [[pos[i]] for i in range(n_agents)]
    agrid = [np.zeros_like(up_inflated) for _ in range(n_agents)]
    step_count = [0] * n_agents

    total_grid = np.zeros_like(up_inflated, dtype=float)
    access = np.zeros(n_agents, dtype=bool)
    na = int(round(share_fraction * n_agents))
    if na:
        access[np.random.choice(n_agents, na, replace=False)] = True

    tasks = 0; fails = 0; collisions = 0

    def new_task(i):
        s, e = F.generate_start_and_end(up_inflated, orig_black, num_agents=1, min_distance=4)
        goals[i] = list(e[0]); pos[i] = list(s[0])
        best_route[i] = []; k_value[i] = 0; path_taken[i] = [pos[i]]
        agrid[i] = np.zeros_like(up_inflated); step_count[i] = 0

    for t in range(steps):
        for i in range(n_agents):
            obstacles = [pos[j] for j in range(n_agents) if j != i] or [[1, 1]]
            up_scaled, up_inf, up_black = F.Update2(orig_scaled, orig_black, obstacles)
            try:
                (cc, cf, success, pt, br, kv, sf, cr) = F.agent_navigation_step(
                    agent_id=i, routes=routes, best_route=best_route[i], end_point=goals[i],
                    current_position=pos[i], up_black_ranges=up_black, k_value=k_value[i],
                    agent_collision_grid=agrid[i], up_inflated_matrix=up_inf,
                    iteration_collision_grid=np.zeros_like(up_inf), path_taken=path_taken[i],
                    up_scaled_matrix=up_scaled,
                    collision_grid=(total_grid if access[i] else None),
                    db_alpha=(1.0 if access[i] else 0.0),
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
                    if return_history:
                        history.append((int(tasks), int((total_grid > 0).sum())))
                new_task(i)
            elif step_count[i] > MAX_TASK_STEPS:
                fails += 1; new_task(i)

    res = dict(map=map_id, dataset=dataset, n_agents=n_agents, share_fraction=share_fraction,
               steps=steps, seed=seed, n_access=int(access.sum()),
               tasks_completed=int(tasks), fails=int(fails), collisions=int(collisions),
               coll_obstacle=int(coll_type[0]), coll_robot=int(coll_type[1]),
               rrt_calls=int(F.RRT_CALL_COUNT), wall_s=round(time.perf_counter() - _t0, 2))
    if return_grid:
        res["collision_grid"] = total_grid; res["obstacles"] = up_inflated
    if return_history:
        res["history"] = history
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="map1"); ap.add_argument("--dataset", default="original")
    ap.add_argument("--agents", type=int, default=3); ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--share", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    r = run_episode_rrtbc(a.map, a.dataset, a.agents, a.steps, a.seed, a.share)
    r["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(r))
