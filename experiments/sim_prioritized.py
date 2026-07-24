#!/usr/bin/env python3
"""
PRIORITIZED PLANNING baseline (Revision 1; Reviewer A comment 6, Reviewer B comment 3).
A mainstream multi-agent path-finding method: decoupled prioritized planning with
space-time A* and a reservation table (Erdmann & Lozano-Perez 1987; Silver 2005) --
the standard suboptimal member of the CBS family of coordination methods.

Protocol parity with sim_rrtbc.py / sim_rrt_online.py:
  * same maps, same start/goal sampler (generate_start_and_end), same 2000-tick
    scheduler with one move per robot per tick, same per-task step cap (300),
    same task-resampling on completion/failure, seeded and deterministic;
  * metrics: tasks_completed, fails, collisions (blocked-move events at execution
    time -- with a consistent reservation table these are structurally ~0, which is
    the point of the comparison), planner calls in the rrt_calls column, wall_s.

IMPORTANT difference, stated openly in the manuscript: prioritized planning is NOT
communication-free -- it coordinates through a shared space-time reservation table,
i.e., centralized information. It bounds what coordination-with-shared-state can
achieve on the same tasks, whereas the hybrid framework operates decentralized.
"""
import os, sys, time, json, argparse, random, heapq

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_CODE, "src"))
sys.path.insert(0, _HERE)
import Functions_code_RRT as FR
from scenario import _BR

MAX_TASK_STEPS = 300
ASTAR_MAX_EXPANSIONS = 60000
MOVES = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def _build_maps(map_id):
    mat = np.loadtxt(os.path.join(_CODE, "maps", f"{map_id}.txt"), delimiter="\t")
    black = _BR[map_id]
    _, up_inflated, up_black = FR.Update1(mat, black, [[0, 0]])
    return up_inflated, mat, black, up_black


def _free_grid(up_black):
    free = np.zeros((50, 50), dtype=bool)   # free[x, y]
    for x in range(50):
        for y in range(50):
            free[x, y] = not FR.is_point_inside_black_ranges3(x, y, up_black)
    return free


def _octile(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy)


def space_time_astar(start, goal, t0, free, cell_res, edge_res, park_res, horizon=MAX_TASK_STEPS):
    """A* over (x, y, t) with wait moves. cell_res: {(x,y,t)} occupied; edge_res:
    {(x1,y1,x2,y2,t)} swap conflicts; park_res: {(x,y): t_from} cells permanently
    reserved from t_from on (a robot parked at its plan end). Returns list of cells
    [start at t0, ..., goal] or None."""
    sx, sy = start
    gx, gy = goal
    openq = [(_octile(start, goal), 0, sx, sy, t0)]
    came = {}
    g_cost = {(sx, sy, t0): 0}
    expansions = 0
    while openq:
        f, g, x, y, t = heapq.heappop(openq)
        if g_cost.get((x, y, t), 1e18) < g:
            continue
        if (x, y) == (gx, gy):
            # reconstruct
            path, cur = [], (x, y, t)
            while cur in came:
                path.append((cur[0], cur[1]))
                cur = came[cur]
            path.append((sx, sy))
            return path[::-1]
        expansions += 1
        if expansions > ASTAR_MAX_EXPANSIONS or g > horizon:
            return None
        for dx, dy in MOVES:
            nx, ny, nt = x + dx, y + dy, t + 1
            if not (0 <= nx < 50 and 0 <= ny < 50) or not free[nx, ny]:
                continue
            if (nx, ny, nt) in cell_res:
                continue
            if (nx, ny, x, y, t) in edge_res:      # head-on swap
                continue
            pk = park_res.get((nx, ny))
            if pk is not None and nt >= pk:
                continue
            ng = g + 1
            if ng < g_cost.get((nx, ny, nt), 1e18):
                g_cost[(nx, ny, nt)] = ng
                came[(nx, ny, nt)] = (x, y, t)
                heapq.heappush(openq, (ng + _octile((nx, ny), (gx, gy)), ng, nx, ny, nt))
    return None


def run_episode_prioritized(map_id="map2", n_agents=5, steps=2000, seed=0):
    random.seed(seed); np.random.seed(seed)
    up_inflated, mat, black, up_black = _build_maps(map_id)
    _, _, static_black = FR.Update1(mat, black, [[0, 0]])
    free = _free_grid(static_black)
    _t0 = time.perf_counter()

    starts, ends = FR.generate_start_and_end(up_inflated, black, num_agents=n_agents, min_distance=4)
    pos = [(int(round(s[0])), int(round(s[1]))) for s in starts]
    goals = [(int(round(e[0])), int(round(e[1]))) for e in ends]
    plans = [None] * n_agents          # plans[i]: list of cells, plans[i][0] == pos at plan time
    plan_idx = [0] * n_agents
    step_count = [0] * n_agents
    tasks = 0; fails = 0; collisions = 0; planner_calls = 0

    def reservations(exclude):
        """Space-time reservations induced by all robots' current plans except `exclude`,
        expressed relative to the CURRENT tick (t=0 == now)."""
        cell_res, edge_res, park_res = set(), set(), {}
        for j in range(n_agents):
            if j == exclude:
                continue
            pj = plans[j]
            if not pj:
                park_res[pos[j]] = 0
                continue
            rem = pj[plan_idx[j]:]
            for k, c in enumerate(rem):
                cell_res.add((c[0], c[1], k))
                if k + 1 < len(rem):
                    n = rem[k + 1]
                    edge_res.add((c[0], c[1], n[0], n[1], k))
            park_res[rem[-1]] = len(rem) - 1
        return cell_res, edge_res, park_res

    def new_task(i):
        s, e = FR.generate_start_and_end(up_inflated, black, num_agents=1, min_distance=4)
        pos_i = (int(round(s[0][0])), int(round(s[0][1])))
        goals[i] = (int(round(e[0][0])), int(round(e[0][1])))
        plans[i] = None; plan_idx[i] = 0; step_count[i] = 0
        return pos_i

    for t in range(steps):
        for i in range(n_agents):
            if plans[i] is None:
                cell_res, edge_res, park_res = reservations(i)
                planner_calls += 1
                p = space_time_astar(pos[i], goals[i], 0, free, cell_res, edge_res, park_res)
                if p is None:
                    fails += 1
                    pos[i] = new_task(i)
                    continue
                plans[i] = p; plan_idx[i] = 0
            # execute next planned cell
            if plan_idx[i] + 1 < len(plans[i]):
                nxt = plans[i][plan_idx[i] + 1]
                if any(pos[j] == nxt for j in range(n_agents) if j != i):
                    # blocked-move event at execution time (analog of the hybrid's
                    # predicted blocked-move): count it, wait, and replan
                    collisions += 1
                    plans[i] = None
                    step_count[i] += 1
                    if step_count[i] > MAX_TASK_STEPS:
                        fails += 1; pos[i] = new_task(i)
                    continue
                pos[i] = nxt
                plan_idx[i] += 1
            step_count[i] += 1
            if pos[i] == goals[i]:
                tasks += 1
                pos[i] = new_task(i)
            elif step_count[i] > MAX_TASK_STEPS:
                fails += 1
                pos[i] = new_task(i)

    return dict(map=map_id, dataset="prioritized", n_agents=n_agents, share_fraction=0.0,
                steps=steps, seed=seed, n_access=0,
                tasks_completed=int(tasks), fails=int(fails), collisions=int(collisions),
                coll_obstacle=0, coll_robot=int(collisions),
                rrt_calls=int(planner_calls), wall_s=round(time.perf_counter() - _t0, 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="map2"); ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(run_episode_prioritized(a.map, a.agents, a.steps, a.seed)))
