#!/usr/bin/env python3
"""
Route-library transform baselines for Revision 1 (Reviewer B, comment 3; Reviewer A,
comment 6 context). Given the Map2-naive library (generated on Map1) deployed on Map2,
build the three "simple but important" baselines the review asks for:

  * FILTER  -- drop every transferred route that intersects a deployment-map obstacle
               (validity filtering against the DEPLOYMENT map, not the native map).
  * REPAIR  -- keep each invalid route's valid segments and reconnect the gaps offline
               with the same RRT planner running on the deployment map.
  * REGEN   -- re-plan each invalid route end-to-end on the deployment map with the
               same RRT planner (regenerating only the invalid part of the library).

All transforms are OFFLINE, applied once to the library before deployment, and cached
to disk so every episode (any seed) reuses the identical transformed library. The
transform RNG is seeded independently of episode seeds (TRANSFORM_SEED) so the cached
libraries are reproducible. Valid routes are always passed through untouched.
"""
import os, sys, pickle, random

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_CODE, "src"))
sys.path.insert(0, _HERE)

import Functions_code as F
from scenario import build_scenario, _BR

TRANSFORM_SEED = 20260702      # fixed: library transforms are offline preprocessing
CACHE_DIR = os.path.join(_HERE, "lib_cache")


def deployment_statics(map_id):
    """Static (agent-free) obstacle ranges + inflated matrix of the deployment map."""
    mat = np.loadtxt(os.path.join(_CODE, "maps", f"{map_id}.txt"), delimiter="\t")
    black = _BR[map_id]
    _, up_inflated, up_black = F.Update1(mat, black, [[0, 0]])
    return up_inflated, up_black


def _route_invalid(route, up_black):
    return any(F.is_point_inside_black_ranges(p[0], p[1], up_black) for p in route)


def _valid_segments(route, up_black):
    """Split a route into maximal contiguous segments of obstacle-free points."""
    segs, cur = [], []
    for p in route:
        if F.is_point_inside_black_ranges(p[0], p[1], up_black):
            if len(cur) > 1:
                segs.append(cur)
            cur = []
        else:
            cur.append(p)
    if len(cur) > 1:
        segs.append(cur)
    return segs


def transform_filter(routes, up_black):
    return [r for r in routes if not _route_invalid(r, up_black)]


def transform_repair(routes, up_inflated, up_black):
    """Reconnect the valid segments of each invalid route with offline RRT on the
    deployment map; a repaired route is kept only if the reconnections succeed and
    the result is fully valid. Routes that cannot be repaired are dropped (they
    would otherwise steer robots into obstacles)."""
    out = []
    for r in routes:
        if not _route_invalid(r, up_black):
            out.append(r)
            continue
        segs = _valid_segments(r, up_black)
        if not segs:
            continue
        repaired, ok = list(segs[0]), True
        for seg in segs[1:]:
            try:
                bridge, _, _ = F.Apply_RRT(repaired[-1], seg[0], up_inflated, up_black)
            except Exception:      # RRT found no path within its budget
                ok = False
                break
            if not bridge:
                ok = False
                break
            repaired.extend(bridge)
            repaired.extend(seg)
        if ok and not _route_invalid(repaired, up_black):
            out.append(repaired)
    return out


def transform_regen(routes, up_inflated, up_black):
    """Re-plan each invalid route start->end with the same RRT planner on the
    deployment map (endpoints projected to the nearest valid cell if needed)."""
    def _valid_pt(p):
        return not F.is_point_inside_black_ranges(p[0], p[1], up_black)

    def _project(p):
        if _valid_pt(p):
            return list(p[:2])
        best, bd = None, 1e9
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                q = [p[0] + dx, p[1] + dy]
                if 0 <= q[0] <= 49 and 0 <= q[1] <= 49 and _valid_pt(q):
                    d = dx * dx + dy * dy
                    if d < bd:
                        best, bd = q, d
        return best

    out = []
    for r in routes:
        if not _route_invalid(r, up_black):
            out.append(r)
            continue
        s, g = _project(r[0]), _project(r[-1])
        if s is None or g is None:
            continue
        try:
            fresh, _, _ = F.Apply_RRT(s, g, up_inflated, up_black)
        except Exception:          # RRT found no path within its budget
            continue
        full = [s] + list(fresh) + [g]
        if fresh and not _route_invalid(full, up_black):
            out.append(full)
    return out


def get_variant_routes(map_id, dataset, variant):
    """Return the transformed library for (deployment map, dataset, variant), cached.
    variant in {"filter", "repair", "regen"}."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{map_id}_{dataset}_{variant}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    random.seed(TRANSFORM_SEED); np.random.seed(TRANSFORM_SEED)
    routes, _, _, _ = build_scenario(map_id, dataset)   # native-filtered, as deployed
    up_inflated, up_black = deployment_statics(map_id)
    if variant == "filter":
        tr = transform_filter(routes, up_black)
    elif variant == "repair":
        tr = transform_repair(routes, up_inflated, up_black)
    elif variant == "regen":
        tr = transform_regen(routes, up_inflated, up_black)
    else:
        raise ValueError(variant)
    with open(cache, "wb") as f:
        pickle.dump(tr, f)
    return tr


if __name__ == "__main__":
    for v in ("filter", "repair", "regen"):
        r = get_variant_routes("map2", "original", v)
        print(f"map2/original/{v}: {len(r)} routes")
