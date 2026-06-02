#!/usr/bin/env python3
"""
Scenario builder for the FAITHFUL RRT+BC reproducible study.
Reuses the original student code verbatim: the trained Behavior-Cloning network,
the RRT planner, the route datasets (offline RRT expert paths), and the exact
obstacle layouts (black_ranges) extracted from Main_iter.py. Nothing about the
navigation method is re-implemented here -- only the data are assembled.
"""
import os, re, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)          # repository root
sys.path.insert(0, os.path.join(_CODE, "src"))

import Functions_code as F   # the original RRT + BC + navigation primitives


def _extract_black_ranges():
    """Pull the two hand-authored obstacle layouts out of Main_iter.py verbatim."""
    src = open(os.path.join(_CODE, "legacy", "Main_iter.py"), encoding="utf-8").read()
    out = {}
    for name, key in [("original_black_ranges_1", "map1"), ("original_black_ranges_2", "map2")]:
        m = re.search(name + r"\s*=\s*(\[.*?\n    \])", src, re.DOTALL)
        if not m:
            raise RuntimeError(f"could not extract {name}")
        out[key] = eval(m.group(1))      # list literal (comments are fine for eval)
    return out

_BR = _extract_black_ranges()

# Each route dataset was *generated on* (and, per the original pipeline, pre-filtered
# against) a specific map: the "original" library was built on Map1, the "alternative"
# library was adapted on Map2. Main_iter.py filters routes against that NATIVE map and
# only then deploys them ("filter routes on map1 and after that choose map2", line 349).
# Filtering against the deployment map instead would silently delete the very Map1 routes
# that intersect Map2 obstacles -- erasing the environment-mismatch / skill-transfer effect.
_NATIVE_MAP = {"original": "map1", "alternative": "map2"}


def build_scenario(map_id="map1", dataset="original"):
    """Return (routes, up_inflated_matrix, original_scaled_matrix, original_black_ranges)
    exactly as Main_iter.py prepares them, for the chosen DEPLOYMENT map and route dataset.
    Routes are filtered against their NATIVE map (where they were generated), not the
    deployment map, faithfully reproducing the student's preprocessing."""
    import pandas as pd
    mat = np.loadtxt(os.path.join(_CODE, "maps", f"{map_id}.txt"), delimiter="\t")
    black = _BR[map_id]
    csv = "original_data.csv" if dataset == "original" else "alternative_data.csv"
    df = pd.read_csv(os.path.join(_CODE, "data", csv)).dropna()
    if dataset == "original":
        df = df[~((df.iloc[:, -2:] < -2) | (df.iloc[:, -2:] > 2)).any(axis=1)]
        df = df.iloc[:, :2]
        routes = F.extract_routes(df)
    else:
        routes = F.extract_routes_newdata(df)
    # filter against the dataset's native map (Map1 for original, Map2 for alternative)
    native = _NATIVE_MAP[dataset]
    nmat = np.loadtxt(os.path.join(_CODE, "maps", f"{native}.txt"), delimiter="\t")
    _, _, native_black = F.Update1(nmat, _BR[native], [[0, 0]])
    routes = F.filter_routes_through_black_ranges_and_distance(routes, native_black)
    # deployment-map config space (used for navigation/collision checks)
    _, up_inflated_matrix, up_black = F.Update1(mat, black, [[0, 0]])
    return routes, up_inflated_matrix, mat, black


if __name__ == "__main__":
    for mp, ds in [("map1", "original"), ("map2", "alternative")]:
        r, ui, om, ob = build_scenario(mp, ds)
        print(f"{mp}/{ds}: routes={len(r)} matrix={om.shape} black_rows={len(ob)}")
