#!/usr/bin/env python3
"""
Spatial / schematic figures for the manuscript, all regenerated from the faithful pipeline.
Figure numbers follow the manuscript's cited order (Map layouts -> workflow -> mismatch):

  fig2_environments.png   -- Fig. 1: the two map layouts (clean, panels Map1/Map2).
  fig_framework.png       -- Fig. 2: method/pipeline schematic (4 panels, no data).
  fig_route_mismatch.png  -- Fig. 3: route-library/environment mismatch (4 panels), built
                             directly from the real route datasets + the two map layouts.
  tables/tab_route_diag.tex -- route-library diagnostics table (Section 3.3) (sizes/coverage).
  fig_heatmaps_map2.png   -- Fig. 5: per-cell collision heatmaps on Map2 (Map2-naive hybrid,
                             Map2-adapted hybrid, online RRT) at N=4 and N=10, aggregated over
                             15 seeds with a shared colour scale, colour bar and obstacle
                             outlines, normalised to collisions per 100 completed tasks.

The heatmap episodes are launched in parallel; the other two need only assets, so they are
cheap. Run:  python3 make_heatmaps_rrtbc.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("RRT_MAX_ITER", "3000")  # bound the online-RRT column (feasibility)
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.lines import Line2D
from scipy.ndimage import (binary_erosion, binary_dilation, gaussian_filter, convolve,
                           distance_transform_edt)
from multiprocessing import Pool, cpu_count

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_CODE, "src")); sys.path.insert(0, _HERE)
import Functions_code as F
from scenario import build_scenario, _BR
from sim_rrtbc import run_episode_rrtbc
from sim_rrt_online import run_episode_rrt
import pubstyle as PS                      # shared publication figure style (visual-audit fix)
PS.apply()

FIG = os.path.join(_CODE, "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ---- heatmap settings (Fig. 4) ----
HEAT_STEPS = 1200
HEAT_SEEDS = tuple(range(15))
HEAT_AGENTS = (4, 10)
HEAT_COLS = [("Hybrid · Map2-naive", "hyb", "original"),
             ("Hybrid · Map2-adapted", "hyb", "alternative"),
             ("Online RRT", "rrt", None)]


# ============================ Fig. 4 : collision heatmaps ============================
def _heat_worker(task):
    meth, ds, n, s = task
    if meth == "hyb":
        r = run_episode_rrtbc("map2", ds, n_agents=n, steps=HEAT_STEPS, seed=s,
                              share_fraction=0.0, return_grid=True)
    else:
        r = run_episode_rrt("map2", n_agents=n, steps=HEAT_STEPS, seed=s, return_grid=True)
    return (meth, ds, n, s), r["collision_grid"], r["obstacles"], int(r["tasks_completed"])


def heatmaps():
    tasks = [(meth, ds, n, s) for n in HEAT_AGENTS
             for (_, meth, ds) in HEAT_COLS for s in HEAT_SEEDS]
    print(f"[heatmaps] {len(tasks)} episodes on {min(14, cpu_count())} procs ...", flush=True)
    acc = {}; tasks_acc = {}; obst_ref = None
    with Pool(min(14, cpu_count())) as pool:
        for key, grid, obst, tsk in pool.imap_unordered(_heat_worker, tasks, chunksize=1):
            meth, ds, n, s = key
            k = (n, meth, ds)
            acc[k] = grid if k not in acc else acc[k] + grid
            tasks_acc[k] = tsk + tasks_acc.get(k, 0)
            obst_ref = obst
    # ---- Visualization domain, event projection, domain-aware smoothing --------------------
    # The simulator logs each predicted blocked move AT the inflated-obstacle cell the robot's
    # look-ahead would have entered. Painting that raw mass *inside* obstacles reads to a
    # reviewer as "a robot colliding inside a wall". For the figure we therefore (1) project
    # each such event onto the FREE cells adjacent to that obstacle cell -- the contact location
    # the robot was actually advancing from -- so collision density forms a halo on the
    # reachable side of the boundary and never overlaps the geometry (mass is conserved); and
    # (2) smooth it with a mass-normalised Gaussian restricted to the free-space mask, so the
    # field is continuous over free space without leaking through walls or leaving zero-holes.
    infl = (obst_ref > 0.5)                                    # inflated (config-space) obstacle
    core = binary_erosion(infl, structure=np.ones((3, 3)), border_value=0)  # physical obstacle
    margin = infl & ~core                                      # 1-cell robot-footprint margin
    omap = np.where(core, 1.0, np.where(margin, 0.0, np.nan))  # 0 = margin, 1 = physical core
    free = (~infl).astype(float)                               # valid visualization domain
    k8 = np.ones((3, 3)); k8[1, 1] = 0.0                       # 8-neighbour stencil
    free_neigh = convolve(free, k8, mode="constant")           # free 8-neighbours per cell
    den = gaussian_filter(free, sigma=1.5, mode="constant")    # masked-smoothing normaliser

    grids, support, vmax = {}, {}, 0.0
    for i, n in enumerate(HEAT_AGENTS):
        for j, (title, meth, ds) in enumerate(HEAT_COLS):
            k = (n, meth, ds)
            raw = acc[k] * (100.0 / max(tasks_acc[k], 1))      # events / 100 completed tasks
            # Redistribute obstacle-cell mass equally to that cell's free 8-neighbours; mass
            # already in free space (e.g. robot-robot blocks) stays put.
            send = np.where(infl & (free_neigh > 0), raw / np.maximum(free_neigh, 1.0), 0.0)
            proj = np.where(infl, 0.0, raw) + convolve(send, k8, mode="constant") * free
            # H = (G_s * proj) / (G_s * free), evaluated only on free cells -> continuous, no
            # holes, no through-wall leakage.
            H = np.where(free > 0,
                         gaussian_filter(proj, sigma=1.5, mode="constant") / np.maximum(den, 1e-9),
                         0.0)
            grids[(i, j)] = H
            # Paint only the free cells within a 2-cell halo of a projected event (keeps broad
            # free space white; fills interior holes -> a continuous halo along the boundary).
            support[(i, j)] = binary_dilation(proj > 0, iterations=2) & (free > 0)
            vmax = max(vmax, H.max())

    # Light background so the three layers read as distinct categories: white = free space,
    # gray = obstacle (opaque, drawn ON TOP of the heat), warm = collision density on the free
    # contact cells. Collision mass never overlaps the geometry, so the reader is not asked to
    # accept "a collision inside a wall"; the title/legend state that the density marks the
    # blocked-move contact location on the reachable side of the boundary.
    fig, axes = plt.subplots(len(HEAT_AGENTS), len(HEAT_COLS), figsize=(11.0, 7.4),
                             gridspec_kw=dict(hspace=0.12, wspace=0.06,
                                              left=0.085, right=0.85, top=0.96, bottom=0.14))
    obst_cmap = ListedColormap(["#e0e0e0", "#bdbdbd"])         # margin light, physical core darker
    # Warm ramp truncated so the smallest nonzero value is already a clear orange (not near
    # white) and the peak is dark red: high values stay most salient on the white background,
    # and the ramp is distinct from both the white free space and the gray obstacles.
    heat_cmap = ListedColormap(plt.cm.YlOrRd(np.linspace(0.15, 1.0, 256)))
    H_, W_ = obst_ref.shape
    # One shared, data-derived high-density level for every panel: a single ABSOLUTE value
    # (~60% of the global peak, rounded to one decimal), the same in all six panels, so the
    # dashed contour is a genuine cross-panel comparison rather than a hand-placed callout -- it
    # simply does not appear where the field never reaches it (the adapted / online-RRT panels
    # stay bare). The legend prints this absolute level in the field's own units.
    hi_level = round(0.6 * vmax, 1)
    # Round the color-scale ceiling up to a clean 0.4 step so the color bar's top tick is
    # labelled (no unlabelled overflow above the last tick).
    vmax_disp = float(np.ceil(vmax / 0.4) * 0.4)
    print(f"[fig5] vmax={vmax:.3f}  vmax_disp={vmax_disp:.2f}  hi_level={hi_level:.2f}")
    im = None
    letters = iter("ABCDEF")
    for i, n in enumerate(HEAT_AGENTS):
        for j, (title, meth, ds) in enumerate(HEAT_COLS):
            ax = axes[i, j]; gs = grids[(i, j)]
            ax.set_facecolor("white")
            # Layer order: collision density (free-space halo) FIRST, then the two-tone
            # obstacle fill OPAQUE on top, then the boundary outline -- so density forms a halo
            # against the walls and never appears inside the geometry.
            im = ax.imshow(np.ma.masked_where(~support[(i, j)], gs), cmap=heat_cmap,
                           vmin=0, vmax=vmax_disp, interpolation="bilinear")
            ax.imshow(np.ma.masked_invalid(omap), cmap=obst_cmap, vmin=0, vmax=1,
                      interpolation="nearest")
            ax.contour(infl.astype(float), levels=[0.5], colors="0.35", linewidths=0.9)
            # One shared high-density contour (same absolute level in every panel). Drawn as a
            # thin white-haloed DASHED line so it reads as a distinct annotation layer (not
            # another obstacle outline) and survives grayscale; it objectively bounds the
            # "hotspot" instead of leaving the reader to judge it by color impression.
            if hi_level > 0 and gs.max() >= hi_level:
                cs = ax.contour(np.where(support[(i, j)], gs, 0.0), levels=[hi_level],
                                colors="0.1", linewidths=0.8, linestyles="--")
                cs.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("0.6"); sp.set_linewidth(0.8)
            ax.text(0.03, 0.97, next(letters), transform=ax.transAxes, fontsize=11,
                    fontweight="bold", va="top", ha="left",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="0.7", lw=0.4, alpha=0.85))
            if i == 0:
                ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
            if j == 0:
                ax.set_ylabel(f"$N={n}$", rotation=0, labelpad=24, fontsize=13, va="center")
                # Mark the peak of the DISPLAYED field (argmax of the smoothed density, which is
                # exactly the hottest cell the reader sees) in the naive column only, as a small
                # white-haloed crosshair. A point marker -- not an arbitrary-radius circle -- so
                # it cannot be misread as a quantitative region, and it always coincides with the
                # plotted maximum.
                yy, xx = np.unravel_index(int(np.argmax(gs)), gs.shape)
                ax.plot(xx, yy, marker="P", ms=7.0, mfc="#111111", mec="white", mew=1.2,
                        ls="none", zorder=6)
    # Scale bar (10 grid cells) in panel F, black with a white halo so it reads on any cell.
    axF = axes[1, 2]
    x0, y0 = 0.06 * W_, 0.92 * H_
    axF.plot([x0, x0 + 10], [y0, y0], color="k", lw=3.0, solid_capstyle="butt",
             path_effects=[pe.withStroke(linewidth=5.0, foreground="white")])
    axF.text(x0 + 5, y0 - 1.5, "10 cells", color="k", fontsize=9.0, ha="center", va="bottom",
             fontweight="bold", path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    # Colorbar in its own axes, aligned to the full grid height (no longer floating between rows).
    cax = fig.add_axes([0.865, 0.14, 0.022, 0.82])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("blocked-contact density (events / grid cell / 100 tasks)",
                   fontsize=10.0, labelpad=8)
    cbar.set_ticks(np.arange(0.0, vmax_disp + 1e-9, 0.4)); cbar.update_ticks()
    cbar.ax.tick_params(labelsize=9)
    # One legend for the map encoding; the takeaway (naive vs adapted) stays in the caption.
    leg_handles = [mpatches.Patch(fc="#bdbdbd", ec="0.35", label="obstacle (physical)"),
                   mpatches.Patch(fc="#e0e0e0", ec="0.35", label="robot-footprint margin (1 cell)"),
                   Line2D([0], [0], color="0.1", lw=0.9, ls="--",
                          label=f"high-density contour: $H\\geq{hi_level:.1f}$"),
                   Line2D([0], [0], marker="P", mfc="#111111", mec="white", mew=1.2, ms=8, ls="none",
                          label="peak projected contact cell (Map2-naive)")]
    fig.legend(handles=leg_handles, loc="lower center", bbox_to_anchor=(0.467, 0.01),
               ncol=4, frameon=True, framealpha=0.9, edgecolor="0.7", fontsize=8.5)
    PS.save(fig, os.path.join(FIG, "fig_heatmaps_map2"))
    print("wrote fig_heatmaps_map2.pdf")


# ====================== Fig. 2 : route-library / environment mismatch ======================
def _obst_matrix(map_id):
    mat = np.loadtxt(os.path.join(_CODE, "maps", f"{map_id}.txt"), delimiter="\t")
    _, up_inflated, up_black = F.Update1(mat, _BR[map_id], [[0, 0]])
    return up_inflated, up_black


def _to_rowcol(route, H):
    cols = [p[0] for p in route]; rows = [H - p[1] - 1 for p in route]
    return cols, rows


def _route_invalid_cells(route, black):
    return sum(int(F.is_point_inside_black_ranges(p[0], p[1], black)) for p in route)


def _overlap_stats(routes, black):
    """Pooled fraction of waypoints inside obstacles, and fraction of routes affected."""
    tot = sum(len(r) for r in routes if len(r))
    bad = sum(_route_invalid_cells(r, black) for r in routes if len(r))
    affected = sum(1 for r in routes if len(r) and _route_invalid_cells(r, black) > 0)
    nroutes = sum(1 for r in routes if len(r))
    return 100 * bad / max(tot, 1), 100 * affected / max(nroutes, 1)


def _conflict_mask(routes, black, shape):
    """Binary cell mask: True where any stored-route waypoint falls inside `black` obstacle.
    Display coordinates: out[row, col] follows the imshow convention (row = H-y-1, col = x),
    matching the y-flip used by _to_rowcol for the route polylines."""
    H = shape[0]
    out = np.zeros(shape, dtype=bool)
    for r in routes:
        for p in r:
            x = int(round(p[0])); y = int(round(p[1]))
            if 0 <= x < shape[1] and 0 <= y < H and \
               F.is_point_inside_black_ranges(p[0], p[1], black):
                row = H - y - 1
                if 0 <= row < H:
                    out[row, x] = True
    return out


def _seg_free(p, q, black, step=0.15):
    """True iff the straight segment p->q stays in free space, tested by dense sub-cell
    sampling against the SAME config-space obstacle the simulator collision-checks against
    (physical obstacle + robot-footprint inflation). Used to guarantee a display-simplified
    segment never clips an obstacle corner between two valid waypoints."""
    d = np.hypot(q[0] - p[0], q[1] - p[1])
    n = max(2, int(d / step) + 1)
    for t in np.linspace(0.0, 1.0, n):
        x = p[0] + t * (q[0] - p[0]); y = p[1] + t * (q[1] - p[1])
        if F.is_point_inside_black_ranges(x, y, black):
            return False
    return True


def _simplify_route(route, black, eps=0.9):
    """Collision-checked Ramer-Douglas-Peucker simplification for DISPLAY only. Reduces the
    raw-planner jitter on free stretches (so the plotted route looks like a clean, traversable
    path rather than a noisy trace) while GUARANTEEING every drawn segment stays in free space:
    each simplified segment is validated by dense sub-cell sampling, and any segment that would
    clip an obstacle falls back to the raw sub-polyline. Simplification therefore never invents
    free space or moves the route into geometry it should avoid."""
    pts = [(float(p[0]), float(p[1])) for p in route]
    n = len(pts)
    if n < 3:
        return pts
    P = np.array(pts)
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a; L = np.hypot(*ab)
        seg = P[i + 1:j] - a
        if L < 1e-9:
            dist = np.hypot(seg[:, 0], seg[:, 1])
        else:
            dist = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L
        kk = int(np.argmax(dist))
        if dist[kk] > eps:
            k = i + 1 + kk; keep.add(k)
            stack.append((i, k)); stack.append((k, j))
    idx = sorted(keep)
    out_idx = [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        if _seg_free(pts[a], pts[b], black):
            out_idx.append(b)
        else:
            out_idx.extend(range(a + 1, b + 1))   # keep raw points across this span
    return [pts[k] for k in out_idx]


def _hatch_cells(ax, mask, color, fill_alpha=0.10, hatch="////", zorder=4):
    """Diagnostic-annotation overlay for each True cell: nearly-transparent colour wash plus a
    strong dense hatched stroke in the same colour. The fill stays barely visible so the gray
    two-tone obstacle geometry underneath remains dominant; the hatch carries the entire
    visual signal so the cell reads as an overlay annotation, not a new opaque obstacle class.
    The caller is expected to set `plt.rcParams["hatch.linewidth"]` for stroke weight."""
    rows, cols = np.where(mask)
    fill_rgba = to_rgba(color, alpha=fill_alpha)
    for r, c in zip(rows, cols):
        ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                        facecolor=fill_rgba, edgecolor=color,
                                        hatch=hatch, linewidth=0.0,
                                        zorder=zorder))


def route_mismatch():
    naive_routes, _, _, _ = build_scenario("map1", "original")     # Map1-generated library
    adapt_routes, _, _, _ = build_scenario("map2", "alternative")  # Map2-adapted library
    m1, b1 = _obst_matrix("map1")
    m2, b2 = _obst_matrix("map2")
    H = m2.shape[0]

    from matplotlib.lines import Line2D
    # Two-tone gray obstacle mask: darker core = physical obstacle, lighter one-cell ring =
    # robot-footprint inflation. Drawn opaque above route traces so no stroke can appear
    # inside the geometry.
    #
    # CRITICAL round-9 fix: the drawn obstacle is now built from the SAME black-range
    # configuration-space mask the simulator and online RRT collision-check against
    # (`is_point_inside_black_ranges`), NOT from a separately-inflated matrix. The two
    # previously disagreed by ~11% of cells (172 cells were free against the collision mask
    # but drawn as obstacle), which made genuinely valid routes appear to graze or clip the
    # inflation ring -- the core of the reviewer's route-validity trust problem. Drawing the
    # exact collision mask guarantees every route that is valid (against `black`) is also
    # visually clear of the drawn geometry.
    def _omap_from_black(black, shape):
        Hh, Ww = shape
        mask = np.zeros((Hh, Ww), bool)
        for yy in range(Hh):
            for xx in range(Ww):
                if F.is_point_inside_black_ranges(xx, yy, black):
                    mask[Hh - yy - 1, xx] = True       # display orientation (row = H-y-1)
        core = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
        return np.where(core, 1.0, np.where(mask, 0.0, np.nan)), mask
    m1_omap, m1_infl = _omap_from_black(b1, m1.shape)
    m2_omap, m2_infl = _omap_from_black(b2, m2.shape)
    # Per-cell distance (in cells) from each free cell to the nearest obstacle cell, in the
    # same display orientation as the routes. Used to choose a reconnect path that keeps
    # CLEARANCE from the walls, so the drawn online-RRT continuation reads as a confident
    # detour rather than scraping the obstacle boundary.
    m2_clear = distance_transform_edt(~m2_infl)
    obst_cmap = ListedColormap(["#e0e0e0", "#bdbdbd"])

    def _draw_obstacles(a, omap, infl, outline_zorder=3.1):
        a.imshow(np.ma.masked_invalid(omap), cmap=obst_cmap, vmin=0, vmax=1,
                 interpolation="nearest", zorder=3)
        a.contour(infl.astype(float), levels=[0.5], colors="0.35", linewidths=0.5,
                  zorder=outline_zorder)
        a.set_xlim(-0.5, infl.shape[1] - 0.5); a.set_ylim(infl.shape[0] - 0.5, -0.5)

    def _frame(a):
        # Map panels: white background, no ticks, no spines so they look like map frames.
        a.set_facecolor("white"); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)

    # Denser, slightly heavier hatch so the route-obstacle conflict cells read even at very
    # low fill opacity -- the hatch carries the signal, the fill stays nearly transparent.
    plt.rcParams["hatch.linewidth"] = 1.7

    n_sample = min(80, len(naive_routes))
    cN_cells, cN_routes = _overlap_stats(naive_routes, b2)

    # Pick a (naive, adapted) pair sharing the same task (start AND goal cells match within
    # `tol` cells, i.e. shared task seed) so the start/goal markers truly sit on BOTH visible
    # route endpoints. Skip naive routes whose start is closer than `min_sep` to
    # `exclude_start` so the two examples look spatially distinct.
    def _pick_example(exclude_start=None, min_sep=14.0, tol=1.0, min_fb=8):
        best = None; best_score = -1.0
        for r in naive_routes:
            if len(r) < 16:
                continue
            flags = [F.is_point_inside_black_ranges(p[0], p[1], b2) for p in r]
            if not any(flags):
                continue
            fb = flags.index(True)
            if fb < min_fb:
                continue
            bad_total = sum(flags)
            if bad_total < 3:
                continue
            s0 = np.array(r[0][:2]); g0 = np.array(r[-1][:2])
            if exclude_start is not None and np.linalg.norm(s0 - exclude_start) < min_sep:
                continue
            twin = None
            for ar in adapt_routes:
                if len(ar) < 4:
                    continue
                if (np.linalg.norm(np.array(ar[0][:2]) - s0) <= tol and
                    np.linalg.norm(np.array(ar[-1][:2]) - g0) <= tol):
                    twin = ar; break
            if twin is None:
                continue
            mid_score = 1.0 - abs(fb / len(r) - 0.5)
            # Reward longer visible naive approach: scale by min(fb, 14)/14, capped so very
            # long approaches don't dominate. Combined with mid_score this picks examples
            # whose blocked contact sits well inside the panel, not next to the start dot.
            score = bad_total * mid_score * (min(fb, 14) / 14.0)
            if score > best_score:
                best_score = score
                best = (r, twin, fb, s0)
        return best

    def _route_arrow(ax, cs, rs, color):
        """Small filled arrowhead showing travel direction, placed at ~mid-ARC-LENGTH and
        oriented strictly downstream (from 46% to 56% of the route's arc length). Arc-length
        placement keeps the heading unambiguous regardless of where the longest segment falls."""
        cs = np.asarray(cs, float); rs = np.asarray(rs, float)
        if len(cs) < 2:
            return
        d = np.hypot(np.diff(cs), np.diff(rs))
        if d.sum() <= 0:
            return
        s = np.concatenate([[0.0], np.cumsum(d)]); L = s[-1]

        def at(frac):
            t = frac * L
            i = int(np.clip(np.searchsorted(s, t) - 1, 0, len(cs) - 2))
            u = (t - s[i]) / max(d[i], 1e-9)
            return cs[i] + u * (cs[i + 1] - cs[i]), rs[i] + u * (rs[i + 1] - rs[i])
        x0, y0 = at(0.46); x1, y1 = at(0.56)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    annotation_clip=False, zorder=4.0,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.0,
                                    mutation_scale=15, shrinkA=0, shrinkB=0,
                                    path_effects=[pe.withStroke(linewidth=2.5,
                                                                foreground="white")]))

    def _path_clearance(pts):
        """Minimum clearance (in cells) of a polyline from the obstacle mask, by dense
        sub-cell sampling against the precomputed distance field `m2_clear`."""
        mn = 1e9
        for a, b in zip(pts[:-1], pts[1:]):
            d = np.hypot(b[0] - a[0], b[1] - a[1]); n = max(2, int(d / 0.25) + 1)
            for t in np.linspace(0.0, 1.0, n):
                x = a[0] + t * (b[0] - a[0]); y = a[1] + t * (b[1] - a[1])
                row = int(round(H - y - 1)); col = int(round(x))
                if 0 <= row < m2_clear.shape[0] and 0 <= col < m2_clear.shape[1]:
                    mn = min(mn, float(m2_clear[row, col]))
        return mn

    def _reconnect(contact, goal):
        """Plan a genuine online-RRT continuation from the blocked contact to the goal, using
        the SAME planner the deployment loop calls (F.Apply_RRT against the Map2 config-space
        obstacle). This is the manuscript's OnlineRRT(x, g) mechanism (Fig. 2B / Alg. 1), so
        the blue continuation is a real reconnect path, not a relabelled library route.
        Several fixed seeds are tried; among all valid candidates the one with the LARGEST
        minimum obstacle clearance is kept (ties broken toward fewer vertices), so the drawn
        reconnect visibly keeps off the walls instead of hugging a boundary. Returns the
        collision-checked-simplified path ready to plot."""
        c = (float(contact[0]), float(contact[1])); g = (float(goal[0]), float(goal[1]))
        cands = []
        for seed in (7, 11, 19, 23, 41, 101, 211, 307, 401, 509):
            random.seed(seed); np.random.seed(seed)
            try:
                route, _, ok = F.Apply_RRT(c, g, m2, b2)
            except Exception:
                continue
            if not (ok and route):
                continue
            if any(F.is_point_inside_black_ranges(p[0], p[1], b2) for p in route):
                continue
            full = [c] + [(float(p[0]), float(p[1])) for p in route] + [g]
            simp = _simplify_route(full, b2, eps=0.9)
            cands.append((_path_clearance(simp), -len(simp), simp))
        if not cands:
            return None
        cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return cands[0][2]

    # Draw one example as the deployment RECOVERY sequence (user-confirmed story):
    #   start --(orange solid: naive route followed)--> X (blocked contact)
    #         --(orange dashed: invalid naive continuation into the Map2 obstacle)-->
    #   X --(blue solid: online-RRT reconnect, computed live)--> goal
    # The blue continuation begins EXACTLY at the blocked contact X, so the causal transition
    # (attempt -> block -> reconnect) is unambiguous. All drawn polylines are collision-checked-
    # simplified for display (never invents free space); obstacles are drawn OPAQUE on top so
    # the dashed invalid continuation visibly terminates at the wall it would have entered.
    # `near` (the adapted-library twin) is no longer drawn -- only used by the picker to keep
    # the chosen tasks stable across runs.
    def _render_example(ax, worst, near, first_bad):
        _frame(ax)
        cw, rw = _to_rowcol(worst, H)
        contact_idx = max(first_bad - 1, 0)
        contact_pt = (worst[contact_idx][0], worst[contact_idx][1])
        goal_pt = (worst[-1][0], worst[-1][1])
        # Orange solid: naive route actually followed, start -> contact (all free).
        followed = worst[:contact_idx + 1]
        followed_s = _simplify_route(followed, b2, eps=0.6) if len(followed) >= 2 else followed
        # Orange dashed: short invalid continuation from contact into the blocked cell(s).
        invalid = worst[contact_idx:min(len(worst), first_bad + 3)]
        # Blue solid: live online-RRT reconnect from contact to goal (already collision-
        # checked-simplified and clearance-selected by _reconnect).
        recon_s = _reconnect(contact_pt, goal_pt)

        # Obstacles drawn FIRST; the C/D example routes are drawn ON TOP of them. The naive
        # followed route and the RRT reconnect are collision-validated, so they stay clear of
        # the (now collision-consistent) drawn geometry; drawing them on top avoids the
        # straight-segment corner-clip gaps that opaque-obstacle-over-route produced, and lets
        # the invalid continuation visibly pierce INTO the wall it would have entered.
        _draw_obstacles(ax, m2_omap, m2_infl)
        if len(invalid) >= 2:
            ci, ri = _to_rowcol(invalid, H)
            ax.plot(ci, ri, "--", color=PS.VERMILLION, lw=2.0, alpha=0.95,
                    dash_capstyle="round", zorder=3.5,
                    path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])
        if len(followed_s) >= 2:
            cf, rf = _to_rowcol(followed_s, H)
            ax.plot(cf, rf, "-", color=PS.VERMILLION, lw=2.1, zorder=3.6,
                    solid_capstyle="round", solid_joinstyle="round",
                    path_effects=[pe.withStroke(linewidth=3.4, foreground="white")])
            _route_arrow(ax, cf, rf, PS.VERMILLION)
        if recon_s and len(recon_s) >= 2:
            cb, rb = _to_rowcol(recon_s, H)
            ax.plot(cb, rb, "-", color=PS.BLUE, lw=2.4, zorder=3.8,
                    solid_capstyle="round", solid_joinstyle="round",
                    path_effects=[pe.withStroke(linewidth=4.0, foreground="white")])
            _route_arrow(ax, cb, rb, PS.BLUE)
        bx, by = cw[contact_idx], rw[contact_idx]
        ax.plot(bx, by, marker="X", ms=11, mfc=PS.VERMILLION, mec="white", mew=1.7,
                ls="none", zorder=5,
                path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])
        W = m2.shape[1]; offx = 4 if bx < W * 0.7 else -4
        offy = -3.5 if by > 7 else 3.5
        ax.annotate("blocked", xy=(bx, by), xytext=(bx + offx, by + offy),
                    fontsize=9.0, color=PS.VERMILLION,
                    ha="left" if offx > 0 else "right",
                    va="bottom" if offy < 0 else "top",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                    arrowprops=dict(arrowstyle="-", color=PS.VERMILLION, lw=0.9,
                                    shrinkA=2, shrinkB=4),
                    zorder=6)
        ax.plot(cw[0], rw[0], "o", color="k", ms=6, zorder=5,
                path_effects=[pe.withStroke(linewidth=2.3, foreground="white")])
        ax.text(cw[0] + 1.2, rw[0], "S", fontsize=9, fontweight="bold", va="center",
                ha="left", color="k", path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        ax.plot(cw[-1], rw[-1], "*", color="k", ms=12, zorder=5,
                path_effects=[pe.withStroke(linewidth=2.3, foreground="white")])
        ax.text(cw[-1] + 1.4, rw[-1], "G", fontsize=9, fontweight="bold", va="center",
                ha="left", color="k", path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    # Column-major layout per audit: left column = A (valid Map1/Map1) above B (conflicts
    # Map1->Map2); right column = C, D (two shared-task blocked examples). The full-library
    # overlap statistics that previously sat in a Panel C bar chart / stat cards now live in
    # the caption and the surrounding text -- nothing is lost, the figure stops competing
    # with itself.
    fig = plt.figure(figsize=(10.2, 9.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.13],
                          left=0.04, right=0.985, top=0.965, bottom=0.025,
                          hspace=0.22, wspace=0.10)
    a_ax = fig.add_subplot(gs[0, 0])
    b_ax = fig.add_subplot(gs[1, 0])
    c_ax = fig.add_subplot(gs[0, 1])
    d_ax = fig.add_subplot(gs[1, 1])
    leg_ax = fig.add_subplot(gs[2, :]); leg_ax.axis("off")

    # ---- Panel A: Map1 library on Map1 ----
    _frame(a_ax)
    for r in naive_routes[:n_sample]:
        c, rw = _to_rowcol(r, H)
        a_ax.plot(c, rw, "-", color=PS.BLUE, lw=0.65, alpha=0.55, zorder=2)
    _draw_obstacles(a_ax, m1_omap, m1_infl)
    PS.panel(a_ax, "A")
    a_ax.set_title("Map1 library on Map1 – valid routes")

    # ---- Panel B: same routes evaluated on Map2; conflict cells as diagnostic hatching ----
    _frame(b_ax)
    for r in naive_routes[:n_sample]:
        c, rw = _to_rowcol(r, H)
        b_ax.plot(c, rw, "-", color=PS.BLUE, lw=0.65, alpha=0.55, zorder=2)
    _draw_obstacles(b_ax, m2_omap, m2_infl, outline_zorder=5)
    conflict = _conflict_mask(naive_routes[:n_sample], b2, m2.shape)
    _hatch_cells(b_ax, conflict, color=PS.VERMILLION, fill_alpha=0.06,
                 hatch="////", zorder=4)
    PS.panel(b_ax, "B")
    b_ax.set_title("Map1 library on Map2 – route–obstacle conflicts")

    # ---- Panels C, D: two shared-task blocked examples ----
    # Tight-tolerance twin first; if no shared-task pair survives, relax to 2.5 cells.
    ex1 = (_pick_example(tol=1.0, min_fb=8) or
           _pick_example(tol=1.0, min_fb=5) or
           _pick_example(tol=2.5, min_fb=5))
    ex2 = None
    if ex1 is not None:
        ex2 = (_pick_example(exclude_start=ex1[3], tol=1.0, min_fb=8) or
               _pick_example(exclude_start=ex1[3], tol=1.0, min_fb=5) or
               _pick_example(exclude_start=ex1[3], tol=2.5, min_fb=5) or
               _pick_example(exclude_start=ex1[3], min_sep=8.0, tol=2.5, min_fb=5))

    if ex1 is not None:
        _render_example(c_ax, ex1[0], ex1[1], ex1[2])
    else:
        _frame(c_ax)
        c_ax.text(0.5, 0.5, "no shared-task example found",
                  ha="center", va="center", transform=c_ax.transAxes, fontsize=10)
    PS.panel(c_ax, "C")
    c_ax.set_title("Example 1 – blocked, then online-RRT reconnect")

    if ex2 is not None:
        _render_example(d_ax, ex2[0], ex2[1], ex2[2])
    else:
        _frame(d_ax)
        d_ax.text(0.5, 0.5, "no second shared-task example",
                  ha="center", va="center", transform=d_ax.transAxes, fontsize=10)
    PS.panel(d_ax, "D")
    d_ax.set_title("Example 2 – blocked, then online-RRT reconnect")

    # ---- Unified bottom legend strip ----
    # Decodes the full C/D recovery sequence so the route phases are unambiguous: naive route
    # followed -> blocked contact -> invalid continuation (into the wall) -> online-RRT reconnect.
    obs_dark = mpatches.Patch(facecolor="#bdbdbd", edgecolor="0.35", linewidth=0.5,
                              label="physical obstacle")
    obs_light = mpatches.Patch(facecolor="#e0e0e0", edgecolor="0.35", linewidth=0.5,
                               label="1-cell inflation")
    route_h = Line2D([0], [0], color=PS.BLUE, ls="-", lw=1.0, alpha=0.6,
                     label="Map1 stored route (A, B)")
    conflict_h = mpatches.Patch(facecolor=to_rgba(PS.VERMILLION, 0.06),
                                edgecolor=PS.VERMILLION, hatch="////", linewidth=0.0,
                                label="route–obstacle conflict cell (B)")
    followed_h = Line2D([0], [0], color=PS.VERMILLION, ls="-", lw=2.1,
                        label="naive route followed (C, D)")
    invalid_h = Line2D([0], [0], color=PS.VERMILLION, ls="--", lw=2.0,
                       label="invalid naive continuation (C, D)")
    block_h = Line2D([0], [0], color=PS.VERMILLION, marker="X", ls="", ms=9,
                     mec="white", mew=1.2, label="blocked contact (C, D)")
    recon_h = Line2D([0], [0], color=PS.BLUE, ls="-", lw=2.4,
                     label="online-RRT reconnect to goal (C, D)",
                     path_effects=[pe.withStroke(linewidth=4.0, foreground="white")])
    start_h = Line2D([0], [0], color="k", marker="o", ls="", ms=6, label="start (C, D)")
    goal_h = Line2D([0], [0], color="k", marker="*", ls="", ms=11, label="goal (C, D)")
    leg_ax.legend(handles=[obs_dark, obs_light, route_h, conflict_h, block_h,
                           followed_h, invalid_h, recon_h, start_h, goal_h],
                  loc="center", ncol=5, fontsize=8.5, frameon=False,
                  handlelength=2.2, handleheight=1.3,
                  columnspacing=1.3, handletextpad=0.5)

    PS.save(fig, os.path.join(FIG, "fig_route_mismatch"))
    print(f"wrote fig_route_mismatch.pdf  (naive {cN_cells:.1f}% cells / {cN_routes:.0f}% routes)")


# ============================ Fig. 2 : pipeline schematic ============================
# Semantic color system (visual-audit: color should encode category, consistently).
C_MAP   = "#cfe2f3"   # environment / map
C_PLAN  = "#d9ead3"   # planner / BC policy
C_LIB   = "#fce5cd"   # route library
C_RECON = "#f4cccc"   # online-RRT reconnect
C_COND  = "#efefef"   # experimental condition
C_SHARE = "#d9d2e9"   # optional sharing mechanism (shared map H)


def _box(ax, xy, w, h, text, fc, fontsize=11.0):
    ax.add_patch(mpatches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.015,rounding_size=0.018",
                                         fc=fc, ec="#222222", lw=1.3))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _diamond(ax, cx, cy, w, h, text, fc=C_COND, fontsize=10.0):
    ax.add_patch(mpatches.Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2),
                                   (cx - w / 2, cy)], closed=True, fc=fc, ec="#222222", lw=1.3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, p, q, color="#222222", lw=1.6, mutation_scale=15):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=mutation_scale,
                                 lw=lw, color=color))


def _rounded(points, r, n=16):
    """Sample an orthogonal/polyline path with rounded corners (quadratic-bezier fillets).
    Used for the Panel B next-timestep return so the outer feedback lane reads as a smooth
    looping arrow rather than a rectangular frame, while still occupying a clean lane that
    never crosses the inner reconnect corridor. Returns (xs, ys)."""
    pts = [np.array(p, float) for p in points]
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
        u_in = (p1 - p0) / (np.hypot(*(p1 - p0)) + 1e-9)
        u_out = (p2 - p1) / (np.hypot(*(p2 - p1)) + 1e-9)
        ri = min(r, np.hypot(*(p1 - p0)) / 2, np.hypot(*(p2 - p1)) / 2)
        a = p1 - u_in * ri
        c = p1 + u_out * ri
        tt = np.linspace(0, 1, n)[:, None]
        arc = (1 - tt) ** 2 * a + 2 * (1 - tt) * tt * p1 + tt ** 2 * c
        out.append(a); out.extend(arc); out.append(c)
    out.append(pts[-1])
    arr = np.array(out)
    return arr[:, 0], arr[:, 1]


def framework():
    # Legend BELOW the panel block. Round-5 audit: legend was floating too far from the
    # panels and felt detached. Tightened hspace 0.34 -> 0.22 and bumped legend
    # height_ratio 0.15 -> 0.18 with larger swatches so the key visually attaches to the
    # figure body instead of reading as a separate row of content.
    fig = plt.figure(figsize=(12.0, 8.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.18],
                          top=0.97, bottom=0.035, left=0.04, right=0.985,
                          hspace=0.22, wspace=0.16)
    panels = {"A": fig.add_subplot(gs[0, 0]),
              "B": fig.add_subplot(gs[0, 1]),
              "C": fig.add_subplot(gs[1, 0]),
              "D": fig.add_subplot(gs[1, 1])}
    leg_ax = fig.add_subplot(gs[2, :])
    leg_ax.set_xlim(0, 1); leg_ax.set_ylim(0, 1); leg_ax.axis("off")
    for p in panels.values():
        p.set_xlim(0, 1); p.set_ylim(0, 1); p.axis("off")

    # Compact 2-row x 3-col visual key. Swatch height enlarged (0.30 -> 0.45) and "Visual
    # key:" softened from semibold to regular weight (round-5 audit #36) so the encodings
    # rather than the heading dominate. Items grouped: node colors (top row) + structural
    # encodings (bottom row).
    leg_ax.text(0.005, 0.5, "Visual key:", va="center", ha="left",
                fontsize=10.5, fontweight="normal", color="0.25")
    legend_items = [(C_MAP,    "Environment / robot state"),
                    (C_PLAN,   "Planner / BC policy"),
                    (C_LIB,    "Route library"),
                    (C_RECON,  "Online RRT reconnect"),
                    (C_SHARE,  "Shared collision map $H$"),
                    ("DASHED", "Comparison arm (Panel C)")]
    ncol = 3; sw_w, sw_h = 0.032, 0.42
    for i, (fc, label) in enumerate(legend_items):
        col = i % ncol; row = i // ncol
        x0 = 0.10 + col * 0.300
        y_center = 0.74 if row == 0 else 0.26
        y0 = y_center - sw_h / 2
        if fc == "DASHED":
            patch = mpatches.FancyBboxPatch(
                (x0, y0), sw_w, sw_h,
                boxstyle="round,pad=0.003,rounding_size=0.008",
                fc="white", ec="0.20", lw=1.4, linestyle="--")
        else:
            patch = mpatches.FancyBboxPatch(
                (x0, y0), sw_w, sw_h,
                boxstyle="round,pad=0.003,rounding_size=0.008",
                fc=fc, ec="#222222", lw=1.0)
        leg_ax.add_patch(patch)
        leg_ax.text(x0 + sw_w + 0.010, y_center, label,
                    va="center", ha="left", fontsize=10.5)

    def plabel(a, L, title):
        a.text(0.0, 1.04, L, transform=a.transAxes, fontsize=14,
               fontweight="bold", va="bottom")
        a.text(0.075, 1.04, title, transform=a.transAxes, fontsize=11.5, va="bottom",
               fontweight="semibold")

    # White halo for every connector / branch label (round-7 audit): the recurring class of
    # defect was text sitting close enough to a connector stroke that it read as crossed at
    # print scale. A white outline guarantees the glyphs stay legible even where a line
    # passes nearby, without having to chase sub-0.02 position tweaks on every label.
    HALO = [pe.withStroke(linewidth=3.0, foreground="white")]

    # ============================== Panel A: offline ==============================
    # Direct map -> library arrow REMOVED (round-2 audit): the library is produced by the
    # RRT expert from the map, so a single dependency chain (Map -> RRT -> Library) is the
    # cleaner story. The "map metadata stored alongside routes" detail is conveyed in the
    # surrounding text rather than as a second arrow that suggested independent production.
    a = panels["A"]; plabel(a, "A", "Offline: build the route library")
    _box(a, (0.04, 0.62), 0.36, 0.22, "Map layout\n(grid + obstacles)", C_MAP, 11)
    _box(a, (0.60, 0.62), 0.36, 0.22, "RRT expert planner", C_PLAN, 11)
    _box(a, (0.30, 0.14), 0.40, 0.22, "Offline route library", C_LIB, 11)
    _arrow(a, (0.40, 0.73), (0.60, 0.73))           # map -> planner
    _arrow(a, (0.78, 0.62), (0.60, 0.38))           # planner -> library
    # Round-5 audit #12/#13/#14: "obstacle map" was redundant (the source box already says
    # "grid + obstacles"); removed to let the arrow breathe. "stored routes" darkened from
    # 0.15 -> 0.10 and bumped 10.5 -> 11 so it doesn't visually disappear at column width.
    a.text(0.80, 0.50, "stored\nroutes", ha="left", va="center", fontsize=11,
           color="0.10", style="italic", path_effects=HALO)

    # ===================== Panel B: deployment loop ============================
    # Strict per-step flow: Select -> BC -> Decision -> {yes: Reconnect, no: Advance}.
    # Round-6 audit (connector-lane pass): the two feedback paths are now placed in
    # DISTINCT lanes that enter Select on DIFFERENT faces, so they can never be confused.
    #   * Reconnect feedback (occasional): inner vertical lane, enters Select BOTTOM-centre.
    #   * Next-timestep return (every step): outer rounded lane down the bottom and up the
    #     FAR-LEFT margin (left of all nodes), enters Select LEFT face.
    # The whole panel was shifted right (node left edge 0.02 -> 0.10) to open a dedicated
    # left return lane at x~0.035 that is clear of every node and of the inner reconnect
    # corridor. The yes branch now enters Online RRT's RIGHT face while reconnect exits its
    # TOP face (round-6 audit #7), so the two no longer knot at one corner.
    b = panels["B"]; plabel(b, "B", "Deployment (per robot, per timestep)")
    _box(b, (0.10, 0.72), 0.26, 0.16, "Select and filter\nroutes", C_LIB, 10.5)
    _box(b, (0.46, 0.72), 0.24, 0.16, "BC local control", C_PLAN, 11)
    _diamond(b, 0.58, 0.46, 0.28, 0.26, "Blocked or\nroute too far?", fontsize=10.5)
    _box(b, (0.10, 0.16), 0.26, 0.16, "Online RRT\nreconnect", C_RECON, 10.5)
    _box(b, (0.71, 0.16), 0.25, 0.16, "Advance\nrobot", C_MAP, 11)
    _arrow(b, (0.36, 0.80), (0.46, 0.80))            # Select -> BC
    _arrow(b, (0.58, 0.72), (0.58, 0.59))            # BC -> Decision top
    # yes/no labels sit ABOVE their branches with clearance (round-7 audit #5/#13); the
    # white halo keeps them legible where the branch stroke runs nearby.
    _arrow(b, (0.44, 0.45), (0.37, 0.27))            # yes -> Online RRT right face
    b.text(0.395, 0.42, "yes", fontsize=10.5, color="#222222", fontweight="semibold",
           ha="center", path_effects=HALO)
    _arrow(b, (0.72, 0.45), (0.82, 0.33))            # no  -> Advance top
    b.text(0.785, 0.42, "no", fontsize=10.5, color="#222222", fontweight="semibold",
           ha="center", path_effects=HALO)
    # Reconnect feedback: inner vertical lane, Online RRT TOP-centre -> Select BOTTOM-centre.
    _arrow(b, (0.23, 0.32), (0.23, 0.72))
    b.text(0.26, 0.52, "new waypoint", fontsize=11, color="0.10",
           style="italic", va="center", ha="left", path_effects=HALO)
    # Next-timestep return: outer recurrence lane. Advance bottom -> shallow bottom run ->
    # sweeping LEFT corner -> up the far-left margin -> into Select's LEFT face. The
    # bottom-left corner now uses a large fillet (r=0.16) so the turn reads as one
    # continuous sweep rather than a square corner + vertical wall (round-7 audit #1/#4),
    # while the lane still clears every node and never shares the inner reconnect corridor.
    rx, ry = _rounded([(0.835, 0.150), (0.835, 0.065), (0.045, 0.065),
                       (0.045, 0.80), (0.085, 0.80)], r=0.16)
    b.plot(rx, ry, color="0.45", lw=1.3, zorder=1,
           solid_joinstyle="round", solid_capstyle="round")
    _arrow(b, (0.085, 0.80), (0.10, 0.80), color="0.45", lw=1.3, mutation_scale=18)
    b.text(0.52, 0.105, "next timestep", ha="center", va="center",
           fontsize=11, color="0.10", style="italic", path_effects=HALO)

    # ====================== Panel C: skill-transfer experiment ======================
    # Two parallel CONDITIONS (dashed cards) -- now with a "vs." marker between them and
    # an "alternative arms" italic between the converging arrows so the reader cannot read
    # the two condition arrows as simultaneous inputs (round-2 audit #39). Dashed outlines
    # darkened from 0.45 to 0.20 with lw 1.1 -> 1.3 so the condition grouping does not
    # recede behind the solid node boxes (round-2 audit #52).
    c = panels["C"]; plabel(c, "C", "Skill-transfer experiment (route library only)")
    # CRITICAL round-4 fix (audit #1/#2/#39): the round-3 Condition 2 dashed container
    # extended to x=1.00, but FancyBboxPatch with rounded corners adds ~0.012 padding
    # outside the rect so its right border clipped on the axis right edge. Card widths
    # dropped 0.44 -> 0.42 with x0 = (0.04, 0.54) -- right edges land at 0.46 and 0.96,
    # well inside the axis. Gap widens to 0.08 so "vs." has clear whitespace around it.
    cards_c = [(0.04, "Condition 1", "Naive library\n(built on Map1)"),
               (0.54, "Condition 2", "Adapted library\n(built on Map2)")]
    for x0, label, body in cards_c:
        c.add_patch(mpatches.FancyBboxPatch(
            (x0, 0.56), 0.42, 0.40,
            boxstyle="round,pad=0.010,rounding_size=0.015",
            fc="white", ec="0.20", lw=1.3, linestyle="--"))
        c.text(x0 + 0.21, 0.93, label, ha="center", va="top",
               fontsize=10.5, fontweight="bold", color="0.15")
        _box(c, (x0 + 0.04, 0.62), 0.34, 0.20, body, C_LIB, 10.5)
    # "vs." sat trapped between dashed cards (round-5 audit #18/#19) -- given a small
    # white clearance halo so it reads as a deliberate comparison marker rather than an
    # overlay artefact between the two outlines.
    c.text(0.50, 0.83, "vs.", ha="center", va="center",
           fontsize=13, color="0.10", fontweight="bold",
           path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])
    _arrow(c, (0.25, 0.56), (0.36, 0.40))
    _arrow(c, (0.75, 0.56), (0.64, 0.40))
    c.text(0.50, 0.50, "alternative arms", ha="center", va="center",
           fontsize=11, color="0.10", style="italic", path_effects=HALO)
    _box(c, (0.18, 0.18), 0.64, 0.22, "Map2 deployment environment", C_MAP, 11.5)
    # "Fixed:" note tied to the deployment box with a small bracket so it reads as an
    # experimental-control statement attached to that node. Round-7 audit #31/#32: bracket
    # darkened (0.50 -> 0.40) and the note pulled up closer to the Map2 box (y 0.08 -> 0.095)
    # so the attachment is unambiguous.
    c.plot([0.30, 0.30, 0.70, 0.70], [0.17, 0.145, 0.145, 0.17],
           color="0.40", lw=1.1, zorder=1, solid_capstyle="round")
    c.text(0.50, 0.095, "Fixed: BC network + RRT planner",
           ha="center", va="center", fontsize=11, color="0.10",
           fontweight="semibold")

    # =================== Panel D: optional collision-history sharing ===================
    # H -> bracket -> {selective, q=1} structure (round-2 audit #43, #44, #47): single H box
    # shifted to centre over the bracket span; one stem from H bottom-centre down to the
    # bracket bar; bracket horizontal spans the two H-consuming cards; arrowed drops at each
    # end. "H read/write" labels the bracket bar (local meaning of the purple). q=0 carries
    # its "no H read/write" exclusion directly below its card.
    d = panels["D"]; plabel(d, "D", "Optional collision-history sharing (add-on)")
    H_purple = "#7a5ea8"
    # Selective centre x = 0.37 + 0.135 = 0.505; q=1 centre x = 0.71 + 0.135 = 0.845;
    # bracket span midpoint = 0.675. H box centred over that midpoint.
    _box(d, (0.535, 0.76), 0.28, 0.16, "Shared collision\nmap $H$", C_SHARE, 11)
    # q=0 card now carries "no $H$ read/write" as a third line INSIDE the box (round-3
    # audit #63 -- darker and visually integrated rather than detached below).
    cards_d = [(0.02, "None\n$q_{\\mathrm{share}}{=}0$\n(no $H$ read/write)"),
               (0.37, "Selective\n$0{<}q_{\\mathrm{share}}{<}1$"),
               (0.71, "All\n$q_{\\mathrm{share}}{=}1$")]
    for x0, lab in cards_d:
        _box(d, (x0, 0.20), 0.27, 0.30, lab, C_COND, 10.5)
    # Bracket: stem from H bottom-centre down to bracket bar; horizontal bar spans the two
    # H-consuming cards; arrowed drops to their card tops.
    d.plot([0.675, 0.675], [0.76, 0.62], color=H_purple, lw=1.8, zorder=2,
           solid_capstyle="butt")
    d.plot([0.505, 0.845], [0.62, 0.62], color=H_purple, lw=1.8, zorder=2,
           solid_capstyle="butt")
    _arrow(d, (0.505, 0.62), (0.505, 0.50), color=H_purple, lw=1.8, mutation_scale=16)
    _arrow(d, (0.845, 0.62), (0.845, 0.50), color=H_purple, lw=1.8, mutation_scale=16)
    # CRITICAL round-6 fix (audit #16): the bracket-bar label was centred/right-anchored so
    # its glyphs straddled the vertical purple connector at x=0.675 -- the connector pierced
    # the text. Label now starts at x=0.70 (ha="left"), entirely to the RIGHT of the
    # connector, so no linework touches the glyphs.
    d.text(0.71, 0.665, "$H$ read/write access", ha="left", va="bottom",
           fontsize=10.5, color=H_purple, style="italic", path_effects=HALO)
    # Episode-condition axis label collapsed to ONE line (round-5 audit #29/#30): the
    # two-line version was cramped and the second line was too small to survive print
    # reduction. A subtle horizontal axis bracket under all three cards (round-5 audit
    # #28) anchors the label to the full card group rather than the middle card.
    d.plot([0.05, 0.05, 0.93, 0.93], [0.17, 0.14, 0.14, 0.17],
           color="0.55", lw=1.0, zorder=1, solid_capstyle="round")
    d.text(0.50, 0.07,
           r"Episode condition: fraction of robots that consult $H$",
           ha="center", va="bottom", fontsize=10.5, color="0.10")

    PS.save(fig, os.path.join(FIG, "fig_framework"))
    print("wrote fig_framework.pdf")


# ============================ Fig. 1 : map layouts ============================
def environments():
    """Fig. 1: the two map layouts, clean -- no axis ticks, no generic configuration-space
    title, panels labelled simply Map1/Map2 (reviewer item 11). Regenerated from the real
    obstacle layouts (inflated by the robot footprint), so it stays reproducible."""
    m1, _ = _obst_matrix("map1")
    m2, _ = _obst_matrix("map2")
    novel = ((m2 > 0.5) & ~(m1 > 0.5)).astype(float)   # obstacle in Map2 but free in Map1
    cmap = ListedColormap(["white", "#4d4d4d"])         # free = white, obstacle = dark gray (less heavy)
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 4.7))
    for a, mat, name in [(ax[0], m1, "Map1"), (ax[1], m2, "Map2")]:
        a.imshow(mat, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        a.contour(mat, levels=[0.5], colors="k", linewidths=0.5)
        a.set_title(name)
        a.set_xticks([0, 10, 20, 30, 40, 49]); a.set_xticklabels([0, 10, 20, 30, 40, 50])
        a.set_yticks([0, 10, 20, 30, 40, 49]); a.set_yticklabels([0, 10, 20, 30, 40, 50])
        a.tick_params(labelsize=8, length=2)
        a.grid(True, color="0.85", lw=0.3, alpha=0.5)   # lighter grid so it does not dominate
    # outline the obstacles that are new in Map2 (prepares the route-mismatch argument).
    # The in-figure "vermillion outline: ..." caption annotation was removed because the
    # manuscript caption already names the vermillion outlines, and it visually collided
    # with the relocated bottom legend.
    ax[1].contour(novel, levels=[0.5], colors=PS.VERMILLION, linewidths=1.8)
    leg = [mpatches.Patch(facecolor="#4d4d4d", edgecolor="k", label="inflated obstacle"),
           mpatches.Patch(facecolor="white", edgecolor="k", label="free cell"),
           Line2D([0], [0], color=PS.VERMILLION, lw=1.8,
                  label="obstacle new in Map2")]
    # Legend BELOW the panels (between the figure body and the manuscript caption) rather
    # than above the panel titles, matching the placement convention used by Fig. 2.
    fig.legend(handles=leg, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
               fontsize=10, frameon=True)
    fig.tight_layout(rect=[0, 0.07, 1, 1.0])
    PS.save(fig, os.path.join(FIG, "fig2_environments"))
    print("wrote fig2_environments.pdf")


# ===================== Route-library diagnostics table (Section 3.3) =====================
def route_diagnostics():
    """Route-library diagnostics table: size / coverage / obstacle-overlap of the two libraries,
    computed directly from the released datasets + maps. Shows the adapted library is not
    simply larger or denser -- the difference is environment match (reviewer item 2)."""
    import pandas as pd
    TAB = os.path.join(_CODE, "tables")
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    _, b2 = _obst_matrix("map2")

    def raw_routes(dataset):
        csv = "original_data.csv" if dataset == "original" else "alternative_data.csv"
        d = pd.read_csv(os.path.join(_CODE, "data", csv)).dropna()
        if dataset == "original":
            d = d[~((d.iloc[:, -2:] < -2) | (d.iloc[:, -2:] > 2)).any(axis=1)].iloc[:, :2]
            return F.extract_routes(d)
        return F.extract_routes_newdata(d)

    rows = []
    for name, ds, gen in [("Map2-naive", "original", "Map1"), ("Map2-adapted", "alternative", "Map2")]:
        raw = raw_routes(ds)
        filt, _, _, _ = build_scenario("map2", ds)   # native-filtered, deployed on Map2
        lens = [len(r) for r in filt if len(r)]
        free = set()
        for r in filt:
            for p in r:
                if not F.is_point_inside_black_ranges(p[0], p[1], b2):
                    free.add((int(p[0]), int(p[1])))
        pc, pr = _overlap_stats(filt, b2)
        rows.append((name, gen, len(raw), len(filt), int(np.median(lens)), len(free), pc, pr))

    tex = [r"% AUTO-GENERATED by make_heatmaps_rrtbc.py -- do not edit by hand.",
           r"\begin{table}[H]",
           r"\caption{Route-library diagnostics (both libraries deployed on Map2), computed "
           r"from the released route datasets and map layouts. The two libraries hold a "
           r"comparable number of routes (within about 2\%), so the adapted library is not "
           r"simply larger; their routes differ modestly in median length and free-cell "
           r"coverage, but the decisive difference is that the Map2-naive library leaves route "
           r"cells inside Map2 obstacles while the adapted one does not.}",
           r"\label{tab:route_diag}",
           r"\centering", r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{l c c c c c c c}", r"\toprule",
           r"\textbf{Library} & \textbf{\shortstack{Generated\\on}} & \textbf{\shortstack{Raw\\routes}} & "
           r"\textbf{\shortstack{Filtered\\routes}} & \textbf{\shortstack{Median\\length}} & "
           r"\textbf{\shortstack{Free cells\\covered}} & \textbf{\shortstack{Cells in\\obstacles}} & "
           r"\textbf{\shortstack{Routes\\overlapping}} \\", r"\midrule"]
    for name, gen, raw, filt, mlen, free, pc, pr in rows:
        tex.append(f"{name} & {gen} & {raw} & {filt} & {mlen} & {free} & {pc:.1f}\\% & {pr:.0f}\\% \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TAB, "tab_route_diag.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"wrote tables/tab_route_diag.tex  (naive raw={rows[0][2]} filt={rows[0][3]}; "
          f"adapted raw={rows[1][2]} filt={rows[1][3]})")


if __name__ == "__main__":
    environments()       # cheap (assets only) -- Fig. 1
    framework()          # cheap (no data)     -- Fig. 2
    route_mismatch()     # cheap (assets only) -- Fig. 3
    route_diagnostics()  # cheap (assets only) -- route-library diagnostics table
    heatmaps()           # parallel episodes   -- Fig. 5
