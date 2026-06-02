#!/usr/bin/env python3
"""
Analysis for the faithful RRT+BC campaign.
Reads results_rrtbc.csv (hybrid skill-transfer + collision-history sharing) and, if
present, results_rrt_online.csv (online-RRT baseline). Reports collisions BOTH absolutely
and per completed task (the workload-fair metric the thesis itself argued for), runs the
statistics, writes stats_summary_rrtbc.md, and regenerates the figures.
"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle as PS                      # shared publication figure style (visual-audit fix)
PS.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(FIG, exist_ok=True)
TAB = os.path.join(os.path.dirname(HERE), "tables")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


def cliffs_delta(a, b):
    """Cliff's delta effect size of a vs b; +1 means every a exceeds every b."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    gt = sum(int(x > y) for x in a for y in b)
    lt = sum(int(x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def bootstrap_ci(x, n=5000, seed=0):
    """Percentile bootstrap 95% CI for the mean (deterministic seed)."""
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (preserves input order)."""
    pvals = list(pvals); m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m; running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def _machine():
    """Report the CPU model and Python version actually running this analysis, so the
    hardware-sensitive runtime numbers are attributable (no invented hardware)."""
    import platform, re
    cpu = ""
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.lower().startswith("model name"):
                cpu = ln.split(":", 1)[1].strip(); break
    except Exception:
        pass
    cpu = cpu or platform.processor() or "unspecified CPU"
    # Clean publication-unfriendly raw notation: drop (R)/(TM), turn "@ 3.60GHz" into
    # "(3.60 GHz)", collapse whitespace.
    cpu = re.sub(r"\((?:R|TM|r|tm)\)", "", cpu)
    cpu = re.sub(r"\s*@\s*([\d.]+)\s*GHz", r", \1 GHz", cpu)
    cpu = re.sub(r"\s{2,}", " ", cpu).strip()
    return cpu, platform.python_version()

df = pd.read_csv(os.path.join(HERE, "results_rrtbc.csv"))
df["coll_per_task"] = df["collisions"] / df["tasks_completed"].clip(lower=1)
df["fail_rate"] = df["fails"] / (df["tasks_completed"] + df["fails"]).clip(lower=1)
rrt_path = os.path.join(HERE, "results_rrt_online.csv")
rrt = pd.read_csv(rrt_path) if os.path.exists(rrt_path) else None
if rrt is not None:
    rrt["coll_per_task"] = rrt["collisions"] / rrt["tasks_completed"].clip(lower=1)
    rrt["fail_rate"] = rrt["fails"] / (rrt["tasks_completed"] + rrt["fails"]).clip(lower=1)

out = ["# RRT + Behavior-Cloning campaign — statistics\n"]
out.append(f"Source: results_rrtbc.csv ({len(df)} episodes). "
           f"Faithful method: RRT expert routes + trained BC local policy + route reuse. "
           f"Collisions = blocked-move (3-step-lookahead-into-obstacle) events. "
           f"Episodes run to steps={int(df['steps'].mode()[0])} "
           f"(~{int(df[df.n_agents==5].tasks_completed.mean())} completed tasks at 5 agents), "
           f"so the shared collision database fills and comparisons have power.\n")


def ms(x):
    return f"{np.mean(x):.2f} ± {np.std(x, ddof=1):.2f}"


# ---------------- SKILL TRANSFER (Map2) ----------------
st = df[df.exp == "T0_skill_transfer_map2"]
if len(st):
    out.append("\n## Skill transfer on Map2 (hybrid; Map2-naive 'original' vs Map2-adapted 'alternative')")
    def _row(n, g):
        """(n, coll/task mean, lo, hi, fail-rate mean, lo, hi) with bootstrap 95% CIs."""
        cl, ch = bootstrap_ci(g.coll_per_task)
        fl, fh = bootstrap_ci(g.fail_rate)
        return (n, g.coll_per_task.mean(), cl, ch, g.fail_rate.mean(), fl, fh)

    rows_o, rows_a, rows_r = [], [], []
    for n in sorted(st.n_agents.unique()):
        o = st[(st.n_agents == n) & (st.dataset == "original")]
        a = st[(st.n_agents == n) & (st.dataset == "alternative")]
        red = 100 * (o.collisions.mean() - a.collisions.mean()) / o.collisions.mean()
        try:
            p = mannwhitneyu(o.collisions, a.collisions, alternative="greater").pvalue
        except ValueError:
            p = float("nan")
        out.append(f"- {n} agents: original coll {ms(o.collisions)} (per-task {o.coll_per_task.mean():.2f}, "
                   f"fail-rate {o.fail_rate.mean():.2f}); alternative coll {ms(a.collisions)} "
                   f"(per-task {a.coll_per_task.mean():.2f}, fail-rate {a.fail_rate.mean():.2f}); "
                   f"adaptation cuts collisions {red:.0f}% (MWU original>alt p={p:.3g}).")
        rows_o.append(_row(n, o))
        rows_a.append(_row(n, a))
        if rrt is not None:
            rr = rrt[(rrt["map"] == "map2") & (rrt.n_agents == n)]
            if len(rr):
                rows_r.append(_row(n, rr))

    # figure: collisions-per-task and fail-rate vs fleet size, 3 methods, with
    # bootstrap 95% CI bands so the seed-to-seed uncertainty is visible (not only in the table).
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.1))
    ao, aa = np.array(rows_o), np.array(rows_a)
    # Shorter legend labels for this figure (the global PS labels stay verbose for tables).
    short_labels = {PS.M_NAIVE["label"]: "Hybrid: naive",
                    PS.M_ADAPT["label"]: "Hybrid: adapted",
                    PS.M_ONLINE["label"]: "Online RRT"}
    series = [(ao, PS.M_NAIVE), (aa, PS.M_ADAPT)]
    if rows_r:
        series.append((np.array(rows_r), PS.M_ONLINE))
    for arr, s in series:
        lab = short_labels.get(s["label"], s["label"])
        ax[0].plot(arr[:, 0], arr[:, 1], color=s["color"], marker=s["marker"], ls=s["ls"],
                   lw=1.8, ms=6.5, label=lab)
        ax[0].fill_between(arr[:, 0], arr[:, 2], arr[:, 3], color=s["color"], alpha=0.30, lw=0)
        ax[1].plot(arr[:, 0], 100 * arr[:, 4], color=s["color"], marker=s["marker"], ls=s["ls"],
                   lw=1.8, ms=6.5)
        ax[1].fill_between(arr[:, 0], 100 * arr[:, 5], 100 * arr[:, 6], color=s["color"],
                           alpha=0.30, lw=0)
    for a in ax:
        a.set_xlabel("number of robots ($N$)"); PS.fleet_xaxis(a)
        a.grid(axis="y", alpha=0.18); a.set_axisbelow(True)
    ax[0].set_ylabel("collisions per completed task")
    ax[0].set_title("Collisions per completed task")
    ax[1].set_ylabel("task-failure rate (%)")
    ax[1].set_title("Task-failure rate")
    # Y-axis headroom -- the top naive point currently sits near the upper boundary.
    ax[0].set_ylim(0, max(ao[:, 1].max() * 1.18, 0.5))
    ax[1].set_ylim(0, max(100 * ao[:, 4].max() * 1.18, 5))
    PS.panel(ax[0], "A"); PS.panel(ax[1], "B")
    # Vertical comparison bracket at N=4 between the naive value (top) and adapted value
    # (bottom). Neutral dark gray so it does NOT visually claim the result for any one series,
    # explicit "vs naive" wording so the comparison is unambiguous. The percentage is computed
    # from the data, not hardcoded.
    if 4 in ao[:, 0]:
        idx = list(ao[:, 0]).index(4)
        y_top = ao[idx, 1]; y_bot = aa[idx, 1]
        pct = 100 * (y_top - y_bot) / max(y_top, 1e-9)
        x_b = 4 + 0.30
        ax[0].annotate("", xy=(x_b, y_top), xytext=(x_b, y_bot),
                       arrowprops=dict(arrowstyle="|-|", color="0.25", lw=1.3,
                                       shrinkA=0, shrinkB=0, mutation_scale=6))
        ax[0].text(x_b + 0.18, 0.5 * (y_top + y_bot),
                   f"{pct:.0f}% lower\nvs naive at $N{{=}}4$",
                   fontsize=9.0, color="0.20", ha="left", va="center",
                   linespacing=1.2)
    # Shared method legend below both panels, no frame (cleaner publication look).
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04),
               frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 1.0]); PS.save(fig, os.path.join(FIG, "fig_skill_transfer"))


# ---------------- SHARING ablation (Map1 / Map2) ----------------
for mp in ("map1", "map2"):
    ab = df[df.exp == f"S1_ablation_{mp}"]
    if not len(ab):
        continue
    out.append(f"\n## Collision-history sharing — ablation on {mp} (5 agents)")
    none = ab[ab.share_fraction == 0.0]
    sel = ab[ab.share_fraction == 0.6]
    full = ab[ab.share_fraction == 1.0]
    for name, g in [("none", none), ("selective 0.6", sel), ("full 1.0", full)]:
        out.append(f"- {name}: collisions {ms(g.collisions)} | per-task {g.coll_per_task.mean():.2f} | "
                   f"tasks {ms(g.tasks_completed)}")
    for name, g in [("selective", sel), ("full", full)]:
        red = 100 * (none.collisions.mean() - g.collisions.mean()) / none.collisions.mean()
        p = mannwhitneyu(none.collisions, g.collisions, alternative="greater").pvalue
        out.append(f"- {name} reduces collisions {red:.0f}% vs none (MWU none>{name} p={p:.3g}).")
    p_sf = mannwhitneyu(full.collisions, sel.collisions, alternative="greater").pvalue
    out.append(f"- full>selective one-sided p={p_sf:.3g} "
               f"({'selective better (migration under full)' if full.collisions.mean()>sel.collisions.mean() else 'no migration'}).")

# ---------------- SHARING sweep (Map1) ----------------
sw = df[df.exp == "S2_sweep_map1"]
if len(sw):
    out.append("\n## Access-fraction sweep (Map1, 5 agents)")
    xs, ys = [], []
    for p in sorted(sw.share_fraction.unique()):
        g = sw[sw.share_fraction == p]
        out.append(f"- p={p}: collisions {ms(g.collisions)} | per-task {g.coll_per_task.mean():.2f}")
        xs.append(p); ys.append(g.coll_per_task.mean())
    rho, pr = spearmanr(sw.share_fraction, sw.coll_per_task)
    out.append(f"- Spearman ρ(p, collisions-per-task) = {rho:.2f} (p={pr:.3g}); minimum at p={xs[int(np.argmin(ys))]}.")
    # Discrete points with 95% CI (no connecting line, so no spurious dose-response trend);
    # y starts at 0; no-sharing baseline + Spearman annotation make the flat result explicit.
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    fracs = sorted(sw.share_fraction.unique())
    means, los, his = [], [], []
    for p in fracs:
        g = sw[sw.share_fraction == p].coll_per_task
        lo, hi = bootstrap_ci(g); means.append(g.mean()); los.append(lo); his.append(hi)
    means, los, his = np.array(means), np.array(los), np.array(his)
    base = means[fracs.index(0.0)]
    ax.axhline(base, color=PS.GRAY, ls="--", lw=1.3, label="no-sharing baseline ($q_{\\mathrm{share}}=0$)")
    ax.errorbar(fracs, means, yerr=[means - los, his - means], fmt="o", color=PS.BLUE,
                capsize=4, ms=7, lw=0, elinewidth=1.6, label="mean (95% CI)")
    ax.set_ylim(0, max(his) * 1.18)
    ax.set_xticks(fracs)
    ax.set_xlabel("sharing access fraction $q_{\\mathrm{share}}$")
    ax.set_ylabel("collisions per completed task")
    ax.set_title("No dose-response across sharing access fraction")
    ax.annotate(f"Spearman $\\rho={rho:.2f}$, $p={pr:.2f}$\n(no monotonic trend)",
                xy=(0.5, 0.9), xycoords="axes fraction", ha="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
    ax.legend(loc="lower right")
    fig.tight_layout(); PS.save(fig, os.path.join(FIG, "fig_sharing_sweep"))

# ---------------- SHARING scalability (Map1) ----------------
sc = df[df.exp == "S3_scalability_map1"]
if len(sc):
    out.append("\n## Sharing scalability (Map1, none vs selective 0.6)")
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for sh, s in [(0.0, PS.S_NONE), (0.6, PS.S_SEL)]:
        g = sc[sc.share_fraction == sh]
        ns = sorted(g.n_agents.unique())
        t_m, t_lo, t_hi, c_m, c_lo, c_hi = [], [], [], [], [], []
        for n in ns:
            gt = g[g.n_agents == n].tasks_completed; gc = g[g.n_agents == n].coll_per_task
            t_m.append(gt.mean()); lo, hi = bootstrap_ci(gt); t_lo.append(lo); t_hi.append(hi)
            c_m.append(gc.mean()); lo, hi = bootstrap_ci(gc); c_lo.append(lo); c_hi.append(hi)
        lab = "no sharing" if sh == 0.0 else "selective ($q_{\\mathrm{share}}=0.6$)"
        ax[0].plot(ns, t_m, color=s["color"], marker=s["marker"], ls=s["ls"], label=lab)
        ax[0].fill_between(ns, t_lo, t_hi, color=s["color"], alpha=0.18, lw=0)
        ax[1].plot(ns, c_m, color=s["color"], marker=s["marker"], ls=s["ls"], label=lab)
        ax[1].fill_between(ns, c_lo, c_hi, color=s["color"], alpha=0.18, lw=0)
        rho, pr = spearmanr(g.n_agents, g.tasks_completed)
        out.append(f"- {lab}: ρ(fleet, tasks)={rho:.2f} (p={pr:.2g}); "
                   f"coll-per-task {g.coll_per_task.min():.2f}..{g.coll_per_task.max():.2f}")
    for a in ax:
        a.set_xlabel("number of robots ($N$)"); PS.fleet_xaxis(a)
    ax[0].set_ylabel("tasks completed (fixed budget)"); ax[0].set_title("Throughput")
    ax[1].set_ylabel("collisions per completed task"); ax[1].set_title("Collision rate")
    PS.panel(ax[0], "A"); PS.panel(ax[1], "B")
    ax[1].annotate("curves overlap within\nseed variability\n(selective effect n.s.)",
                   xy=(0.04, 0.96), xycoords="axes fraction", va="top", fontsize=9, color=PS.GRAY)
    ax[0].legend(loc="upper left")
    fig.tight_layout(); PS.save(fig, os.path.join(FIG, "fig_sharing_scalability"))

# ============ Fig 5 / Fig 7 / planning-cost table (need instrumented columns) ============
_INSTR = ("wall_s" in df.columns and rrt is not None and "wall_s" in rrt.columns)
if _INSTR:
    for d in (df, rrt):
        d["obst_per_task"] = d["coll_obstacle"] / d["tasks_completed"].clip(lower=1)
        d["robot_per_task"] = d["coll_robot"] / d["tasks_completed"].clip(lower=1)
        d["rrt_per_task"] = d["rrt_calls"] / d["tasks_completed"].clip(lower=1)
    st = df[df.exp == "T0_skill_transfer_map2"]   # re-slice so it carries the new columns
    METH = [("Map2-naive hybrid", PS.ORANGE, "o", lambda n: st[(st.n_agents == n) & (st.dataset == "original")]),
            ("Map2-adapted hybrid", PS.BLUE, "s", lambda n: st[(st.n_agents == n) & (st.dataset == "alternative")]),
            ("Online RRT", PS.BLACK, "^", lambda n: rrt[(rrt["map"] == "map2") & (rrt.n_agents == n)])]
    METH_LS = ["-", "-", "--"]   # online dashed (redundant with color/marker for accessibility)
    NS = sorted(st.n_agents.unique())

    # ---- Fig 7: planning-cost / efficiency trade-off (runtime vs collision rate) ----
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for (label, col, mk, sel), ls in zip(METH, METH_LS):
        xs, ys, xe, ylo, yhi = [], [], [], [], []
        for n in NS:
            g = sel(n)
            if len(g):
                xs.append(g.wall_s.mean()); xe.append(g.wall_s.std(ddof=1))
                ys.append(g.coll_per_task.mean())
                lo, hi = bootstrap_ci(g.coll_per_task); ylo.append(ys[-1] - lo); yhi.append(hi - ys[-1])
        ax.errorbar(xs, ys, xerr=xe, yerr=[ylo, yhi], fmt=mk + ls, color=col, label=label,
                    capsize=3, elinewidth=1.1, alpha=0.9)
        for n, x, y in zip(NS, xs, ys):
            ax.annotate(f"$N{{=}}{n}$", (x, y), fontsize=9, xytext=(5, 4),
                        textcoords="offset points", color=col)
    ax.set_xlabel("runtime / episode (s)")
    ax.set_ylabel("collisions per completed task")
    # Title gets generous top padding and stands alone; the points/error-bar legend that used to
    # collide with it has moved into the LaTeX caption, and the lower-left blue annotation is
    # removed, leaving only the clean method legend inside the axes (figure-review fix).
    ax.set_title("Planning-cost vs collision-rate trade-off (Map2)", pad=14)
    ax.legend(loc="upper right")
    fig.tight_layout(); PS.save(fig, os.path.join(FIG, "fig_planning_cost"))

    # ---- Fig 6: collision-type composition (static-obstacle vs robot-robot) ----
    import matplotlib.patches as mpatches
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9))
    short = ["Naive", "Adapted", "Online"]
    for ax, n in zip(axes[:2], (4, 10)):
        obst, robot, tot, lo_e, hi_e = [], [], [], [], []
        for label, col, mk, sel in METH:
            g = sel(n)
            obst.append(g.obst_per_task.mean()); robot.append(g.robot_per_task.mean())
            t = g.coll_per_task.mean(); lo, hi = bootstrap_ci(g.coll_per_task)
            tot.append(t); lo_e.append(t - lo); hi_e.append(hi - t)
        x = np.arange(3)
        ax.bar(x, obst, color=PS.C_STATIC["color"], hatch="//", edgecolor="white", linewidth=0.5)
        ax.bar(x, robot, bottom=obst, color=PS.C_ROBOT["color"])
        ax.errorbar(x, tot, yerr=[lo_e, hi_e], fmt="none", ecolor="0.2", capsize=3, elinewidth=1.1)
        ax.set_xticks(x); ax.set_xticklabels(short)
        ax.set_title(f"$N={n}$"); ax.grid(axis="y", alpha=.25)
        for xi, t in zip(x, tot):
            ax.text(xi, t + max(tot) * 0.05, f"{t:.2f}", ha="center", fontsize=8.5)
        ax.set_ylim(0, max(np.array(tot) + np.array(hi_e)) * 1.2)
        ax.set_ylabel("collisions per completed task")
    # Panel C: naive-library component vs N, showing robot--robot overtakes static near N=8
    naive = lambda n: st[(st.n_agents == n) & (st.dataset == "original")]
    s_st = [naive(n).obst_per_task.mean() for n in NS]
    s_rb = [naive(n).robot_per_task.mean() for n in NS]
    axes[2].plot(NS, s_st, color=PS.C_STATIC["color"], marker="o", ls="-")
    axes[2].plot(NS, s_rb, color=PS.C_ROBOT["color"], marker="s", ls="--")
    axes[2].axvline(8, color="0.6", ls=":", lw=1.2)
    axes[2].annotate("robot--robot\novertakes near $N{=}8$", xy=(8, max(s_rb) * 0.45),
                     fontsize=10, color="0.2", ha="center")
    PS.fleet_xaxis(axes[2]); axes[2].set_xlabel("number of robots ($N$)")
    axes[2].set_ylabel("collisions per completed task")
    axes[2].set_title("Map2-naive: components vs $N$"); axes[2].grid(alpha=.25)
    for a, L in zip(axes, "ABC"):
        PS.panel(a, L)
    leg = [mpatches.Patch(facecolor=PS.C_STATIC["color"], hatch="//", edgecolor="white", label="static-obstacle"),
           mpatches.Patch(facecolor=PS.C_ROBOT["color"], label="robot--robot")]
    fig.legend(handles=leg, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.94]); PS.save(fig, os.path.join(FIG, "fig_collision_types"))

    # ---- Planning-cost table (exact runtime / RRT-call numbers behind Fig 5) ----
    _cpu, _py = _machine()
    pl = [r"% AUTO-GENERATED by analyze_rrtbc.py -- do not edit by hand.",
          r"\begin{table}[H]",
          r"\caption{Planning cost on Map2 (15 seeds, 2000 steps). Runtime is mean\,$\pm$\,SD "
          r"wall-clock for the navigation loop on one pinned CPU thread (" + _cpu +
          r", Python " + _py + r"), excluding scenario construction, figure rendering and disk "
          r"I/O; the RRT-calls-per-task column is the number of online RRT planner invocations "
          r"per completed task. Online RRT attains the lowest collision rate but at far higher "
          r"planning cost; the adapted hybrid reuses routes and calls the planner rarely.}",
          r"\label{tab:planning_cost}",
          r"\centering", r"\small", r"\begin{tabular}{c l c c c c}", r"\toprule",
          r"\boldmath$N$ & \textbf{Method} & \textbf{Runtime (s)} & \textbf{RRT calls/task} & "
          r"\textbf{Coll./task} & \textbf{Fail rate} \\", r"\midrule"]
    for n in NS:
        for k, (label, col, mk, sel) in enumerate(METH):
            g = sel(n)
            if not len(g):
                continue
            ncell = f"\\multirow{{3}}{{*}}{{{n}}}" if k == 0 else ""
            pl.append(f"{ncell} & {label} & ${g.wall_s.mean():.1f}\\pm{g.wall_s.std(ddof=1):.1f}$ & "
                      f"{g.rrt_per_task.mean():.2f} & "
                      f"{g.coll_per_task.mean():.2f} & {g.fail_rate.mean():.2f} \\\\")
        pl.append(r"\midrule" if n != NS[-1] else r"\bottomrule")
    pl += [r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TAB, "tab_planning_cost.tex"), "w") as f:
        f.write("\n".join(pl) + "\n")

    # ---- Collision-composition table at ALL fleet sizes (appendix; supports Fig 7's
    #      "at every fleet size" claim with the full breakdown, not just N=4 and N=10) ----
    cc = [r"% AUTO-GENERATED by analyze_rrtbc.py -- do not edit by hand.",
          r"\begin{table}[H]",
          r"\caption{Collision composition on Map2 at all fleet sizes (15 seeds, 2000 steps): "
          r"static-obstacle and robot--robot collisions per completed task, and the "
          r"static-obstacle fraction. The static-obstacle component is exclusive to the "
          r"Map2-naive library and roughly constant across $N$ (it dominates at small fleets "
          r"and is overtaken by robot--robot congestion at the largest fleets); the adapted "
          r"hybrid and online RRT incur no obstacle collisions at any $N$. Full-fleet "
          r"complement to Figure~\ref{fig:coll_types}.}",
          r"\label{tab:coll_comp}",
          r"\centering", r"\footnotesize", r"\begin{tabular}{c l c c c}", r"\toprule",
          r"\boldmath$N$ & \textbf{Method} & \textbf{Static-obstacle/task} & "
          r"\textbf{Robot--robot/task} & \textbf{Static fraction} \\", r"\midrule"]
    for n in NS:
        for k, (label, col, mk, sel) in enumerate(METH):
            g = sel(n)
            if not len(g):
                continue
            ob = g.obst_per_task.mean(); rb = g.robot_per_task.mean()
            frac = ob / (ob + rb) if (ob + rb) > 0 else 0.0
            ncell = f"\\multirow{{3}}{{*}}{{{n}}}" if k == 0 else ""
            cc.append(f"{ncell} & {label} & {ob:.2f} & {rb:.2f} & {100*frac:.0f}\\% \\\\")
        cc.append(r"\midrule" if n != NS[-1] else r"\bottomrule")
    cc += [r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TAB, "tab_coll_composition.tex"), "w") as f:
        f.write("\n".join(cc) + "\n")
    print("wrote tables/tab_planning_cost.tex + tab_coll_composition.tex "
          "+ fig_planning_cost.png + fig_collision_types.png")
else:
    print("[skip] instrumented columns (wall_s/coll_obstacle/rrt_calls) not yet in CSVs "
          "-> Fig 5/7 and planning-cost table will generate after the instrumented re-run.")

# ============ Fig 8: throughput-dependence of collision-history sharing ============
thr_path = os.path.join(HERE, "results_throughput.csv")
if os.path.exists(thr_path):
    from sim_rrtbc import run_episode_rrtbc   # for Panel A live H-population history
    thr = pd.read_csv(thr_path)
    thr["coll_per_task"] = thr["collisions"] / thr["tasks_completed"].clip(lower=1)
    horizons = sorted(thr.steps.unique())
    # Taller panels (+~14%) so panel A's note, panel B's legend and the panel labels sit
    # comfortably above the print-size lower limit (figure-review fix).
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 4.9))

    # Panel A: shared-map population vs completed tasks (single representative run, seed 0)
    hist = run_episode_rrtbc("map1", "original", n_agents=5, steps=max(horizons),
                             seed=0, share_fraction=1.0, return_history=True)["history"]
    if hist:
        ht = np.array(hist)
        ax[0].plot(ht[:, 0], ht[:, 1], "-", color=PS.BLUE, lw=2.0)
        ax[0].set_xlabel("completed tasks"); ax[0].set_ylabel("populated cells in $H$")
        ax[0].set_title("$H$ population"); ax[0].grid(alpha=.25)
        ax[0].text(0.05, 0.96, "representative run (seed 0)", transform=ax[0].transAxes,
                   va="top", fontsize=10, color="0.4")

    # Panel B: collision rate vs episode length for the three sharing levels (bootstrap 95% CI)
    for p, s in [(0.0, PS.S_NONE), (0.6, PS.S_SEL), (1.0, PS.S_FULL)]:
        stps = sorted(thr[thr.share_fraction == p].steps.unique())
        m, elo, ehi = [], [], []
        for stp in stps:
            gc = thr[(thr.share_fraction == p) & (thr.steps == stp)].coll_per_task
            mm = gc.mean(); lo, hi = bootstrap_ci(gc); m.append(mm); elo.append(mm - lo); ehi.append(hi - mm)
        plain = {0.0: "none (0)", 0.6: "selective (0.6)", 1.0: "full (1.0)"}[p]
        ax[1].errorbar(stps, m, yerr=[elo, ehi], marker=s["marker"], ls=s["ls"], capsize=3,
                       color=s["color"], label=plain)
    ax[1].set_xlabel("episode length (steps)"); ax[1].set_ylabel("collisions per completed task")
    ax[1].set_title("Collision rate"); ax[1].grid(alpha=.25); ax[1].set_xticks(horizons)
    ax[1].legend(title="$q_{\\mathrm{share}}$", fontsize=10, title_fontsize=10)

    # Panel C: full-sharing collision reduction (%) vs horizon, with bootstrap 95% CI (gray band)
    xs, ys, los, his = [], [], [], []
    for steps in horizons:
        a = thr[(thr.share_fraction == 0.0) & (thr.steps == steps)].collisions.values
        b = thr[(thr.share_fraction == 1.0) & (thr.steps == steps)].collisions.values
        if not (len(a) and len(b)):
            continue
        rng = np.random.default_rng(0)
        reds = [100 * (rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()) / rng.choice(a, len(a)).mean()
                for _ in range(4000)]
        xs.append(steps); ys.append(100 * (a.mean() - b.mean()) / a.mean())
        los.append(np.percentile(reds, 2.5)); his.append(np.percentile(reds, 97.5))
    xs = np.array(xs); ys = np.array(ys)
    ax[2].axhline(0, color="k", lw=1.6)
    ax[2].fill_between(xs, los, his, color=PS.GRAY, alpha=0.30, lw=0, label="95% CI")
    ax[2].plot(xs, ys, "o-", color=PS.BLUE, lw=2.0)
    ax[2].set_xticks(horizons)
    ax[2].set_xlabel("episode length (steps)")
    ax[2].set_ylabel("full-sharing collision reduction (%)")
    ax[2].set_title("Full-sharing reduction"); ax[2].grid(alpha=.25); ax[2].legend(loc="lower right", fontsize=10)
    for a, L in zip(ax, "ABC"):
        PS.panel(a, L)
    fig.tight_layout(rect=[0, 0, 1, 1.0]); PS.save(fig, os.path.join(FIG, "fig_sharing_throughput"))
    print("wrote fig_sharing_throughput.pdf")
else:
    print("[skip] results_throughput.csv not present yet -> Fig 8 generates after the throughput campaign.")

# ================= LaTeX TABLES (regenerated from the same data) =================
def _fmtp(p):
    if p != p:                       # NaN -> baseline row
        return "---"
    if p < 1e-5:
        return r"$<\!10^{-5}$"
    if p < 1e-3:
        return r"$<\!0.001$"
    return f"${p:.3f}$"


# ---- Table 2: full skill-transfer numerical results (3 methods x fleet size) ----
if len(st):
    lines = [
        r"% AUTO-GENERATED by analyze_rrtbc.py -- do not edit by hand.",
        r"\begin{table}[H]",
        r"\caption{Full skill-transfer results on Map2 (15 seeds, 2000 scheduler steps). "
        r"Tasks is the mean number of completed tasks (rounded to an integer); collisions are "
        r"mean\,$\pm$\,SD; collisions per completed task are reported with a percentile "
        r"bootstrap 95\% CI. The Reduction column is the percentage reduction in mean "
        r"collisions relative to the Map2-naive hybrid; the Mann--Whitney $U$ $p$-value "
        r"(one-sided, naive\,$>$\,method) and Cliff's $\delta$ are computed against that same "
        r"baseline. The Map2-adapted hybrid and the online RRT baseline both improve on the "
        r"naive library at every fleet size, with $\delta=1$ and $p<10^{-5}$ throughout.}",
        r"\label{tab:skill_transfer}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{c l c c c c c c c}",
        r"\toprule",
        r"\boldmath$N$ & \textbf{Method} & \textbf{Tasks} & \textbf{Collisions} & "
        r"\textbf{Coll./task [95\% CI]} & \textbf{Fail rate} & \textbf{Reduction} & "
        r"\boldmath$p$ & \boldmath$\delta$ \\",
        r"\midrule",
    ]
    method_rows = [("Map2-naive hybrid", "original", "rrtbc"),
                   ("Map2-adapted hybrid", "alternative", "rrtbc"),
                   ("Online RRT", None, "rrt")]
    for n in sorted(st.n_agents.unique()):
        o = st[(st.n_agents == n) & (st.dataset == "original")]
        base = o.collisions.mean()
        for label, ds, kind in method_rows:
            if kind == "rrtbc":
                g = st[(st.n_agents == n) & (st.dataset == ds)]
            else:
                g = rrt[(rrt["map"] == "map2") & (rrt.n_agents == n)] if rrt is not None else o.iloc[0:0]
            if not len(g):
                continue
            lo, hi = bootstrap_ci(g.coll_per_task)
            if ds == "original":
                red = pstr = dstr = "---"
            else:
                red = f"{100*(base-g.collisions.mean())/base:.0f}\\%"
                pstr = _fmtp(mannwhitneyu(o.collisions, g.collisions, alternative="greater").pvalue)
                dstr = f"${cliffs_delta(o.collisions, g.collisions):.2f}$"
            ncell = f"\\multirow{{3}}{{*}}{{{n}}}" if label.startswith("Map2-naive") else ""
            lines.append(
                f"{ncell} & {label} & {g.tasks_completed.mean():.0f} & "
                f"${g.collisions.mean():.0f}\\pm{g.collisions.std(ddof=1):.0f}$ & "
                f"{g.coll_per_task.mean():.2f} [{lo:.2f}, {hi:.2f}] & "
                f"{g.fail_rate.mean():.2f} & {red} & {pstr} & {dstr} \\\\")
        lines.append(r"\midrule" if n != sorted(st.n_agents.unique())[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TAB, "tab_skill_transfer.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tables/tab_skill_transfer.tex")


# ---- Table 4: collision-history sharing statistical summary (Map1, Map2) ----
# Pre-compute the four directional ablation p-values (Map1/Map2 x selective/full) and Holm-
# correct across them (reviewer item 9): the modest Map1 full-sharing effect must be reported
# both nominally and after multiple-comparison correction. Map1 full is the primary test.
_ab_tests = []   # (map, fraction, nominal p)
for mp in ("map1", "map2"):
    ab = df[df.exp == f"S1_ablation_{mp}"]
    if not len(ab):
        continue
    none = ab[ab.share_fraction == 0.0]
    for p in (0.6, 1.0):
        g = ab[ab.share_fraction == p]
        if len(g) and len(none):
            _ab_tests.append((mp, p, mannwhitneyu(none.collisions, g.collisions, alternative="greater").pvalue))
_holm_adj = holm([t[2] for t in _ab_tests]) if _ab_tests else []
_nom = {(mp, p): pv for (mp, p, pv) in _ab_tests}
_holm = {(mp, p): hp for (mp, p, _), hp in zip(_ab_tests, _holm_adj)}
out.append("\n## Sharing multiple-comparison correction (Holm across the 4 ablation tests)")
for (mp, p, pv), hp in zip(_ab_tests, _holm_adj):
    out.append(f"- {mp} share={p}: nominal p={pv:.3g}, Holm p={hp:.3g}, "
               f"{'sig' if hp < 0.05 else 'n.s.'} after correction.")

sl = [
    r"% AUTO-GENERATED by analyze_rrtbc.py -- do not edit by hand.",
    r"\begin{table}[H]",
    r"\caption{Collision-history sharing on the native route library of each map (five "
    r"robots, 15 seeds, 2000 steps). Selective sharing preserves throughput and produces "
    r"small changes in collisions. The Change column and the significance test refer to the "
    r"mean per-episode \emph{collision count} (the quantity tested); Coll./task is the "
    r"corresponding workload-normalized rate, which changes by a similar amount. $p$ is the "
    r"one-sided Mann--Whitney $U$ test against no sharing; $p_{\mathrm{Holm}}$ applies a "
    r"Holm--Bonferroni correction across the four ablation comparisons, and the Significant "
    r"column is judged after correction. Only Map1 full participation ($\qshare=1$) is "
    r"nominally significant, and it does not survive correction---so the broader sharing "
    r"analysis supports only a limited, layout-dependent effect.}",
    r"\label{tab:sharing_stats}",
    r"\centering",
    r"\small",
    r"\begin{tabular}{l c c c c c c c c}",
    r"\toprule",
    r"\textbf{Map} & \boldmath$\qshare$ & \textbf{Collisions} & \textbf{Coll./task} & "
    r"\textbf{Tasks} & \textbf{Change} & \boldmath$p$ & \boldmath$p_{\mathrm{Holm}}$ & "
    r"\textbf{Significant} \\",
    r"\midrule",
]
for mp in ("map1", "map2"):
    ab = df[df.exp == f"S1_ablation_{mp}"]
    if not len(ab):
        continue
    none = ab[ab.share_fraction == 0.0]
    for k, p in enumerate((0.0, 0.6, 1.0)):
        g = ab[ab.share_fraction == p]
        chg = 100 * (none.collisions.mean() - g.collisions.mean()) / none.collisions.mean()
        if p == 0.0:
            chgs = pv = pvh = sig = "---"
        else:
            chgs = f"$-{chg:.1f}\\%$" if chg >= 0 else f"$+{-chg:.1f}\\%$"
            pv = _fmtp(_nom[(mp, p)]); pvh = _fmtp(_holm[(mp, p)])
            sig = "yes" if _holm[(mp, p)] < 0.05 else "no"
        mapcell = f"\\multirow{{3}}{{*}}{{{'Map1' if mp=='map1' else 'Map2'}}}" if k == 0 else ""
        sl.append(f"{mapcell} & {p:.1f} & ${g.collisions.mean():.0f}\\pm{g.collisions.std(ddof=1):.0f}$ & "
                  f"{g.coll_per_task.mean():.2f} & {g.tasks_completed.mean():.0f} & {chgs} & {pv} & {pvh} & {sig} \\\\")
    sl.append(r"\midrule" if mp == "map1" else r"\bottomrule")
sl += [r"\end{tabular}", r"\end{table}"]
with open(os.path.join(TAB, "tab_sharing_stats.tex"), "w") as f:
    f.write("\n".join(sl) + "\n")
print("wrote tables/tab_sharing_stats.tex")

with open(os.path.join(HERE, "stats_summary_rrtbc.md"), "w") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
print("\nFigures ->", FIG)
print("Tables  ->", TAB)
