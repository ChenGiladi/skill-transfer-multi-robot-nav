#!/usr/bin/env python3
"""
Analysis for the Revision-1 experiment campaign (reviewer-requested baselines).
Reads results_revision.csv (R1 library variants, R2 scripted connector, R3 prioritized
planning, R4 online-RRT compute-budget caps, R5 Map3 transfer), plus the reference
conditions from results_rrtbc.csv (T0 hybrid on Map2) and results_rrt_online.csv
(online RRT at the paper cap of 3000). Writes tab_lib_baselines.tex,
tab_map3_transfer.tex, fig_budget_frontier and stats_summary_revision.md.
"""
import os, sys, argparse
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle as PS                      # shared publication figure style
PS.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(FIG, exist_ok=True)
TAB = os.path.join(os.path.dirname(HERE), "tables")
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


def perm_test(a, b, n=10000, seed=0):
    """One-sided permutation p-value for mean(a) > mean(b)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b]); rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        rng.shuffle(pooled)
        if pooled[:len(a)].mean() - pooled[len(a):].mean() >= obs:
            cnt += 1
    return (cnt + 1) / (n + 1)


def _fmtp(p):
    if p != p:                       # NaN -> baseline row
        return "---"
    if p < 1e-5:
        return r"$<\!10^{-5}$"
    if p < 1e-3:
        return r"$<\!0.001$"
    return f"${p:.3f}$"


def _mwu_greater(a, b):
    """One-sided Mann-Whitney U p (a > b); NaN when the test is undefined."""
    try:
        return mannwhitneyu(a, b, alternative="greater").pvalue
    except ValueError:
        return float("nan")


def ms(x):
    return f"{np.mean(x):.2f} ± {np.std(x, ddof=1):.2f}"


def _derive(d):
    d["coll_per_task"] = d["collisions"] / d["tasks_completed"].clip(lower=1)
    d["fail_rate"] = d["fails"] / (d["tasks_completed"] + d["fails"]).clip(lower=1)
    d["rrt_per_task"] = d["rrt_calls"] / d["tasks_completed"].clip(lower=1)
    return d


# ---------------- inputs (each optional; missing input -> clear skip, never crash) ----
ap = argparse.ArgumentParser()
ap.add_argument("--input", default=os.path.join(HERE, "results_revision.csv"),
                help="revision campaign CSV (default: results_revision.csv)")
args = ap.parse_args()


def _load(path, label):
    if not os.path.exists(path):
        print(f"[skip] {label} not found at {path} -> outputs that need it are skipped.")
        return None
    return _derive(pd.read_csv(path))


rev = _load(args.input, "results_revision.csv (revision campaign)")
ref = _load(os.path.join(HERE, "results_rrtbc.csv"), "results_rrtbc.csv (T0 reference hybrids)")
if ref is not None:
    ref = ref[ref.exp == "T0_skill_transfer_map2"]
rrt = _load(os.path.join(HERE, "results_rrt_online.csv"), "results_rrt_online.csv (online RRT, paper cap 3000)")
if rrt is not None:
    rrt = rrt[rrt["map"] == "map2"]


def _rev(exp, method, n):
    if rev is None:
        return None
    return rev[(rev.exp == exp) & (rev.method == method) & (rev.n_agents == n)]


def _ref(dataset, n):
    if ref is None:
        return None
    return ref[(ref.dataset == dataset) & (ref.n_agents == n)]


def _seeds(d):
    """Max seeds per condition in a dataframe (for honest caption seed counts)."""
    if d is None or not len(d):
        return 0
    keys = [k for k in ("exp", "method", "dataset", "n_agents") if k in d.columns]
    return int(d.groupby(keys).seed.nunique().max())


NS = sorted(set(
    (list(rev.n_agents.unique()) if rev is not None else []) +
    (list(ref.n_agents.unique()) if ref is not None else [])))

S_REV, S_REF = _seeds(rev), _seeds(ref)
_seed_note = "" if (S_REF in (0, S_REV)) else f"; the reference hybrid and cap-3000 rows use {S_REF} seeds"

out = ["# Revision-1 campaign — statistics\n"]
out.append(f"Source: {os.path.basename(args.input)} "
           f"({0 if rev is None else len(rev)} episodes, {S_REV} seeds per condition), "
           f"reference conditions from results_rrtbc.csv (T0, {S_REF} seeds) and "
           f"results_rrt_online.csv (paper cap 3000). All episodes: 2000 scheduler steps.\n")

# =================== Table: library-level baselines on Map2 ===================
# Order matters: naive baseline first, then the offline library transforms, then the
# adapted hybrid, the scripted connector, prioritized planning and the online-RRT caps.
T1_METHODS = [
    ("Map2-naive hybrid",       True,  lambda n: _ref("original", n)),
    (r"\;+ Filtered library",   False, lambda n: _rev("R1_libvariants_map2", "lib_filter", n)),
    (r"\;+ Repaired library",   False, lambda n: _rev("R1_libvariants_map2", "lib_repair", n)),
    (r"\;+ Regenerated library", False, lambda n: _rev("R1_libvariants_map2", "lib_regen", n)),
    ("Map2-adapted hybrid",     False, lambda n: _ref("alternative", n)),
    ("Scripted connector (adapted)", False, lambda n: _rev("R2_scripted_map2", "scripted_alternative", n)),
    ("Prioritized planning",    False, lambda n: _rev("R3_prioritized_map2", "prioritized", n)),
    ("Online RRT (cap 3000)",   False, lambda n: (rrt[rrt.n_agents == n] if rrt is not None else None)),
    ("Online RRT (cap 1000)",   False, lambda n: _rev("R4_online_cap1000_map2", "online_cap1000", n)),
    ("Online RRT (cap 300)",    False, lambda n: _rev("R4_online_cap300_map2", "online_cap300", n)),
]

if rev is not None and ref is not None and len(ref):
    lines = [
        r"% AUTO-GENERATED by analyze_revision.py -- do not edit by hand.",
        r"\begin{table}[H]",
        r"\caption{Library-level baselines on Map2 (" + f"{S_REV} seeds, 2000 steps{_seed_note}" + r"). "
        r"Tasks is the mean number of completed tasks; collisions are mean\,$\pm$\,SD per "
        r"episode; collisions per completed task carry a percentile bootstrap 95\% CI. "
        r"Reduction, $p$ and $p_{\mathrm{perm}}$ compare each method against the Map2-naive "
        r"hybrid on mean per-episode collision counts: Reduction is the percentage reduction "
        r"in mean collisions, $p$ is a one-sided Mann--Whitney $U$ test "
        r"(naive\,$>$\,method) and $p_{\mathrm{perm}}$ is a one-sided permutation test "
        r"($10^{4}$ resamples) in the same direction. The filtered, repaired and regenerated "
        r"rows apply offline transforms to the naive library before deployment; the scripted "
        r"row replaces the trained BC connector with a greedy non-learning controller on the "
        r"adapted library; prioritized planning coordinates the fleet through a shared "
        r"space-time reservation table, so unlike the hybrid it is not communication-free; "
        r"the online RRT rows replan continuously under the stated iteration cap.}",
        r"\label{tab:lib_baselines}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{c l c c c c c c c}",
        r"\toprule",
        r"\boldmath$N$ & \textbf{Method} & \textbf{Tasks} & \textbf{Collisions} & "
        r"\textbf{Coll./task [95\% CI]} & \textbf{Fail rate} & \textbf{Reduction} & "
        r"\boldmath$p$ & \boldmath$p_{\mathrm{perm}}$ \\",
        r"\midrule",
    ]
    for n in NS:
        base = _ref("original", n)
        if base is None or not len(base):
            continue
        bcoll = base.collisions.mean()
        rows = []
        for label, is_base, sel in T1_METHODS:
            g = sel(n)
            if g is None or not len(g):
                continue
            lo, hi = bootstrap_ci(g.coll_per_task)
            if is_base:
                red = pstr = ppstr = "---"
            else:
                red = f"{100*(bcoll-g.collisions.mean())/bcoll:.0f}\\%"
                pstr = _fmtp(_mwu_greater(base.collisions, g.collisions))
                ppstr = _fmtp(perm_test(base.collisions.values, g.collisions.values))
            rows.append(
                f"{label} & {g.tasks_completed.mean():.0f} & "
                f"${g.collisions.mean():.0f}\\pm{g.collisions.std(ddof=1):.0f}$ & "
                f"{g.coll_per_task.mean():.2f} [{lo:.2f}, {hi:.2f}] & "
                f"{g.fail_rate.mean():.2f} & {red} & {pstr} & {ppstr} \\\\")
        for k, row in enumerate(rows):
            ncell = f"\\multirow{{{len(rows)}}}{{*}}{{{n}}}" if k == 0 else ""
            lines.append(f"{ncell} & {row}")
        lines.append(r"\midrule" if n != NS[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TAB, "tab_lib_baselines.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tables/tab_lib_baselines.tex")
else:
    print("[skip] tab_lib_baselines.tex needs both the revision CSV and results_rrtbc.csv.")

# =================== Table: Map3 transfer ===================
if rev is not None and len(rev[rev.exp == "R5_transfer_map3"]):
    r5 = rev[rev.exp == "R5_transfer_map3"]
    ns5 = sorted(r5.n_agents.unique())
    lines = [
        r"% AUTO-GENERATED by analyze_revision.py -- do not edit by hand.",
        r"\begin{table}[H]",
        r"\caption{Transfer on Map3 (corridor/bottleneck layout; " + f"{S_REV} seeds, 2000 steps" + r"). "
        r"The Map3-naive hybrid deploys the Map1-generated route library on Map3 unchanged; "
        r"the Map3-adapted hybrid uses a library generated on Map3 itself. Collisions are "
        r"mean\,$\pm$\,SD per episode and collisions per completed task carry a percentile "
        r"bootstrap 95\% CI. Reduction, $p$, $p_{\mathrm{perm}}$ and Cliff's $\delta$ are "
        r"computed against the Map3-naive hybrid on per-episode collision counts, with $p$ "
        r"from a one-sided Mann--Whitney $U$ test (naive\,$>$\,adapted) and "
        r"$p_{\mathrm{perm}}$ from a one-sided permutation test ($10^{4}$ resamples) in the "
        r"same direction.}",
        r"\label{tab:map3_transfer}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{c l c c c c c c c c}",
        r"\toprule",
        r"\boldmath$N$ & \textbf{Method} & \textbf{Tasks} & \textbf{Collisions} & "
        r"\textbf{Coll./task [95\% CI]} & \textbf{Fail rate} & \textbf{Reduction} & "
        r"\boldmath$p$ & \boldmath$p_{\mathrm{perm}}$ & \boldmath$\delta$ \\",
        r"\midrule",
    ]
    for n in ns5:
        o = r5[(r5.n_agents == n) & (r5.method == "hybrid_original")]
        a = r5[(r5.n_agents == n) & (r5.method == "hybrid_map3lib")]
        rows = []
        for label, g, is_base in [("Map3-naive hybrid", o, True),
                                  ("Map3-adapted hybrid", a, False)]:
            if not len(g):
                continue
            lo, hi = bootstrap_ci(g.coll_per_task)
            if is_base or not len(o):
                red = pstr = ppstr = dstr = "---"
            else:
                red = f"{100*(o.collisions.mean()-g.collisions.mean())/o.collisions.mean():.0f}\\%"
                pstr = _fmtp(_mwu_greater(o.collisions, g.collisions))
                ppstr = _fmtp(perm_test(o.collisions.values, g.collisions.values))
                dstr = f"${cliffs_delta(o.collisions, g.collisions):.2f}$"
            rows.append(
                f"{label} & {g.tasks_completed.mean():.0f} & "
                f"${g.collisions.mean():.0f}\\pm{g.collisions.std(ddof=1):.0f}$ & "
                f"{g.coll_per_task.mean():.2f} [{lo:.2f}, {hi:.2f}] & "
                f"{g.fail_rate.mean():.2f} & {red} & {pstr} & {ppstr} & {dstr} \\\\")
        for k, row in enumerate(rows):
            ncell = f"\\multirow{{{len(rows)}}}{{*}}{{{n}}}" if k == 0 else ""
            lines.append(f"{ncell} & {row}")
        lines.append(r"\midrule" if n != ns5[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TAB, "tab_map3_transfer.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tables/tab_map3_transfer.tex")
else:
    print("[skip] tab_map3_transfer.tex needs R5_transfer_map3 rows in the revision CSV.")

# =================== Figure: compute-budget frontier on Map2 ===================
# PS colors where defined (naive / adapted / online cap 3000); the reduced caps and
# prioritized planning take distinct matplotlib-default (tab:) colors and markers.
FRONTIER = [
    ("Map2-naive hybrid",    lambda n: _ref("original", n),
     dict(color=PS.M_NAIVE["color"], marker=PS.M_NAIVE["marker"], ls=PS.M_NAIVE["ls"])),
    ("Map2-adapted hybrid",  lambda n: _ref("alternative", n),
     dict(color=PS.M_ADAPT["color"], marker=PS.M_ADAPT["marker"], ls=PS.M_ADAPT["ls"])),
    ("online RRT, cap 3000", lambda n: (rrt[rrt.n_agents == n] if rrt is not None else None),
     dict(color=PS.M_ONLINE["color"], marker=PS.M_ONLINE["marker"], ls=PS.M_ONLINE["ls"])),
    ("online RRT, cap 1000", lambda n: _rev("R4_online_cap1000_map2", "online_cap1000", n),
     dict(color="tab:red", marker="v", ls="--")),
    ("online RRT, cap 300",  lambda n: _rev("R4_online_cap300_map2", "online_cap300", n),
     dict(color="tab:purple", marker="<", ls="--")),
    ("prioritized planning", lambda n: _rev("R3_prioritized_map2", "prioritized", n),
     dict(color="tab:green", marker="P", ls="-.")),
]

_have_frontier = any(
    sel(n) is not None and len(sel(n)) for _, sel, _ in FRONTIER for n in NS)
if _have_frontier and NS:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for label, sel, sty in FRONTIER:
        xs, ys, xe, ylo, yhi, fs, ns_used = [], [], [], [], [], [], []
        for n in NS:
            g = sel(n)
            if g is None or not len(g):
                continue
            xs.append(g.wall_s.mean()); xe.append(g.wall_s.std(ddof=1))
            ys.append(g.coll_per_task.mean())
            lo, hi = bootstrap_ci(g.coll_per_task)
            ylo.append(ys[-1] - lo); yhi.append(hi - ys[-1])
            fs.append(g.fail_rate.mean())
            ns_used.append(n)
        if not xs:
            continue
        ax.errorbar(xs, ys, xerr=xe, yerr=[ylo, yhi], fmt=sty["marker"] + sty["ls"],
                    color=sty["color"], label=label, capsize=3, elinewidth=1.1,
                    lw=1.8, ms=6.5, alpha=0.9)
        if 10 in ns_used:
            i = ns_used.index(10)
            ax.annotate(f"$N{{=}}10$; fail {fs[i]:.2f}", (xs[i], ys[i]), fontsize=8.5,
                        xytext=(6, 4), textcoords="offset points", color=sty["color"])
    ax.set_xlabel("runtime / episode (s)")
    ax.set_ylabel("collisions per completed task")
    ax.set_title("Compute budget vs collision rate (Map2)", pad=12)
    ax.legend(loc="best", fontsize=9.5)
    fig.tight_layout()
    base = os.path.join(FIG, "fig_budget_frontier")
    fig.savefig(base + ".png", dpi=300)   # PS.save writes the PDF and closes the figure
    PS.save(fig, base)
    print("wrote figures/fig_budget_frontier.pdf + .png")
else:
    print("[skip] fig_budget_frontier needs at least one frontier method with data.")

# =================== stats_summary_revision.md ===================

# ---------------- R1: library variants ----------------
_r1_ps = []   # (variant, N, nominal MWU p) for the Holm family
if rev is not None and len(rev[rev.exp == "R1_libvariants_map2"]) and ref is not None:
    out.append("\n## R1 library variants on Map2 (offline filter / repair / regen of the naive library)")
    for n in NS:
        base = _ref("original", n)
        if base is None or not len(base):
            continue
        parts = []
        for v in ("filter", "repair", "regen"):
            g = _rev("R1_libvariants_map2", f"lib_{v}", n)
            if g is None or not len(g):
                continue
            red = 100 * (base.collisions.mean() - g.collisions.mean()) / base.collisions.mean()
            p = _mwu_greater(base.collisions, g.collisions)
            pp = perm_test(base.collisions.values, g.collisions.values)
            _r1_ps.append((v, n, p))
            parts.append(f"{v} coll {ms(g.collisions)} ({red:+.0f}% vs naive, p={p:.3g}, p_perm={pp:.3g})")
        out.append(f"- {n} agents: naive coll {ms(base.collisions)}; " + "; ".join(parts) + ".")
    # library sizes (only from existing caches; never trigger the offline transform here)
    sizes = []
    for v in ("filter", "repair", "regen"):
        cache = os.path.join(HERE, "lib_cache", f"map2_original_{v}.pkl")
        if os.path.exists(cache):
            try:
                import pickle
                with open(cache, "rb") as f:
                    sizes.append(f"{v}: {len(pickle.load(f))} routes")
            except Exception as e:
                sizes.append(f"{v}: cache unreadable ({e})")
    if sizes:
        out.append("- Library sizes after transform (from lib_cache): " + "; ".join(sizes) + ".")
    else:
        out.append("- Library sizes: no lib_cache entries found, skipped.")
else:
    out.append("\n## R1 library variants: no data yet (needs revision CSV + results_rrtbc.csv), skipped.")

# ---------------- R2: scripted connector vs trained BC ----------------
if rev is not None and len(rev[rev.exp == "R2_scripted_map2"]) and ref is not None:
    out.append("\n## R2 scripted connector vs trained BC connector (Map2, adapted library)")
    bc_wins = 0; total = 0
    for n in NS:
        sc = _rev("R2_scripted_map2", "scripted_alternative", n)
        bc = _ref("alternative", n)
        if sc is None or bc is None or not len(sc) or not len(bc):
            continue
        p_sb = _mwu_greater(sc.collisions, bc.collisions)   # scripted worse than BC?
        p_bs = _mwu_greater(bc.collisions, sc.collisions)   # BC worse than scripted?
        total += 1; bc_wins += int(sc.collisions.mean() > bc.collisions.mean())
        out.append(f"- {n} agents: scripted coll {ms(sc.collisions)} vs BC coll {ms(bc.collisions)}; "
                   f"MWU scripted>BC p={p_sb:.3g}, BC>scripted p={p_bs:.3g}.")
    sc_o = rev[(rev.exp == "R2_scripted_map2") & (rev.method == "scripted_original")]
    if len(sc_o):
        na = ref[ref.dataset == "original"]
        out.append(f"- Naive-library check: scripted (original) coll {ms(sc_o.collisions)} vs "
                   f"BC naive hybrid {ms(na.collisions)} pooled over N "
                   f"(per-N table rows use the adapted pair above).")
    if total:
        verdict = ("outperforms" if bc_wins == total else
                   ("mostly outperforms" if bc_wins > total / 2 else "does not outperform"))
        out.append(f"- Verdict: the trained BC connector {verdict} the scripted greedy controller "
                   f"on mean collisions at {bc_wins}/{total} fleet sizes.")
else:
    out.append("\n## R2 scripted vs BC: no data yet, skipped.")

# ---------------- R3 + R4: compute-budget frontier numbers ----------------
if _have_frontier:
    out.append("\n## R3+R4 compute-budget frontier (Map2): runtime, coll/task, tasks per wall-clock second")
    for n in (4, 10):
        if n not in NS:
            continue
        out.append(f"- N={n}:")
        for label, sel, _ in FRONTIER:
            g = sel(n)
            if g is None or not len(g):
                continue
            tps = g.tasks_completed.mean() / max(g.wall_s.mean(), 1e-9)
            out.append(f"  - {label}: runtime {g.wall_s.mean():.1f} ± {g.wall_s.std(ddof=1):.1f} s | "
                       f"coll/task {g.coll_per_task.mean():.2f} | tasks/s {tps:.2f}")
else:
    out.append("\n## R3+R4 frontier: no data yet, skipped.")

# ---------------- R5: Map3 transfer ----------------
_r5_ps = []   # (N, nominal MWU p) for the Holm family
if rev is not None and len(rev[rev.exp == "R5_transfer_map3"]):
    out.append("\n## R5 transfer on Map3 (corridor/bottleneck; Map1 library vs Map3 library)")
    r5 = rev[rev.exp == "R5_transfer_map3"]
    for n in sorted(r5.n_agents.unique()):
        o = r5[(r5.n_agents == n) & (r5.method == "hybrid_original")]
        a = r5[(r5.n_agents == n) & (r5.method == "hybrid_map3lib")]
        if not (len(o) and len(a)):
            continue
        red = 100 * (o.collisions.mean() - a.collisions.mean()) / o.collisions.mean()
        p = _mwu_greater(o.collisions, a.collisions)
        pp = perm_test(o.collisions.values, a.collisions.values)
        _r5_ps.append((n, p))
        out.append(f"- {n} agents: naive coll {ms(o.collisions)} (per-task {o.coll_per_task.mean():.2f}); "
                   f"adapted coll {ms(a.collisions)} (per-task {a.coll_per_task.mean():.2f}); "
                   f"adaptation cuts collisions {red:.0f}% (MWU naive>adapted p={p:.3g}, p_perm={pp:.3g}).")
else:
    out.append("\n## R5 Map3 transfer: no data yet, skipped.")

# ---------------- Holm within families ----------------
out.append("\n## Holm within families")
if _r1_ps:
    adj = holm([p for _, _, p in _r1_ps])
    surv = sum(1 for h in adj if h < 0.05)
    out.append(f"- R1 family ({len(_r1_ps)} tests: 3 variants x fleet sizes): "
               f"{surv}/{len(_r1_ps)} survive Holm at 0.05.")
    for (v, n, p), h in zip(_r1_ps, adj):
        out.append(f"  - {v} N={n}: nominal p={p:.3g}, Holm p={h:.3g}, "
                   f"{'sig' if h < 0.05 else 'n.s.'} after correction.")
else:
    out.append("- R1 family: no tests available, skipped.")
if _r5_ps:
    adj = holm([p for _, p in _r5_ps])
    surv = sum(1 for h in adj if h < 0.05)
    out.append(f"- R5 family ({len(_r5_ps)} tests, one per fleet size): "
               f"{surv}/{len(_r5_ps)} survive Holm at 0.05.")
    for (n, p), h in zip(_r5_ps, adj):
        out.append(f"  - N={n}: nominal p={p:.3g}, Holm p={h:.3g}, "
                   f"{'sig' if h < 0.05 else 'n.s.'} after correction.")
else:
    out.append("- R5 family: no tests available, skipped.")

with open(os.path.join(HERE, "stats_summary_revision.md"), "w") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
print("\nFigures ->", FIG)
print("Tables  ->", TAB)
