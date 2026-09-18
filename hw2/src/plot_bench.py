#!/usr/bin/env python3
"""Plot the sweeps produced by poisson.sbatch / bench_poisson_mpi.

    python3 plot_bench.py ../plots

Reads the four result files produced in this study:

    bench_1node.txt   1 node,  unpinned, P <= 128   (sizes 32..8192)
    bench_2node.txt   2 nodes, unpinned, P <= 256   (P/2 ranks per node)
    bench_k15.txt     1 node,  pinned,   P <= 64    (Nx = 32768 only)
    bench_sweep.txt   1 node,  pinned,   P <= 64    (the original sweep)

Figures:
  1. error vs nx              -- all sizes incl. 32768, with an O(h^2) guide
  2. time vs P               -- one panel per grid size, 1-node vs 2-node
  3. GFLOPS vs P             -- same layout, shared scale
  4. efficiency vs work/rank -- one panel per configuration

ENCODING.  Colour is the transpose strategy (two validated categorical
slots); the node count rides on line style AND marker shape, so the two
configurations stay distinguishable without spending more hues and
without relying on either channel alone.

The 1-node/2-node comparison panels show only the two UNPINNED runs:
no binding policy works across nodes on this cluster, so the multi-node
runs had to be unpinned, and the matching single-node sweep was re-run
unpinned to keep the comparison like-for-like.  The 32768 panel carries
the pinned single-node run, which is the only data at that size; it is
labelled as such rather than silently mixed in.
"""
import sys, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
INK_2   = "#52514e"
GRID    = "#e3e2de"
GUIDE   = "#9a9892"
SERIES  = {1: "#2a78d6", 2: "#eb6834"}     # validated categorical slots 1, 2
SNAME   = {1: "dealing-staggered", 2: "crystal-router"}
SHORT   = {1: "dealing", 2: "crystal"}

# configuration -> (line style, marker, human label)
CFG = {
    "1n":  ((0, ()),      "o", "1 node"),
    "2n":  ((0, (5, 2)),  "s", "2 nodes (P/2 each)"),
    "pin": ((0, ()),      "o", "1 node, pinned"),
}
MARKER_BY_N = {32: "o", 128: "s", 512: "^", 2048: "D", 8192: "v", 32768: "P"}
SUP2, MINUS, TIMES, DIV, APPROX = "²", "−", "×", "÷", "≈"
MDASH, CDOT, SUB2 = "—", "·", "₂"


def load(path, cfg):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path, errors="replace"):
        if "BENCH" not in line or "SKIPPED" in line:
            continue
        f = line.split()
        try:
            N, P, s = int(f[1]), int(f[3]), int(f[4])
        except (ValueError, IndexError):
            continue
        kv, toks = {}, f[6:]
        for i in range(0, len(toks) - 1, 2):
            try:
                kv[toks[i]] = float(toks[i + 1])
            except ValueError:
                pass
        rows.append(dict(N=N, P=P, s=s, cfg=cfg, **kv))
    return rows


def style(ax, xlabel, ylabel, title=None, sub=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=(16 if sub else 8))
    if sub:
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, color=INK_2, fontsize=8.5)


def newfig(*a, **kw):
    fig, ax = plt.subplots(*a, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def series(rows, N, cfg, s, key):
    pts = sorted([(r["P"], r[key]) for r in rows
                  if r["N"] == N and r["cfg"] == cfg and r["s"] == s])
    return (tuple(zip(*pts)) if pts else ((), ()))


# ----------------------------------------------------------------- fig 1
def fig_error(rows, out):
    """Error is bit-identical across P, strategy AND configuration."""
    sizes = sorted({r["N"] for r in rows})
    errs = []
    for N in sizes:
        e = sorted({round(r["err"], 18) for r in rows if r["N"] == N})
        assert len(e) == 1, "error varies within Nx=%d: %s" % (N, e)
        errs.append(e[0])
    nx = [N - 1 for N in sizes]
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]

    fig, ax = newfig(figsize=(8.2, 5.8))
    guide = [0.06 * errs[0] * (sizes[0] / N) ** 2 for N in sizes]
    ax.plot(nx, guide, color=GUIDE, linewidth=1.5, linestyle=(0, (5, 4)), zorder=2)
    ax.plot(nx, errs, color=SERIES[1], linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(nx)
    ax.set_xticklabels([str(v) for v in nx])
    ax.set_ylim(min(guide) * 0.3, max(errs) * 8)
    style(ax, "nx  (= ny, interior unknowns per direction)",
          "max |U " + MINUS + " u(x,y)|   (global)",
          "Discretisation error converges at O(h" + SUP2 + ")")
    # The band between the two lines is crowded with value and ratio
    # labels, so identify the lines from a legend in the empty
    # upper-right corner rather than with floating text.
    leg = ax.legend(handles=[
        Line2D([], [], color=SERIES[1], linewidth=2, marker="o", markersize=8,
               markeredgecolor=SURFACE, label="measured error"),
        Line2D([], [], color=GUIDE, linewidth=1.5, linestyle=(0, (5, 4)),
               label="slope " + MINUS + "2   (exact O(h" + SUP2 + "))")],
        loc="upper right", frameon=False, fontsize=10, labelcolor=INK)
    for i, (x, y) in enumerate(zip(nx, errs)):
        off, ha, va = (((11, 13), "left", "bottom") if i == 0
                       else ((-11, -7), "right", "top"))
        ax.annotate("%.2e" % y, (x, y), textcoords="offset points",
                    xytext=off, ha=ha, va=va, color=INK_2, fontsize=8.5)
    for i in range(len(nx) - 1):
        xm = (nx[i] * nx[i + 1]) ** 0.5
        ym = (errs[i] * errs[i + 1]) ** 0.5
        ax.annotate(DIV + "%.1f" % ratios[i], (xm, ym),
                    textcoords="offset points", xytext=(12, 7), ha="left",
                    color=INK_2, fontsize=8.5)
    fig.text(0.012, 0.015,
             "Identical for every P, both routers and every node count " + MDASH +
             " the parallel solve reproduces the serial result to the last digit.\n"
             "Each 4" + TIMES + " refinement divides the error by " + APPROX +
             "16, as second order requires, out to nx = 32767.",
             color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


# ------------------------------------------------------------- figs 2, 3
def panels(rows, out, key, ylabel, title, subtitle, shared, ideal=False):
    sizes = sorted({r["N"] for r in rows})
    fig, axes = newfig(2, 3, figsize=(14.2, 8.2))
    axes = axes.ravel()
    vals = [r[key] for r in rows]
    ylo, yhi = min(vals) * 0.6, max(vals) * 1.8

    for k, N in enumerate(sizes):
        ax = axes[k]
        have = [c for c in ("1n", "2n", "pin")
                if any(r["N"] == N and r["cfg"] == c for r in rows)]
        # Show the like-for-like unpinned pair; fall back to pinned only
        # where that is the only run at this size.
        cfgs = [c for c in have if c != "pin"] or ["pin"]
        for cfg in cfgs:
            ls, mk, _ = CFG[cfg]
            for s in (1, 2):
                xs, ys = series(rows, N, cfg, s, key)
                if not xs:
                    continue
                ax.plot(xs, ys, color=SERIES[s], linewidth=2, linestyle=ls,
                        marker=mk, markersize=6.5, markeredgecolor=SURFACE,
                        markeredgewidth=1.1, zorder=3)
        if ideal:
            ref = [r[key] for r in rows
                   if r["N"] == N and r["P"] == 1 and r["s"] == 1
                   and r["cfg"] == cfgs[0]]
            if ref:
                Ps = sorted({r["P"] for r in rows if r["N"] == N})
                ax.plot(Ps, [ref[0] / p for p in Ps], color=GUIDE,
                        linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        Ps = sorted({r["P"] for r in rows if r["N"] == N})
        ax.set_xticks(Ps)
        ax.set_xticklabels([str(p) for p in Ps], fontsize=8)
        if shared:
            ax.set_ylim(ylo, yhi)
        sub = "1 node, pinned (only data at this size)" if cfgs == ["pin"] else None
        style(ax, "P (MPI ranks)" if k >= 3 else "",
              ylabel if k % 3 == 0 else "", "Nx = Ny = %d" % N, sub)

    handles = []
    for cfg in ("1n", "2n"):
        ls, mk, lab = CFG[cfg]
        for s in (1, 2):
            handles.append(Line2D([], [], color=SERIES[s], linewidth=2,
                                  linestyle=ls, marker=mk, markersize=7,
                                  markeredgecolor=SURFACE,
                                  label="%s, %s" % (SHORT[s], lab)))
    if ideal:
        handles.append(Line2D([], [], color=GUIDE, linewidth=1.4,
                              linestyle=(0, (5, 4)), label="ideal 1/P"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                     fontsize=10, labelcolor=INK, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(title, color=INK, fontsize=13.5, x=0.012, ha="left", y=0.988)
    fig.text(0.012, 0.952, subtitle, color=INK_2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.935))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


# ----------------------------------------------------------------- fig 4
def fig_efficiency(rows, out):
    """One panel per configuration: if the efficiency loss were purely a
    per-rank problem-size effect, each panel's curves would collapse."""
    cfgs = [c for c in ("1n", "2n", "pin")
            if any(r["cfg"] == c for r in rows)]
    fig, axes = newfig(1, len(cfgs), figsize=(6.0 * len(cfgs), 6.0))
    if len(cfgs) == 1:
        axes = [axes]
    base = {(r["N"], r["s"], r["cfg"]): r["total"] for r in rows if r["P"] == 1}

    for ax, cfg in zip(axes, cfgs):
        ax.axhline(100, color=GUIDE, linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
        for s in (1, 2):
            for N in sorted({r["N"] for r in rows if r["cfg"] == cfg}):
                b = base.get((N, s, cfg))
                if not b:
                    continue
                pts = sorted([((N - 1) * (N - 1) / r["P"],
                               100 * b / (r["P"] * r["total"]))
                              for r in rows
                              if r["N"] == N and r["s"] == s and r["cfg"] == cfg])
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=SERIES[s], linewidth=1.1, alpha=0.45, zorder=3)
                ax.plot(xs, ys, color=SERIES[s], linestyle="none",
                        marker=MARKER_BY_N[N], markersize=7.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=4)
        ax.set_xscale("log")
        ax.set_ylim(0, 135)
        name = {"1n": "1 node (unpinned)", "2n": "2 nodes, P/2 each (unpinned)",
                "pin": "1 node (pinned)"}[cfg]
        style(ax, "grid points per rank,   nx" + CDOT + "ny / P",
              "parallel efficiency  T(1) / (P" + CDOT + "T(P))    [%]"
              if cfg == cfgs[0] else "", name)
        ax.annotate("ideal (100%)", (ax.get_xlim()[0] * 2.0, 100),
                    textcoords="offset points", xytext=(0, 6),
                    color=INK_2, fontsize=8.5)

    hs = [Line2D([], [], color=SERIES[s], linewidth=2, marker="o", markersize=7,
                 markeredgecolor=SURFACE, label=SNAME[s]) for s in (1, 2)]
    hs += [Line2D([], [], linestyle="none", label="")]
    hs += [Line2D([], [], color=INK_2, linestyle="none", marker=MARKER_BY_N[N],
                  markersize=7, markeredgecolor=SURFACE, label="Nx = %d" % N)
           for N in sorted(MARKER_BY_N)]
    leg = fig.legend(handles=hs, loc="lower center", ncol=9, frameon=False,
                     fontsize=9.5, labelcolor=INK, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Efficiency against per-rank problem size",
                 color=INK, fontsize=13.5, x=0.008, ha="left", y=0.975)
    fig.text(0.008, 0.895,
             "Colour = strategy, shape = grid size.  Along each curve P grows to "
             "the LEFT (fewer points per rank).\nAbove 100% is superlinear speedup: "
             "the P=1 baseline thrashes cache that the split problem fits into.",
             color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 0.86))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


# ----------------------------------------------------------------- fig 5
def fig_comm(rows, out):
    """Time spent in the distributed transpose -- the communication cost.

    `transpose` is the whole data-movement step (message time + relay
    packing + the local transpose_real work), i.e. what the solver
    actually pays to redistribute, not just the time on the wire.  Each
    panel is annotated with that cost as a share of the whole solve, so
    the absolute curve and the relative burden are both readable.
    """
    sizes = sorted({r["N"] for r in rows})
    fig, axes = newfig(2, 3, figsize=(14.2, 8.2))
    axes = axes.ravel()

    for k, N in enumerate(sizes):
        ax = axes[k]
        have = [c for c in ("1n", "2n", "pin")
                if any(r["N"] == N and r["cfg"] == c for r in rows)]
        cfgs = [c for c in have if c != "pin"] or ["pin"]
        fracs = []
        for cfg in cfgs:
            ls, mk, _ = CFG[cfg]
            for st in (1, 2):
                xs, ys = series(rows, N, cfg, st, "transpose")
                if not xs:
                    continue
                ax.plot(xs, ys, color=SERIES[st], linewidth=2, linestyle=ls,
                        marker=mk, markersize=6.5, markeredgecolor=SURFACE,
                        markeredgewidth=1.1, zorder=3)
                fracs += [100.0 * r["transpose"] / r["total"] for r in rows
                          if r["N"] == N and r["cfg"] == cfg and r["s"] == st
                          and r["P"] > 1]
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        Ps = sorted({r["P"] for r in rows if r["N"] == N})
        ax.set_xticks(Ps)
        ax.set_xticklabels([str(p) for p in Ps], fontsize=8)
        sub = "1 node, pinned (only data at this size)" if cfgs == ["pin"] else None
        style(ax, "P (MPI ranks)" if k >= 3 else "",
              "time in distributed transpose [s]" if k % 3 == 0 else "",
              "Nx = Ny = %d" % N, sub)
        if fracs:
            # Drop below the panel subtitle where one is present.
            ax.text(0.03, 0.86 if sub else 0.94,
                    "share of solve: %.0f%% \u2013 %.0f%%  (P > 1)"
                    % (min(fracs), max(fracs)),
                    transform=ax.transAxes, color=INK_2, fontsize=8.5,
                    va="top")

    handles = []
    for cfg in ("1n", "2n"):
        ls, mk, lab = CFG[cfg]
        for st in (1, 2):
            handles.append(Line2D([], [], color=SERIES[st], linewidth=2,
                                  linestyle=ls, marker=mk, markersize=7,
                                  markeredgecolor=SURFACE,
                                  label="%s, %s" % (SHORT[st], lab)))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=10, labelcolor=INK, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Communication cost: time in the distributed transpose",
                 color=INK, fontsize=13.5, x=0.012, ha="left", y=0.988)
    fig.text(0.012, 0.952,
             "Whole data-movement step (message time + relay packing + local "
             "reorder).  Own y-scale per panel; note it is the ONLY part of the "
             "solve that grows with P.",
             color=INK_2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.935))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


# ----------------------------------------------------------------- fig 6
def fig_eta(rows, out, cfg="2n"):
    """Weak scaling on ONE log-log axes: communication time vs P, with
    each line holding eta = Nx*Ny/P (work per rank) fixed.

    Restricted to the 2-node configuration -- the case where the
    transpose actually crosses a network, so the cost models below are
    the relevant ones, and with one configuration the band is legible.

    With each pair exchanging eta/P values,

        dealing  T ~ (P-1)*alpha + eta*beta          -> LINEAR in P
        crystal  T ~ 2*log2(P)*alpha + (eta*log2(P)/2)*beta
                                                     -> LOG in P

    so at constant eta the dealing lines should approach slope 1 once
    latency dominates, while crystal stays far shallower -- paying for
    it in bytes, which grow as eta*log2(P)/2 rather than eta.

    CAVEAT: the sweep steps grid size by 16x and P by 2x, so only every
    4th P lands on a given eta; each line carries 2-3 points.  The slope
    between them is meaningful, the curvature is not.
    """
    fig, ax = newfig(figsize=(11.6, 7.2))
    groups = {}
    for r in rows:
        if r["cfg"] != cfg:
            continue
        groups.setdefault((r["N"] * r["N"] // r["P"], r["s"]), []).append(
            (r["P"], r["transpose"]))

    # Keep only eta values that are anchored at P=1, i.e. eta = Nx*Ny of
    # an actual serial run.  Those are the genuine weak-scaling series:
    # each starts from a measured single-rank baseline and grows P with
    # the problem.  Other eta values are mid-sweep coincidences with no
    # serial anchor.
    etas = sorted({e for (e, st) in groups
                   if all(len(groups.get((e, k), [])) >= 2 for k in (1, 2))
                   and min(P for P, _ in groups[(e, 1)]) == 1})
    for e in etas:
        for st in (1, 2):
            xs, ys = zip(*sorted(groups[(e, st)]))
            ax.plot(xs, ys, color=SERIES[st], linewidth=1.8, marker="o",
                    markersize=5.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.0, alpha=0.9, zorder=3)
        xs, ys = zip(*sorted(groups[(e, 1)]))
        ax.annotate(r"$2^{%d}$" % (e.bit_length() - 1), (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(8, 0), ha="left",
                    va="center", color=INK_2, fontsize=8.5)

    def med_slope(pred):
        """Slopes over the PLOTTED lines only, so the box matches the band."""
        out = {1: [], 2: []}
        for (e, st), pts in groups.items():
            if e not in etas or not pred(e) or len(pts) < 2:
                continue
            pts = sorted(pts)
            for i in range(len(pts) - 1):
                (x0, y0), (x1, y1) = pts[i], pts[i + 1]
                if y0 > 0 and y1 > 0:
                    out[st].append(math.log(y1 / y0) / math.log(x1 / x0))
        return {k: (sorted(v)[len(v) // 2] if v else float("nan"))
                for k, v in out.items()}

    cut = 2 ** 16
    lo, hi = med_slope(lambda e: e < cut), med_slope(lambda e: e >= cut)
    nlo = sum(1 for e in etas if e < cut)
    nhi = len(etas) - nlo
    ax.text(0.985, 0.03,
            "median measured slope (plotted lines only)\n"
            r"small $\eta\ (< 2^{16})$, %d series:   dealing %+.2f    crystal %+.2f"
            "\n"
            r"large $\eta\ (\geq 2^{16})$, %d series:   dealing %+.2f    crystal %+.2f"
            % (nlo, lo[1], lo[2], nhi, hi[1], hi[2]),
            transform=ax.transAxes, ha="right", va="bottom",
            color=INK_2, fontsize=9,
            bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.5"))

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    allp = sorted({r["P"] for r in rows if r["cfg"] == cfg})
    ax.set_xticks(allp)
    ax.set_xticklabels([str(v) for v in allp], fontsize=9)
    style(ax, "P (MPI ranks)", "time in distributed transpose [s]",
          "Weak scaling: communication cost at constant work per rank")

    hs = [Line2D([], [], color=SERIES[st], linewidth=2, marker="o",
                 markersize=7, markeredgecolor=SURFACE, label=SNAME[st])
          for st in (1, 2)]
    ax.legend(handles=hs, loc="upper left", frameon=False, fontsize=10,
              labelcolor=INK)
    fig.text(0.012, 0.022,
             r"2 nodes, P/2 ranks each, unpinned.  Each line holds "
             r"$\eta = N_x N_y/P$ fixed (labelled $2^k$ at its right end); "
             "colour is the strategy.\n"
             r"Only every 4th P lands on a given $\eta$, so each line carries "
             "2-3 points: read the slopes, not the curvature.",
             color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


def main():
    dst = sys.argv[1] if len(sys.argv) > 1 else "../plots"
    os.makedirs(dst, exist_ok=True)
    r1 = load("bench_1node.txt", "1n")
    r2 = load("bench_2node.txt", "2n")
    # Pinned runs (bench_sweep.txt, bench_k15.txt) are deliberately left
    # out: the plots compare the two unpinned configurations only.
    rows = r1 + r2
    print("loaded: 1node=%d  2node=%d  total=%d" % (len(r1), len(r2), len(rows)))

    fig_error(rows, dst + "/1_error_vs_nx.png")
    panels(rows, dst + "/2_time_vs_P.png", "total", "total solve time [s]",
           "Strong scaling: time to solution",
           "Best of nrep, critical-path rank.  Each panel has its OWN y-scale; "
           "the dashed grey line is that panel's ideal 1/P.",
           shared=False, ideal=True)
    panels(rows, dst + "/3_gflops_vs_P.png", "gflops", "GFLOPS",
           "Strong scaling: achieved GFLOPS",
           "Same flop model as the serial solver: 2" + CDOT + "10" + CDOT + "nx"
           + CDOT + "ny" + CDOT + "(log" + SUB2 + "Nx + log" + SUB2 + "Ny).  "
           "Shared scale across panels.",
           shared=True, ideal=False)
    fig_efficiency(rows, dst + "/4_efficiency_vs_work_per_rank.png")
    fig_comm(rows, dst + "/5_communication_vs_P.png")
    fig_eta(rows, dst + "/6_comm_constant_eta.png")


main()
