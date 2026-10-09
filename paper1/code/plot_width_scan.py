"""Figure 8: GW signal vs log-normal width sigma, from results/width_scan.csv.

Clusters: NL-A+F (with module F's burst efficiency), NL-A without it thin dotted; shot noise.
Solid: on the published bound; dashed: beta_f = 1e-2.  Vertical lines: sigma = 0.5/x_cl, where
the coherent cluster signal has halved.  Triangles: monochromatic (left) and Choptuik (right).
Below 1e-30 the coherent signal is numerical noise (it has vanished); the axis stops there.
Style: figstyle.py.
"""
import os
import sys
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs             # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.patches import Patch     # noqa: E402

fs.use()
OUT = fs.OUT
RED, GREEN = fs.COL["NL-A"], fs.COL["SHOT"]

rows = list(csv.DictReader(open(os.path.join(OUT, "width_scan.csv"))))
fig, axs = plt.subplots(1, 3, figsize=(fs.WIDE, 3.1), sharey=True)
fig.subplots_adjust(left=0.096, right=0.99, top=0.915, bottom=0.315, wspace=0.06)
for ax, M in zip(axs, ("1.0e+02", "1.0e+04", "1.0e+06")):
    ax.axvspan(0.1, 3.0, color="0.88", lw=0, zorder=0)
    for lab, ls in (("2012.08151 bound", "-"), ("saturated (beta=1e-2)", "--")):
        sub = [r for r in rows if r["M_g"] == M and r["label"] == lab]
        for ch, col, lw, lsx in (("NL-A+F", RED, 1.5, ls), ("NL-A", RED, 0.7, ":"), ("SHOT", GREEN, 1.5, ls)):
            pts = sorted((float(r["sigma"]), float(r["int_h2_damped"])) for r in sub
                         if r["channel"] == ch and r["sigma"] not in ("mono", "choptuik"))
            s, y = np.array(pts).T
            ax.loglog(s, y, ls=lsx, color=col, lw=lw, zorder=3)
            if ch != "NL-A":
                mono = [float(r["int_h2_damped"]) for r in sub if r["channel"] == ch and r["sigma"] == "mono"]
                chop = [float(r["int_h2_damped"]) for r in sub if r["channel"] == ch and r["sigma"] == "choptuik"]
                ax.plot([s[0] / 3], mono, "<", color=col, ms=4.5, zorder=4)
                ax.plot([s[-1] * 2.2], chop, ">", color=col, ms=4.5, zorder=4)
        xcl = float(sub[0]["x_cl"])
        ax.axvline(0.5 / xcl, color=RED, lw=0.6, ls=ls, alpha=0.6, zorder=1)
    ax.axhline(fs.LIMIT, color="k", lw=0.7, ls="-.")
    ax.text(2.5e-7, fs.LIMIT * 2.5, r"$\Delta N_{\rm eff}<0.3$", fontsize=7, va="bottom", zorder=2,
            bbox=dict(facecolor="w", edgecolor="none", pad=0.6))
    ax.set_title(rf"$M_{{\rm in}}=10^{{{int(np.log10(float(M)))}}}$ g")
    ax.set_xlabel(r"log-normal width $\sigma$")
    ax.set_xlim(1.5e-8, 4.0)
    ax.set_xticks(10.0 ** np.arange(-7, 1, 2))
    ax.set_xticks(10.0 ** np.arange(-8, 1), minor=True)
    ax.set_xticklabels([], minor=True)
    ax.set_ylim(1e-30, 1e-4)
    ax.set_yticks(10.0 ** np.arange(-30, -3, 4))
    ax.set_yticks(10.0 ** np.arange(-30, -3), minor=True)
    ax.set_yticklabels([], minor=True)
axs[0].set_ylabel(r"$\int \mathrm{d}\ln f\;\Omega_{\rm GW}h^2$ today")
leg = [(Line2D([], [], color=RED, lw=1.5), "clusters (NL-A+F)"),
       (Line2D([], [], color=RED, lw=0.7, ls=":"), "clusters without burst efficiency"),
       (Line2D([], [], color=GREEN, lw=1.5), "shot noise"),
       (Line2D([], [], color="k", lw=1.2), "on the published bound"),
       (Line2D([], [], color="k", lw=1.2, ls="--"), r"at $\beta_f=10^{-2}$"),
       (Line2D([], [], color=RED, lw=0.6, alpha=0.6), r"$\sigma=0.5/x_{\rm cl}$"),
       (Line2D([], [], color="0.3", marker="<", ls="none", ms=4.5), "monochromatic"),
       (Line2D([], [], color="0.3", marker=">", ls="none", ms=4.5), "critical collapse"),
       (Patch(color="0.88", lw=0), r"explosion times spread over $>H^{-1}/3$")]
fig.legend([h for h, _ in leg], [l for _, l in leg], loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
           handlelength=2.4, columnspacing=2.0)
fs.save(fig, "width_scan")
print("written")
