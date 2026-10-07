"""Figure: GW signal vs log-normal width sigma, from results/width_scan.csv.

Clusters: NL-A+F (with module F's burst efficiency) solid, NL-A without it thin dotted.
Vertical ticks: sigma = 0.5/x_cl, where the coherent cluster signal has halved.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
LIMIT = 5.6e-6 * 0.3

rows = list(csv.DictReader(open(os.path.join(OUT, "width_scan.csv"))))
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
for ax, M in zip(axs, ("1.0e+02", "1.0e+04", "1.0e+06")):
    for lab, ls in (("2012.08151 bound", "-"), ("saturated (beta=1e-2)", "--")):
        sub = [r for r in rows if r["M_g"] == M and r["label"] == lab]
        short = "old bound" if lab.startswith("2012") else "saturated"
        for ch, col, lw, name in (("NL-A+F", "tab:red", 2, "clusters"), ("NL-A", "tab:red", 0.8, None),
                                  ("SHOT", "tab:green", 2, "shot noise")):
            pts = [(float(r["sigma"]), float(r["int_h2_damped"])) for r in sub
                   if r["channel"] == ch and r["sigma"] not in ("mono", "choptuik")]
            s, y = np.array(pts).T
            ax.loglog(s, np.maximum(y, 1e-40), ls if name else ":", color=col, lw=lw,
                      label=f"{name}, {short}" if name else None)
            mono = [float(r["int_h2_damped"]) for r in sub if r["channel"] == ch and r["sigma"] == "mono"]
            chop = [float(r["int_h2_damped"]) for r in sub if r["channel"] == ch and r["sigma"] == "choptuik"]
            if name:
                ax.plot([s[0] / 3], mono, "<", color=col, ms=7)
                ax.plot([s[-1] * 3], chop, ">", color=col, ms=7)
        xcl = float(sub[0]["x_cl"])
        ax.axvline(0.5 / xcl, color="tab:red", lw=0.8, ls=ls, alpha=0.5)
    ax.axhline(LIMIT, color="k", lw=0.8, ls="-.")
    ax.text(2e-7, LIMIT * 2, r"$\Delta N_{\rm eff}$ limit", fontsize=7)
    ax.axvspan(0.1, 3.0, color="0.85", alpha=0.5, lw=0)
    ax.text(0.12, 3e-33, "explosion times\nspread > H$^{-1}$/3:\nshot noise O(1)", fontsize=6, color="0.3")
    ax.set_title(rf"$M_{{\rm in}}=10^{{{int(np.log10(float(M)))}}}$ g  "
                 "(◀ monochromatic, ▶ Choptuik)", fontsize=9)
    ax.set_xlabel(r"log-normal width $\sigma$")
    ax.set_ylim(1e-34, 1e-4)
axs[0].set_ylabel(r"$\int d\ln f\,\Omega_{\rm GW}h^2$ (damped)")
axs[0].plot([], [], ":", color="tab:red", lw=0.8, label="clusters without module F")
axs[0].plot([], [], "-", color="tab:red", lw=0.8, alpha=0.5, label=r"$\sigma = 0.5/x_{\rm cl}$")
axs[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"width_scan.{ext}"), dpi=160)
print("written")
