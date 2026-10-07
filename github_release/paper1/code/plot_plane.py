"""
Module G figure: the (M_in, beta_f) plane for monochromatic PBH reheating.

(a) peak Omega_GW h^2 of the cluster channel with module F (NL-A+F), with contours of its
    peak frequency; (b) the margin below Delta N_eff, LIMIT / int Omega (NL-B+F, the upper
    bracket); (c) the shot-noise floor of a critical-collapse population (peak);
(d) detectability of NL-A+F: max over detectors of R = Omega/Omega_PLI (SNR 1, 1 yr), with
    each detector's R = 10 contour (SNR 10 in one year; detectors.py, detect_map.py); dashed:
    the same contours with shock-limited sound (NL-A+F-sh, sound_lifetime.py).
Lines: beta_c (PBH domination), the 2012.08151 bound, where our linear extrapolation
(LIN-UV, damping on) reaches Delta N_eff, and (dash-dot) where half the holes would have
merged in early binaries before evaporating (mergers.py, alpha = 0.4; an estimate), (dotted)
where a quarter would, enough for the merged holes to dominate again (mergers.py (4)).
Hatched: excluded by merged relics (relics.py, 2412.01890).  The grid stops at 3e8 g,
inside the BBN mass limit (~5e8 g).
From results/plane_scan.csv and plane_spectra.npz.
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.tri import Triangulation   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase_a as pa              # noqa: E402
import cosmology as co            # noqa: E402
import relics as rl               # noqa: E402
import detect_map as dmap         # noqa: E402
import mergers as mg              # noqa: E402

OUT = pa.OUT
LIMIT = 5.6e-6 * 0.3
rows = list(csv.DictReader(open(os.path.join(OUT, "plane_scan.csv"))))


def field(ch, col):
    pts = [(float(r["M_g"]), float(r["beta"]), float(r[col])) for r in rows if r["channel"] == ch]
    return np.array(pts).T


M, B, pkA = field("NL-A+F", "peak_h2_damped")
_, _, fA = field("NL-A+F", "f_peak")
_, _, IB = field("NL-B+F", "int_h2_damped")
_, _, IU = field("LIN-UV", "int_h2_damped")
_, _, pkS = field("SHOT-Ch", "peak_h2_damped")
assert np.allclose(M, np.repeat(dmap.M, dmap.B.shape[1]), rtol=1e-3) and np.allclose(B, dmap.B.ravel(), rtol=1e-3)
R = dmap.ratio(dmap.CH["NL-A+F"])
Rmax = np.max([R[n] for n in R], axis=0).ravel()
tri = Triangulation(np.log10(M), np.log10(B))
Ms = np.geomspace(1.0, 3e8, 200)
b_rel = np.array([rl.beta_relic(m) for m in Ms])
# early-binary merger contours (mergers.py, alpha = 0.4): P(< t_ev) = 1/2, and the merged
# fraction above which the merged holes dominate again before evaporating (P ~ 0.24)
bg = np.geomspace(1e-14, 1.0, 600)
b_merge = np.array([bg[np.argmax(mg.merged_fraction(m, bg)[0] >= 0.5)] for m in Ms])
P_RED = mg.redomination_fraction()
b_redom = np.array([bg[np.argmax(mg.merged_fraction(m, bg)[0] >= P_RED)] for m in Ms])

fig, axs = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
axs = axs.ravel()
panels = ((axs[0], np.log10(np.maximum(pkA, 1e-40)), np.arange(-26, -8, 1), "viridis",
           r"(a) $\log_{10}$ peak $\Omega_{\rm GW}h^2$, clusters with burst efficiency"),
          (axs[1], np.log10(LIMIT / np.maximum(IB, 1e-60)), np.arange(3, 30, 1), "magma_r",
           r"(b) $\log_{10}$ (margin below $\Delta N_{\rm eff}$), NL-B with burst efficiency"),
          (axs[2], np.log10(np.maximum(pkS, 1e-80)), np.arange(-60, -14, 3), "cividis",
           r"(c) $\log_{10}$ peak $\Omega_{\rm GW}h^2$, shot noise, Choptuik"),
          (axs[3], np.log10(np.maximum(Rmax, 1e-30)), np.arange(-12, 8, 1), "RdYlGn",
           r"(d) $\log_{10}$ max$_{\rm det}\,\Omega/\Omega_{\rm PLI}$ (SNR 1, 1 yr), clusters"))
for ax, zz, lev, cmap, title in panels:
    cs = ax.tricontourf(tri, zz, levels=lev, cmap=cmap, extend="both")
    fig.colorbar(cs, ax=ax, pad=0.01)
    ax.plot(np.log10(Ms), np.log10([co.beta_crit(m) for m in Ms]), "-", color="0.4", lw=1.2)
    ax.plot(np.log10(Ms), np.log10(1.1e-6 * (Ms / 1e4) ** (-17 / 24)), "r--", lw=1.5)
    ax.tricontour(tri, np.log10(np.maximum(IU, 1e-300)), levels=[np.log10(LIMIT)], colors="k", linewidths=1.0,
                  linestyles=":")
    ok = np.isfinite(b_rel)
    ax.fill_between(np.log10(Ms[ok]), np.log10(b_rel[ok]), 0.0, facecolor="none", edgecolor="k", hatch="///",
                    lw=0.0, alpha=0.6)
    ax.plot(np.log10(Ms[ok]), np.log10(b_rel[ok]), "k-", lw=1.0)
    ax.plot(np.log10(Ms), np.log10(b_merge), "-.", color="tab:blue", lw=1.1)
    ax.plot(np.log10(Ms), np.log10(b_redom), ":", color="tab:blue", lw=1.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, np.log10(3e8))
    ax.set_ylim(np.log10(np.min(B)), 0)
    ax.text(0.1, -9.2, "red dashed: published induced-GW bound\nblack dotted: our linear extrapolation\n"
            "reaches $\\Delta N_{\\rm eff}$ (damping on)\ngrey: $\\beta_c$ (PBH domination)\n"
            "hatched: merged relics (Holst et al.)\nblue dash-dot: half the holes merge (estimate)\n"
            "blue dotted: a quarter merge; merged holes re-dominate",
            fontsize=6.5, color="k", va="top")
cf = axs[0].tricontour(tri, np.log10(fA), levels=[-4, -3, -2, -1, 0, 1, 2, 3, 4, 5], colors="w",
                       linewidths=0.6, alpha=0.7)
axs[0].clabel(cf, fmt=lambda v: f"{10 ** v:g} Hz", fontsize=6)
cols = {"LISA": "tab:blue", "DECIGO": "tab:purple", "BBO": "k", "ET": "tab:orange", "CE": "tab:red", "HLVK": "tab:brown"}
for n, c in cols.items():
    if np.max(R[n]) >= dmap.RHO:
        axs[3].tricontour(tri, np.log10(np.maximum(R[n].ravel(), 1e-30)), levels=[np.log10(dmap.RHO)], colors=c,
                          linewidths=1.3)
        axs[3].plot([], [], color=c, lw=1.3, label=f"{n}, SNR 10")
# the same contours with shock-limited sound (sound_lifetime.py), dashed
Rs = dmap.ratio(dmap.CH["NL-A+F-sh"])
for n, c in cols.items():
    if np.max(Rs[n]) >= dmap.RHO:
        axs[3].tricontour(tri, np.log10(np.maximum(Rs[n].ravel(), 1e-30)), levels=[np.log10(dmap.RHO)], colors=c,
                          linewidths=1.3, linestyles="--")
        axs[3].plot([], [], color=c, lw=1.3, ls="--", label=f"{n}, shock-limited")
axs[3].legend(fontsize=7, loc="upper right")
for ax in axs[2:]:
    ax.set_xlabel(r"$\log_{10} M_{\rm in}$ [g]")
for ax in axs[::2]:
    ax.set_ylabel(r"$\log_{10}\beta_f$")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"plane.{ext}"), dpi=150)
print("written")
