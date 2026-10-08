"""
Module G figure (Fig. 9): the (M_in, beta_f) plane for monochromatic PBH reheating.

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

The scan grid (plane_scan.py: 22 masses by 16 abundances, from 3 beta_c to 1) is drawn through
a monotone cubic (PCHIP) interpolation along each grid direction, which passes through every
computed value and does not overshoot; the peak frequency, quantized by the frequency grid,
is smoothed before its contours are drawn.  Table VI (detect_map.py) interpolates the same
grid log-linearly.  From results/plane_scan.csv and plane_spectra.npz.  Style: figstyle.py.
"""
import os
import sys
import csv
import numpy as np
from scipy.interpolate import PchipInterpolator, RectBivariateSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs             # noqa: E402
import phase_a as pa              # noqa: E402  (sets sys.path for cosmology)
import cosmology as co            # noqa: E402
import relics as rl               # noqa: E402
import detect_map as dmap         # noqa: E402
import mergers as mg              # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.colors import TwoSlopeNorm   # noqa: E402
from matplotlib.lines import Line2D          # noqa: E402
from matplotlib.patches import Patch         # noqa: E402

fs.use()
OUT = fs.OUT
LIMIT = fs.LIMIT
rows = list(csv.DictReader(open(os.path.join(OUT, "plane_scan.csv"))))
NM, NB = dmap.B.shape


def field(ch, col):
    pts = [(float(r["M_g"]), float(r["beta"]), float(r[col])) for r in rows if r["channel"] == ch]
    return np.array(pts).T


M, B, pkA = field("NL-A+F", "peak_h2_damped")
_, _, fA = field("NL-A+F", "f_peak")
_, _, IB = field("NL-B+F", "int_h2_damped")
_, _, IU = field("LIN-UV", "int_h2_damped")
_, _, pkS = field("SHOT-Ch", "peak_h2_damped")
assert np.allclose(M, np.repeat(dmap.M, NB), rtol=1e-3) and np.allclose(B, dmap.B.ravel(), rtol=1e-3)
grid = lambda v: v.reshape(NM, NB)

# the scan grid: log10 M uniform; along beta, log10 beta = (1 - s) log10(3 beta_c), s uniform in [0, 1]
LM = np.log10(dmap.M)
S = np.linspace(0.0, 1.0, NB)
LB0 = np.log10(dmap.B[:, 0])
assert np.allclose(np.log10(dmap.B), LB0[:, None] * (1.0 - S[None, :]), atol=1e-6)
LMf = np.linspace(LM[0], LM[-1], 360)
Sf = np.linspace(0.0, 1.0, 260)
LB0f = np.log10([3.0 * co.beta_crit(10.0 ** u) for u in LMf])
X = np.repeat(LMf[:, None], Sf.size, axis=1)
Y = LB0f[:, None] * (1.0 - Sf[None, :])


def fine(F):
    """A field on the scan grid on the fine grid: PCHIP along beta, then along M."""
    G = PchipInterpolator(S, F, axis=1)(Sf)
    return PchipInterpolator(LM, G, axis=0)(LMf)


def smooth(F, s):
    """A field with grid noise (the peak frequency), as a smoothing bicubic spline."""
    return RectBivariateSpline(LM, S, F, s=s)(LMf, Sf)


lgA = fine(grid(np.log10(np.maximum(pkA, 1e-40))))
lgMarg = fine(grid(np.log10(LIMIT / np.maximum(IB, 1e-60))))
lgS = fine(grid(np.log10(np.maximum(pkS, 1e-80))))
lgU = fine(grid(np.log10(np.maximum(IU, 1e-300))))
lgF = smooth(grid(np.log10(fA)), s=NM * NB * 0.04 ** 2)       # ~ quantization of the frequency grid
R = dmap.ratio(dmap.CH["NL-A+F"])
Rs = dmap.ratio(dmap.CH["NL-A+F-sh"])
lgR = fine(np.clip(np.log10(np.maximum(np.max([R[n] for n in R], axis=0), 1e-30)), -13.0, 8.0))

# overlays: beta_c, the published bound, merged relics, early-binary mergers
Ms = np.geomspace(1.0, 3e8, 400)
lMs = np.log10(Ms)
b_c = np.array([co.beta_crit(m) for m in Ms])
b_rel = np.array([rl.beta_relic(m) for m in Ms])
bg = np.geomspace(1e-14, 1.0, 600)


def merge_line(P0):
    """beta_f at which the merged fraction reaches P0 (mergers.py), interpolated in log beta."""
    out = []
    for m in Ms:
        P = mg.merged_fraction(m, bg)[0]
        j = int(np.argmax(P >= P0))
        if j == 0:
            out.append(np.nan)
            continue
        t = (P0 - P[j - 1]) / (P[j] - P[j - 1])
        out.append(np.log10(bg[j - 1]) + t * (np.log10(bg[j]) - np.log10(bg[j - 1])))
    return np.array(out)


l_merge = merge_line(0.5)
l_redom = merge_line(mg.redomination_fraction())
BLUE, RED = fs.COL["LIN-NL"], "#CC3311"

fig, axs = plt.subplots(2, 2, figsize=(fs.WIDE, 6.55), sharex=True, sharey=True)
fig.subplots_adjust(left=0.07, right=0.958, top=0.965, bottom=0.165, wspace=0.14, hspace=0.17)
axs = axs.ravel()
panels = ((lgA, np.arange(-26, -8, 1), "viridis", None,
           r"(a) $\log_{10}$ peak $\Omega_{\rm GW}h^2$, clusters (NL-A+F)"),
          (lgMarg, np.arange(3, 30, 1), "magma_r", None,
           r"(b) $\log_{10}$ margin below $\Delta N_{\rm eff}$, upper bracket (NL-B+F)"),
          (lgS, np.arange(-60, -14, 3), "cividis", None,
           r"(c) $\log_{10}$ peak $\Omega_{\rm GW}h^2$, shot noise (critical collapse)"),
          (lgR, np.arange(-12, 8, 1), "BrBG", TwoSlopeNorm(vmin=-12, vcenter=1, vmax=7),
           r"(d) $\log_{10}\,\mathrm{max}_{\rm det}\,\Omega/\Omega_{\rm PLI}$ (SNR 1, 1 yr), clusters"))
for ax, (zz, lev, cmap, norm, title) in zip(axs, panels):
    cs = ax.contourf(X, Y, zz, levels=lev, cmap=cmap, norm=norm, extend="both")
    cb = fig.colorbar(cs, ax=ax, pad=0.015, fraction=0.06, aspect=22)
    cb.ax.tick_params(labelsize=7, direction="out", length=2.5, right=True, left=False)
    cb.ax.minorticks_off()
    ax.plot(lMs, np.log10(b_c), "-", color="0.45", lw=0.9)
    ax.plot(lMs, np.log10(1.1e-6 * (Ms / 1e4) ** (-17 / 24)), "--", color=RED, lw=1.2)
    ax.contour(X, Y, lgU, levels=[np.log10(LIMIT)], colors="k", linewidths=0.9, linestyles=":")
    ok = np.isfinite(b_rel)
    ax.fill_between(lMs[ok], np.log10(b_rel[ok]), 0.0, facecolor="none", edgecolor="k", hatch="////",
                    lw=0.0, alpha=0.55)
    ax.plot(lMs[ok], np.log10(b_rel[ok]), "k-", lw=0.9)
    ax.plot(lMs, l_merge, "-.", color=BLUE, lw=1.0)
    ax.plot(lMs, l_redom, ":", color=BLUE, lw=1.3)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, np.log10(3e8))
    ax.set_ylim(np.log10(np.min(dmap.B)), 0)
    ax.set_xticks(np.arange(0, 9))
    ax.set_yticks(np.arange(-12, 1, 2))
    ax.tick_params(which="minor", top=False, bottom=False, left=False, right=False)
cf = axs[0].contour(X, Y, lgF, levels=np.arange(-4, 6), colors="w", linewidths=0.6, alpha=0.85,
                    negative_linestyles="solid")
# labels where the contours cross a band 1.3 dex above the lower edge of the grid (dark colours)
band = lambda x: np.interp(x, LMf, LB0f) + 1.3
pos = []
for segs in cf.allsegs:
    v = np.concatenate(segs) if segs else np.empty((0, 2))
    if len(v):
        i = int(np.argmin(np.abs(v[:, 1] - band(v[:, 0]))))
        if abs(v[i, 1] - band(v[i, 0])) < 0.3:
            pos.append(tuple(v[i]))
axs[0].clabel(cf, fmt=lambda v: rf"$10^{{{int(round(v))}}}$ Hz", fontsize=6, inline_spacing=2, manual=pos)
DET = {"DECIGO": "#AA3377", "BBO": "k", "ET": fs.COL["NL-B"], "CE": RED}
for n, c in DET.items():
    for RR, ls in ((R, "-"), (Rs, "--")):
        if np.max(RR[n]) >= dmap.RHO:
            lr = fine(np.clip(np.log10(np.maximum(RR[n], 1e-30)), -3.0, 5.0))
            axs[3].contour(X, Y, lr, levels=[np.log10(dmap.RHO)], colors=c, linewidths=1.1, linestyles=ls)
for ax in axs[2:]:
    ax.set_xlabel(r"$\log_{10}(M_{\rm in}/{\rm g})$")
for ax in axs[::2]:
    ax.set_ylabel(r"$\log_{10}\beta_f$")
lines = [(Line2D([], [], color="0.45", lw=0.9), r"$\beta_c$: PBH domination"),
         (Line2D([], [], color=RED, ls="--", lw=1.2), "published induced-GW bound"),
         (Line2D([], [], color="k", ls=":", lw=0.9), r"linear extrapolation reaches $\Delta N_{\rm eff}$"),
         (Patch(facecolor="none", edgecolor="k", hatch="////", lw=0.6), "excluded by merged relics"),
         (Line2D([], [], color=BLUE, ls="-.", lw=1.0), "half the holes merge (estimate)"),
         (Line2D([], [], color=BLUE, ls=":", lw=1.3), "merged holes dominate again")]
fig.legend([h for h, _ in lines], [l for _, l in lines], loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, 0.045), handlelength=2.6, columnspacing=1.8)
dets = [(Line2D([], [], color=c, lw=1.1), n) for n, c in DET.items()]
dets += [(Line2D([], [], color="0.3", lw=1.1), "(d): SNR 10, sound lives a Hubble time"),
         (Line2D([], [], color="0.3", lw=1.1, ls="--"), "shock-limited sound")]
fig.legend([h for h, _ in dets], [l for _, l in dets], loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.4)
fs.save(fig, "plane")
print("written")
