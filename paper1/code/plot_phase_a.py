"""
Figs. 6 and 7 of the paper.

Fig. 6 (phase_a_spectra): GW spectra today on the published bound at 10^4 g, for a
monochromatic and a critical-collapse population.  The spectra are recomputed here with
phase_a.run_benchmark on a finer frequency grid (NX points; phase_a.py stores 90, enough for
the peaks of Table IV) so that the curves are smooth.  Each spectrum ends at its cutoff,
k = 2 c_s k_max (k_max = k_PBH, or k_NL for LIN-NL), above which no pair of sound waves can
source it (figstyle.until_cutoff).  The axis reaches down to 1e-50 so that the suppressed
critical-collapse spectra are seen whole, down to their cutoff.  The detector sensitivity curves are drawn up to
the Delta N_eff limit, above which no signal is allowed; where a tabulated band ends below
it (DECIGO and BBO at 10 Hz, CE at 5 Hz) a vertical line marks the edge of the band.
Fig. 7 (phase_a_beta_scan): integrated signal against beta_f, from results/beta_scan.csv.
The linear extrapolation grows as beta_f^(16/3) without limit; it is drawn up to TOP_UV and
ends in an arrow.
"""
import os
import sys
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs   # noqa: E402
import phase_a as pa    # noqa: E402
import detectors as dt  # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.patches import Patch     # noqa: E402

fs.use()
OUT = fs.OUT
COL = fs.COL
NX = 1200            # frequency points of Fig. 6
SNR = 10.0           # sensitivity curves drawn for SNR 10 in one year (detectors.py)
TOP_UV = 1e1         # Fig. 7: the linear extrapolation is drawn up to here, then an arrow
M_FID, B_FID = 1e4, 1.1e-6
# line style of each variant: (linestyle, linewidth)
STY = {"base": ("-", 1.5), "LIN-UV": ("--", 1.2), "F": ("-.", 1.1), "puff": (":", 1.5),
       "sh": ("-", 0.75), "undamped": (":", 1.1)}


def spectra(pop):
    """{name: (f [Hz], Omega_GW h^2 today)} for Fig. 6; all damped except SHOT-u."""
    _, f, res, _, _ = pa.run_benchmark(M_FID, B_FID, pop, nx=NX)
    _, _, resF, _, _ = pa.run_benchmark(M_FID, B_FID, pop, nx=NX, hydro=True, only=("NL-A", "NL-B"),
                                        damped_only=True)
    out = {n: (f, v[1]) for n, v in res.items()}
    out["SHOT-u"] = (f, res["SHOT"][0])
    out.update({n + "+F": (f, v[1]) for n, v in resF.items()})
    if pop == "mono":
        _, fp, rp, _, _ = pa.run_benchmark(M_FID, B_FID, pop, xi=1.8, nx=NX, hydro=True, only=("NL-A",),
                                           damped_only=True)
        out["NL-A+F-puff"] = (fp, rp["NL-A"][1])
        _, fs_, rs, _, _ = pa.run_benchmark(M_FID, B_FID, pop, nx=NX, hydro=True, lifetime=True,
                                            only=("NL-A", "LIN-NL"), damped_only=True)
        out["NL-A+F-sh"], out["LIN-NL-sh"] = (fs_, rs["NL-A"][1]), (fs_, rs["LIN-NL"][1])
    return out


def below(f, o, ceil, fmin):
    """
    A sensitivity curve up to ceil: the part below it, ending on it.  Where the tabulated band
    ends below ceil, a vertical line up to ceil marks the band edge (not at fmin, the frame).
    """
    lf, lo, lc = np.log(f), np.log(o), np.log(ceil)
    ok = np.nonzero(lo <= lc)[0]
    i, j = ok[0], ok[-1]
    assert np.all(np.diff(ok) == 1)               # one interval below ceil
    x, y = list(lf[i:j + 1]), list(lo[i:j + 1])
    if i > 0:                                     # crosses ceil on the left arm
        t = (lc - lo[i - 1]) / (lo[i] - lo[i - 1])
        x, y = [lf[i - 1] + t * (lf[i] - lf[i - 1])] + x, [lc] + y
    elif f[0] > 1.01 * fmin:                      # lower band edge
        x, y = [lf[0]] + x, [lc] + y
    if j < len(f) - 1:
        t = (lc - lo[j]) / (lo[j + 1] - lo[j])
        x, y = x + [lf[j] + t * (lf[j + 1] - lf[j])], y + [lc]
    else:                                         # upper band edge
        x, y = x + [lf[-1]], y + [lc]
    return np.exp(x), np.exp(y)


# detector labels: (f, Omega, ha, va), next to the arm or the minimum they name
DET_LABEL = {"LISA": (0.055, 1e-9, "right", "center"), "DECIGO": (0.14, 2e-15, "center", "bottom"),
             "BBO": (0.18, 3e-17, "center", "top"), "ET": (1.7, 1e-9, "right", "center"),
             "CE": (560.0, 1e-9, "left", "center")}


def detectors(ax, fmin):
    """Power-law-integrated sensitivity curves (Schmitz), for SNR 10 in one year, up to the limit."""
    for name in ("LISA", "DECIGO", "BBO", "ET", "CE"):
        f, o = dt.CURVES[name]
        ax.loglog(*below(f, SNR * o, fs.LIMIT, fmin), color=fs.DET, lw=0.7, zorder=1)
        x, y, ha, va = DET_LABEL[name]
        ax.text(x, y, name, ha=ha, va=va, fontsize=7, color="0.3", zorder=1)


def line(ax, x, y, col, sty, **kw):
    ls, lw = STY[sty]
    return ax.loglog(x, fs.until_cutoff(x, y), color=col, ls=ls, lw=lw, **kw)


def fig_spectra():
    FMIN, YLO, YHI, H = 1e-3, 1e-50, 1e-4, 4.6        # H: figure height [in]
    fig, axs = plt.subplots(1, 2, figsize=(fs.WIDE, H), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.99, top=1 - 0.235 / H, bottom=1.15 / H, wspace=0.05)
    for ax, pop in zip(axs, ("mono", "choptuik")):
        s = spectra(pop)
        detectors(ax, FMIN)
        ax.axhline(fs.LIMIT, color="k", lw=0.7, ls="-.", zorder=1)
        ax.text(1.4e-3, fs.LIMIT * 2.5, r"$\Delta N_{\rm eff}<0.3$", fontsize=7, va="bottom")
        # thick channels first, the variants on top, LIN-NL (equal to NL-A below its cutoff) last
        line(ax, *s["LIN-UV"], COL["LIN-UV"], "LIN-UV", zorder=3)
        for n in ("NL-B", "NL-A", "SHOT"):
            line(ax, *s[n], COL[n], "base", zorder=3)
        if pop == "mono":
            fA, yA = s["NL-A+F"]
            ysh = np.interp(np.log(fA), np.log(s["NL-A+F-sh"][0]), s["NL-A+F-sh"][1])
            top, bot = fs.until_cutoff(fA, yA), fs.until_cutoff(fA, ysh)
            ax.fill_between(fA, bot, top, where=np.isfinite(top) & np.isfinite(bot), color=COL["NL-A"],
                            alpha=0.14, lw=0, zorder=2)
            line(ax, *s["NL-A+F-sh"], COL["NL-A"], "sh", zorder=4)
            line(ax, *s["NL-A+F-puff"], COL["NL-A"], "puff", zorder=4)
            line(ax, *s["LIN-NL-sh"], COL["LIN-NL"], "sh", zorder=5)
        for n in ("NL-B+F", "NL-A+F"):
            line(ax, *s[n], COL[n], "F", zorder=4)
        line(ax, *s["SHOT-u"], COL["SHOT"], "undamped", zorder=4)
        line(ax, *s["LIN-NL"], COL["LIN-NL"], "base", zorder=5)
        ax.set_title("monochromatic" if pop == "mono" else "critical collapse (Choptuik)")
        ax.set_xlabel(r"$f$ [Hz]")
        ax.set_xlim(FMIN, 1e4)
        ax.set_xticks(10.0 ** np.arange(-3, 4, 2))       # labels at odd decades: none at the shared edge
        ax.set_xticks(10.0 ** np.arange(-3, 5), minor=True)
        ax.set_xticklabels([], minor=True)
        ax.set_ylim(YLO, YHI)
        ax.set_yticks(10.0 ** np.arange(-50, -4, 5))
        ax.set_yticks(10.0 ** np.arange(-50, -3), minor=True)
        ax.set_yticklabels([], minor=True)
    axs[0].set_ylabel(r"$\Omega_{\rm GW}h^2$ today")
    # what the right panel shows
    axs[1].text(0.3, 1e-26, "coherent channels suppressed:\nthe holes evaporate at different times",
                ha="center", va="bottom", fontsize=7, color="0.2")
    axs[1].text(1.1e3, 1e-22, "shot noise, about as on the left", ha="right", va="bottom", fontsize=7,
                color=COL["SHOT"])
    # legends: channels (colour) and variants (line style)
    ch = [Line2D([], [], color=COL["LIN-UV"], ls="--", lw=1.2)] + \
         [Line2D([], [], color=COL[n], lw=1.5) for n in ("LIN-NL", "NL-A", "NL-B", "SHOT")]
    fig.legend(ch, ["LIN-UV", "LIN-NL", "NL-A", "NL-B", "SHOT"], loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.49 / H), handlelength=2.6, columnspacing=2.2)
    # variants, filled column by column (every curve is damped except the dotted green)
    var = [(Line2D([], [], color="k", ls="-.", lw=1.1), "with burst efficiency (module F)"),
           (Line2D([], [], color=COL["NL-A"], ls=":", lw=1.5), "NL-A with burst efficiency, puff-up 1.8"),
           ((Patch(color=COL["NL-A"], alpha=0.14, lw=0), Line2D([], [], color=COL["NL-A"], lw=0.75)),
            "NL-A with burst efficiency, shock-limited sound"),
           (Line2D([], [], color=COL["LIN-NL"], lw=0.75), "LIN-NL, shock-limited sound"),
           (Line2D([], [], color=COL["SHOT"], ls=":", lw=1.1), "SHOT without diffusion damping"),
           (Line2D([], [], color=fs.DET, lw=0.7), "detector sensitivity, SNR 10 in one year")]
    fig.legend([h for h, _ in var], [l for _, l in var], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=3.0)
    fs.save(fig, "phase_a_spectra")


def rising_to(x, y, top):
    """A rising curve up to y = top, ending on it (interpolated in log-log)."""
    keep = y < top
    xe = np.exp(np.interp(np.log(top), np.log(y), np.log(x)))
    return np.r_[x[keep], xe], np.r_[y[keep], top]


def fig_beta():
    rows = list(csv.DictReader(open(os.path.join(OUT, "beta_scan.csv"))))
    fig, axs = plt.subplots(2, 2, figsize=(fs.WIDE, 5.1), sharey=True)
    fig.subplots_adjust(left=0.102, right=0.99, top=0.96, bottom=0.165, wspace=0.05, hspace=0.32)
    curves = (("LIN-UV", "1.0", "LIN-UV"), ("LIN-NL", "1.0", "base"), ("NL-A", "1.0", "base"),
              ("NL-B", "1.0", "base"), ("NL-A+F", "1.0", "F"), ("NL-B+F", "1.0", "F"),
              ("NL-A+F", "1.8", "puff"), ("NL-B+F", "1.8", "puff"))
    for ax, M in zip(axs.ravel(), (1.0, 1e2, 1e4, 1e6)):
        sub = [r for r in rows if abs(float(r["M_g"]) / M - 1) < 1e-6]
        b_old = float(sub[0]["beta_2012_08151"])
        for ch, xi, sty in curves:
            pts = sorted((float(r["beta"]), float(r["int_h2_damped"])) for r in sub
                         if r["channel"] == ch and r["xi"] == xi)
            bb, yy = np.array(pts).T
            ls, lw = STY[sty]
            if ch == "LIN-UV" and yy.max() > TOP_UV:
                bb, yy = rising_to(bb, yy, TOP_UV)
                b0 = np.exp(np.interp(np.log(TOP_UV / 10.0 ** 0.6), np.log(yy), np.log(bb)))
                ax.annotate("", xy=(bb[-1], yy[-1]), xytext=(b0, TOP_UV / 10.0 ** 0.6),
                            arrowprops=dict(arrowstyle="-|>", color=COL[ch], lw=lw, mutation_scale=8,
                                            shrinkA=0, shrinkB=0), zorder=3)
            ax.loglog(bb, yy, ls=ls, lw=lw if ch != "LIN-NL" else 1.2, color=COL[ch],
                      zorder=4 if ch == "LIN-NL" else 3)
        ax.axhline(fs.LIMIT, color="k", lw=0.7, ls="-.")
        ax.axvline(b_old, color="0.45", lw=0.7, ls=":")
        ax.text(b_old * 1.25, 2e-27, "published\nbound", fontsize=7, color="0.3", va="bottom")
        ax.text(0.03, 0.95, rf"$M_{{\rm in}}={'1' if M == 1 else f'10^{{{int(round(np.log10(M)))}}}'}$ g",
                transform=ax.transAxes, va="top", fontsize=8.5)
        ax.set_xlim(None, 1.6)
        ax.set_ylim(1e-28, 1e3)
        ax.set_yticks(10.0 ** np.arange(-28, 3, 4))
        ax.set_xlabel(r"$\beta_f$")
    axs[0, 0].text(1.15 * float([r for r in rows if r["M_g"] == "1.0e+00"][0]["beta"]), fs.LIMIT * 3,
                   r"$\Delta N_{\rm eff}<0.3$", fontsize=7, va="bottom")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\int \mathrm{d}\ln f\;\Omega_{\rm GW}h^2$ today")
    names = {"LIN-UV": "LIN-UV", "LIN-NL": "LIN-NL", "NL-A": "NL-A", "NL-B": "NL-B"}
    hl = []
    for ch, xi, sty in curves:
        ls, lw = STY[sty]
        lab = names.get(ch) or (ch.replace("+F", " with burst efficiency") + (", puff-up 1.8" if xi == "1.8" else ""))
        hl.append((Line2D([], [], color=COL[ch], ls=ls, lw=lw), lab))
    # filled column by column: rows LIN-UV, NL-A, NL-A+F, puff-up and LIN-NL, NL-B, NL-B+F, puff-up
    fig.legend([h for h, _ in hl], [l for _, l in hl], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6)
    fs.save(fig, "phase_a_beta_scan")


if __name__ == "__main__":
    fig_spectra()
    fig_beta()
    print("figures written to", OUT)
