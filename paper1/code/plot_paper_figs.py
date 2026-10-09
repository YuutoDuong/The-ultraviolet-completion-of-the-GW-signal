"""
Paper figures 1, 2, 3 and 5:
 1  scales vs M_in along the 2012.08151 bound: k_NL, k_cl, k_D, k_PBH in units of k_eva
 2  coherent and shot-noise suppression factors vs k for several mass functions
 3  radiation spectra handed over at evaporation, fiducial point, by channel
 5  sound-energy budget K vs beta at 10^4 g: linear extrapolation vs clusters
Writes ../results/fig1_scales, fig2_suppression, fig3_spectra, fig4_energy (.png/.pdf) and
the data of Figs. 1 and 5 (fig1_scales.txt, fig4_energy.txt).  Style: figstyle.py.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs         # noqa: E402
import phase_a as pa          # noqa: E402
import cosmology as co        # noqa: E402
import clusters as cl         # noqa: E402
import suddenness as sd       # noqa: E402
import damping as dm          # noqa: E402
import acoustic_gw as ag      # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.patches import Patch     # noqa: E402

fs.use()
OUT = fs.OUT
COL = fs.COL
PURPLE = "#AA3377"


def fig1():
    Ms = np.geomspace(1.0, 3e8, 300)
    rows = []
    for M in Ms:
        b = 1.1e-6 * (M / 1e4) ** (-17 / 24)
        c = cl.Clusters(M, b)
        s = c.sc
        xNL = float(np.exp(np.interp(0.0, np.log(c.P_lin(np.geomspace(c.k_eva, c.k_PBH, 400))),
                                     np.log(np.geomspace(1.0, c.k_PBH / c.k_eva, 400)))))
        rows.append((M, s["T_evap"], xNL, 1.0 / (c.R_cl(c.N_star) * c.k_eva),
                     float(dm.kD_over_aH(s["T_evap"])), c.k_PBH / c.k_eva, s["k_eq"] / c.k_eva))
    a = np.array(rows)
    fig, ax = plt.subplots(figsize=(fs.NARROW, 3.0))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.14)
    ax.fill_between(a[:, 0], 1, a[:, 4], where=a[:, 4] < a[:, 3], color=PURPLE, alpha=0.12, lw=0)
    ax.loglog(a[:, 0], a[:, 5], color="k", lw=1.5)
    ax.loglog(a[:, 0], a[:, 4], color=PURPLE, lw=1.5)
    ax.loglog(a[:, 0], a[:, 3], color=COL["NL-A"], lw=1.5)
    ax.loglog(a[:, 0], a[:, 2], color=COL["LIN-NL"], lw=1.5)
    ax.loglog(a[:, 0], a[:, 6], color="0.45", lw=1.0, ls="--")
    ax.axhline(1.0, color="0.5", lw=0.7, ls=":")
    ax.set_xlim(1.0, 3e8)
    ax.set_ylim(0.3, 3e10)
    ax.set_xlabel(r"$M_{\rm in}$ [g] on the published bound")
    ax.set_ylabel(r"$k/k_{\rm eva}$")
    h = [Line2D([], [], color="k", lw=1.5), Line2D([], [], color=PURPLE, lw=1.5),
         Line2D([], [], color=COL["NL-A"], lw=1.5), Line2D([], [], color=COL["LIN-NL"], lw=1.5),
         Line2D([], [], color="0.45", lw=1.0, ls="--"), Patch(color=PURPLE, alpha=0.12, lw=0)]
    ax.legend(h, [r"$k_{\rm PBH}$ (PBH spacing)", r"$k_D$ (diffusion damping)", r"$k_{\rm cl}$ (typical cluster)",
                  r"$k_{\rm NL}$ (non-linear)", r"$k_{\rm eq}$", "cluster sound damped"], loc="upper left")
    fs.save(fig, "fig1_scales")
    np.savetxt(os.path.join(OUT, "fig1_scales.txt"), a,
               header="M_g T_evap x_NL x_cl x_D x_PBH x_eq   (x = k/k_eva)")


def fig2():
    x = np.geomspace(1.0, 1e6, 2000)
    fig, ax = plt.subplots(figsize=(fs.NARROW, 3.65))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.98, bottom=0.335)
    h, lab = [], []

    def add(handle, text):
        h.append(handle)
        lab.append(text)
    add(ax.loglog(x, sd.S_mono(x), "k-", lw=1.8)[0], "monochromatic (exact, this work)")
    add(ax.loglog(x, (np.sqrt(2 / 3) * x) ** (-1 / 3), "k:", lw=0.9)[0],
        r"Inomata et al.: $(\sqrt{2/3}\,x)^{-1/3}$")
    for sg, col, txt in ((1e-4, COL["NL-B"], r"10^{-4}"), (1e-2, fs.MAGENTA, r"10^{-2}"), (0.1, fs.CYAN, "0.1")):
        m = sd.LogNormalMF(sg, truncate=None).tabulate(x[0], x[-1])
        add(ax.loglog(x, np.maximum(m.S_coh(x), 1e-30), color=col, lw=1.3)[0],
            rf"log-normal, $\sigma={txt}$ (coherent)")
        mt = sd.LogNormalMF(sg).tabulate(x[0], x[-1])
        ax.loglog(x, np.maximum(mt.S_coh(x), 1e-30), color=col, lw=0.8, ls=":")
    ch = sd.choptuik()
    add(ax.loglog(x, ch.S_coh(x), color=COL["NL-A"], lw=1.5)[0], "critical collapse (coherent)")
    xg = x[x >= 2.0]                     # a large-x fit: from x = 2, where it is below 1
    add(ax.loglog(xg, 2.3 * xg ** (-4 / 3), color=COL["NL-A"], ls=":", lw=0.9)[0],
        r"Gouttenoire et al.: $2.3\,x^{-4/3}$")
    add(ax.loglog(x, np.sqrt(ch.S_inc2(x)), color=COL["SHOT"], lw=1.5, ls="--")[0],
        "shot noise (any width)")
    add(Line2D([], [], color="0.4", lw=0.8, ls=":"), r"log-normals cut at $5\sigma$")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(1e-9, 3)
    ax.set_xlabel(r"$x = k/k_{\rm eva}$")
    ax.set_ylabel("sound amplitude / sudden limit")
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.53, 0.0), fontsize=7,
               columnspacing=1.0, handlelength=2.2)
    fs.save(fig, "fig2_suppression")


def fig3():
    fig, axs = plt.subplots(1, 2, figsize=(fs.WIDE, 3.05), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.865, bottom=0.145, wspace=0.05)
    for ax, pop in zip(axs, ("mono", "choptuik")):
        ch, info = pa.channel_spectra(1e4, 1e-6, pop)
        chF, _ = pa.channel_spectra(1e4, 1e-6, pop, hydro="energy")
        z = np.geomspace(1.0, 3 * info["xPBH"], 4000)
        xD = float(dm.kD_over_aH(info["s"]["T_evap"]))
        # the scales, labelled just above the frame.  The panel ends at k_PBH, where every spectrum
        # ends; k_D is 10% below it at 10^4 g, so the two share the label at the right edge.
        for xv, lab, ha in ((info["xNL"], r"$k_{\rm NL}$", "center"), (info["xcl"], r"$k_{\rm cl}$", "center"),
                            (info["xPBH"], r"$k_D,\,k_{\rm PBH}$", "right")):
            ax.text(xv, 1.015, lab, transform=ax.get_xaxis_transform(), fontsize=7.5, ha=ha, va="bottom")
        for xv in (info["xNL"], info["xcl"], xD):
            ax.axvline(xv, color="0.65", lw=0.6, ls=":", zorder=1)
        for name, ls, lw in (("LIN-UV", "--", 1.2), ("NL-B", "-", 1.5), ("NL-A", "-", 1.5), ("SHOT", "-", 1.5)):
            ax.loglog(z, fs.until_cutoff(z, ch[name][0](z)), color=COL[name], ls=ls, lw=lw, label=name, zorder=3)
        for name in ("NL-A", "NL-B"):
            ax.loglog(z, fs.until_cutoff(z, chF[name][0](z)), color=COL[name], ls="-.", lw=1.1, zorder=4,
                      label=name + " with burst efficiency")
        ax.set_title(("monochromatic" if pop == "mono" else "critical collapse (Choptuik)") +
                     r", $M_{\rm in}=10^4$ g, $\beta_f=10^{-6}$", pad=14)
        ax.set_xlabel(r"$k/k_{\rm eva}$")
        ax.set_xlim(0.5, info["xPBH"])   # the panel ends at k_PBH, where the spectra end
        ax.set_ylim(1e-19, 1e5)          # low enough that every spectrum is seen to end at k_PBH
        ax.set_yticks(10.0 ** np.arange(-16, 5, 4))
        ax.set_yticks(10.0 ** np.arange(-19, 6), minor=True)
        ax.set_yticklabels([], minor=True)
    axs[0].set_ylabel(r"$\mathcal{P}_{\delta_r}(k)$ handed to radiation")
    h, l = axs[0].get_legend_handles_labels()
    order = [l.index(n) for n in ("LIN-UV", "NL-A", "NL-B", "SHOT", "NL-A with burst efficiency",
                                  "NL-B with burst efficiency")]
    axs[1].legend([h[i] for i in order], [l[i] for i in order], loc="upper left", fontsize=7)
    fs.save(fig, "fig3_spectra")


def fig4():
    M = 1e4
    bc = co.beta_crit(M)
    betas = np.geomspace(3 * bc, 3e-2, 66)
    K = {n: [] for n in ("LIN-UV", "NL-A", "NL-B", "LIN-NL", "NL-A+F", "NL-B+F")}
    for b in betas:
        ch, info = pa.channel_spectra(M, b, "mono")
        chF, _ = pa.channel_spectra(M, b, "mono", hydro="energy")     # energy, not GW-equivalent
        ch.update({n + "+F": chF[n] for n in ("NL-A", "NL-B")})
        z = np.exp(np.linspace(np.log(0.5), np.log(1.2 * info["xPBH"]), 6000))
        for n in K:
            K[n].append(ag.GAM * ag.PV_OVER_PDELTA * np.trapezoid(ch[n][0](z), np.log(z)))
    np.savetxt(os.path.join(OUT, "fig4_energy.txt"), np.column_stack([betas] + [K[n] for n in K]),
               header="beta " + " ".join(K))
    fig, ax = plt.subplots(figsize=(fs.NARROW, 3.0))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.14)
    for n, ls, lw in (("LIN-UV", "--", 1.2), ("LIN-NL", "-", 1.5), ("NL-B", "-", 1.5), ("NL-A", "-", 1.5),
                      ("NL-B+F", "-.", 1.1), ("NL-A+F", "-.", 1.1)):
        ax.loglog(betas, K[n], color=COL[n], ls=ls, lw=lw, label=n.replace("+F", " with burst efficiency"))
    ax.axhline(1.0, color="k", lw=0.7)
    ax.text(betas[0] * 1.3, 1.8, "sound energy = total energy", fontsize=7, va="bottom")
    ax.axvline(1.1e-6, color="0.45", ls=":", lw=0.7)
    ax.text(1.3e-6, 1.5e-6, "published\nbound", fontsize=7, color="0.3", va="bottom")
    ax.set_xlim(betas[0], betas[-1])     # the curves span the frame
    ax.set_ylim(3e-7, 3e15)
    ax.set_yticks(10.0 ** np.arange(-6, 16, 3))
    ax.set_yticks(10.0 ** np.arange(-6, 16), minor=True)
    ax.set_yticklabels([], minor=True)
    ax.set_xlabel(r"$\beta_f$ (monochromatic, $M_{\rm in}=10^4$ g)")
    ax.set_ylabel(r"$K = \Gamma\langle v^2\rangle$ in the sound field")
    h, l = ax.get_legend_handles_labels()
    order = [l.index(n) for n in ("LIN-UV", "LIN-NL", "NL-A", "NL-B", "NL-A with burst efficiency",
                                  "NL-B with burst efficiency")]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", fontsize=7)
    fs.save(fig, "fig4_energy")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        f()
        print(f.__name__, "done", flush=True)
