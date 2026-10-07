"""
Draft paper figures 1-4 (OUTLINE.md figure table):
 1  scales vs M_in along the 2012.08151 bound: k_NL, k_cl, k_D, k_PBH in units of k_eva
 2  coherent and shot-noise suppression factors vs k for several mass functions
 3  radiation spectra handed over at evaporation, fiducial point, by channel
 4  sound-energy budget K vs beta at 10^4 g: linear extrapolation vs clusters
Writes ../results/fig1_scales, fig2_suppression, fig3_spectra, fig4_energy (.png/.pdf).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase_a as pa          # noqa: E402
import cosmology as co        # noqa: E402
import clusters as cl         # noqa: E402
import suddenness as sd       # noqa: E402
import damping as dm          # noqa: E402
import acoustic_gw as ag      # noqa: E402

OUT = pa.OUT


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=160)
    plt.close(fig)


def fig1():
    Ms = np.geomspace(1.0, 3e8, 36)
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
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(a[:, 0], a[:, 5], "k-", lw=2, label=r"$k_{\rm PBH}$ (PBH spacing)")
    ax.loglog(a[:, 0], a[:, 4], "tab:purple", lw=2, label=r"$k_D$ (diffusion damping)")
    ax.loglog(a[:, 0], a[:, 3], "tab:red", lw=2, label=r"$k_{\rm cl}$ (typical cluster)")
    ax.loglog(a[:, 0], a[:, 2], "tab:blue", lw=2, label=r"$k_{\rm NL}$ (non-linear)")
    ax.loglog(a[:, 0], a[:, 6], "0.5", lw=1, ls="--", label=r"$k_{\rm eq}$")
    ax.axhline(1.0, color="0.5", lw=1, ls=":")
    ax.set_xlabel(r"$M_{\rm in}$ [g]  (on the published bound)")
    ax.set_ylabel(r"$k/k_{\rm eva}$")
    ax.fill_between(a[:, 0], 1, a[:, 4], where=a[:, 4] < a[:, 3], color="tab:purple", alpha=0.12)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Scales at evaporation; shaded: cluster sound damped", fontsize=9)
    save(fig, "fig1_scales")
    np.savetxt(os.path.join(OUT, "fig1_scales.txt"), a,
               header="M_g T_evap x_NL x_cl x_D x_PBH x_eq   (x = k/k_eva)")


def fig2():
    x = np.geomspace(1.0, 1e6, 300)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(x, sd.S_mono(x), "k-", lw=2.5, label="monochromatic (exact, this work)")
    ax.loglog(x, (np.sqrt(2 / 3) * x) ** (-1 / 3), "k:", lw=1, label=r"Inomata et al.: $(\sqrt{2/3}\,x)^{-1/3}$")
    for sg, col in ((1e-4, "tab:orange"), (1e-2, "tab:olive"), (0.1, "tab:cyan")):
        m = sd.LogNormalMF(sg, truncate=None).tabulate(x[0], x[-1])
        ax.loglog(x, np.maximum(m.S_coh(x), 1e-30), color=col, lw=1.5, label=rf"log-normal $\sigma={sg:g}$ (coherent)")
        mt = sd.LogNormalMF(sg).tabulate(x[0], x[-1])
        ax.loglog(x, np.maximum(mt.S_coh(x), 1e-30), color=col, lw=0.8, ls=":")
    ch = sd.choptuik()
    ax.loglog(x, ch.S_coh(x), "tab:red", lw=2, label="Choptuik (coherent)")
    ax.loglog(x, 2.3 * x ** (-4 / 3), "tab:red", ls=":", lw=1, label=r"2605.21474: $2.3\,x^{-4/3}$")
    ax.loglog(x, np.sqrt(ch.S_inc2(x)), "tab:green", lw=1.8, ls="--",
              label="shot noise, Choptuik (single-hole law, any width)")
    ax.text(1.3e3, 2e-9, r"dotted: log-normal cut at $5\sigma$ (edge tail $\propto x^{-4/3}$)", fontsize=7)
    ax.set_ylim(1e-9, 2)
    ax.set_xlabel(r"$x = k/k_{\rm eva}$")
    ax.set_ylabel("sound amplitude / sudden limit")
    fig.set_size_inches(6.4, 5.6)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig2_suppression.{ext}"), dpi=160)
    plt.close(fig)


def fig3():
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, pop in zip(axs, ("mono", "choptuik")):
        ch, info = pa.channel_spectra(1e4, 1e-6, pop)
        chF, _ = pa.channel_spectra(1e4, 1e-6, pop, hydro="energy")
        z = np.geomspace(1.0, 3 * info["xPBH"], 600)
        for name, col, ls in (("LIN-UV", "0.55", "--"), ("NL-A", "tab:red", "-"), ("NL-B", "tab:orange", "-"),
                              ("SHOT", "tab:green", "-")):
            ax.loglog(z, np.maximum(ch[name][0](z), 1e-40), color=col, ls=ls, lw=2, label=name)
        for name, col in (("NL-A", "tab:red"), ("NL-B", "tab:orange")):
            ax.loglog(z, np.maximum(chF[name][0](z), 1e-40), color=col, ls="-.", lw=1.3,
                      label=name + " with burst efficiency (module F)")
        xD = float(dm.kD_over_aH(info["s"]["T_evap"]))
        for xv, lab, ha in ((info["xNL"], r"$k_{\rm NL}$", "right"), (info["xcl"], r"$k_{\rm cl}$", "right"),
                            (xD, r"$k_D$", "right" if xD < info["xPBH"] else "left"),
                            (info["xPBH"], r"$k_{\rm PBH}$", "left" if xD < info["xPBH"] else "right")):
            ax.axvline(xv, color="0.7", lw=0.8, ls=":")
            ax.text(xv, 3e3, lab, fontsize=8, rotation=90, va="top", ha=ha)
        ax.set_title(f"{'monochromatic' if pop == 'mono' else 'Choptuik'}: $10^4$ g, $\\beta_f=10^{{-6}}$",
                     fontsize=9)
        ax.set_xlabel(r"$k/k_{\rm eva}$")
        ax.set_ylim(1e-14, 1e5)
    axs[0].set_ylabel(r"$\mathcal{P}_{\delta_r}(k)$ handed to radiation")
    axs[0].legend(fontsize=7)
    save(fig, "fig3_spectra")


def fig4():
    M = 1e4
    bc = co.beta_crit(M)
    betas = np.geomspace(3 * bc, 3e-2, 14)
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
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for n, col, ls in (("LIN-UV", "0.4", "--"), ("LIN-NL", "tab:blue", "-"), ("NL-A", "tab:red", "-"),
                       ("NL-B", "tab:orange", "-"), ("NL-A+F", "tab:red", "-."), ("NL-B+F", "tab:orange", "-.")):
        ax.loglog(betas, K[n], color=col, ls=ls, lw=2 if "+F" not in n else 1.5,
                  label=n.replace("+F", " with burst efficiency"))
    ax.axhline(1.0, color="k", lw=1)
    ax.text(betas[0], 1.6, "sound energy = total energy", fontsize=8)
    ax.axvline(1.1e-6, color="0.5", ls=":", lw=1)
    ax.text(1.2e-6, 1e-5, "published\nbound", fontsize=8, color="0.3")
    ax.set_xlabel(r"$\beta_f$  ($M_{\rm in}=10^4$ g, monochromatic)")
    ax.set_ylabel(r"$K = \Gamma\langle v^2\rangle$ in the sound field")
    ax.legend(fontsize=8)
    save(fig, "fig4_energy")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        f()
        print(f.__name__, "done", flush=True)
