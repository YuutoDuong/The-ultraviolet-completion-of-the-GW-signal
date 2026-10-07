"""Phase A figures from results/phase_a_spectra.npz and results/beta_scan.csv."""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase_a as pa   # noqa: E402
import detectors as dt  # noqa: E402

OUT = pa.OUT
LIMIT = 5.6e-6 * 0.3
COL = {"LIN-UV": "0.55", "LIN-NL": "tab:blue", "NL-A": "tab:red", "NL-B": "tab:orange", "SHOT": "tab:green",
       "NL-A+F": "tab:red", "NL-B+F": "tab:orange"}
LAB = {"NL-A+F": "NL-A with burst efficiency", "NL-B+F": "NL-B with burst efficiency"}
SNR = 10.0          # PLI curves drawn for SNR 10 in one year (detectors.py)


def detectors(ax):
    """Schmitz PLI curves (detectors.py), scaled to SNR 10 in one year."""
    for name in ("LISA", "DECIGO", "BBO", "ET", "CE"):
        f, o = dt.CURVES[name]
        ax.loglog(f, SNR * o, color="0.6", lw=0.8)
        i = np.argmin(o)
        ax.text(f[i], SNR * o[i] / 12, name, ha="center", va="top", fontsize=7, color="0.4")


def fig_spectra():
    d = np.load(os.path.join(OUT, "phase_a_spectra.npz"))
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)
    M, b = 1e4, 1.1e-6
    for ax, pop in zip(axs, ("mono", "choptuik")):
        key = f"{M:.0e}_{b:.2e}_{pop}_xi1.0"
        f = d[key + "_f"]
        detectors(ax)
        names = ("LIN-UV", "LIN-NL", "NL-A", "NL-B", "SHOT", "NL-A+F", "NL-B+F")
        for n in names:
            y = d[f"{key}_{n}_d"]
            ax.loglog(f, np.maximum(y, 1e-40), color=COL[n],
                      lw=1.4 if n.startswith("LIN") or n.endswith("+F") else 2,
                      ls="--" if n == "LIN-UV" else ("-." if n.endswith("+F") else "-"),
                      label=LAB.get(n, n) + " (damped)")
        if pop == "mono":
            k2 = f"{M:.0e}_{b:.2e}_{pop}_xi1.8"
            ax.loglog(d[k2 + "_f"], d[f"{k2}_NL-A+F_d"], color=COL["NL-A"], ls=":", lw=2,
                      label="NL-A with burst efficiency, puff 1.8 (damped)")
            # shock-limited sound (sound_lifetime.py): the lower end of the cluster signal
            ax.fill_between(f, np.maximum(d[f"{key}_NL-A+F-sh_d"], 1e-40), np.maximum(d[f"{key}_NL-A+F_d"], 1e-40),
                            color=COL["NL-A"], alpha=0.15, lw=0)
            ax.loglog(f, np.maximum(d[f"{key}_NL-A+F-sh_d"], 1e-40), color=COL["NL-A"], lw=1.0,
                      label="NL-A with burst efficiency, shock-limited sound (damped)")
            ax.loglog(f, np.maximum(d[f"{key}_LIN-NL-sh_d"], 1e-40), color=COL["LIN-NL"], lw=0.8, ls=":",
                      label="LIN-NL, shock-limited sound (damped)")
        ax.loglog(f, d[f"{key}_SHOT_u"], color=COL["SHOT"], ls=":", lw=1.2,
                  label="SHOT (undamped)" if pop != "mono" else None)
        ax.axhline(LIMIT, color="k", lw=0.8, ls="-.")
        ax.text(f[1], LIMIT * 2, r"$\Delta N_{\rm eff}<0.3$ (integrated)", fontsize=7)
        ax.set_xlabel("f [Hz]")
        ax.set_title(f"{'monochromatic' if pop == 'mono' else 'Choptuik (critical collapse)'}:  "
                     rf"$M_{{\rm in}}=10^4$ g, $\beta_f=1.1\times10^{{-6}}$ (published bound)", fontsize=9)
        ax.set_ylim(1e-32, 1e-3)
        ax.set_xlim(1e-3, 1e4)
    axs[0].set_ylabel(r"$\Omega_{\rm GW}h^2$ today")
    h0, l0 = axs[0].get_legend_handles_labels()
    h1, l1 = axs[1].get_legend_handles_labels()
    extra = [(h, l) for h, l in zip(h1, l1) if l not in l0]
    fig.legend(h0 + [h for h, _ in extra], l0 + [l for _, l in extra], loc="lower center", ncol=4, fontsize=7.5,
               frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"phase_a_spectra.{ext}"), dpi=160)
    plt.close(fig)


def fig_beta():
    rows = list(csv.DictReader(open(os.path.join(OUT, "beta_scan.csv"))))
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
    for ax, M in zip(axs, (1.0, 1e2, 1e4, 1e6)):
        sub = [r for r in rows if abs(float(r["M_g"]) / M - 1) < 1e-6]
        b_old = float(sub[0]["beta_2012_08151"])
        for (ch, xi, ls) in (("LIN-UV", "1.0", "--"), ("LIN-NL", "1.0", "-"), ("NL-A", "1.0", "-"),
                             ("NL-B", "1.0", "-"), ("NL-A+F", "1.0", "-."), ("NL-A+F", "1.8", ":"),
                             ("NL-B+F", "1.0", "-."), ("NL-B+F", "1.8", ":")):
            pts = [(float(r["beta"]), float(r["int_h2_damped"])) for r in sub
                   if r["channel"] == ch and r["xi"] == xi]
            if not pts:
                continue
            bb, yy = np.array(pts).T
            ax.loglog(bb, yy, ls, color=COL[ch], lw=1.4 if ch.startswith("LIN") or ch.endswith("+F") else 2,
                      label=LAB.get(ch, ch) + (", puff 1.8" if xi == "1.8" else ""))
        ax.axhline(LIMIT, color="k", lw=0.8, ls="-.")
        ax.axvline(b_old, color="0.4", lw=0.8, ls=":")
        ax.text(b_old * 1.2, 1e-26, "published\nbound", fontsize=7, color="0.3")
        ax.set_title(rf"monochromatic, $M_{{\rm in}}=10^{{{int(round(np.log10(M)))}}}$ g", fontsize=9)
        ax.set_xlabel(r"$\beta_f$")
        ax.set_ylim(1e-28, 1e3)
    axs[0].set_ylabel(r"$\int d\ln f\,\Omega_{\rm GW}h^2$ (damped)")
    axs[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"phase_a_beta_scan.{ext}"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_spectra()
    fig_beta()
    print("figures written to", OUT)
