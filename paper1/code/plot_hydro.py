"""Figure 4: module F sound efficiency, from results/hydro_scan.txt and hydro_profiles.npz.

(a) kappa(delta0) for sudden top hats (t = 7R, 25R), the fit of burst_efficiency.py, and
    the three monochromatic release histories placed at delta_eff = A_SRC delta_tot;
(b) kappa(k) across the shell spectrum (hydro_spectrum.py);
(c) shell profiles at t = 7R, normalised to the release.
The colour of a burst strength delta0 is the same in (b) and (c).  Style: figstyle.py.
"""
import os
import re
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs             # noqa: E402
import burst_efficiency as be     # noqa: E402
import hydro_spectrum as hs       # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402

fs.use()
OUT = fs.OUT
txt = open(os.path.join(OUT, "hydro_scan.txt")).read()
rows = re.findall(r"^\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", txt, flags=re.M)
d = np.array([[float(v) for v in r] for r in rows])
src = re.findall(r"^(.+?)\s+x_cl=\S+\s+delta_tot=\s*([0-9.]+)\s+kappa\(7R\)=([0-9.]+)\s+kappa\(25R\)=([0-9.]+)",
                 txt, flags=re.M)
prof = np.load(os.path.join(OUT, "hydro_profiles.npz"))


def delta_colour(d0):
    """One colour per burst strength, ordered: delta0 = 0.1 .. 100."""
    return plt.cm.viridis(0.75 * (np.log10(d0) + 1.0) / 3.0)


def history_label(lab):
    """'fiducial 1e4 g, puff 1.8' -> '10^4 g, puff-up 1.8'."""
    lab = re.sub(r"^(fiducial|saturated)\s+", "", lab.strip())
    lab = re.sub(r"1e(\d+) g", r"$10^{\1}$ g", lab)
    return lab.replace(", no puff", "").replace("puff 1.8", "puff-up 1.8").replace(" (beta=1e-2)", r", $\beta_f=10^{-2}$")


fig, axs = plt.subplots(1, 3, figsize=(fs.WIDE, 2.55))
fig.subplots_adjust(left=0.065, right=0.99, top=0.91, bottom=0.17, wspace=0.32)

ax = axs[0]
dd = np.geomspace(5e-3, 300, 400)
ax.axhline(1.0, color="0.6", lw=0.6)
ax.semilogx(dd, be.kappa_tophat(dd), "k-", lw=1.0, label="fit")
ax.semilogx(d[:, 0], d[:, 1], "o", color=fs.COL["NL-A"], ms=3.6, label=r"top hat, $t=7R$")
ax.semilogx(d[:, 0], d[:, 2], "s", color=fs.COL["LIN-NL"], mfc="none", ms=4.2, mew=0.8, label=r"top hat, $t=25R$")
for (lab, dt, k7, k25), mk in zip(src, ("*", "P", "X")):
    ax.semilogx(be.A_SRC * float(dt), float(k25), mk, color=fs.COL["SHOT"], ms=6.5, mew=0.4, mec="k",
                label=history_label(lab))
ax.set_xlim(5e-3, 300)
ax.set_ylim(-0.04, 1.12)       # a margin below 0: kappa -> 0 at strong bursts
ax.set_xlabel(r"local radiation contrast $\delta$")
ax.set_ylabel(r"$\kappa = E_{\rm ac}/E_{\rm ac}^{\rm linear}$")
ax.legend(loc="lower left", fontsize=6.3, handlelength=1.6, labelspacing=0.25, borderaxespad=0.3)
ax.set_title("efficiency of a burst")

ax = axs[1]
k = np.geomspace(0.05, 3.3, 600)
e_lin, kap = hs.kappa_of_k(k, prof)
for d0 in (1.0, 3.0, 10.0, 30.0):
    ax.semilogx(k, kap[d0], color=delta_colour(d0), lw=1.3, label=rf"$\delta_0={d0:g}$")
ax2 = ax.twinx()
ax2.semilogx(k, k ** 3 * e_lin / np.max(k ** 3 * e_lin), color="0.55", lw=0.9, ls="--")
ax2.set_ylim(-0.04, 1.1)
ax2.set_yticks([])
ax.set_xlim(0.05, 3.3)           # kappa(k) is a ratio to the top-hat template, which vanishes near kR = 4.5
ax.set_ylim(-0.04, 1.1)
ax.set_xlabel(r"$kR$")
ax.set_ylabel(r"$\kappa(k)$ at $t=7R$")
ax.legend(loc="lower left", fontsize=6.3, handlelength=1.6, labelspacing=0.25, borderaxespad=0.3)
ax.set_title("scale dependence")      # dashed: the linear shell power per ln k (caption)

ax = axs[2]
for key in prof.files:
    if key.startswith("top_"):
        d0 = float(key[4:])
        if d0 in (0.1, 1.0, 10.0, 100.0):
            r, rho, v = prof[key]
            ax.plot(r, (rho - 1.0) / d0, lw=1.1, color=delta_colour(d0), label=rf"$\delta_0={d0:g}$")
ax.axhline(0.0, color="0.6", lw=0.5, zorder=0)
ax.set_xlim(0, 12)
ax.set_xlabel(r"$r/R_{\rm cl}$")
ax.set_ylabel(r"$\delta\rho/(\rho_0\,\delta_0)$ at $t = 7R$")
ax.legend(loc="lower right", fontsize=6.3, handlelength=1.6, labelspacing=0.25, borderaxespad=0.3)
ax.set_title("shell profiles")
fs.save(fig, "hydro_efficiency")
print("written")
