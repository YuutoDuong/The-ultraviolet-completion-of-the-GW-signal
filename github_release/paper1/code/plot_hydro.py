"""Figure: module F sound efficiency, from results/hydro_scan.txt and hydro_profiles.npz.

(a) kappa(delta0) for sudden top hats (t = 7R, 25R), the fit of burst_efficiency.py, and
    the three monochromatic release histories placed at delta_eff = A_SRC delta_tot;
(b) kappa(k) across the shell spectrum (hydro_spectrum.py);
(c) shell profiles at t = 7R, normalised to the release.
"""
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import burst_efficiency as be     # noqa: E402
import hydro_spectrum as hs       # noqa: E402

OUT = os.path.join(HERE, "..", "results")
txt = open(os.path.join(OUT, "hydro_scan.txt")).read()
rows = re.findall(r"^\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", txt, flags=re.M)
d = np.array([[float(v) for v in r] for r in rows])
src = re.findall(r"^(.+?)\s+x_cl=\S+\s+delta_tot=\s*([0-9.]+)\s+kappa\(7R\)=([0-9.]+)\s+kappa\(25R\)=([0-9.]+)",
                 txt, flags=re.M)
prof = np.load(os.path.join(OUT, "hydro_profiles.npz"))

fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
ax = axs[0]
ax.semilogx(d[:, 0], d[:, 1], "o", color="tab:red", label=r"top hat, $t=7R$")
ax.semilogx(d[:, 0], d[:, 2], "s", color="tab:blue", mfc="none", label=r"top hat, $t=25R$")
dd = np.geomspace(5e-3, 200, 200)
ax.semilogx(dd, be.kappa_tophat(dd), "k-", lw=1,
            label=rf"fit $[1+(\delta/{be.D1})^{{{be.P}}}]^{{-1}}$")
for (lab, dt, k7, k25), mk in zip(src, ("*", "P", "X")):
    ax.semilogx(be.A_SRC * float(dt), float(k25), mk, color="tab:green", ms=10,
                label=rf"release history: {lab.strip()}, $\delta_{{\rm tot}}={float(dt):.1f}$")
ax.axhline(1.0, color="0.5", lw=0.8)
ax.set_xlabel(r"local radiation contrast $\delta$ (release histories at $0.21\,\delta_{\rm tot}$)")
ax.set_ylabel(r"$\kappa = E_{\rm ac}/E_{\rm ac}^{\rm linear}$")
ax.set_ylim(0, 1.15)
ax.legend(fontsize=6.5)
ax.set_title("Sound efficiency of a cluster burst", fontsize=9)

ax = axs[1]
k = np.geomspace(0.05, 3.3, 200)
e_lin, kap = hs.kappa_of_k(k, prof)
for d0, col in ((1.0, "tab:blue"), (3.0, "tab:orange"), (10.0, "tab:red"), (30.0, "tab:purple")):
    ax.semilogx(k, kap[d0], color=col, lw=1.5, label=rf"$\delta_0={d0:g}$")
ax2 = ax.twinx()
ax2.semilogx(k, k ** 3 * e_lin / np.max(k ** 3 * e_lin), color="0.6", lw=1, ls="--")
ax2.set_yticks([])
ax.text(0.06, 0.05, "dashed: linear shell power per ln k", fontsize=7, color="0.4")
ax.set_xlabel(r"$kR$")
ax.set_ylabel(r"$\kappa(k)$ at $t=7R$")
ax.set_ylim(0, 1.1)
ax.legend(fontsize=7, loc="upper right")
ax.set_title("Scale dependence across the shell spectrum", fontsize=9)

ax = axs[2]
for key in prof.files:
    if key.startswith("top_"):
        d0 = float(key[4:])
        if d0 in (0.1, 1.0, 10.0, 100.0):
            r, rho, v = prof[key]
            ax.plot(r, (rho - 1.0) / d0, lw=1.5, label=rf"$\delta_0={d0:g}$")
ax.set_xlim(0, 12)
ax.set_xlabel(r"$r/R_{\rm cl}$")
ax.set_ylabel(r"$\delta\rho/(\rho_0\,\delta_0)$ at $t = 7R$")
ax.legend(fontsize=8)
ax.set_title("Shell profiles, normalised to the release", fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"hydro_efficiency.{ext}"), dpi=160)
print("written")
