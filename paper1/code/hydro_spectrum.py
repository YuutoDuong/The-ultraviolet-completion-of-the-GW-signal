"""
Scale dependence of the module F sound efficiency: kappa(k) from the saved profiles.

For linear sound in a p = rho/3 fluid, d_t delta = -(4/3) div v and d_t v = -(1/4) grad delta,
so each Fourier mode conserves e(k) = |delta~(k)|^2 + (16/3) |v~_L(k)|^2.  Spherical profiles:
    delta~(k) = 4 pi int r^2 delta(r) j0(kr) dr,   v~_L(k) = 4 pi int r^2 v(r) j1(kr) dr.
kappa(k) = e_NL(k) / (e_lin(k) delta0^2/delta_lin^2), with the delta0 = 0.01 top hat as the
linear template (kappa = 0.996 there).  Profiles are at t = 7R (hydro_scan.py).  Uses the
rest-frame rho, like hydro1d.acoustic_energy, so kappa(k -> 0) is not the conserved
monopole for strong bursts; the main band kR ~ 0.3-1.5 is what the GW signal sees.
Writes ../results/hydro_spectrum.txt.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
DELTAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def e_of_k(r, rho, v, k):
    d = rho - 1.0
    dr = np.gradient(r)
    kr = np.outer(k, r)
    j0 = np.sinc(kr / np.pi)
    j1 = np.where(kr > 1e-6, (np.sin(kr) / np.maximum(kr, 1e-6) - np.cos(kr)) / np.maximum(kr, 1e-6), kr / 3)
    dk = 4 * np.pi * (j0 * (r ** 2 * d * dr)).sum(axis=1)
    vk = 4 * np.pi * (j1 * (r ** 2 * v * dr)).sum(axis=1)
    return dk ** 2 + (16.0 / 3.0) * vk ** 2


def kappa_of_k(k, prof=None):
    """(e_lin(k) per delta^2, {delta0: kappa(k)}) for the saved top hats."""
    prof = prof if prof is not None else np.load(os.path.join(OUT, "hydro_profiles.npz"))
    r0, rho0, v0 = prof["top_0.01"]
    e_lin = e_of_k(r0, rho0, v0, k) / 0.01 ** 2
    out = {}
    for d0 in DELTAS:
        r, rho, v = prof[f"top_{d0}"]
        out[d0] = e_of_k(r, rho, v, k) / (d0 ** 2 * e_lin)
    return e_lin, out


def main():
    k = np.geomspace(0.05, 30.0, 60)          # in units of 1/R
    e_lin, kap = kappa_of_k(k)
    lines = ["kappa(k) for sudden top hats at t = 7R; k in 1/R; linear template delta0 = 0.01",
             f"{'kR':>7s} " + " ".join(f"{d:>8g}" for d in DELTAS) + "   e_lin"]
    for i, kk in enumerate(k):
        if i % 3 == 0:
            lines.append(f"{kk:7.3f} " + " ".join(f"{kap[d][i]:8.4f}" for d in DELTAS) + f"   {e_lin[i]:.3e}")
    w = e_lin * k ** 3
    lines.append("energy-weighted mean over k^3 e_lin dlnk: " +
                 " ".join(f"{np.trapezoid(kap[d] * w, np.log(k)) / np.trapezoid(w, np.log(k)):.4f}" for d in DELTAS))
    print("\n".join(lines))
    with open(os.path.join(OUT, "hydro_spectrum.txt"), "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
