"""
Table VIII of the paper: our damping scale k_D/aH against Domenech & Chluba (2503.13670).

Their Eq. (2.13) with the exponents of their text and Fig. 1 (their Eqs. 2.11 and 2.13 have
the sign of m flipped relative to their own text), neutrino-dominated viscosity:
    gamma = H/k_D = 1.2e-8 / sqrt(3 - m) (T/T_EW)^p (g_*/106.75)^(-1/4),
below T_EW: k_D ~ T^(5/2), so gamma ~ T^(-3/2), m = 2 (massive mediator, sigma ~ G_F^2 T^2);
above T_EW: k_D ~ T^(1/2), so gamma ~ T^(+1/2), m = -2 (gauge interaction, sigma ~ 1/T^2).
Ours: damping.kD_over_aH.  Writes ../results/damping_compare.txt.
(First run as a one-off script on 2026-10-04; saved here 2026-10-07 unchanged.)
"""
import io
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import damping as dm      # noqa: E402
import cosmology as co    # noqa: E402

TEW = 100.0


def dc_kD_over_H(T):
    g = co.g_star(T)
    if T < TEW:
        gam = 1.2e-8 / np.sqrt(3 - 2) * (T / TEW) ** (-1.5)
    else:
        gam = 1.2e-8 / np.sqrt(3 + 2) * (T / TEW) ** (0.5)
    return 1.0 / (gam * (g / 106.75) ** (-0.25))


if __name__ == "__main__":
    out = ["T [GeV]    ours k_D/aH    2503.13670    ours/theirs"]
    for T in (0.004, 0.01, 0.045, 0.3, 1.0, 10.0, 30.0, 79.0, 100.0, 1e3, 3e4, 1e5, 3e7, 3e10):
        a = float(dm.kD_over_aH(T))
        b = dc_kD_over_H(T)
        out.append(f"{T:9.3g} {a:13.3e} {b:13.3e} {a / b:12.3f}")
    print("\n".join(out))
    io.open(os.path.join(HERE, "..", "results", "damping_compare.txt"), "w", encoding="utf-8",
            newline="\n").write("\n".join(out) + "\n")
