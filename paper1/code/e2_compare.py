"""
E2 validation: the induced-GW kernel (sigw_kernel.py) vs (i) 2409.12125's resonant
formula, (ii) its "factor 2" statement, (iii) our sound-shell result (acoustic_gw.py),
all on the same linear monochromatic input.  Writes ../results/e2_compare.txt.
"""
import os
import sys
import numpy as np
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sigw_kernel as sk     # noqa: E402
import acoustic_gw as ag     # noqa: E402

CS = sk.CS
lines = []


def say(s):
    print(s)
    lines.append(s)


# ---------------------------------------------------- E2-1: closed forms of C, S
for q, c in ((0.3, 5.0), (-0.7, 2.0), (1e-3, 50.0), (2.0, 300.0)):
    Cq, Sq = sk._CS_int(np.array([q]), c)
    Cn = quad(lambda x: np.cos(q * x) / (x + c), 0, np.inf, weight=None, limit=2000)[0] if False else None
    # oscillatory quadrature (QAWF) for the Fourier integrals
    Cn = quad(lambda x: 1.0 / (x + c), 0, np.inf, weight="cos", wvar=abs(q))[0]
    Sn = np.sign(q) * quad(lambda x: 1.0 / (x + c), 0, np.inf, weight="sin", wvar=abs(q))[0]
    say(f"E2-1 q={q:+.3g} c={c:g}:  C closed {Cq[0]:+.6e} quad {Cn:+.6e} | S closed {Sq[0]:+.6e} quad {Sn:+.6e}")

# ---------------------------------------------------- input: linear monochromatic
C, x_eq, x_uv = 1.0, 1.0, 1.0e4
TS = lambda kap: 1.0 / (5.0 + kap ** 2 / C)          # limits 1/5 and C/kappa^2
Sinom = lambda z: (np.sqrt(2.0 / 3.0) * z) ** (-1.0 / 3.0)
P_Phi = lambda z: TS(z / x_eq) ** 2 * (2 / (3 * np.pi)) * (z / x_uv) ** 3 * Sinom(z) ** 2
P_delta = lambda z: (4.0 / 9.0) * np.asarray(z) ** 4 * P_Phi(z)


def theta_uv(xr):
    """2409.12125 Eq. (E7) by quadrature; xr = k/k_uv."""
    r = 1.0 / xr
    if r >= (1 + 1 / CS) / 2:
        s0 = 1.0
    elif r >= 1 / (2 * CS):
        s0 = 2 * r - 1 / CS
    else:
        return 0.0
    return quad(lambda s: (s * s - 1) ** 2 / (1 - CS ** 2 * s * s) ** (5 / 3), -s0, s0)[0]


peak = C ** 4 * CS ** (7 / 3) * (CS ** 2 - 1) ** 2 / (576 * 6 ** (1 / 3) * np.pi) * (x_eq / x_uv) ** 8 * x_uv ** (17 / 3)
say(f"\nInput: C={C}, k_eq = k_eva, k_uv = {x_uv:.0e} k_eva.  2409.12125 Omega_res^peak = {peak:.4e}")
say(f"{'k/k_uv':>7s} {'D&T res formula':>16s} {'E2 res-only':>12s} {'E2 full':>12s} {'full/res':>9s} "
    f"{'SSM':>12s} {'SSM/E2full':>11s} {'SSM/formula':>11s}")
xs = np.array([0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1]) * x_uv
res_only = sk.omega_gw(xs, P_Phi, x_uv=x_uv, resonant_only=True)
full = sk.omega_gw(xs, P_Phi, x_uv=x_uv, resonant_only=False)
ssm = ag.P_GW(xs, lambda z: ag.PV_OVER_PDELTA * np.where(z <= x_uv, P_delta(z), 0.0), breaks=[x_uv], n=400)
for x, r, f, s in zip(xs, res_only, full, ssm):
    xr = x / x_uv
    formula = peak * xr ** (11 / 3) * theta_uv(xr)
    say(f"{xr:7.2f} {formula:16.4e} {r:12.4e} {f:12.4e} {f / r:9.3f} {s:12.4e} {s / f:11.3f} "
        f"{(s / formula) if formula > 0 else float('nan'):11.3f}")
i_pk = np.argmax(full)
say(f"\nPeaks: E2 full {full.max():.4e} at k/k_uv={xs[i_pk] / x_uv:.2f};  SSM {ssm.max():.4e} at "
    f"{xs[np.argmax(ssm)] / x_uv:.2f};  ratio SSM/E2 at peak = {ssm.max() / full.max():.3f}")
with open(os.path.join(HERE, "..", "results", "e2_compare.txt"), "w", newline="\n") as fh:
    fh.write("\n".join(lines) + "\n")
