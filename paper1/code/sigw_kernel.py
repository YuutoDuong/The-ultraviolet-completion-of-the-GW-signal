"""
Module E2 -- the induced-GW (SIGW) calculation for a sudden matter-to-radiation
transition, implemented independently of the sound-shell model, to settle the
normalization question P1-1 (RECHECK.md).

Follows Domenech & Trankle 2409.12125 (App. D-E) and Domenech, Lin & Sasaki
2012.08151, keeping ALL terms of the oscillatory kernel, not just the resonant one.

----------------------------------------------------------------------------
FORMULAE (2409.12125 numbering)
----------------------------------------------------------------------------
(D.1)  Omega_GW(k) = (k^2/12 H^2) Pbar_h,  H = 1/taubar in radiation domination.
(4.4)  Pbar_h = (c_s^4/2048) (x_eva^8 / xbar^2) int dv int du (4v^2 - (1+v^2-u^2)^2)^2
                Ibar^2_osc(u,v) P_Phi(uk) P_Phi(vk),     x_eva = 2k/k_eva,
       where P_Phi is the potential spectrum at the onset of oscillation (all
       suppression and transfer factors included).  Hence
         Omega_GW(k) = (c_s^4 / 96) (k/k_eva)^8 int dv int du (...)^2 Ibar^2 P_Phi P_Phi.
(D.8)  I_osc = int_0^inf dx sin(xbb - x) sin(c_s u x) sin(c_s v x) / (x + c),  c = x_eva/2,
       so I_osc = A sin(xbb) + B cos(xbb) with
         A =  (1/4)[C(1+a-b) + C(1-a+b) - C(1+a+b) - C(1-a-b)],
         B = -(1/4)[S(1+a-b) + S(1-a+b) - S(1+a+b) - S(1-a-b)],
       a = c_s u, b = c_s v, and the oscillation average Ibar^2 = (A^2 + B^2)/2.
       C(q) = int cos(q x)/(x+c) = -cos(|q|c) Ci(|q|c) - sin(|q|c) si(|q|c),
       S(q) = int sin(q x)/(x+c) = sign(q) [sin(|q|c) Ci(|q|c) - cos(|q|c) si(|q|c)],
       si = Si - pi/2.  (Checked against direct quadrature in TEST E2-1.)
       The resonant-only approximation keeps the Ci of 1-a-b in A: Ibar^2 = Ci^2/32,
       which is 2409.12125 Eq. (E2).
Conversion to our variables: delta_r = (2/3)(q/k_eva)^2 Phi_osc, so
       P_Phi(q) = (9/4) (k_eva/q)^4 P_delta_r(q).
"""

import numpy as np
from scipy.special import sici

CS = 1.0 / np.sqrt(3.0)


def _CS_int(q, c):
    """C(q) and S(q) of the module docstring, vectorised; q may be negative."""
    aq = np.abs(q) * c
    aq = np.maximum(aq, 1e-300)
    si, ci = sici(aq)
    sim = si - 0.5 * np.pi
    Cq = -np.cos(aq) * ci - np.sin(aq) * sim
    Sq = np.sign(q) * (np.sin(aq) * ci - np.cos(aq) * sim)
    return Cq, Sq


def Ibar2(u, v, c, resonant_only=False):
    """Oscillation-averaged kernel squared, Eq. (D.8)."""
    a, b = CS * u, CS * v
    if resonant_only:
        _, ci = sici(np.maximum(np.abs(1 - a - b) * c, 1e-300))
        return ci ** 2 / 32.0
    C1, S1 = _CS_int(1 + a - b, c)
    C2, S2 = _CS_int(1 - a + b, c)
    C3, S3 = _CS_int(1 + a + b, c)
    C4, S4 = _CS_int(1 - a - b, c)
    A = 0.25 * (C1 + C2 - C3 - C4)
    B = -0.25 * (S1 + S2 - S3 - S4)
    return 0.5 * (A ** 2 + B ** 2)


def omega_gw(x, P_Phi, x_uv=None, resonant_only=False, ns=48, nt_per=400):
    """
    Omega_GW at emission (radiation era, saturated) for x = k/k_eva.
    P_Phi(z): potential spectrum at oscillation onset, z = q/k_eva.
    x_uv: optional hard cutoff of P_Phi in units of k_eva (applied to u x and v x).
    Integration in t = u+v and s = u-v (du dv = dt ds / 2), with the resonance
    t = 1/c_s treated as a breakpoint and a log-spaced grid around it.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    sn, sw = np.polynomial.legendre.leggauss(ns)
    tres = 1.0 / CS
    for i, xi in enumerate(x):
        c = xi           # c = x_eva/2 = k/k_eva
        tmax = 2.0 * x_uv / xi if x_uv is not None else 200.0
        tmax = max(tmax, 1.0 + 1e-9)
        # t-grid: log-spaced distances from the resonance on both sides, plus far range
        pieces = []
        lo_end = min(tres, tmax)
        d = np.geomspace(1e-12, 1.0, nt_per)
        if lo_end > 1.0:
            tl = tres - d * (tres - 1.0)
            tl = tl[(tl > 1.0) & (tl < lo_end)]
            pieces.append(np.sort(np.concatenate(([1.0], tl, [lo_end]))))
        if tmax > tres:
            th = tres + np.geomspace(1e-12, tmax - tres, nt_per)
            pieces.append(np.concatenate(([tres], th)))
        w = x_uv / xi if x_uv is not None else np.inf
        total = 0.0
        for tg in pieces:
            tg = np.unique(tg)
            vals = np.zeros_like(tg)
            for j, t in enumerate(tg):
                # the cutoff u, v <= w is a pair of straight lines in (t, s): fold it into
                # the s-range so Gauss-Legendre never straddles a discontinuity
                s_lo, s_hi = max(-1.0, t - 2 * w), min(1.0, 2 * w - t)
                if s_hi <= s_lo:
                    continue
                s = 0.5 * (s_hi - s_lo) * sn + 0.5 * (s_hi + s_lo)
                u, v = 0.5 * (t + s), 0.5 * (t - s)
                f = (4 * v ** 2 - (1 + v ** 2 - u ** 2) ** 2) ** 2 * Ibar2(u, v, c, resonant_only)
                vals[j] = 0.5 * (s_hi - s_lo) * np.dot(sw, f * P_Phi(u * xi) * P_Phi(v * xi))
            total += np.trapezoid(vals, tg) * 0.5
        out[i] = (CS ** 4 / 96.0) * xi ** 8 * total
    return out


def P_Phi_from_Pdelta(P_delta):
    """Wrap a radiation-contrast spectrum P_delta_r(z) as P_Phi(z) = (9/4) z^-4 P_delta_r."""
    return lambda z: 2.25 * np.asarray(z, dtype=float) ** -4 * P_delta(z)
