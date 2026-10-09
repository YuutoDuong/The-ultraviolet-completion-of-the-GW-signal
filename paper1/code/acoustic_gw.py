"""
Module E -- gravitational waves from the sound field left by evaporation.

Sound-shell model (Hindmarsh 1608.04735; Hindmarsh & Hijazi 1909.10040, Eqs.
GWGroRat and the spectral density function), in the form used by Ning et al.
2512.21151 Eq. (GW spectrum).  2512.21151 shows on the lattice that for small
amplitudes this acoustic GW is the fluid part of the standard induced GW, which is
what lets one machinery serve the linear, the cluster and the shot-noise channels.

----------------------------------------------------------------------------
DERIVATION / CONVENTIONS
----------------------------------------------------------------------------
(1) Sudden hand-over: radiation starts with the density field and zero velocity.
    Plane-wave amplitudes v_q = (1/2)(q.v - c_s lambda) e^{i w t}, lambda = delta/Gamma
    (1909.10040 Eq. PlaWavCoeCor), so the dimensionless velocity power is
        P_v(q) = c_s^2/(2 Gamma^2) P_delta(q) = (3/32) P_delta(q),    Gamma = 4/3,
    with <v^2> = int dlnq P_v.  (TEST E1: time-averaged standing wave.)
(2) GW power per ln k after production has saturated, radiation domination:
        P_GW(x) = 3 Gamma^2 (H L) (x)^3/(2 pi^2) Pt_GW(x) * Upsilon,
        Pt_GW(x) = 1/(4 pi x c_s) ((1-c_s^2)/c_s^2)^2
                   int_{z-}^{z+} dz/z (z-z+)^2 (z-z-)^2/(z+ + z- - z) Pb(z) Pb(z+ + z- - z),
        z_pm = x(1 pm c_s)/(2 c_s),   Pb(z) = pi^2 P_v(z)/z^3.
    Length unit L = 1/k_eva and H L = 1 (sound waves start at evaporation), so
    x = k/k_eva and z = q/k_eva.  Upsilon = 1 for a long-lived source in radiation
    domination (2007.08537; 2512.21151 below its Eq. for P_GW).  A source that lives only
    until y_end = a/a_ev (sound_lifetime.py: shock formation) gives 1 - 1/y_end instead;
    P_GW(..., y_end=...) and Upsilon(p, y_end=...) apply it.
(3) Damping (damping.py): each velocity mode decays at Gamma_q/H = (q/k_D)^2 with
    comoving k_D^2 ~ a^-p.  A GW mode built from (q, q~) then accumulates
        Upsilon_once(X) = int_1^inf dy y^-2 exp[-2 X (y^p - 1)/p],  X = (q^2+q~^2)/k_D^2,
    which is 1 for X -> 0 and 1/(2X) for X >> 1 (TEST E4).  This applies when all the
    sound is made at once (monochromatic holes; the coherent part of any population).
    When sound is made continuously over a time T (incoherent explosions of an
    extended mass function), the steady state holds less energy at any moment:
        Upsilon_cont = Upsilon_once / (1 + G_eff H T),  G_eff = 2 g g~/(g + g~),
    g = (q/k_D)^2.  Exact in both limits (Phase A note, PHASE_A.md Sec. 3).
(4) Today: Omega_GW,0 h^2 = Omega_r0 h^2 (g_*/g_*0) (g_s0/g_s)^(4/3) P_GW,
    Omega_r0 h^2 = 4.18e-5 (photons + three massless neutrinos), g_*0 = 3.36.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import cosmology as co  # noqa: E402

CS = 1.0 / np.sqrt(3.0)
GAM = 4.0 / 3.0
PV_OVER_PDELTA = CS ** 2 / (2.0 * GAM ** 2)           # 3/32
OMEGA_R0_H2 = 4.18e-5
GSTAR0 = 3.36


class Upsilon:
    """Tabulated Upsilon_once(X) for one damping exponent p, Eq. (3).  With y_end the source
    switches off at y = y_end (sound_lifetime.py), and the integral stops there."""

    def __init__(self, p=1.0, n=400, y_end=np.inf):
        self.p, self.y_end = p, y_end
        lnX = np.linspace(np.log(1e-12), np.log(1e12), n)
        # y = 1 + u on a log grid in u, so the region u ~ 1/(2X) is resolved for any X
        s = np.linspace(np.log(1e-16), np.log(1e6), 8000)
        if np.isfinite(y_end):
            s_end = np.log(y_end - 1.0)
            s = np.append(s[s < s_end], s_end)          # grid ends exactly at the cutoff
        u = np.exp(s)
        ypm1 = np.expm1(p * np.log1p(u))                 # y^p - 1 without cancellation
        vals = []
        for X in np.exp(lnX):
            integrand = np.exp(-2.0 * X * ypm1 / p) / (1.0 + u) ** 2 * u
            vals.append(np.trapezoid(integrand, s))
        self._lnX, self._lnU = lnX, np.log(np.maximum(vals, 1e-300))

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        lnU = np.interp(np.log(np.clip(X, 1e-12, 1e12)), self._lnX, self._lnU)
        out = np.exp(lnU)
        big = X > 1e12
        out = np.where(big, 1.0 / (2.0 * np.maximum(X, 1e-300)), out)
        return out


_GL_CACHE = {}


def _gl(n):
    if n not in _GL_CACHE:
        _GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GL_CACHE[n]


def P_GW(x, Pv, x_D=None, ups=None, mode="once", HT=1.0, breaks=(), n=200, y_end=None):
    """
    Eq. (2)-(3).  x: array of k/k_eva.  Pv: callable, dimensionless velocity power at
    z = q/k_eva.  x_D: k_D/k_eva at evaporation (None = no damping).  ups: an Upsilon
    instance for the damping exponent.  mode: 'once' or 'cont'.  breaks: z-values
    where Pv has kinks or cutoffs; the z-integral is split there for accuracy.
    y_end: the source switches off at y = a/a_ev = y_end (None: lives for ever); with
    damping, ups must have been built with the same y_end.
    """
    if y_end is not None and x_D is not None and getattr(ups, "y_end", np.inf) != y_end:
        raise ValueError("P_GW: ups was built for a different source lifetime")
    life = 1.0 if (y_end is None or x_D is not None) else 1.0 - 1.0 / y_end
    x = np.atleast_1d(np.asarray(x, dtype=float))
    nodes, wts = _gl(n)
    pref = (1.0 / (4.0 * np.pi * CS)) * ((1.0 - CS ** 2) / CS ** 2) ** 2
    out = np.zeros_like(x)
    for i, xi in enumerate(x):
        zm, zp = xi * (1 - CS) / (2 * CS), xi * (1 + CS) / (2 * CS)
        zs = zm + zp
        # split points: breaks and their mirror images z -> zs - z
        cuts = [zm, zp]
        for b in breaks:
            for c in (b, zs - b):
                if zm < c < zp:
                    cuts.append(c)
        cuts = np.unique(cuts)
        tot = 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            z = 0.5 * (b - a) * nodes + 0.5 * (b + a)
            zt = zs - z
            Pb = np.pi ** 2 * Pv(z) / z ** 3
            Pbt = np.pi ** 2 * Pv(zt) / zt ** 3
            f = (z - zp) ** 2 * (z - zm) ** 2 / (z * zt) * Pb * Pbt
            if x_D is not None:
                X = (z ** 2 + zt ** 2) / x_D ** 2
                U = ups(X)
                if mode == "cont":
                    g, gt = (z / x_D) ** 2, (zt / x_D) ** 2
                    U = U / (1.0 + 2.0 * g * gt / (g + gt) * HT)
                f = f * U
            tot += 0.5 * (b - a) * np.dot(wts, f)
        Pt = pref * tot / xi
        out[i] = 3.0 * GAM ** 2 * xi ** 3 / (2.0 * np.pi ** 2) * Pt * life
    return out


def today_h2(P, T_evap):
    """Eq. (4): present-day Omega_GW h^2 from the saturated P_GW at temperature T_evap."""
    g = co.g_star(T_evap)
    return OMEGA_R0_H2 * (g / GSTAR0) * (co.GS0 / g) ** (4.0 / 3.0) * np.asarray(P)


def J_powerlaw(nidx, m=20001):
    """
    Closed-form reduction of Eq. (2) for P_v = z^n (TEST E3):
        P_GW = K(n) x^(2n-1),  K = 3 Gamma^2/(8 pi^3 c_s) ((1-c_s^2)/c_s^2)^2 c_s^(3-2n) J,
        J = pi^4 int dw w^(n-4) (1-w)^(n-4) (w - w+)^2 (w - w-)^2,  w_pm = (1 pm c_s)/2.
    """
    wm, wp = (1 - CS) / 2, (1 + CS) / 2
    w = np.linspace(wm, wp, m)
    J = np.pi ** 4 * np.trapezoid(w ** (nidx - 4) * (1 - w) ** (nidx - 4) * (w - wp) ** 2 * (w - wm) ** 2, w)
    return 3 * GAM ** 2 / (8 * np.pi ** 3 * CS) * ((1 - CS ** 2) / CS ** 2) ** 2 * CS ** (3 - 2 * nidx) * J
