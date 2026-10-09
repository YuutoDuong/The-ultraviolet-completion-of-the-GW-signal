"""
Stage 1 -- the physical scale hierarchy of a PBH-dominated (early matter) era.

Everything here is derived from three inputs: the PBH initial mass M_in, the
formation abundance beta_f = rho_PBH/rho_tot at formation, and standard
cosmology.  No fitting formula is copied from a paper; every printed number
traces to an equation written out below, so every one of them can be checked
independently.  See RECHECK.md for the list of things that SHOULD be checked.

Companion to evaporation.py (Stage 0), which supplies k_univ/k_eva.

----------------------------------------------------------------------------
DERIVATIONS
----------------------------------------------------------------------------
Reduced Planck mass M_pl, non-reduced m_pl = sqrt(8 pi) M_pl.  rho = 3 M_pl^2 H^2.

(1) Formation by horizon collapse during radiation domination
        M_in = gamma * (4pi/3) rho H_f^-3 = gamma * 4 pi M_pl^2 / H_f
             =>  H_f = gamma m_pl^2 / (2 M_in)
        t_f  = 1 / (2 H_f)                                            [RD]

(2) Hawking evaporation, Schwarzschild, grey-body factor G
        T_BH  = m_pl^2 / (8 pi M)
        dM/dt = -(G g_H / 30720 pi) m_pl^4 / M^2
        tau   = 10240 pi M_in^3 / (G g_H m_pl^4)
    Normalisation check, g_H = 108, G = 3.8:  tau(1e10 g) = 410 s, and
    T_BH(1e13 g) = 1.06 GeV.  Both are standard values.  -> TEST 1, TEST 2

(3) Exact matter+radiation background, y = a/a_eq
        H(y)   = H_eq sqrt( (y^-3 + y^-4) / 2 )
        H_eq t = F(y) = (2 sqrt2 / 3) [ sqrt(1+y) (y-2) + 2 ]
    Limits: H t -> 1/2 (y<<1), 2/3 (y>>1).                    -> TEST 3
    rho_tot(eq) = 2 rho_PBH(eq) gives
        H_eq = sqrt2 beta_f^2 H_f
    and F(y) at y = beta_f reproduces t_f = 1/(2 H_f) identically. -> TEST 4
    y_evap solves  F(y) = H_eq (t_f + tau).

(4) PBH domination happens iff y_evap > 1, i.e.
        beta_f > beta_c = [ F(1) / (sqrt2 H_f (t_f + tau)) ]^(1/2)
    With F(1) = 0.552285 this is a closed form, not a root find. -> TEST 8

(5) Comoving scales.  Internal normalisation a_eq = 1, so k_eq = H_eq.
        a_f    = beta_f          k_f    = beta_f H_f
        a_evap = y_evap          k_eva  = y_evap H_evap
        k_PBH  = a_f / dbar_f,  with (4pi/3) dbar^3 n_PBH = 1
               =>  k_PBH / k_f = (beta_f / gamma)^(1/3)
    That definition of dbar is the one for which the PBH shot-noise spectrum
    takes its standard normalisation
        P_S(k) = (2 / 3 pi) (k / k_PBH)^3,   k < k_PBH,
    since P_S = k^3 / (2 pi^2 nbar) and nbar = 3 k_PBH^3 / (4 pi).
    This module VERIFIES that 2/(3pi) rather than assuming it.  -> TEST 5,6,7

(6) Non-linear scales.  Both defined by sqrt(P_delta(k)) = 1 at evaporation,
    so that the two are directly comparable.
    Isocurvature: PBH density contrast obeys the Meszaros equation, whose
    growing mode in a radiation+matter universe is exactly D+(y) = 1 + 3y/2:
        sqrt(2/3pi) (k/k_PBH)^(3/2) D+(y_evap) = 1
        =>  k_NL^iso / k_PBH = [ sqrt(3 pi / 2) / D+(y_evap) ]^(2/3)
    so the number of non-linear decades depends ONLY on how long eMD lasts:
        R = k_PBH / k_NL^iso = [ D+(y_evap) / sqrt(3 pi/2) ]^(2/3)
          -> (0.691 y_evap)^(2/3)  for y_evap >> 1.             -> TEST 12
    Adiabatic: during MD, delta = -(2/5) (k/aH)^2 zeta, so
        k_NL^adia / k_eva = [ 5 / (2 sqrt(A_s)) ]^(1/2) ~ 234
    independent of (M_in, beta_f).                              -> TEST 9

(7) Present-day frequency of a comoving mode k
        f_0(k) = (k / a_evap) (a_evap / a_0) / (2 pi)
        a_evap / a_0 = (g_s0 / g_s(T_evap))^(1/3) T_0 / T_evap
    Check: the horizon mode at T = 1 GeV, g_* = 100, lands at 2.6e-8 Hz, the
    standard textbook value.                                    -> TEST 10
"""

import numpy as np
from scipy.optimize import brentq

# --------------------------------------------------------------- constants
M_PL      = 2.435323e18        # reduced Planck mass                   [GeV]
M_PL_NR   = 1.220890e19        # non-reduced = sqrt(8 pi) M_PL         [GeV]
GRAM      = 5.609588e23        # GeV per gram
HBAR      = 6.582119569e-25    # [GeV s]
GEV_TO_HZ = 1.0 / HBAR         # Hz per GeV
KB        = 8.617333262e-14    # [GeV / K]
T0        = 2.7255 * KB        # CMB temperature today                 [GeV]
GS0       = 3.9310             # g_{*s} today (photons + 3 nu)

# ------------------- model parameters a reader may legitimately change ----
GAMMA   = 0.2      # M_in / M_horizon at formation; w^{3/2} for RD collapse
GSTAR_H = 108.0    # spin-weighted Hawking degrees of freedom (all SM)
CAL_G   = 3.8      # grey-body enhancement of dM/dt
A_S     = 2.1e-9   # primordial curvature amplitude, assumed scale invariant

# BBN lower bound on the reheating temperature after evaporation.
T_BBN   = 4.0e-3   # [GeV]
# CMB upper bound on the inflationary Hubble rate (r < 0.036).
H_INF_MAX = 6.1e13  # [GeV]

# Effective relativistic dof of the SM.  Coarse but monotone; only the BBN
# boundary of the parameter plane is sensitive to it -- see RECHECK item 6.
_LOGT = np.array([-4.0, -3.0, -2.0, -1.0, -0.8, -0.5, 0.0, 1.0, 2.0, 3.0, 4.0])
_GST = np.array([3.36, 10.75, 10.75, 17.25, 24.0, 61.75,
                 75.75, 86.25, 96.25, 106.75, 106.75])


def g_star(T_gev):
    """Effective relativistic dof at temperature T [GeV].  Vectorised."""
    g = np.interp(np.log10(T_gev), _LOGT, _GST)
    return float(g) if np.ndim(g) == 0 else g


# ------------------------------------------------------ single-hole physics

def T_hawking(M_g):
    """Hawking temperature [GeV] of a Schwarzschild hole of mass M_g grams."""
    return M_PL_NR ** 2 / (8.0 * np.pi * M_g * GRAM)


def lifetime(M_g, g_H=GSTAR_H, cal_G=CAL_G):
    """PBH lifetime [GeV^-1].  Eq. (2)."""
    M = M_g * GRAM
    return 10240.0 * np.pi * M ** 3 / (cal_G * g_H * M_PL_NR ** 4)


def lifetime_seconds(M_g, g_H=GSTAR_H, cal_G=CAL_G):
    return lifetime(M_g, g_H, cal_G) * HBAR


def H_form(M_g, gamma=GAMMA):
    """Hubble rate at formation [GeV].  Eq. (1)."""
    return gamma * M_PL_NR ** 2 / (2.0 * M_g * GRAM)


def T_form(M_g, gamma=GAMMA):
    """Temperature at formation [GeV], solved with a running g_*."""
    rho = 3.0 * M_PL ** 2 * H_form(M_g, gamma) ** 2
    T = (30.0 * rho / (np.pi ** 2 * 106.75)) ** 0.25
    for _ in range(4):                      # fixed point on g_*(T)
        T = (30.0 * rho / (np.pi ** 2 * g_star(T))) ** 0.25
    return T


def M_min_inflation(gamma=GAMMA, H_inf=H_INF_MAX):
    """Smallest PBH mass [g] that can form after inflation:  H_f <= H_inf."""
    return gamma * M_PL_NR ** 2 / (2.0 * H_inf) / GRAM


# ------------------------------------------------------ exact background

_C = 2.0 * np.sqrt(2.0) / 3.0


def F_of_y(y):
    """
    H_eq * t as a function of y = a/a_eq.  Eq. (3).

    NUMERICS: the textbook form sqrt(1+y)(y-2)+2 is a difference of two
    numbers near 2 whose true value is (3/4) y^2, so it loses all precision
    below y ~ 1e-3 and returns pure roundoff by y ~ 1e-9.  That is the same
    cancellation that silently returned Gamma = 0 in Stage 0.  Substituting
    u = sqrt(1+y) factorises it EXACTLY,
        sqrt(1+y) (y-2) + 2 = (u-1)^2 (u+2),   with  u - 1 = y / (u+1),
    which is cancellation-free at every y and needs no series branch.
    """
    y = np.asarray(y, dtype=float)
    u = np.sqrt(1.0 + y)
    return _C * (y / (u + 1.0)) ** 2 * (u + 2.0)


def y_of_F(Fv):
    """
    Exact inverse of F.  In terms of u = sqrt(1+y),
        c = 3 F / (2 sqrt2) = u^3 - 3u + 2
    is a depressed cubic whose relevant root is u >= 1:
        u = 2 cos ( arccos ((c-2)/2) / 3 )     0 <= c <= 4   (y <= 3)
        u = 2 cosh( arccosh((c-2)/2) / 3 )          c >= 4   (y >= 3)

    NUMERICS: for y << 1, c -> 0 and (c-2)/2 -> -1, so forming c - 2 destroys
    the small quantity and arccos near pi then amplifies what is left -- the
    round trip loses 5 significant figures by y = 1e-6.  Writing the cos
    branch through the half angle avoids ever forming c - 2:
        psi = pi - arccos((c-2)/2) = 2 arcsin(sqrt(c)/2)
        u - 1 = sqrt3 sin(psi/3) - 2 sin^2(psi/6)
    whose two terms are O(psi) and O(psi^2), so nothing cancels.  Returning
    y = (u-1)(u+2-1) in terms of u-1 keeps it clean to the end.
    Vectorised, so the whole (M_in, beta_f) map costs a handful of array ops.
    """
    c = 3.0 * np.asarray(Fv, dtype=float) / (2.0 * np.sqrt(2.0))
    psi = 2.0 * np.arcsin(np.sqrt(np.clip(c, 0.0, 4.0)) / 2.0)
    um1_lo = np.sqrt(3.0) * np.sin(psi / 3.0) - 2.0 * np.sin(psi / 6.0) ** 2
    # above c = 4 the subtraction c - 2 is harmless and the root is hyperbolic
    um1_hi = 2.0 * np.cosh(
        np.arccosh(np.maximum((c - 2.0) / 2.0, 1.0)) / 3.0) - 1.0
    um1 = np.where(c >= 4.0, um1_hi, um1_lo)
    return um1 * (2.0 + um1)


def H_over_Heq(y):
    """H(y) / H_eq for a matter + radiation universe."""
    return np.sqrt((y ** -3.0 + y ** -4.0) / 2.0)


def _t_evap(M_g, gamma=GAMMA, g_H=GSTAR_H, cal_G=CAL_G):
    """Cosmic time at which the holes finish evaporating [GeV^-1]."""
    return 1.0 / (2.0 * H_form(M_g, gamma)) + lifetime(M_g, g_H, cal_G)


def y_evap_of(M_g, beta_f, gamma=GAMMA, g_H=GSTAR_H, cal_G=CAL_G):
    """a_evap / a_eq.  Values below 1 mean the PBHs never came to dominate."""
    Hf = H_form(M_g, gamma)
    y = y_of_F(np.sqrt(2.0) * beta_f ** 2 * Hf * _t_evap(M_g, gamma, g_H, cal_G))
    return float(y) if np.ndim(y) == 0 else y


def y_evap_brentq(M_g, beta_f, gamma=GAMMA, g_H=GSTAR_H, cal_G=CAL_G):
    """
    Reference implementation of y_evap by root find, kept only so the analytic
    inversion above can be checked against it (TEST 11c).  Not used elsewhere.
    """
    Hf = H_form(M_g, gamma)
    target = np.sqrt(2.0) * beta_f ** 2 * Hf * _t_evap(M_g, gamma, g_H, cal_G)
    lo, hi = np.log(1e-24), np.log(1e40)
    return float(np.exp(brentq(lambda u: float(F_of_y(np.exp(u))) - target,
                               lo, hi, rtol=1e-14, maxiter=300)))


def beta_crit(M_g, gamma=GAMMA, g_H=GSTAR_H, cal_G=CAL_G):
    """
    Smallest beta_f for which the PBHs dominate before evaporating.  Eq. (4).
    Closed form: set y_evap = 1 and invert.
    """
    Hf = H_form(M_g, gamma)
    return np.sqrt(float(F_of_y(1.0))
                   / (np.sqrt(2.0) * Hf * _t_evap(M_g, gamma, g_H, cal_G)))


# ------------------------------------------------------ the scale hierarchy

def poisson_amplitude_coefficient():
    """
    Derive the coefficient of P_S(k) = C (k/k_PBH)^3 from shot noise alone:
    P_S = k^3 / (2 pi^2 nbar) with nbar = 3 k_PBH^3 / (4 pi).  Returns C,
    which must equal 2/(3 pi) = 0.2122 for the convention used here to be the
    one in the literature.  TEST 5 checks it.
    """
    return (1.0 / (2.0 * np.pi ** 2)) * (4.0 * np.pi / 3.0)


D_PLUS = lambda y: 1.0 + 1.5 * y          # exact Meszaros growing mode


def scales_array(M_g, beta_f, gamma=GAMMA, g_H=GSTAR_H, cal_G=CAL_G, A_s=A_S,
                 k_univ_over_keva=10.0):
    """
    Every scale of the problem, vectorised.  M_g [g] and beta_f broadcast
    against one another, so a full (M_in, beta_f) map is a single call.

    k_univ_over_keva is the onset of the universal k^(-4/3) suppression.  It
    is NOT derivable here: it comes from Stage 0's TEST 6 and depends on the
    mass-function width.  Default 10.0 is the Choptuik critical-collapse value.

    Comoving k are normalised to a_eq = 1 and are meaningful only as ratios;
    the physical content is in the f_* frequencies [Hz].

    Points with y_evap < 1 never had an eMD era.  They are returned rather
    than masked -- 'dominates' says which is which -- because the hierarchy
    statements below are only meaningful where it is True.
    """
    # Broadcast up front so that EVERY returned array has the map's shape,
    # including the ones that happen to depend on M_in alone (beta_c, T_f).
    M_g, beta_f = np.broadcast_arrays(np.asarray(M_g, dtype=float),
                                      np.asarray(beta_f, dtype=float))

    Hf = H_form(M_g, gamma)
    tau = lifetime(M_g, g_H, cal_G)
    t_ev = 1.0 / (2.0 * Hf) + tau
    Heq = np.sqrt(2.0) * beta_f ** 2 * Hf
    y = y_of_F(Heq * t_ev)
    Hev = Heq * H_over_Heq(y)

    # reheating temperature after evaporation, fixed point on g_*(T)
    rho = 3.0 * M_PL ** 2 * Hev ** 2
    T = (30.0 * rho / (np.pi ** 2 * 106.75)) ** 0.25
    for _ in range(4):
        T = (30.0 * rho / (np.pi ** 2 * g_star(T))) ** 0.25

    # comoving wavenumbers, a_eq = 1
    k_eq = Heq
    k_f = beta_f * Hf
    k_eva = y * Hev
    k_PBH = k_f * (beta_f / gamma) ** (1.0 / 3.0)
    k_NL_iso = k_PBH * (np.sqrt(1.5 * np.pi) / D_PLUS(y)) ** (2.0 / 3.0)
    k_NL_ad = k_eva * np.sqrt(5.0 / (2.0 * np.sqrt(A_s)))
    k_univ = k_eva * k_univ_over_keva

    # redshift to today; a_evap = y in these units
    a_ratio = (GS0 / g_star(T)) ** (1.0 / 3.0) * T0 / T
    to_hz = a_ratio * GEV_TO_HZ / (2.0 * np.pi * y)

    R = k_PBH / k_NL_iso          # non-linear decades, the Stage 1 headline
    return dict(
        M_g=M_g, beta_f=beta_f, H_f=Hf, T_f=T_form(M_g, gamma),
        T_BH=T_hawking(M_g), tau_s=tau * HBAR,
        beta_c=beta_crit(M_g, gamma, g_H, cal_G),
        y_evap=y, dominates=(y > 1.0),
        N_efolds_eMD=np.where(y > 1.0, np.log(np.maximum(y, 1.0)), 0.0),
        H_evap=Hev, T_evap=T, a_evap_over_a0=a_ratio,
        k_eq=k_eq, k_f=k_f, k_eva=k_eva, k_PBH=k_PBH,
        k_NL_iso=k_NL_iso, k_NL_adia=k_NL_ad, k_univ=k_univ,
        R_nl=R, decades_nl=np.log10(R),
        f_eq=k_eq * to_hz, f_eva=k_eva * to_hz, f_PBH=k_PBH * to_hz,
        f_NL_iso=k_NL_iso * to_hz, f_NL_adia=k_NL_ad * to_hz,
        f_univ=k_univ * to_hz,
    )


def scales(M_g, beta_f, **kw):
    """
    Scalar convenience wrapper.  Delegates to scales_array so there is exactly
    one implementation of the physics; TEST 17 checks the two agree.
    """
    out = scales_array(float(M_g), float(beta_f), **kw)
    d = {k: (bool(v) if k == "dominates" else float(v))
         for k, v in out.items()}
    return d


# ------------------------------------------------------ parameter-space cuts

def allowed(M_g, beta_f, gamma=GAMMA, g_H=GSTAR_H, cal_G=CAL_G):
    """
    (dominates, post_BBN, post_inflation) for one point of the plane.
    A point is viable only if all three are True.
    """
    s = scales(M_g, beta_f, gamma, g_H, cal_G)
    dom = s["dominates"]
    bbn = dom and s.get("T_evap", 0.0) > T_BBN
    inf = M_g >= M_min_inflation(gamma)
    return dom, bbn, inf
