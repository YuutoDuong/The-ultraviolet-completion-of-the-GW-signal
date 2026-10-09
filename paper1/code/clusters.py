"""
Module B -- PBH clusters at evaporation: linear spectrum, Press-Schechter statistics,
and the halo-model density power spectrum that replaces the linear one above k_NL.

Conventions are Stage 1's (cosmology.py): comoving k with a_eq = 1, so every
wavenumber is used only as a ratio; n_bar = 3 k_PBH^3 / (4 pi).

----------------------------------------------------------------------------
DERIVATIONS
----------------------------------------------------------------------------
(1) Linear PBH density contrast at evaporation per unit initial isocurvature S:
        delta(k)/S = (2/3) (k/k_eva)^2 T_S(k/k_eq)
    (sub-horizon Poisson equation; 2605.21474 Eq. Poisson_NL).  Transfer function
        T_S(kappa) = [sqrt5 + kappa/sqrt(C)]^-2,   limits 1/5 and C/kappa^2.
    C = 9/8 makes the kappa >> 1 limit equal the exact Meszaros growth 1 + 3y/2 -> 3y/2
    that Stage 1 uses (TEST B2); 2605.21474 fit C ~ 1 numerically.  [RECHECK P1-4]
(2) Shot-noise seed P_S = (2/3pi)(k/k_PBH)^3 (Stage 1 TEST 5).  Linear spectrum
        P_lin(k) = P_S(k) [delta(k)/S]^2.
(3) Mass assignment: sharp-k filter with M = 6 pi^2 rho R^3 (2412.01890 Eq. R_M), so
        N = (9 pi/2) (k_PBH R)^3,   sigma^2(N) = int_0^{1/R} dlnk P_lin.
    For kappa >> 1 this is D^2/N with D = 3y/2, i.e. 2412.01890 Eq. (sigmaM2) with
    mu = 1 (TEST B3).
(4) Press-Schechter: nu = delta_c/sigma, mass fraction per ln N
        dF/dlnN = sqrt(2/pi) nu |dln sigma/dlnN| exp(-nu^2/2),
    bias b = 1 + (nu^2 - 1)/delta_c (Mo & White).  N_* defined by nu(N_*) = 1.
(5) Clusters: uniform spheres of density Delta rho_bar, Delta = 18 pi^2 (2412.01890
    Eq. rhobar_cl, evaluated at evaporation -- the conservative baseline of
    COMPUTE_PLAN), comoving radius R_cl = xi (N/Delta)^(1/3) / k_PBH.
    xi >= 1 is the puff-up while the holes lose mass slowly: a bound cluster losing
    mass adiabatically expands as R ~ 1/M (Jeans; noted for inflaton halos by
    1002.3278 Sec. IV C).  xi ~ 2 for clusters forming near evaporation (puff_factor).
(6) Halo model, dimensionless, k^3/(2 pi^2 n) = P_S(k):
        P_1h(k) = P_S(k) int dlnN (dF/dlnN) N u(k R_cl)^2,
        P_2h(k) = P_lin(k) [int dlnN (dF/dlnN) b u(k R_cl)]^2,
    u = top-hat transform.  For white noise the halo model overshoots P_lin by
    1/delta_c^2 ~ 35% at k << k_NL (TEST B5), so the non-linear coherent spectrum is
        P_coh(k) = min(P_lin, P_1h + P_2h).
    Discreteness (the 1/n term) is NOT in P_coh; suddenness.py adds it separately.
(7) Physical size: R_cl H_ev = R_cl k_eva (comoving R times a_ev H_ev), and the
    virial velocity sigma_v = sqrt(Delta/2) (R_cl H_ev) (matter domination).
"""

import sys
import os
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosmology as co  # noqa: E402  (Stage 1)
import burst_efficiency as be  # noqa: E402  (module F)

DELTA_C = 1.686
DELTA_VIR = 18.0 * np.pi ** 2
C_TS = 9.0 / 8.0


def T_S(kappa, C=C_TS):
    return (np.sqrt(5.0) + np.asarray(kappa, dtype=float) / np.sqrt(C)) ** -2


def u_tophat(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    big = x > 1e-3
    xb = x[big]
    out[big] = 3.0 * (np.sin(xb) - xb * np.cos(xb)) / xb ** 3
    small = ~big
    out[small] = 1.0 - x[small] ** 2 / 10.0
    return out


class Clusters:
    """Everything module B knows at one point (M_in [g], beta_f) of the plane."""

    def __init__(self, M_g, beta_f, Delta=DELTA_VIR, xi=1.0, C=C_TS, nk=6000):
        self.M_g, self.beta_f = M_g, beta_f
        self.sc = sc = co.scales(M_g, beta_f)
        self.Delta, self.xi, self.C = Delta, xi, C
        self.k_eva, self.k_eq, self.k_PBH = sc["k_eva"], sc["k_eq"], sc["k_PBH"]
        # cumulative sigma^2 on a log grid, Eq. (3)
        lnk = np.linspace(np.log(self.k_eva) - 25.0, np.log(self.k_PBH) + 2.0, nk)
        k = np.exp(lnk)
        P = self.P_lin(k)
        cum = np.concatenate(([0.0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(lnk))))
        self._lnk, self._lncum = lnk, np.log(np.maximum(cum, 1e-300))
        self.N_star = self._solve_N_star()
        self.k_star = self.k_of_N_sharp(self.N_star)

    # ---------------------------------------------------------- linear
    def growth(self, k):
        """delta(k)/S at evaporation, Eq. (1)."""
        return (2.0 / 3.0) * (k / self.k_eva) ** 2 * T_S(k / self.k_eq, self.C)

    def P_S(self, k):
        return (2.0 / (3.0 * np.pi)) * (np.asarray(k, dtype=float) / self.k_PBH) ** 3

    def P_lin(self, k):
        return self.P_S(k) * self.growth(k) ** 2

    # ---------------------------------------------------------- Press-Schechter
    def k_of_N_sharp(self, N):
        return self.k_PBH * (9.0 * np.pi / (2.0 * np.asarray(N, dtype=float))) ** (1.0 / 3.0)

    def sigma2(self, N):
        lnkc = np.log(self.k_of_N_sharp(N))
        return np.exp(np.interp(lnkc, self._lnk, self._lncum))

    def nu(self, N):
        return DELTA_C / np.sqrt(self.sigma2(N))

    def _solve_N_star(self):
        f = lambda lnN: 0.5 * np.log(self.sigma2(np.exp(lnN))) - np.log(DELTA_C)
        return float(np.exp(brentq(f, -5.0, 250.0)))

    def dF_dlnN(self, N):
        """PS mass fraction per ln N, Eq. (4); |dln sigma/dlnN| by finite difference."""
        N = np.asarray(N, dtype=float)
        h = 1e-3
        dls = 0.5 * (np.log(self.sigma2(N * np.exp(h))) - np.log(self.sigma2(N * np.exp(-h)))) / (2 * h)
        nu = self.nu(N)
        return np.sqrt(2.0 / np.pi) * nu * np.abs(dls) * np.exp(-0.5 * nu ** 2)

    def bias(self, N):
        nu = self.nu(N)
        return 1.0 + (nu ** 2 - 1.0) / DELTA_C

    # ---------------------------------------------------------- clusters
    def R_cl(self, N):
        """Comoving cluster radius, Eq. (5)."""
        return self.xi * (np.asarray(N, dtype=float) / self.Delta) ** (1.0 / 3.0) / self.k_PBH

    def R_cl_H(self, N):
        """Physical radius in Hubble units at evaporation, Eq. (7)."""
        return self.R_cl(N) * self.k_eva

    def sigma_v(self, N):
        """Virial velocity at virialisation (before any puff-up), Eq. (7)."""
        return np.sqrt(self.Delta / 2.0) * self.R_cl_H(N) / self.xi

    def lnN_grid(self, n=600, lo=0.0, hi_decades=4.0):
        return np.linspace(lo, np.log(self.N_star) + hi_decades * np.log(10.0), n)

    def kappa_burst(self, N, gw=True):
        """Burst efficiency of a cluster of N holes (burst_efficiency.py): GW-equivalent
        kappa_gw by default, the energy efficiency with gw=False."""
        return be.kappa_cluster(self.Delta / self.xi ** 3, 1.0 / self.R_cl_H(N), gw=gw)

    def P_halo(self, k, n=600, hydro=False):
        """
        (P_1h, P_2h) dimensionless, Eq. (6).  hydro=True weights each cluster in the
        one-halo term by kappa_gw (for GW spectra), hydro="energy" by the energy kappa.
        """
        k = np.atleast_1d(np.asarray(k, dtype=float))
        lnN = self.lnN_grid(n)
        N = np.exp(lnN)
        dF = self.dF_dlnN(N)
        b = self.bias(N)
        uk = u_tophat(np.outer(k, self.R_cl(N)))           # (nk, nN)
        w1 = dF * N * (self.kappa_burst(N, gw=(hydro is True)) if hydro else 1.0)
        I1 = np.trapezoid(w1 * uk ** 2, lnN, axis=1)
        I2 = np.trapezoid(dF * b * uk, lnN, axis=1)
        return self.P_S(k) * I1, self.P_lin(k) * I2 ** 2

    def P_coh(self, k):
        """Non-linear coherent spectrum, Eq. (6): min(P_lin, P_1h + P_2h)."""
        p1, p2 = self.P_halo(k)
        return np.minimum(self.P_lin(k), p1 + p2)

    def mass_fraction_above(self, Nmin):
        lnN = np.linspace(np.log(Nmin), np.log(self.N_star) + 6 * np.log(10), 800)
        return np.trapezoid(self.dF_dlnN(np.exp(lnN)), lnN)


def puff_factor(t_form_over_tau=1.0):
    """
    Adiabatic expansion of a virialised cluster while its holes lose mass, Eq. (5):
    t_dyn = sqrt(3/(4 pi G Delta rho)) = 0.159 t_form in matter domination; mass loss
    stays adiabatic until 1/(3 s) ~ 1/t_dyn(M), with t_dyn ~ M^-2 and M = (s/tau)^(1/3);
    the release then becomes impulsive at M_ad = (t_dyn0/(3 tau))^(1/5), and R ~ 1/M.
    """
    t_dyn0 = np.sqrt(3.0 * 6.0 * np.pi / (4.0 * np.pi * DELTA_VIR)) * t_form_over_tau
    M_ad = (t_dyn0 / 3.0) ** 0.2
    return 1.0 / M_ad


# ---------------------------------------------- Holst, Krnjaic & Xiao formulas

def holst_R_and_sigma_v(m_g, t_i_s, N_i, mu=1.0):
    """
    2412.01890 Eqs. (rhobar_cl), (R_cl), (sigma_v_cl) in their variables, evaluated
    from the definitions; returns (R_i / r_s^cl, sigma_v).  TEST B6 checks the
    prefactors 1e6 and 1e-3 they quote for m = 1e9 g, t_i = 1e-20 s.
    """
    G = 1.0 / co.M_PL_NR ** 2                       # GeV^-2
    t_i = t_i_s / co.HBAR                           # GeV^-1
    m = m_g * co.GRAM
    rho = DELTA_VIR / (6 * np.pi * G * t_i ** 2 * DELTA_C ** 3) * (mu / N_i) ** 1.5
    M = N_i * m
    R = (3 * M / (4 * np.pi * rho)) ** (1.0 / 3.0)
    return R / (2 * G * M), np.sqrt(G * M / R)
