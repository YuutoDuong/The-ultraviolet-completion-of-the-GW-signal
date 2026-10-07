"""
Endpoint dynamics of an evaporating PBH population with an extended mass function.

Reproduces from first principles the universal suppression of the Newtonian
potential reported in:
  - Gouttenoire, Leister & Schwaller, arXiv:2605.21474 / 2605.21477
  - He, Ma, Sasaki & Takhistov, arXiv:2606.09804

Claim under test
----------------
For a mass-loss law  dM/dt = -kappa * M^(-alpha)  (Hawking: alpha = 2), ANY
finite-width initial mass function is driven to a universal low-mass tail
f(M,t) ~ M^alpha, giving

    n_c(s)   ~ s^1                        linear depletion; absent if monochromatic
    rho_c(s) ~ s^(1 + 1/(alpha+1))
    S_Phi(k) ~ k^(-(alpha+2)/(alpha+1))   -> k^(-4/3) for Hawking

versus monochromatic rho_c ~ s^(1/(alpha+1)), S_Phi ~ k^(-1/(alpha+1)).
Since Omega_GW ~ S_Phi^4, finite width costs an extra factor k^-4.

NUMERICS: everything near the endpoint is parameterised by s = t_evap - t, never
by t. Reconstructing t = t_evap - s and differencing loses all precision once
s/t_evap < 1e-12, which silently returns Gamma = 0 and breaks the root find.
In terms of s the initial-mass map carries no cancellation:
    M0(M,s) = [ M^(a+1) + M_max^(a+1) - (a+1) kappa s ]^(1/(a+1)).

Units: kappa = 1, effective maximum initial mass M_max = 1. This sector is scale
free. Convention: f(M,t) is a COMOVING NUMBER density per unit M; the
mass-weighted log distribution plotted in the literature is M^2 * f(M).
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


# --------------------------------------------------------------------------
# Characteristics of the evaporation flow in mass space
# --------------------------------------------------------------------------

def lifetime(M0, alpha=2.0, kappa=1.0):
    """Lifetime of a single hole of initial mass M0."""
    return M0 ** (alpha + 1.0) / ((alpha + 1.0) * kappa)


def M0_of_M_s(M, s, M_max=1.0, alpha=2.0, kappa=1.0):
    """
    Initial mass of a hole that has mass M at time t = t_evap - s.
    He et al. Eq. (S.14) with t eliminated in favour of s.
    """
    M = np.asarray(M, dtype=float)
    # Clamp at zero: at s = t_evap exactly (i.e. t = 0) A vanishes analytically,
    # but roundoff can push it slightly negative. The integrand reaches ~40
    # decades below M_top, where M^(a+1) is far smaller than that roundoff, so an
    # unclamped A produces a negative base and a silent NaN in rho_plateau.
    A = max(M_max ** (alpha + 1.0) - (alpha + 1.0) * kappa * s, 0.0)
    return (M ** (alpha + 1.0) + A) ** (1.0 / (alpha + 1.0))


def M_top_s(s, alpha=2.0, kappa=1.0):
    """Largest surviving mass at s = t_evap - t."""
    return ((alpha + 1.0) * kappa * max(s, 0.0)) ** (1.0 / (alpha + 1.0))


def f_of_M_s(M, s, f0, M_max=1.0, alpha=2.0, kappa=1.0):
    """
    Evolved mass function, He et al. Eq. (S.15): f(M,t) = (M/M0)^alpha f0(M0).
    Solves the mass-space continuity equation d_t f + d_M(Mdot f) = 0 along
    characteristics; no assumption is made about f0.
    """
    M = np.asarray(M, dtype=float)
    M0 = M0_of_M_s(M, s, M_max, alpha, kappa)
    return (M / M0) ** alpha * f0(M0)


def f_of_M_t(M, t, f0, M_max=1.0, alpha=2.0, kappa=1.0):
    """Wrapper in absolute time; use only well away from the endpoint."""
    return f_of_M_s(M, lifetime(M_max, alpha, kappa) - t, f0, M_max, alpha, kappa)


def M_top(t, M_max=1.0, alpha=2.0, kappa=1.0):
    return M_top_s(lifetime(M_max, alpha, kappa) - t, alpha, kappa)


# --------------------------------------------------------------------------
# Initial mass functions
# --------------------------------------------------------------------------

def make_powerlaw(p=1.0 + 1.0 / 0.36, M_cut=1.0):
    """
    Critical-collapse (Choptuik) infrared tail. The MASS-weighted log
    distribution obeys psi(M) = dbeta/dlnM ~ M^p, p = 1 + 1/gamma_M ~ 3.78.
    Since psi = M^2 f0, the number density per unit M is f0 ~ M^(p-2).
    Sharp UV cutoff at M_cut: the signal-maximising choice of arXiv:2605.21477.
    """
    def f0(M):
        M = np.asarray(M, dtype=float)
        scalar = (M.ndim == 0)
        Ma = np.atleast_1d(M)
        out = np.zeros_like(Ma)
        ok = (Ma > 0) & (Ma <= M_cut)
        out[ok] = Ma[ok] ** (p - 2.0)
        return float(out[0]) if scalar else out
    return f0, M_cut


def make_lognormal(sigma=0.1, M_c=1.0, n_sigma=5.0):
    """
    Log-normal number density per unit M. No hard maximum exists, so an
    effective M_max is imposed at M_c*exp(n_sigma*sigma); He et al. note this is
    set by the finite cosmological volume. Insensitivity to n_sigma is checked.
    """
    M_max = M_c * np.exp(n_sigma * sigma)

    def f0(M):
        M = np.asarray(M, dtype=float)
        scalar = (M.ndim == 0)
        Ma = np.atleast_1d(M)
        out = np.zeros_like(Ma)
        ok = (Ma > 0) & (Ma <= M_max)
        out[ok] = np.exp(-np.log(Ma[ok] / M_c) ** 2 / (2 * sigma ** 2)) / (
            np.sqrt(2 * np.pi) * sigma * Ma[ok]
        )
        return float(out[0]) if scalar else out

    return f0, M_max


# --------------------------------------------------------------------------
# Comoving number and energy density, as functions of s = t_evap - t
# --------------------------------------------------------------------------

def _moment_s(s, f0, M_max, order, alpha=2.0, kappa=1.0, rtol=1e-11, decades=40.0):
    """
    int_0^{M_top} M^order f(M,s) dM, evaluated in log M: near the endpoint the
    integrand collapses onto a narrow window below M_top.
    order = 0 -> number density, order = 1 -> energy density.
    """
    Mt = M_top_s(s, alpha, kappa)
    if Mt <= 0.0:
        return 0.0
    lo, hi = np.log(Mt) - decades, np.log(Mt)

    def integrand(u):
        M = np.exp(u)
        return M ** (order + 1.0) * float(f_of_M_s(M, s, f0, M_max, alpha, kappa))

    val, _ = quad(integrand, lo, hi, limit=400, epsabs=0.0, epsrel=rtol)
    return val


def n_comoving_s(s, f0, M_max, alpha=2.0, kappa=1.0):
    return _moment_s(s, f0, M_max, 0, alpha, kappa)


def rho_comoving_s(s, f0, M_max, alpha=2.0, kappa=1.0):
    return _moment_s(s, f0, M_max, 1, alpha, kappa)


def rho_comoving_mono_s(s, alpha=2.0, kappa=1.0):
    """Monochromatic reference: rho_c ~ M(t) ~ s^(1/(alpha+1))."""
    return ((alpha + 1.0) * kappa * max(s, 0.0)) ** (1.0 / (alpha + 1.0))


# --------------------------------------------------------------------------
# Collective decay rate and potential suppression
# --------------------------------------------------------------------------

def _rho_s(f0, M_max, alpha, kappa, monochromatic):
    if monochromatic:
        return lambda s: rho_comoving_mono_s(s, alpha, kappa)
    return lambda s: rho_comoving_s(s, f0, M_max, alpha, kappa)


def Gamma_collective_s(s, f0, M_max, alpha=2.0, kappa=1.0, monochromatic=False,
                       dlog=1e-4):
    """
    Collective rate Gamma = -dln(rho_c)/dt = +dln(rho_c)/ds at s = t_evap - t.
    NOT the single-hole rate: the mass-loss law convolved with the surviving
    population. Develops a pole (alpha+2)/((alpha+1) s).
    Differenced in ln s so the step stays relatively sized at any depth.
    """
    rho = _rho_s(f0, M_max, alpha, kappa, monochromatic)
    sp, sm = s * np.exp(dlog), s * np.exp(-dlog)
    dlnrho_dlns = (np.log(rho(sp)) - np.log(rho(sm))) / (2.0 * dlog)
    return dlnrho_dlns / s


def S_Phi(k_over_keva, f0, M_max, alpha=2.0, kappa=1.0, monochromatic=False,
          H_evap=1.0):
    """
    S_Phi(k) = rho_c(s_dec) / rho_c(plateau), decoupling set by
    Gamma(s_dec) = k / a(s_dec).

    On subhorizon scales during eMD the Poisson equation gives Phi ~ rho_c, so
    the potential tracks the comoving PBH density until the mode oscillation
    outruns the evaporation rate; thereafter the radiation perturbation takes
    over and the suppression saturates. Near the endpoint a ~ a_evap, so k/a is
    held fixed and quoted as k/k_eva with k_eva = a_evap H_evap.
    """
    t_evap = lifetime(M_max, alpha, kappa)
    rho = _rho_s(f0, M_max, alpha, kappa, monochromatic)
    rho_plateau = rho(t_evap)
    target = k_over_keva * H_evap

    def gap(s):
        return Gamma_collective_s(s, f0, M_max, alpha, kappa, monochromatic) - target

    # Gamma ~ C/s near the endpoint with C = O(1), so the root sits near
    # s ~ C/target. Bracket adaptively around that rather than at a fixed
    # fraction of t_evap: for broad mass functions t_evap is large and a fixed
    # lower bound can land ABOVE the root, silently returning NaN.
    s_guess = 2.0 / target
    lo, hi = 1e-5 * s_guess, min(0.5 * t_evap, 1e5 * s_guess)
    if not (gap(lo) > 0.0 > gap(hi)):
        return np.nan
    s_dec = brentq(gap, lo, hi, rtol=1e-12, maxiter=400)
    return rho(s_dec) / rho_plateau
