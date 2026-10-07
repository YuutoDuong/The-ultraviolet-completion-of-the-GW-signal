"""
Diffusion (Silk-type) damping of sound waves in the radiation era after PBH evaporation.

Gouttenoire, Leister & Schwaller (2605.21474, Sec. 6.3, l.1923-1937) name this cutoff
k_D and leave it "for future work".  Phase A needs it because it decides whether the
cluster scale is acoustically alive at all.

----------------------------------------------------------------------------
DERIVATION
----------------------------------------------------------------------------
(1) A sound wave of physical wavenumber k/a in a fluid with shear viscosity eta
    (bulk viscosity zero for radiation) loses AMPLITUDE at the rate
        Gamma_k = (2/3) (eta / w) (k/a)^2,          w = rho + p = (4/3) rho.
    This is the standard acoustic attenuation k^2 (4 eta/3) / (2 w); it matches the
    integrand of 2605.21474 Eq. (kD_def), k_D^-2 = int deta 2 eta / (3 a w).
(2) Define the comoving damping scale through Gamma_k / H = (k / k_D)^2:
        (k_D / aH)^2 = 1 / [ (2/3) (eta / w) H ].
(3) Kinetic theory for a massless species j with mean free path l_j:
        eta_j = (4/15) rho_j l_j.
    Neutrinos (T < T_W): l_nu = 1 / Gamma_nu, Gamma_nu = c_nu G_F^2 T^5 (g_*(T)/10.75).
        c_nu = 1 puts Gamma_nu = H at T = 1.5 MeV, the textbook decoupling
        temperature (TEST D1).  The g_* factor scales the density of scattering
        targets above the MeV scale.                          [RECHECK P1-3]
    Charged/strongly coupled plasma, all T: eta_EM = c_eta T^3 with c_eta = 400,
        the order of the Arnold-Moore-Yaffe electroweak-plasma value, dominated by
        right-handed leptons (hypercharge only).              [RECHECK P1-3]
    Above T_W the neutrinos are as strongly coupled as other leptons, so their
    long-mean-free-path term is switched off.
(4) Time dependence in radiation domination (T ~ 1/a): comoving k_D^2 ~ a^-p with
        p = 5 if neutrinos dominate eta,   p = 1 if eta_EM dominates.
    Used by acoustic_gw.Upsilon_eff for damping during GW production.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import cosmology as co  # noqa: E402  (Stage 1)

G_F = 1.1663787e-5      # Fermi constant [GeV^-2]
C_NU = 1.0              # neutrino-rate normalisation, see (3)
C_ETA = 400.0           # eta_EM / T^3, see (3)
T_W = 80.0              # [GeV] above this, no separate long-lived neutrino term


def rho_rad(T, g=None):
    g = co.g_star(T) if g is None else g
    return (np.pi ** 2 / 30.0) * g * T ** 4


def hubble_rad(T, g=None):
    """H in radiation domination [GeV]."""
    return np.sqrt(rho_rad(T, g) / 3.0) / co.M_PL


def gamma_nu(T, c_nu=C_NU):
    """Neutrino interaction rate [GeV], Eq. (3)."""
    return c_nu * G_F ** 2 * T ** 5 * (co.g_star(T) / 10.75)


def eta_nu(T, c_nu=C_NU):
    rho_nu = (7.0 / 8.0) * 6.0 * (np.pi ** 2 / 30.0) * T ** 4
    eta = (4.0 / 15.0) * rho_nu / gamma_nu(T, c_nu)
    return np.where(T < T_W, eta, 0.0)


def eta_em(T, c_eta=C_ETA):
    return c_eta * T ** 3


def kD_over_aH(T, c_nu=C_NU, c_eta=C_ETA):
    """Comoving damping wavenumber in units of aH at temperature T [GeV], Eq. (2)."""
    T = np.asarray(T, dtype=float)
    w = (4.0 / 3.0) * rho_rad(T)
    eta = eta_nu(T, c_nu) + eta_em(T, c_eta)
    return np.sqrt(1.0 / ((2.0 / 3.0) * (eta / w) * hubble_rad(T)))


def damping_exponent(T, c_nu=C_NU, c_eta=C_ETA):
    """p in k_D^2 ~ a^-p, Eq. (4): 5 if neutrino viscosity dominates, else 1."""
    return np.where(eta_nu(T, c_nu) > eta_em(T, c_eta), 5.0, 1.0)


def neutrino_decoupling_T(c_nu=C_NU):
    """Temperature [GeV] where Gamma_nu = H.  TEST D1 wants ~1.5 MeV."""
    from scipy.optimize import brentq
    f = lambda lt: np.log(gamma_nu(10.0 ** lt, c_nu) / hubble_rad(10.0 ** lt))
    return 10.0 ** brentq(f, -4.0, 0.0)
