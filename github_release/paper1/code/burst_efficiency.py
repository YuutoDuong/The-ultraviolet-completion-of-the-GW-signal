"""
Module F result in closed form: the sound efficiency of a cluster burst.

kappa = E_ac / E_ac,linear, the quadratic sound energy of the outgoing shell over the
linear-acoustics prediction (hydro1d.py, hydro_scan.py; p = rho/3, spherical, no gravity).

(1) Sudden top-hat release of local radiation contrast delta (results/hydro_scan.txt,
    t = 25R, each run normalised by a linear run at the same resolution):
        kappa(delta) = 1 / (1 + (delta/D1)^P),   D1 = 1.92, P = 0.915,
    within 3% for 0.01 <= delta <= 100.  Strong bursts: kappa ~ delta^-0.9, so the sound
    energy grows like the burst energy, not its square.
(2) The monochromatic release history (dE ~ s^(-2/3) ds) over the last s0 = 10 R/c of the
    holes' lives.  The earlier release is slow on the cluster's sound-crossing time and
    leaves adiabatically.  The released contrast is
        delta_tot = (Delta/xi^3) (s0/tau)^(1/3),   tau/R = (2/3) x_cl   (R_phys H_ev = 1/x_cl).
    The three source runs of hydro_scan.py follow (1) at delta_eff = A_SRC delta_tot with
    A_SRC = 0.21 (within 3%).
(3) GW output beyond the energy (results/hydro_gw_shift.txt).  A strong burst moves the
    shell's power to longer wavelengths (GW peak kR 2.1 -> 1.1-1.7), and long sound waves
    make GWs more efficiently, so the GW integral is R_shift kappa^2 times the linear one:
        R_shift(delta_tot) = (1 + delta_tot/R_C)^R_Q,   R_C = 8.1, R_Q = 0.43,
    fitted to the three release histories (within 1%; delta_tot = 3.3-42; top hats give
    1.02 at delta = 0.1 up to 2.9 at delta = 100).  kappa_gw = kappa sqrt(R_shift) is the
    amplitude factor that reproduces the GW integral with the linear spectral shape; the
    true peak frequency is lower by up to 2x.
(4) Scale dependence (hydro_spectrum.py): the loss sits at kR > 2, at and above the peak of
    the linear shell power; (3) is its net effect on the GW signal.
"""
import numpy as np

D1, P = 1.92, 0.915
A_SRC = 0.21
R_C, R_Q = 8.1, 0.43
S0 = 10.0                       # light-crossing times of the final release (hydro_scan.S0)


def kappa_tophat(delta):
    """Sound efficiency of a sudden top-hat release of contrast delta, Eq. (1)."""
    return 1.0 / (1.0 + (np.asarray(delta, dtype=float) / D1) ** P)


def delta_tot(Delta_eff, x_cl):
    """Radiation contrast released over the final s0 = 10 R/c, Eq. (2)."""
    return Delta_eff * (S0 / ((2.0 / 3.0) * np.asarray(x_cl, dtype=float))) ** (1.0 / 3.0)


def gw_shift(dtot):
    """GW integral of the non-linear shell over kappa^2 x linear, Eq. (3)."""
    return (1.0 + np.asarray(dtot, dtype=float) / R_C) ** R_Q


def kappa_cluster(Delta_eff, x_cl, gw=True):
    """
    Efficiency of a cluster of density Delta_eff rho_bar and size 1/x_cl, Eq. (2).
    gw=True returns kappa_gw = kappa sqrt(R_shift), Eq. (3), the factor on the power
    spectrum that gives the right GW integral; gw=False the energy efficiency.
    """
    dt = delta_tot(Delta_eff, x_cl)
    k = kappa_tophat(A_SRC * dt)
    return k * np.sqrt(gw_shift(dt)) if gw else k
