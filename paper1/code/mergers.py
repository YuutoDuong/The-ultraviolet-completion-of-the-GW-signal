"""
Binaries that merge before the holes evaporate -- an order-of-magnitude estimate
(RECHECK P1-12; raised by 2506.16154 and by the merger contour of 2605.21477).

----------------------------------------------------------------------------
ESTIMATE
----------------------------------------------------------------------------
(1) Early binaries (Nakamura et al. 1997; Sasaki et al. 2016).  During PBH domination the
    holes are all of the matter, f = 1.  A pair of comoving separation x < xbar decouples
    from the expansion before PBH-radiation equality and forms a binary of semi-major axis
        a = alpha xbar_eq (x/xbar)^4,
    xbar_eq = physical mean separation at equality, n_eq = rho_tot(eq) / (2 M),
    with alpha ~ 0.1-0.4.  Tidal torques of the nearest third hole at distance y give
    j = sqrt(1 - e^2) ~ (x/y)^3.
(2) Peters, high eccentricity, equal masses:  t_merge = (3/170) a^4 j^7 / (G^3 M^3), so
        t_merge = T (x/xbar)^37 (xbar/y)^21,    T = (3/170) (alpha xbar_eq)^4 / (G^3 M^3).
(3) With Poisson-distributed x and y this gives (Sasaki et al. 2016, f = 1)
        dP/dt = (3/58) [ (t/T)^(3/37) - (t/T)^(3/8) ] / t,     t < T,
    so the fraction of holes that have merged by evaporation is
        P(< t_ev) = (37/58) (t_ev/T)^(3/37) - (8/58) (t_ev/T)^(3/8)    (t_ev < T),
    and of order one beyond.  T is proportional to M beta^(-16/3) and t_ev to M^3, so
    t_ev/T ~ M^2 beta^(16/3).  The exponent 3/37 makes the answer insensitive to alpha:
    alpha^4 = 1e-4 ... 2.6e-2 moves P by about x2.
(4) Re-domination (added 2026-10-06).  Merged holes (~1.9 M after ~5% radiated, lifetime
    1.9^3 = 6.9 tau) hold the fraction P of the PBH mass.  After the others evaporate
    (H tau = 2/3) they grow against the radiation and dominate again before evaporating if
    P > 0.241 (redomination_fraction()).  P = 0.241 lies at t_ev/T = 6.8e-6, i.e.
        beta ~ 3.4e-5 (M / 1e4 g)^(-3/8),   a factor 0.107 below the P = 1/2 line.
(5) Merged holes do not go off together: merging at t_m costs t_m/(3 tau) of the mass, so
    they evaporate at 6.86 tau - 5.86 t_m, and with P(<t) ~ t^(3/37) the coherent fraction
    on the cluster scale is |phi| ~ Gamma(1+3/37) (0.85 x_cl)^(-3/37) = 0.56, 0.46, 0.38 for
    x_cl = 1e3, 1e4, 1e5 (|phi|^4 = 0.1 ... 0.02).
(6) GWs of the mergers themselves (one generation, 4.8% of the merging mass), diluted by the
    rest of the matter era: Omega_GW(ev) = 0.048 (3/58)/(3/37 + 2/3) (58/37) P = 0.0052 P,
    i.e. <= 8.5e-8 h^2 today, 20x below the Delta N_eff limit even for P = 1.
Not included: disruption of binaries by the clusters forming at the same time
(Raidal et al. 2018), three-body hardening inside them (2412.01890) and binaries formed in
them (2410.01876).  This is an estimate of where the monochromatic assumption starts to
fail, not a merger model.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import cosmology as co  # noqa: E402

G = 1.0 / co.M_PL_NR ** 2          # GeV^-2


def T_merge_scale(M_g, beta_f, alpha=0.4):
    """T of Eq. (2) [GeV^-1]."""
    M = M_g * co.GRAM
    H_eq = np.sqrt(2.0) * beta_f ** 2 * co.H_form(M_g)
    rho_eq = 3.0 * co.M_PL ** 2 * H_eq ** 2
    xbar = (M / (0.5 * rho_eq)) ** (1.0 / 3.0)
    return (3.0 / 170.0) * (alpha * xbar) ** 4 / (G ** 3 * M ** 3)


def merged_fraction(M_g, beta_f, alpha=0.4):
    """P(< t_ev) of Eq. (3), capped at 1."""
    r = co.lifetime(M_g) / T_merge_scale(M_g, beta_f, alpha)
    P = np.where(r < 1.0, (37.0 / 58.0) * r ** (3.0 / 37.0) - (8.0 / 58.0) * r ** 0.375, 1.0)
    return np.minimum(P, 1.0), r


def redomination_fraction(mass_ratio=1.9):
    """Merged fraction P above which the merged holes dominate again before evaporating, (4)."""
    from scipy.optimize import brentq
    lna = np.linspace(0.0, np.log(60.0), 200001)
    a = np.exp(lna)

    def excess(P):
        H = (2.0 / 3.0) * np.sqrt((1.0 - P) * a ** -4 + P * a ** -3)      # tau = 1
        t = 1.0 + np.concatenate([[0.0], np.cumsum(0.5 * (1 / H[1:] + 1 / H[:-1]) * np.diff(lna))])
        a2 = np.interp(mass_ratio ** 3, t, a)
        return P * a2 / (1.0 - P) - 1.0
    return brentq(excess, 0.01, 0.9)


if __name__ == "__main__":
    print(f"{'M [g]':>8s} {'beta':>9s} {'t_ev/T (a=0.4)':>15s} {'P a=0.4':>8s} {'P a=0.1':>8s}")
    for M, betas in ((1e2, (1e-5, 1e-3, 1e-1)), (1e3, (3e-4, 3e-3)), (1e4, (2e-5, 1e-4, 1e-3)),
                     (1e5, (2e-7, 1e-6, 2.3e-4)), (1e6, (5.5e-8, 1e-6, 5.3e-5)),
                     (3e6, (1.7e-7, 2.1e-5))):
        for b in betas:
            P4, r = merged_fraction(M, b, 0.4)
            P1, _ = merged_fraction(M, b, 0.1)
            print(f"{M:8.0e} {b:9.2e} {float(r):15.2e} {float(P4):8.3f} {float(P1):8.3f}")
