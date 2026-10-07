"""
Validity checks for the fluid treatment (module G input; RECHECK P1-6).

(1) Thermalization of the burst.  The sound at physical wavenumber k is made by the part
    of a hole's life with remaining time s ~ 1/w, w = c_s k; the hole's temperature then
    is T_BH(s) = T_BH,i (tau/s)^(1/3) ~ T_BH,i (w tau)^(1/3).  A quantum of energy E >> T
    loses its energy in a plasma at temperature T over a length (LPM-suppressed splitting)
        l_th ~ sqrt(E/T) / (alpha^2 T),
    so the injection is local on the scale of the wave if l_th << lambda = 2 pi/k.
    alpha = 0.03 (electroweak/strong couplings at these scales) is a choice.  [RECHECK]
(2) Fluid description of the radiation at the cluster scale: the longest mean free
    path (neutrinos below T_W, hypercharge-only leptons above) must be << the cluster
    radius.  Uses damping.py's rates.
(3) Gravitational back-reaction of a cluster: Phi_cl ~ sigma_v^2 << 1.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import cosmology as co   # noqa: E402
import clusters as cl    # noqa: E402
import damping as dm     # noqa: E402

ALPHA = 0.03
CS = 1.0 / np.sqrt(3.0)


def thermalization_ratio(M_g, beta, x):
    """l_th / lambda for the mode x = k/k_eva at evaporation."""
    s = co.scales(M_g, beta)
    T, H = s["T_evap"], s["H_evap"]
    k_phys = x * H                       # k/a at evaporation, since k_eva = a H
    wtau = CS * x * (2.0 / 3.0)
    E = co.T_hawking(M_g) * wtau ** (1.0 / 3.0)
    l_th = np.sqrt(np.maximum(E / T, 1.0)) / (ALPHA ** 2 * T)
    return l_th / (2 * np.pi / k_phys)


def mfp_over_cluster(M_g, beta, xi=1.0):
    """Longest mean free path over the physical cluster radius at evaporation."""
    c = cl.Clusters(M_g, beta, xi=xi)
    s = c.sc
    T, H = s["T_evap"], s["H_evap"]
    R_phys = c.R_cl_H(c.N_star) / H
    if T < dm.T_W:
        l = 1.0 / dm.gamma_nu(T)
    else:
        rho_j = (7 / 8) * 6 * (np.pi ** 2 / 30) * T ** 4
        l = dm.eta_em(T) / ((4 / 15) * rho_j)        # invert eta = (4/15) rho l
    return l / R_phys


if __name__ == "__main__":
    print(f"{'M [g]':>8s} {'beta':>9s} {'T_evap':>9s} {'l_th/lam @k_D':>14s} {'@k_PBH':>9s} "
          f"{'mfp/R_cl':>10s} {'Phi_cl':>9s}")
    for M, b in ((1.0, 30 * co.beta_crit(1.0)), (1e2, 1.1e-6 * (1e-2) ** (-17 / 24)), (1e4, 1.1e-6),
                 (1e6, 1.1e-6 * 1e2 ** (-17 / 24)), (1e8, 1.1e-6 * 1e4 ** (-17 / 24))):
        s = co.scales(M, b)
        xD = float(dm.kD_over_aH(s["T_evap"]))
        xP = s["k_PBH"] / s["k_eva"]
        c = cl.Clusters(M, b)
        print(f"{M:8.0e} {b:9.2e} {s['T_evap']:9.2e} {thermalization_ratio(M, b, xD):14.3f} "
              f"{thermalization_ratio(M, b, xP):9.3f} {mfp_over_cluster(M, b):10.2e} "
              f"{float(c.sigma_v(c.N_star)) ** 2:9.1e}")
