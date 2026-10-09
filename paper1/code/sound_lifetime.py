"""
Module L -- how long the sound field lives before it shocks (added 2026-10-07; RECHECK P1-17).

The GW calculation (acoustic_gw.py) takes the sound waves left at evaporation to live for a
Hubble time (Upsilon = 1), as the induced-GW literature does.  A sound field of rms fluid
velocity U_f and characteristic length L instead steepens into shocks after

    tau_sh = L / U_f                       (Ellis, Lewicki, No 2003.07360; Caprini et al.
                                            1910.13125, with L the mean bubble separation)

and then decays as acoustic turbulence (Dahl, Hindmarsh, Rummukainen, Weir 2112.12013).  A
sound-wave GW source that switches off at tau_sh in radiation domination accumulates

    Upsilon_sh = int_1^y_end dy / y^2 = 1 - 1/y_end,    y_end = sqrt(1 + 2 H tau_sh)

(Guo, Sinha, Vagie, White 2007.08537), i.e. ~ H tau_sh for H tau_sh << 1.  With damping the
cutoff goes inside the Upsilon_once integral: acoustic_gw.Upsilon(p, y_end=...).
(y = a/a_ev starts at 1; a ~ (t - t0)^(1/2) with H = H_ev at y = 1, so y^2 = 1 + 2 H_ev t.)

Choices for the paper's shock-limited estimate:
  U_f = sqrt(<v^2>), <v^2> = int_{z > 1} dln z P_v(z), from the ENERGY spectrum of the channel
        (module F energy efficiency), i.e. the actual sound energy K = Gamma <v^2>; only modes
        inside the horizon at evaporation (z = q/k_eva > 1) count.
  L   = d_*, the mean separation of typical clusters, (4 pi N_*/3)^(1/3) / k_PBH: the analogue
        of the mean bubble separation of phase transitions.  d_* = (4 pi Delta/3)^(1/3) R_cl(N_*)/xi
        ~ 9 R_cl(N_*) ~ 2/k_NL.  In Hubble units at evaporation, L H_ev = d_* k_eva.
Variant (not used for the figures): L = integral scale of P_v (~ R_cl), which makes tau_sh about
ten times shorter.  Not included: GW production by the decaying acoustic turbulence after
tau_sh, which could restore a factor of a few.  The same estimate applies to every channel;
we apply it to the cluster channels and to the linear spectrum cut at k_NL.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import acoustic_gw as ag   # noqa: E402


def d_star(c):
    """Mean separation of typical clusters in units of 1/k_eva (c: clusters.Clusters)."""
    return (4.0 * np.pi * c.N_star / 3.0) ** (1.0 / 3.0) * c.k_eva / c.k_PBH


def rms_velocity(Pdr, x_hi, n=3000):
    """U_f from a radiation-contrast spectrum Pdr(z), modes inside the horizon, z in [1, x_hi]."""
    z = np.exp(np.linspace(0.0, np.log(x_hi), n))
    v2 = np.trapezoid(ag.PV_OVER_PDELTA * Pdr(z), np.log(z))
    return float(np.sqrt(v2))


def integral_scale(Pdr, x_hi, n=3000):
    """P_v-weighted mean of 1/z (the variant length), in units of 1/k_eva."""
    z = np.exp(np.linspace(0.0, np.log(x_hi), n))
    Pv = ag.PV_OVER_PDELTA * Pdr(z)
    return float(np.trapezoid(Pv / z, np.log(z)) / np.trapezoid(Pv, np.log(z)))


def H_tau_sh(L, U_f):
    """Shock-formation time in Hubble times at evaporation."""
    return L / U_f


def y_end(Htau):
    """Scale factor (in units of a_ev) at which the source switches off, radiation domination."""
    return np.sqrt(1.0 + 2.0 * np.asarray(Htau, dtype=float))


def upsilon_sh(Htau):
    """GW suppression of a source that lives for tau_sh (no damping)."""
    return 1.0 - 1.0 / y_end(Htau)
