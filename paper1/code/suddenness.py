"""
Module C -- how sudden is the hand-over of PBH mass to radiation, scale by scale?

Replaces the decoupling-time estimate of Stage 0 (Gamma(s_dec) = k/a) by an exact
statement of linear acoustics, and adds the incoherent (shot-noise) part that a
single-fluid treatment drops.

----------------------------------------------------------------------------
DERIVATION
----------------------------------------------------------------------------
(1) Radiation fluid with an energy source q(x,t), flat space, k >> aH (gravity and
    expansion negligible over the release):
        d_t^2 drho - c_s^2 lap drho = d_t q.
    After the source has switched off, a Fourier mode oscillates with amplitude
        |drho_k| = | int dt q_k(t) e^{-i w t} |,   w = c_s k/a   (integrate by parts).
    An instantaneous release of the same energy gives |q_k(w=0)|.  So the fraction of
    the instantaneous sound amplitude that survives is the normalised Fourier
    transform of the release history at the sound frequency.
(2) One hole, initial mass M0, lifetime tau = M0^3/(3 kappa), s = tau - t:
        q(s) = (M0/3) tau^(-1/3) s^(-2/3),
        |q~(w)| -> M0 * Gamma(1/3) / (3 (w tau)^(1/3))        (w tau >> 1).
    Since M0 tau^(-1/3) = (3 kappa)^(1/3) for every hole, |q~| is the SAME for all
    masses: (Gamma(1/3)/3) (3 kappa / w)^(1/3).
(3) Many holes at positions x_i.  The power of drho_r splits into
        coherent:   [P_PBH(k) - 1/n] |<q~>|^2 / <M0>^2,   <q~> carries e^{-i w tau_i}
        incoherent: (1/n) <|q~|^2> / <M0>^2.
    With (2):
        S_coh(w)   = (Gamma(1/3)/3) (3 kappa/w)^(1/3) |phi(w)| / <M0>,
        S_inc^2(w) = (Gamma(1/3)/3)^2 (3 kappa/w)^(2/3) / <M0>^2
                   = 0.797 (w tau(<M0>))^(-2/3),
    where <.> is the NUMBER-weighted mean and phi(w) = <D(tau)/D_ev e^{-i w tau}> is
    the characteristic function of the lifetime distribution, weighted by the linear
    growth D ~ t^(2/3) reached when each hole goes off (coherent modes only).
    Monochromatic: |phi| = 1 and S_coh = S_inc = Gamma(1/3)/(3 (w tau)^(1/3)).
    The 1/n of the incoherent part is the true number density.  A benchmark's M_in is
    the mass-weighted mean M_mw = <M^2>/<M> (tau(M_mw) = t_eff below), which also sets
    the isocurvature seed of the coherent part, so P_shot = shot_ratio * P_S(M_in) with
    shot_ratio = <M>/M_mw = <M>^2/<M^2> (e^-sigma^2 for a log-normal, 0.93 Choptuik).
    A sharp upper edge of the lifetime distribution gives |phi| ~ p(tau_cut)/w, hence
    S_coh ~ w^(-4/3): the universal suppression of 2606.09804 / 2605.21474, recovered
    here from one line of Fourier analysis.  S_inc keeps the single-hole w^(-1/3):
    THE SHOT-NOISE PART IS NOT UNIVERSALLY SUPPRESSED.
(4) Mapping to k/k_eva: w = c_s x H_ev with x = k/k_eva and H_ev = 2/(3 t_eff),
    t_eff^(1/3) = mass-weighted mean of tau^(1/3) (the <eta_eva> of 2605.21474), so a
    monochromatic population has t_eff = tau and w tau = (2/(3 sqrt3)) x.

Units for extended mass functions follow evaporation.py: kappa = 1, f0 = number
density per unit M, M_max = largest initial mass.
"""

import sys
import os
import numpy as np
from scipy.special import gamma as Gamma
from scipy.integrate import quad

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import evaporation as ev  # noqa: E402  (Stage 0)

CS = 1.0 / np.sqrt(3.0)
G13 = Gamma(1.0 / 3.0) / 3.0          # 0.89298, single-hole coefficient in (2)


def single_hole_S(wtau):
    """
    |q~(w)| / M0 for one hole, Eq. (2), asymptotic in w tau >> 1.  Capped at 1, the
    instantaneous limit, so that w tau < 1 is never credited with more than sudden.
    """
    return np.minimum(1.0, G13 * np.asarray(wtau, dtype=float) ** (-1.0 / 3.0))


def S_mono(x, Ht=2.0 / 3.0):
    """Monochromatic suppression at x = k/k_eva; H_ev tau = 2/3 in matter domination."""
    return single_hole_S(CS * np.asarray(x, dtype=float) * Ht)


def S_mono_numeric(wtau, n=400001, taper=0.3):
    """
    Direct Fourier transform of the single-hole history with a smooth (Gaussian) switch
    -on at the formation end, for validating Eq. (2).  Uses s = tau u^3, which turns
    s^(-2/3) ds into 3 tau^(1/3) du and removes the endpoint singularity.
    The taper models the gradual build-up of a coherent mode; its effect falls off as
    exp(-(w tau taper)^2/4), negligible for w tau > ~30.
    """
    u = np.linspace(0.0, 1.0, n)
    s = u ** 3                                  # time before the end, units of tau
    window = np.exp(-(s / taper) ** 2)          # ~1 at the explosion, ->0 at formation
    # q ds / M0 = (1/3) s^(-2/3) ds = du  exactly, with tau = 1
    return abs(np.trapezoid(window * np.exp(1j * wtau * s), u))


class MassFunction:
    """
    Number-density mass function f0(M0) on (0, M_max], kappa = 1 (evaporation.py units).
    Pre-computes the moments and the lifetime distribution needed by Eqs. (3)-(4).
    """
    HT = 1.0    # explosion-time spread in Hubble units (broad populations)

    def __init__(self, f0, M_max, name="", n_grid=200001):
        self.f0, self.M_max, self.name = f0, M_max, name
        M = np.linspace(0.0, M_max, n_grid)[1:]
        f = np.asarray(f0(M), dtype=float)
        self.M, self.fM = M, f
        self.n = np.trapezoid(f, M)
        self.rho = np.trapezoid(f * M, M)
        self.M_num = self.rho / self.n                           # number-weighted <M0>
        self.M_mw = np.trapezoid(f * M * M, M) / self.rho        # mass-weighted, = M_in
        self.shot_ratio = self.M_num / self.M_mw                 # Eq. (3), true 1/n
        tau = ev.lifetime(M)                                     # M^3/3
        self.tau = tau
        # Eq. (4): t_eff^(1/3) = mass-weighted mean of tau^(1/3)
        self.t_eff = (np.trapezoid(f * M * tau ** (1.0 / 3.0), M) / self.rho) ** 3
        self.H_ev = 2.0 / (3.0 * self.t_eff)
        self.tau_mean_num = ev.lifetime(self.M_num)

    def omega(self, x):
        """Physical sound frequency [code units] of the mode x = k/k_eva."""
        return CS * np.asarray(x, dtype=float) * self.H_ev

    def phi(self, w, growth=True):
        """
        |phi(w)|, Eq. (3): number-weighted characteristic function of lifetimes, each
        hole weighted by the linear growth (t/t_eff)^(2/3) reached when it goes off.
        Evaluated on the mass grid; valid while the grid resolves the phase, i.e.
        w * dtau << 1 near M_max.  phi_asym() takes over beyond that.
        """
        weight = (self.tau / self.t_eff) ** (2.0 / 3.0) if growth else 1.0
        integrand = self.fM * weight * np.exp(-1j * w * self.tau)
        return abs(np.trapezoid(integrand, self.M)) / self.n

    def phi_edge(self, w, growth=True):
        """
        Asymptotic |phi| from the sharp edge at tau_cut = tau(M_max):
            |phi| -> p(tau_cut) * D(tau_cut)/D_ev / w,
            p(tau) dtau = f0 dM / n,  dtau/dM = M^2 (kappa = 1).
        """
        tc = ev.lifetime(self.M_max)
        p_edge = float(self.f0(self.M_max)) / self.M_max ** 2 / self.n
        weight = (tc / self.t_eff) ** (2.0 / 3.0) if growth else 1.0
        return p_edge * weight / np.asarray(w, dtype=float)

    def tabulate(self, x_lo, x_hi, n=None, per_decade=100):
        """
        Cache log S_coh (default arguments) on a log grid covering [x_lo, x_hi]; S_coh
        depends on x = k/k_eva only, so one table serves every (M_in, beta).  Extends an
        existing table when the range grows.  n is ignored (LogNormalMF signature).
        """
        tab = getattr(self, "_tab", None)
        lo, hi = np.log(x_lo), np.log(x_hi)
        if tab is not None and tab[0][0] <= lo + 1e-9 and tab[0][-1] >= hi - 1e-9:
            return self
        if tab is not None:
            lo, hi = min(lo, tab[0][0]), max(hi, tab[0][-1])
        lx = np.linspace(lo, hi, int((hi - lo) / np.log(10) * per_decade) + 2)
        self._tab = None
        self._tab = (lx, np.log(np.maximum(self.S_coh(np.exp(lx)), 1e-300)))
        return self

    def S_coh(self, x, growth=True, w_switch=None):
        """Coherent suppression at x = k/k_eva, Eq. (3); numeric phi, then the edge law."""
        tab = getattr(self, "_tab", None)
        if tab is not None and growth and w_switch is None:
            return np.exp(np.interp(np.log(np.asarray(x, dtype=float)), tab[0], tab[1]))
        w = self.omega(x)
        if w_switch is None:
            dtau = np.max(np.diff(self.tau))
            w_switch = 0.05 / dtau
        ph = np.where(w < w_switch,
                      np.vectorize(lambda ww: self.phi(ww, growth))(np.minimum(w, w_switch)),
                      self.phi_edge(w, growth))
        return np.minimum(1.0, G13 * (3.0 / w) ** (1.0 / 3.0) * ph / self.M_num)

    def S_inc2(self, x):
        """Incoherent (shot-noise) suppression squared, Eq. (3): 0.797 (w tau(<M0>))^(-2/3)."""
        w = self.omega(x)
        return np.minimum(1.0, (G13 * (3.0 / w) ** (1.0 / 3.0) / self.M_num) ** 2)


class LogNormalMF:
    """
    Log-normal number density in M with median M_c and width sigma, any sigma down to
    ~1e-8, integrated in z = ln(M/M_c)/sigma so narrow distributions cost nothing.
    Units: tau_c = tau(M_c) = 1.  Same interface as MassFunction (omega, S_coh, S_inc2).

        tau(z) = e^(3 sigma z),  <M>_num = M_c e^(sigma^2/2),  M_mw = M_c e^(3 sigma^2/2),
        t_eff  = mass-weighted mean of tau^(1/3), cubed = e^(9 sigma^2/2) = tau(M_mw),
        shot_ratio = <M>_num/M_mw = e^(-sigma^2),
        S_coh  = G13 (w tau_c)^(-1/3) e^(-sigma^2/2) |phi(w)|,
        S_inc^2 = G13^2 (w tau_c)^(-2/3) e^(-sigma^2),
    phi = growth-weighted characteristic function of lifetimes, Eq. (3).
    truncate = n_sigma truncates the upper tail as evaporation.make_lognormal does
    (n_sigma = 5); the sharp edge then gives |phi| ~ 1/w at large w.  truncate=None
    keeps the full Gaussian tail; |phi| then falls faster than any power of w, and
    where the quadrature cannot resolve the phase |phi| is set to 0.
    """

    def __init__(self, sigma, truncate=5.0, z_lo=-9.0):
        self.sigma, self.trunc = sigma, truncate
        self.z_lo, self.z_hi = z_lo, (truncate if truncate is not None else 9.0)
        self.shot_ratio = np.exp(-sigma ** 2)
        self.t_eff = np.exp(4.5 * sigma ** 2)
        self.H_ev = 2.0 / (3.0 * self.t_eff)
        self.name = f"log-normal sigma={sigma:g}" + ("" if truncate else " (untruncated)")
        z = np.linspace(self.z_lo, self.z_hi, 20001)
        self._norm = np.trapezoid(np.exp(-0.5 * z ** 2), z)

    def omega(self, x):
        return CS * np.asarray(x, dtype=float) * self.H_ev

    def _phi_one(self, w):
        s = self.sigma
        span = w * (np.exp(3 * s * self.z_hi) - np.exp(3 * s * self.z_lo))   # total phase
        n = int(min(4e6, max(4001, 40 * span)))
        if n >= 4e6:
            if self.trunc is None:
                return 0.0          # unresolved and edge-free: below exp(-10^3)
            # unresolved: interior has cancelled; keep the edge term, Eq. (3)
            te = np.exp(3 * s * self.z_hi)
            ge = np.exp(-0.5 * self.z_hi ** 2) * (te / self.t_eff) ** (2.0 / 3.0)
            return ge / (w * 3 * s * te) / self._norm
        z = np.linspace(self.z_lo, self.z_hi, n)
        tau = np.exp(3 * s * z)
        g = np.exp(-0.5 * z ** 2) * (tau / self.t_eff) ** (2.0 / 3.0)
        return abs(np.trapezoid(g * np.exp(-1j * w * tau), z)) / self._norm

    def phi(self, w):
        w = np.atleast_1d(np.asarray(w, dtype=float))
        if getattr(self, "_tab", None) is not None:
            lw, lp = self._tab
            return np.exp(np.interp(np.log(w), lw, lp))
        return np.array([self._phi_one(wi) for wi in w])

    def tabulate(self, x_lo, x_hi, n=160):
        """Pre-compute log|phi| on a log grid so S_coh on many x is cheap (log-log interp)."""
        if getattr(self, "_tab_key", None) == (x_lo, x_hi, n):
            return self
        self._tab, self._tab_key = None, (x_lo, x_hi, n)
        lw = np.linspace(np.log(self.omega(x_lo)), np.log(self.omega(x_hi)), n)
        ph = np.array([self._phi_one(np.exp(v)) for v in lw])
        self._tab = (lw, np.log(np.maximum(ph, 1e-300)))
        return self

    @property
    def HT(self):
        """Spread of explosion times in Hubble units, for the continuous-creation damping."""
        return min(1.0, 2.0 * self.sigma)

    def S_coh(self, x):
        w = self.omega(x)
        return np.minimum(1.0, G13 * w ** (-1.0 / 3.0) * np.exp(-0.5 * self.sigma ** 2) * self.phi(w))

    def S_inc2(self, x):
        w = self.omega(x)
        return np.minimum(1.0, G13 ** 2 * w ** (-2.0 / 3.0) * np.exp(-self.sigma ** 2))


def choptuik(p=1.0 + 1.0 / 0.36, M_cut=1.0):
    f0, M_max = ev.make_powerlaw(p=p, M_cut=M_cut)
    return MassFunction(f0, M_max, name=f"Choptuik p={p:.2f}")


def lognormal(sigma, M_c=1.0, n_sigma=5.0):
    f0, M_max = ev.make_lognormal(sigma=sigma, M_c=M_c, n_sigma=n_sigma)
    return MassFunction(f0, M_max, name=f"log-normal sigma={sigma}")


# ------------------------------------------------- cross-check against Stage 0

def S_coh_from_stage0(mf, x, s_max_frac=0.6, n=20001):
    """
    Independent route to |Q~(w)|/rho: Fourier-transform the population release rate
    Q(s) = d rho_c/ds built from Stage 0's rho_c(s) (He et al. S.14-S.15), with the
    growth weighting and a smooth window at large s.  Slow; validation only.
    """
    tc = ev.lifetime(mf.M_max)
    s = tc * np.linspace(0.0, s_max_frac, n) ** 2           # dense near the endpoint
    s[0] = 1e-14 * tc
    rho = np.array([ev.rho_comoving_s(si, mf.f0, mf.M_max) for si in s])
    t = tc - s
    weight = np.clip(t / mf.t_eff, 0.0, None) ** (2.0 / 3.0)
    window = np.exp(-(s / (0.5 * s_max_frac * tc)) ** 4)
    Q = np.gradient(rho, s)
    w = mf.omega(x)
    val = np.trapezoid(Q * weight * window * np.exp(1j * w * s), s)
    return abs(val) / mf.rho
