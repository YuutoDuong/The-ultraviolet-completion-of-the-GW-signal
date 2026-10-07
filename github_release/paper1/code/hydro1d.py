"""
Module F -- one cluster's burst in 1D spherical relativistic hydrodynamics.

Question: inside a cluster the radiation contrast at the release is delta_r ~ 3-30,
while the GW machinery (acoustic_gw.py, sigw_kernel.py) is linear in the sound field.
How much sound energy does a strongly over-pressured sphere of radiation really put into
the outgoing shell, compared with linear acoustics?  The answer is an efficiency
kappa(delta) = E_ac(non-linear)/E_ac(linear), applied to each cluster in the one-halo
term with the GW spectral-shift factor of hydro_gw_shift.py (burst_efficiency.py).

----------------------------------------------------------------------------
EQUATIONS
----------------------------------------------------------------------------
Perfect radiation fluid, p = rho/3, flat space (cluster radius << Hubble radius, and the
burst lasts a few sound-crossing times), spherical symmetry.  Conserved variables
    E = T^00 = rho (4 gamma^2 - 1)/3,     S = T^0r = (4/3) rho gamma^2 v,
    dE/dt + r^-2 d(r^2 S)/dr       = q(r,t)       (Hawking energy injected at rest)
    dS/dt + r^-2 d(r^2 (S v + p))/dr = 2 p / r.
Primitive recovery in closed form: with a = S/E,  v = (2 - sqrt(4 - 3a^2))/a,
rho = 3 E (1 - v^2)/(3 + v^2).
Finite volumes on a uniform r-grid, HLLE fluxes with signal speeds
(v +- c_s)/(1 +- v c_s), MUSCL (minmod) reconstruction of (ln rho, v), SSP-RK2.  The
geometric term is discretised as p_i (r_{i+1/2}^2 - r_{i-1/2}^2), which keeps a uniform
fluid at rest exactly static (TEST F1).

Linear acoustic energy of a perturbation (w = rho + p, c_s^2 = 1/3):
    E_ac = int dV [ w v^2/2 + c_s^2 drho^2/(2 w) ],
so a top-hat delta(r) = delta0 Theta(R - r) at rest carries E_ac,lin = delta0^2 rho0 V/8.
The exact linear solution is the spherical N-wave r u = [(r-ct) f(|r-ct|) + (r+ct) f(r+ct)]/2
(TEST F2).
"""

import numpy as np

CS2 = 1.0 / 3.0
CS = np.sqrt(CS2)


def prim(E, S):
    a = np.clip(S / E, -0.999999, 0.999999)
    small = np.abs(a) < 1e-8
    v = np.where(small, 0.75 * a, (2.0 - np.sqrt(4.0 - 3.0 * a * a)) / np.where(small, 1.0, a))
    rho = 3.0 * E * (1.0 - v * v) / (3.0 + v * v)
    return rho, v


def cons(rho, v):
    g2 = 1.0 / (1.0 - v * v)
    return rho * (4.0 * g2 - 1.0) / 3.0, (4.0 / 3.0) * rho * g2 * v


def _minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


class Burst:
    """
    Grid r in [0, r_max] with n cells; ambient rho0 = 1.  Initial perturbation either a
    top-hat overdensity delta0 at rest (sudden release), or a source term releasing
    delta_tot over a time s0 with the single-hole history dE ~ s^(-2/3) ds (monochromatic).
    """

    def __init__(self, R=1.0, r_max=30.0, n=6000, delta0=0.0, edge=1.5,
                 source_delta=0.0, source_s0=0.0):
        self.R, self.n = R, n
        self.rf = np.linspace(0.0, r_max, n + 1)               # faces
        self.r = 0.5 * (self.rf[1:] + self.rf[:-1])            # centres
        self.dr = self.rf[1] - self.rf[0]
        self.V = (self.rf[1:] ** 3 - self.rf[:-1] ** 3) / 3.0  # per 4 pi
        self.A = self.rf ** 2
        w = 0.5 * (1.0 - np.tanh((self.r - R) / (edge * self.dr)))   # smoothed top-hat
        self.shape = w
        rho = 1.0 + delta0 * w
        self.E, self.S = cons(rho, np.zeros(n))
        self.t = 0.0
        self.src_delta, self.src_s0 = source_delta, source_s0
        self.injected = 0.0

    # --- reconstruction and fluxes
    def _faces(self, E, S):
        rho, v = prim(E, S)
        lr = np.log(rho)
        # ghost cells: reflecting at r=0, outflow at r_max
        lr_g = np.concatenate(([lr[1], lr[0]], lr, [lr[-1], lr[-1]]))
        v_g = np.concatenate(([-v[1], -v[0]], v, [v[-1], v[-1]]))
        dl = _minmod(lr_g[1:-1] - lr_g[:-2], lr_g[2:] - lr_g[1:-1])
        dv = _minmod(v_g[1:-1] - v_g[:-2], v_g[2:] - v_g[1:-1])
        # left/right states at faces 0..n (cell i spans faces i, i+1); cells incl. 1 ghost each side
        lrc, vc = lr_g[1:-1], v_g[1:-1]
        lL = (lrc + 0.5 * dl)[:-1]
        vL = np.clip((vc + 0.5 * dv)[:-1], -0.999, 0.999)
        lR = (lrc - 0.5 * dl)[1:]
        vR = np.clip((vc - 0.5 * dv)[1:], -0.999, 0.999)
        return np.exp(lL), vL, np.exp(lR), vR, rho, v

    @staticmethod
    def _flux(rho, v):
        E, S = cons(rho, v)
        p = rho / 3.0
        return S, S * v + p, E, S

    def rhs(self, E, S):
        rL, vL, rR, vR, rho, v = self._faces(E, S)
        FEL, FSL, UEL, USL = self._flux(rL, vL)
        FER, FSR, UER, USR = self._flux(rR, vR)
        lamL = np.minimum((vL - CS) / (1 - vL * CS), (vR - CS) / (1 - vR * CS))
        lamR = np.maximum((vL + CS) / (1 + vL * CS), (vR + CS) / (1 + vR * CS))
        bm, bp = np.minimum(lamL, 0.0), np.maximum(lamR, 0.0)
        den = np.where(bp - bm > 1e-14, bp - bm, 1.0)
        FE = (bp * FEL - bm * FER + bp * bm * (UER - UEL)) / den
        FS = (bp * FSL - bm * FSR + bp * bm * (USR - USL)) / den
        FE[0], FS[0] = 0.0, FS[0]           # no energy flux through r = 0 (A = 0 anyway)
        dE = -(self.A[1:] * FE[1:] - self.A[:-1] * FE[:-1]) / self.V
        p = rho / 3.0
        dS = (-(self.A[1:] * FS[1:] - self.A[:-1] * FS[:-1]) + p * (self.A[1:] - self.A[:-1])) / self.V
        return dE, dS

    def _inject(self, t0, t1):
        """Source released in [t0, t1]: delta_tot [(s(t0)/s0)^(1/3) - (s(t1)/s0)^(1/3)]."""
        if self.src_delta <= 0 or t0 >= self.src_s0:
            return
        s_a = max(self.src_s0 - t0, 0.0)
        s_b = max(self.src_s0 - t1, 0.0)
        dd = self.src_delta * ((s_a / self.src_s0) ** (1 / 3) - (s_b / self.src_s0) ** (1 / 3))
        self.E = self.E + dd * self.shape
        self.injected += dd

    def step(self, cfl=0.4):
        dt = cfl * self.dr
        self._inject(self.t, self.t + 0.5 * dt)          # Strang split: half source
        E0, S0 = self.E, self.S
        k1E, k1S = self.rhs(E0, S0)
        E1, S1 = E0 + dt * k1E, S0 + dt * k1S
        k2E, k2S = self.rhs(E1, S1)
        self.E = 0.5 * (E0 + E1 + dt * k2E)
        self.S = 0.5 * (S0 + S1 + dt * k2S)
        self._inject(self.t + 0.5 * dt, self.t + dt)
        self.t += dt

    def run(self, t_end, cfl=0.4):
        while self.t < t_end - 1e-12:
            self.step(cfl)
        return self

    # --- diagnostics
    def excess_energy(self):
        """int dV (E - 1), per 4 pi; equals the injected energy (TEST F3)."""
        return float(np.sum((self.E - 1.0) * self.V))

    def acoustic_energy(self, r_lo=None, rho_bg=1.0):
        """Quadratic sound energy per 4 pi outside r_lo (default: outside the cluster)."""
        rho, v = prim(self.E, self.S)
        w = (4.0 / 3.0) * rho_bg
        dens = 0.5 * w * v * v + CS2 * (rho - rho_bg) ** 2 / (2.0 * w)
        m = self.r > (r_lo if r_lo is not None else 2.0 * self.R)
        return float(np.sum(dens[m] * self.V[m]))


def linear_acoustic_energy(delta0, R=1.0):
    """E_ac,lin = delta0^2 rho0 V/8, per 4 pi (V/4pi = R^3/3)."""
    return delta0 ** 2 * (R ** 3 / 3.0) / 8.0


def efficiency(delta0, t_end=None, n=6000, r_max=None, R=1.0):
    """kappa(delta0): outgoing-shell sound energy over the linear prediction."""
    t_end = t_end if t_end is not None else 12.0 * R
    r_max = r_max if r_max is not None else R + 1.2 * t_end + 4 * R
    b = Burst(R=R, r_max=r_max, n=n, delta0=delta0).run(t_end)
    lin = Burst(R=R, r_max=r_max, n=n, delta0=1e-4).run(t_end)
    k_lin_num = lin.acoustic_energy() / linear_acoustic_energy(1e-4, R)
    return b.acoustic_energy() / linear_acoustic_energy(delta0, R), k_lin_num, b
