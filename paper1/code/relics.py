"""
Module H: the merged-relic limit on beta_f, from Holst, Krnjaic & Xiao (2412.01890).

Their public data (Zenodo 10.5281/zenodo.17210834, CC BY 4.0, in papers/data/holst_2412.01890;
md5 of the zip d7ca65c15389b1748d1eb98f9d1c7fd2) holds, on a grid of the initial PBH mass
m_i (1e-6..1e10 g, 501 points) and the start of PBH domination t_i (1e-46..1 s, 503 points),
    limit_integral = int dm  (df_BH/dm) / f_limit(m)  (+ the unmerged holes),
the merged relics' abundance weighted by the BBN / CMB / EGB / Voyager limits on evaporating
holes of their mass.  limit_integral >= 1 is excluded (their notebook, cells 32-42).  We read
the saved grid; we do not run their notebook.

Mapping to (M_in, beta_f).  Their background is pure matter domination from t_i with growth
D = a/a_i.  Ours is exact matter + radiation with D+ = 1 + 3y/2, y = a/a_eq, and
H_eq t = F(y) -> (2 sqrt2/3) y^(3/2).  Equal growth at late times gives
    t_i = (2/3)^(3/2) (2 sqrt2 / 3) / H_eq = 0.513 / H_eq,   H_eq = sqrt2 beta_f^2 H_f,
within 7% of our equality time F(1)/H_eq = 0.552/H_eq.  Their evaporation law equals ours
(G = 3.8, g_H = 108).
"""
import os
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import cosmology as co   # noqa: E402

DATA = os.path.join(HERE, "..", "..", "papers", "data", "holst_2412.01890", "results_constraint_plot.npz")
M_I = np.geomspace(1e-6, 1e10, 501)        # their cell 32
T_I = np.geomspace(1e-46, 1e0, 503)
T_I_COEF = (2.0 / 3.0) ** 1.5 * (2.0 * np.sqrt(2.0) / 3.0)      # = 0.513

_z = np.load(DATA)
_L = _z["limit_integral"]
assert _L.shape == (M_I.size, T_I.size)
_interp = RegularGridInterpolator((np.log10(M_I), np.log10(T_I)), np.log10(np.maximum(_L, 1e-30)),
                                  bounds_error=False, fill_value=None)


def t_i_seconds(M_g, beta_f):
    """Start of PBH domination in Holst et al.'s convention [s]."""
    H_eq = np.sqrt(2.0) * np.asarray(beta_f, dtype=float) ** 2 * co.H_form(M_g)
    return T_I_COEF / H_eq * co.HBAR


def limit_integral(M_g, beta_f):
    """Their limit integral at (M_in, beta_f); >= 1 means excluded by the merged relics."""
    pts = np.column_stack([np.broadcast_to(np.log10(M_g), np.shape(beta_f)).ravel(),
                           np.log10(np.ravel(t_i_seconds(M_g, beta_f)))])
    return (10.0 ** _interp(pts)).reshape(np.shape(beta_f))


def beta_relic(M_g, betas=None):
    """Smallest beta_f excluded by the merged relics at M_g (nan if none up to beta = 1)."""
    betas = np.geomspace(co.beta_crit(M_g), 1.0, 400) if betas is None else betas
    L = limit_integral(M_g, betas)
    hit = np.nonzero(L >= 1.0)[0]
    if len(hit) == 0:
        return np.nan
    i = hit[0]
    if i == 0:
        return float(betas[0])
    t = (0.0 - np.log10(L[i - 1])) / (np.log10(L[i]) - np.log10(L[i - 1]))
    return float(10 ** (np.log10(betas[i - 1]) + t * (np.log10(betas[i]) - np.log10(betas[i - 1]))))


if __name__ == "__main__":
    print(f"limit_integral >= 1 in {np.mean(_L >= 1):.1%} of their grid")
    print(f"{'M_in [g]':>10s} {'beta_c':>10s} {'beta_relic':>11s} {'t_i at beta_relic [s]':>22s}")
    for M in np.geomspace(1.0, 3e8, 18):
        b = beta_relic(M)
        print(f"{M:10.3e} {co.beta_crit(M):10.3e} {b:11.3e} {float(t_i_seconds(M, b)) if np.isfinite(b) else np.nan:22.3e}")
