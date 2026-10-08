"""
Module G, part 2: detector sensitivity from the power-law-integrated (PLI) curves of
Schmitz, "New sensitivity curves for gravitational-wave experiments" (JHEP 01 (2021) 097,
arXiv:2002.04615), data Zenodo 10.5281/zenodo.3689582 (CC BY 4.0, in papers/data/schmitz_pls;
md5 of the tarball 02f59c5538b4a2094d84cf4d9b7f0316).

Each file gives log10 f [Hz] and log10 h^2 Omega_PLI for SNR threshold 1 and one year of
observation.  For a weak signal SNR scales linearly with Omega and as sqrt(T), so the
curve for threshold rho and time T is Omega_PLI * rho / sqrt(T / 1 yr).

Detection criterion used here (standard for PLI curves): the spectrum reaches the curve,
    R = max_f  Omega_GW(f) h^2 / Omega_PLI(f) h^2  >=  rho_th sqrt(1 yr / T).
For a power law this is exact by construction; for our broad, smooth spectra it is the usual
approximation (an SNR integral needs the noise curves, the other file in the record).
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "..", "..", "papers", "data", "schmitz_pls")
NAMES = ("LISA", "DECIGO", "BBO", "ET", "CE", "HLVK")


def load(name):
    """(f [Hz], h^2 Omega_PLI) for SNR 1, T = 1 yr."""
    a = np.loadtxt(os.path.join(DIR, f"plis_{name}.dat"), comments="#")
    return 10.0 ** a[:, 0], 10.0 ** a[:, 1]


CURVES = {n: load(n) for n in NAMES}


def omega_pli(name, f):
    """PLI curve at frequencies f (inf outside the detector band)."""
    fc, oc = CURVES[name]
    lo = np.interp(np.log(f), np.log(fc), np.log(oc), left=np.inf, right=np.inf)
    return np.exp(lo)


def detection_ratio(name, f, omega_h2):
    """R = max_f Omega/Omega_PLI (SNR-1, 1 yr); R >= rho_th means detectable at that SNR."""
    return float(np.max(np.asarray(omega_h2) / omega_pli(name, np.asarray(f))))


if __name__ == "__main__":
    for n in NAMES:
        f, o = CURVES[n]
        i = np.argmin(o)
        print(f"{n:7s} band {f[0]:.1e}-{f[-1]:.1e} Hz, minimum h2 Omega_PLI = {o[i]:.2e} at {f[i]:.2e} Hz")
