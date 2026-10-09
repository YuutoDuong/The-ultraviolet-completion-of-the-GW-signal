"""
Module G, part 2: which detectors see the signal where in the (M_in, beta_f) plane.

Reads results/plane_spectra.npz (plane_scan.py) and, for each point and detector, the
detection ratio R = max_f Omega_GW h^2 / Omega_PLI (detectors.py; SNR 1, 1 yr).  A point is
called detectable at R >= RHO = 10 (SNR 10 in one year).  Adds the merged-relic limit of
module H (relics.py): points with beta_f above beta_relic(M_in) are excluded.
Writes ../results/detect_map.csv and prints, per mass, the smallest detectable beta_f, for
the cluster channel with sound that lives a Hubble time (NL-A+F) and with shock-limited
sound (NL-A+F-sh, sound_lifetime.py).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import detectors as dt       # noqa: E402
import relics as rl          # noqa: E402
import mergers as mg         # noqa: E402

OUT = os.path.join(HERE, "..", "results")
RHO = 10.0
z = np.load(os.path.join(OUT, "plane_spectra.npz"))
M, B, F = z["M"], z["beta"], z["f"]
CH = {"NL-A+F": z["NL_ApF"], "NL-B+F": z["NL_BpF"], "SHOT-Ch": z["SHOT_Ch"],
      "NL-A+F-sh": z["NL_ApF_sh"], "NL-B+F-sh": z["NL_BpF_sh"]}
PLI = {n: dt.omega_pli(n, F) for n in dt.NAMES}


def ratio(spec):
    """R[i, j, detector] for one channel."""
    return {n: np.max(spec / PLI[n][None, None, :], axis=2) for n in dt.NAMES}


def threshold(r, b):
    """Smallest beta with R >= RHO, log-interpolated between grid points (None if never).
    The beta grid steps by x2-8, so the first grid point above threshold overestimates it."""
    ok = np.nonzero(r >= RHO)[0]
    if not len(ok):
        return None
    j = ok[0]
    if j == 0:
        return float(b[0])
    x0, x1, y0, y1 = np.log(b[j - 1]), np.log(b[j]), np.log(r[j - 1]), np.log(r[j])
    return float(np.exp(x0 + (np.log(RHO) - y0) * (x1 - x0) / (y1 - y0)))


def mass_edges(rmax):
    """Masses where max_beta R crosses RHO, log-interpolated between grid masses."""
    lm, lr = np.log10(M), np.log10(np.maximum(rmax, 1e-300))
    out = []
    for i in range(M.size - 1):
        if (lr[i] - np.log10(RHO)) * (lr[i + 1] - np.log10(RHO)) < 0:
            out.append(10 ** (lm[i] + (np.log10(RHO) - lr[i]) * (lm[i + 1] - lm[i]) / (lr[i + 1] - lr[i])))
    return out


def main():
    R = {ch: ratio(s) for ch, s in CH.items()}
    b_relic = np.array([rl.beta_relic(m) for m in M])
    rows = ["M_g,beta,relic_excluded," + ",".join(f"R_{ch}_{n}" for ch in CH for n in dt.NAMES)]
    for i in range(M.size):
        for j in range(B.shape[1]):
            excl = bool(np.isfinite(b_relic[i]) and B[i, j] >= b_relic[i])
            vals = ",".join(f"{R[ch][n][i, j]:.3e}" for ch in CH for n in dt.NAMES)
            rows.append(f"{M[i]:.4e},{B[i, j]:.4e},{int(excl)},{vals}")
    with open(os.path.join(OUT, "detect_map.csv"), "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    allowed = np.array([B[i] < (b_relic[i] if np.isfinite(b_relic[i]) else np.inf) for i in range(M.size)])
    for chn, what in (("NL-A+F", "sound lives a Hubble time"), ("NL-A+F-sh", "shock-limited sound")):
        print(f"\ndetectable = R >= {RHO:g} (SNR {RHO:g}, 1 yr); {chn}: clusters with module F, {what};"
              " thresholds log-interpolated between grid points; P = merged fraction there (mergers.py);"
              " * = threshold above the merged-relic limit")
        print(f"{'M_in [g]':>10s} {'beta_relic':>10s} {'smallest detectable beta (detector)':>40s} {'P':>6s} {'best R at beta_max allowed':>28s}")
        for i in range(M.size):
            best = []
            for n in dt.NAMES:
                b = threshold(R[chn][n][i], B[i])
                if b is not None:
                    best.append((b, n))
            jmax = np.nonzero(allowed[i])[0][-1]
            Rbest = max((R[chn][n][i, jmax], n) for n in dt.NAMES)
            star = "*" if best and np.isfinite(b_relic[i]) and min(best)[0] >= b_relic[i] else " "
            s = f"{min(best)[0]:.2e}{star}({', '.join(n for b, n in sorted(best) if b <= 3 * min(best)[0])})" if best else "none"
            P = f"{float(mg.merged_fraction(M[i], min(best)[0])[0]):6.3f}" if best else f"{'':6s}"
            print(f"{M[i]:10.3e} {b_relic[i]:10.2e} {s:>40s} {P} {Rbest[0]:15.2e} ({Rbest[1]})")
        for n in dt.NAMES:
            rmax = np.where(allowed, R[chn][n], 0.0).max(axis=1)
            if rmax.max() >= 1.0:
                print(f"   {n:7s} largest R where the relic limit allows: {rmax.max():.2e};"
                      f" R >= {RHO:g} between {', '.join(f'{e:.1e}' for e in mass_edges(rmax)) or '-'} g")
    for ch in ("SHOT-Ch",):
        top = max((np.max(R[ch][n]), n) for n in dt.NAMES)
        print(f"{ch}: largest R anywhere in the plane = {top[0]:.2e} ({top[1]})")


if __name__ == "__main__":
    main()
