"""
Log-normal width scan: how narrow must the PBH mass function be for the monochromatic
(cluster) result to apply?  For each benchmark and width sigma, the damped GW peak and
integral of the coherent cluster channel (NL-A, and NL-A+F with module F's burst
efficiency), the linear channel cut at k_NL, and the shot-noise channel.  M_in is the
mass-weighted mean mass (suddenness.py Eq. 3).  Writes ../results/width_scan.csv, and
../results/width_trunc_check.txt (5-sigma truncation of the log-normal vs none).
~20 minutes, one thread.

Expectation (suddenness.py Eq. 3): the coherent hand-over at the cluster scale needs
w sigma_tau << 1, i.e. sigma << 1/(3 w tau) ~ 1/x_cl; above that only shot noise is left.
For sigma >~ 0.1 the explosion times spread over more than ~1/3 of a Hubble time, which
the single-epoch mapping of suddenness.py treats only to O(1) (shot-noise channel).
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase_a as pa          # noqa: E402  (sets sys.path for cosmology)
import suddenness as sd       # noqa: E402

SIGMAS = [1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
POINTS = []
for M in (1e2, 1e4, 1e6):
    POINTS.append((M, 1.1e-6 * (M / 1e4) ** (-17 / 24), "2012.08151 bound"))
    POINTS.append((M, 1e-2, "saturated (beta=1e-2)"))


def mf_for(sigma, truncate=5.0):
    if sigma <= 0.1:
        return sd.LogNormalMF(sigma, truncate=truncate)
    return sd.lognormal(sigma)        # M-grid route for broad distributions


def integral(f, d):
    return float(np.trapezoid(d, np.log(f)))


def main():
    t0 = time.time()
    rows = ["M_g,beta,label,sigma,x_cl,channel,peak_h2_damped,f_peak,int_h2_damped"]
    for (M, b, lab) in POINTS:
        for sig in ["mono"] + SIGMAS + ["choptuik"]:
            pop = sig if isinstance(sig, str) else mf_for(sig)
            xs, f, res, info, xD = pa.run_benchmark(M, b, pop, xi=1.0, nx=70, damped_only=True,
                                                    only=("NL-A", "LIN-NL", "SHOT"))
            _, _, resF, _, _ = pa.run_benchmark(M, b, pop, xi=1.0, nx=70, damped_only=True,
                                                hydro=True, only=("NL-A",))
            res["NL-A+F"] = resF["NL-A"]
            for name in ("NL-A", "NL-A+F", "LIN-NL", "SHOT"):
                d = res[name][1]
                rows.append(f"{M:.1e},{b:.4e},{lab},{sig},{info['xcl']:.4e},{name},"
                            f"{d.max():.4e},{f[np.argmax(d)]:.4e},{integral(f, d):.4e}")
            a = rows[-4:]
            print(f"M={M:.0e} {lab:22s} sigma={sig!s:9s} " +
                  "  ".join(r.split(',')[5] + ' ' + r.split(',')[8] for r in a), flush=True)
    with open(os.path.join(pa.OUT, "width_scan.csv"), "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")

    # truncation check: the 5-sigma cut (He et al., evaporation.make_lognormal) vs none
    lines = ["NL-A integral, 5-sigma truncated vs untruncated log-normal",
             f"{'point':34s} {'sigma':>8s} {'trunc 5':>11s} {'none':>11s} {'ratio':>8s}"]
    for (M, b, lab) in [POINTS[2], POINTS[3], POINTS[1]]:
        for sig in (3e-5, 1e-4, 3e-4, 1e-3, 3e-3):
            v = []
            for tr in (5.0, None):
                xs, f, res, info, xD = pa.run_benchmark(M, b, mf_for(sig, tr), xi=1.0, nx=70,
                                                        damped_only=True, only=("NL-A",))
                v.append(integral(f, res["NL-A"][1]))
            lines.append(f"M={M:.0e} {lab:28s} {sig:8.0e} {v[0]:11.4e} {v[1]:11.4e} {v[1] / v[0]:8.4f}")
            print(lines[-1], flush=True)
    with open(os.path.join(pa.OUT, "width_trunc_check.txt"), "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
