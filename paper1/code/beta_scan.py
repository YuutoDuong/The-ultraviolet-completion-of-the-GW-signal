"""
Phase A follow-up: where does the physical (non-linear, damped) monochromatic signal
reach the Delta N_eff limit?  Compares with the 2012.08151 bound, which uses the
linear spectrum extrapolated to k_PBH (our LIN-UV channel).  NL-A+F and NL-B+F are the
cluster channels with module F's non-linear burst efficiency (burst_efficiency.py).

Delta N_eff < 0.3 -> int dlnf Omega_GW,0 h^2 < 5.6e-6 * 0.3 = 1.7e-6.
The beta grid (33 points from 3 beta_c to 3e-2, then six up to beta = 1) contains the
earlier 12-point grid, so the values quoted in the paper are unchanged by the finer grid.
NX = 70 frequencies resolve the integrals to 0.1%, except for LIN-NL: it ends at its peak, at
its cutoff 2 c_s k_NL, and needs NX_NL = 400 (70 points miss its integral by up to x2.5).
Writes ../results/beta_scan.csv.  ~15 minutes, one thread.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import cosmology as co          # noqa: E402
import phase_a as pa            # noqa: E402

LIMIT = 5.6e-6 * 0.3
NX, NX_NL = 70, 400


def integrated(f, h2):
    return float(np.trapezoid(h2, np.log(f)))


rows = ["M_g,beta,beta_over_bc,beta_2012_08151,channel,xi,int_h2_damped,peak_h2_damped,f_peak"]
for M in (1.0, 1e1, 1e2, 1e4, 1e6):
    b_old = 1.1e-6 * (M / 1e4) ** (-17 / 24)
    bc = co.beta_crit(M)
    for beta in list(np.geomspace(3 * bc, 3e-2, 33)) + [0.05, 0.1, 0.17, 0.3, 0.55, 1.0]:   # up to beta = 1
        for xi in (1.0, 1.8):
            xs, f, res, info, xD = pa.run_benchmark(M, beta, "mono", xi=xi, nx=NX, damped_only=True,
                                                    only=("LIN-UV", "NL-A", "NL-B"))
            _, _, resF, _, _ = pa.run_benchmark(M, beta, "mono", xi=xi, nx=NX, hydro=True,
                                                only=("NL-A", "NL-B"), damped_only=True)
            spec = {n: (f, v[1]) for n, v in res.items()}
            spec.update({n + "+F": (f, v[1]) for n, v in resF.items()})
            if xi == 1.0:
                _, fl, rl, _, _ = pa.run_benchmark(M, beta, "mono", nx=NX_NL, damped_only=True, only=("LIN-NL",))
                spec["LIN-NL"] = (fl, rl["LIN-NL"][1])
            for name in ("LIN-UV", "LIN-NL", "NL-A", "NL-B", "NL-A+F", "NL-B+F"):
                if xi == 1.8 and name.startswith("LIN"):
                    continue
                f, d = spec[name]
                I = integrated(f, d)
                rows.append(f"{M:.1e},{beta:.4e},{beta / bc:.3e},{b_old:.3e},{name},{xi},"
                            f"{I:.4e},{d.max():.4e},{f[np.argmax(d)]:.4e}")
                print(rows[-1], "  <-- over limit" if I > LIMIT else "", flush=True)
with open(os.path.join(pa.OUT, "beta_scan.csv"), "w", newline="\n") as fh:
    fh.write("\n".join(rows) + "\n")
