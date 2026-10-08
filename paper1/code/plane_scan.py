"""
Module G, part 1: the (M_in, beta_f) plane for monochromatic PBH reheating.

On a log grid of M_in (1 g .. 3e8 g) and beta_f (3 beta_c .. 1) this records, with damping
and module F on (hydro=True):
  NL-A+F, NL-B+F   the cluster channels (this work; NL-B+F the upper bracket)
  LIN-UV           the linear extrapolation to k_PBH (the 2012.08151 bound uses it)
  SHOT (Choptuik)  the shot-noise floor of a realistic (critical-collapse) population
  NL-A+F-sh, NL-B+F-sh   the cluster channels with shock-limited sound (sound_lifetime.py)
each as integrated and peak Omega_GW h^2 today and the peak frequency, and the full
spectra on a common frequency grid for the detector overlay (detectors.py).  H tau_sh of
the two cluster channels is saved as Htau_A, Htau_B.
Writes ../results/plane_scan.csv and plane_spectra.npz.  ~5 minutes, one thread.
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import phase_a as pa          # noqa: E402  (first: sets sys.path for cosmology)
import cosmology as co        # noqa: E402

M_GRID = np.geomspace(1.0, 3e8, 22)
N_BETA = 16
F_GRID = np.geomspace(1e-7, 1e7, 400)        # Hz, common grid for the saved spectra
SAVED = ("NL-A+F", "NL-B+F", "SHOT-Ch", "NL-A+F-sh", "NL-B+F-sh")


def summary(f, d):
    i = int(np.argmax(d))
    return float(np.trapezoid(d, np.log(f))), float(d[i]), float(f[i])


def on_grid(f, d):
    """Spectrum on F_GRID (log-log interpolation, zero outside the computed range)."""
    out = np.exp(np.interp(np.log(F_GRID), np.log(f), np.log(np.maximum(d, 1e-300))))
    return np.where((F_GRID >= f[0]) & (F_GRID <= f[-1]), out, 0.0)


def main():
    t0 = time.time()
    # the Choptuik coherent factor depends on k/k_eva only: one table for the whole plane
    x_hi = max(co.scales(M, 1.0)["k_PBH"] / co.scales(M, 1.0)["k_eva"] for M in (M_GRID[0], M_GRID[-1]))
    pa.CHOPTUIK.tabulate(1e-2, 10.0 * x_hi)
    print(f"Choptuik table to x = {10 * x_hi:.2e} ({time.time() - t0:.0f} s)", flush=True)
    rows = ["M_g,beta,beta_over_bc,channel,int_h2_damped,peak_h2_damped,f_peak"]
    spec = {n: np.zeros((M_GRID.size, N_BETA, F_GRID.size)) for n in SAVED}
    betas = np.zeros((M_GRID.size, N_BETA))
    htau = {"A": np.zeros((M_GRID.size, N_BETA)), "B": np.zeros((M_GRID.size, N_BETA))}
    for i, M in enumerate(M_GRID):
        bc = co.beta_crit(M)
        for j, b in enumerate(np.geomspace(3 * bc, 1.0, N_BETA)):
            betas[i, j] = b
            _, f, res, info, _ = pa.run_benchmark(M, b, "mono", nx=60, damped_only=True,
                                                  only=("LIN-UV", "NL-A", "NL-B"), hydro=True)
            _, fc, rc, _, _ = pa.run_benchmark(M, b, "choptuik", nx=60, damped_only=True, only=("SHOT",))
            _, fs, rs, ins, _ = pa.run_benchmark(M, b, "mono", nx=60, damped_only=True,
                                                 only=("NL-A", "NL-B"), hydro=True, lifetime=True)
            htau["A"][i, j], htau["B"][i, j] = ins["Htau"]["NL-A"], ins["Htau"]["NL-B"]
            curves = {"LIN-UV": (f, res["LIN-UV"][1]), "NL-A+F": (f, res["NL-A"][1]),
                      "NL-B+F": (f, res["NL-B"][1]), "SHOT-Ch": (fc, rc["SHOT"][1]),
                      "NL-A+F-sh": (fs, rs["NL-A"][1]), "NL-B+F-sh": (fs, rs["NL-B"][1])}
            for name, (ff, d) in curves.items():
                I, pk, fp = summary(ff, d)
                rows.append(f"{M:.4e},{b:.4e},{b / bc:.4e},{name},{I:.4e},{pk:.4e},{fp:.4e}")
                if name in spec:
                    spec[name][i, j] = on_grid(ff, d)
        print(f"M={M:9.3e} done  ({time.time() - t0:.0f} s)", flush=True)
    with open(os.path.join(pa.OUT, "plane_scan.csv"), "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    np.savez(os.path.join(pa.OUT, "plane_spectra.npz"), M=M_GRID, beta=betas, f=F_GRID,
             Htau_A=htau["A"], Htau_B=htau["B"],
             **{n.replace("-", "_").replace("+", "p"): s for n, s in spec.items()})


if __name__ == "__main__":
    main()
