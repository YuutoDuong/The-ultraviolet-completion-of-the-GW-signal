"""
Numbers quoted in the paper for the shock-limited sound (sound_lifetime.py, Sec. VII C).

(1) On the published bound (10^2, 10^4, 10^6 g) and at beta_f = 1 (1 g .. 10^6 g): U_f and
    H tau_sh of the cluster channels (NL-A+F, NL-B+F, with and without puff-up) and of the
    linear spectrum cut at k_NL; Upsilon_sh.
(2) At beta_f = 1: the integrated signal with shock-limited sound and its margin below the
    Delta N_eff limit, for the variants of beta_scan.py.
(3) Range of H tau_sh over the plane grid (plane_spectra.npz, plane_scan.py).
Writes ../results/lifetime_summary.txt.  ~1 minute, one thread.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase_a as pa            # noqa: E402
import sound_lifetime as sl     # noqa: E402

LIMIT = 5.6e-6 * 0.3
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


def integrated(f, d):
    return float(np.trapezoid(d, np.log(f)))


out("(1) U_f and H tau_sh = d_*/U_f;  Upsilon_sh = 1 - (1 + 2 H tau_sh)^(-1/2)")
out(f"{'M [g]':>8s} {'beta':>9s} {'xi':>4s} {'channel':>8s} {'U_f':>7s} {'H tau_sh':>9s} {'Ups_sh':>9s}")
pts = [(1e2, 1.1e-6 * (1e2 / 1e4) ** (-17 / 24)), (1e4, 1.1e-6), (1e6, 1.1e-6 * (1e6 / 1e4) ** (-17 / 24))]
pts += [(m, 1.0) for m in (1.0, 1e1, 1e2, 1e4, 1e6)]
for M, b in pts:
    for xi in (1.0, 1.8):
        ch, info = pa.channel_spectra(M, b, "mono", xi, hydro="energy")
        L = sl.d_star(info["c"])
        for name in ("NL-A", "NL-B", "LIN-NL"):
            if xi == 1.8 and name == "LIN-NL":
                continue
            Uf = sl.rms_velocity(ch[name][0], info["xPBH"])
            Ht = sl.H_tau_sh(L, Uf)
            lab = name if name.startswith("LIN") else name + "+F"
            out(f"{M:8.1e} {b:9.2e} {xi:4.1f} {lab:>8s} {Uf:7.3f} {Ht:9.2e} {float(sl.upsilon_sh(Ht)):9.2e}")

out("\n(2) beta_f = 1: integrated Omega h^2 and margin below Delta N_eff, sound for a Hubble time"
    " vs shock-limited")
out(f"{'M [g]':>8s} {'xi':>4s} {'channel':>8s} {'I (Ups=1)':>10s} {'margin':>9s} {'I (shock)':>10s} {'margin':>9s}")
for M in (1.0, 1e1, 1e2, 1e4, 1e6):
    for xi in (1.0, 1.8):
        _, f, r0, _, _ = pa.run_benchmark(M, 1.0, "mono", xi, nx=70, damped_only=True, hydro=True,
                                          only=("NL-A", "NL-B"))
        _, f1, r1, _, _ = pa.run_benchmark(M, 1.0, "mono", xi, nx=70, damped_only=True, hydro=True,
                                           only=("NL-A", "NL-B"), lifetime=True)
        for name in ("NL-A", "NL-B"):
            I0, I1 = integrated(f, r0[name][1]), integrated(f1, r1[name][1])
            out(f"{M:8.1e} {xi:4.1f} {name + '+F':>8s} {I0:10.2e} {LIMIT / I0:9.2e} {I1:10.2e} {LIMIT / I1:9.2e}")

z = np.load(os.path.join(pa.OUT, "plane_spectra.npz"))
HA, HB = z["Htau_A"], z["Htau_B"]
out("\n(3) plane grid (plane_spectra.npz): H tau_sh of NL-A+F from "
    f"{HA.min():.1e} to {HA.max():.1e} (Upsilon_sh {float(sl.upsilon_sh(HA.min())):.1e} to "
    f"{float(sl.upsilon_sh(HA.max())):.1e}); NL-B+F from {HB.min():.1e} to {HB.max():.1e}")
with open(os.path.join(pa.OUT, "lifetime_summary.txt"), "w", newline="\n") as fh:
    fh.write("\n".join(lines) + "\n")
