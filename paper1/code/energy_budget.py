"""
Energy budget of each channel: kinetic fraction K = Gamma <v^2> = Gamma (3/32) int dlnk P_delta_r.
A physical sound field must have K < 1 (it is a part of the total energy).  The linear
extrapolation to k_PBH behind the 2012.08151 bound violates this by orders of magnitude.
Also prints the rms radiation contrast sqrt(int P_delta_r dlnk).  NL-A+F and NL-B+F use
module F's energy efficiency (burst_efficiency.py, hydro="energy").
Writes ../results/energy_budget.txt.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase_a as pa       # noqa: E402
import acoustic_gw as ag   # noqa: E402

lines = [f"{'benchmark':34s} {'pop':9s} {'channel':7s} {'K = kinetic/total':>18s} {'rms delta_r':>12s}"]
for (M, b, lab) in pa.BENCH:
    for pop in ("mono", "choptuik"):
        ch, info = pa.channel_spectra(M, b, pop)
        chF, _ = pa.channel_spectra(M, b, pop, hydro="energy")
        ch.update({n + "+F": chF[n] for n in ("NL-A", "NL-B")})
        z = np.exp(np.linspace(np.log(0.5), np.log(1.2 * info["xPBH"]), 20000))
        for name, (Pdr, br) in ch.items():
            P = Pdr(z)
            var = np.trapezoid(P, np.log(z))
            K = ag.GAM * ag.PV_OVER_PDELTA * var
            lines.append(f"{lab + f' M={M:.0e} b={b:.1e}':34s} {pop:9s} {name:7s} {K:18.3e} {np.sqrt(var):12.3e}")
            print(lines[-1], flush=True)
with open(os.path.join(pa.OUT, "energy_budget.txt"), "w", newline="\n") as fh:
    fh.write("\n".join(lines) + "\n")
