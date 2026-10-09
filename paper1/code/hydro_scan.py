"""
Module F scan: sound efficiency of a cluster burst vs its local radiation contrast.

(a) Sudden release (top-hat at rest): kappa(delta0) = E_ac / E_ac,linear at t = 7R
    (shell near the collision radius ~R_*/2 for clusters filling ~1/200 of space) and
    t = 25R (late), each normalised by a linear run at the same resolution so the
    scheme's own numerical dissipation cancels.
(b) The monochromatic release history (dE ~ s^(-2/3) ds over the last s0 = 10 R/c) for
    three clusters from Phase A: fiducial 10^4 g with and without puff-up, and the
    saturated 10^2 g case.  delta_tot = Delta (s0/tau)^(1/3), tau/R = (2/3) x_cl.
Writes ../results/hydro_scan.txt and hydro_profiles.npz.  ~10-15 minutes, one thread.
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hydro1d as h   # noqa: E402

OUT = os.path.join(HERE, "..", "results")
T_COLL, T_LATE = 7.0, 25.0
S0 = 10.0                                    # release duration for the source runs
N_PER_R = 200
R_MAX = 1.0 + 1.05 * (T_LATE + S0) + 3.0
N = int(N_PER_R * R_MAX)
lines, prof = [], {}


def say(s):
    print(s, flush=True)
    lines.append(s)


def run_pair(t_off=0.0, **kw):
    """Run to t_off + T_COLL and t_off + T_LATE (t_off = end of the release)."""
    b = h.Burst(R=1.0, r_max=R_MAX, n=N, **kw)
    b.run(t_off + T_COLL)
    e1 = b.acoustic_energy()
    rho, v = h.prim(b.E, b.S)
    snap = (b.r.copy(), rho.copy(), v.copy())
    b.run(t_off + T_LATE)
    return e1, b.acoustic_energy(), snap, b


t0 = time.time()
ref1, ref2, _, _ = run_pair(delta0=1e-4)
L1, L2 = ref1 / 1e-8, ref2 / 1e-8                  # linear E_ac per delta0^2
say(f"linear reference: E_ac/delta^2 = {L1:.5f} (t=7), {L2:.5f} (t=25); exact {h.linear_acoustic_energy(1.0):.5f}")
say(f"{'delta0':>8s} {'kappa(t=7R)':>12s} {'kappa(t=25R)':>13s} {'max v at 7R':>12s}")
for d0 in (0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
    e1, e2, snap, b = run_pair(delta0=d0)
    prof[f"top_{d0}"] = np.array(snap)
    say(f"{d0:8.2f} {e1 / (L1 * d0 ** 2):12.4f} {e2 / (L2 * d0 ** 2):13.4f} {np.max(np.abs(snap[2])):12.4f}")

say("\nMonochromatic release history over the last s0 = 10 R/c (measured after the release ends):")
s0 = S0
ref1, ref2, _, _ = run_pair(t_off=s0, source_delta=1e-4, source_s0=s0)
S1, S2 = ref1 / 1e-8, ref2 / 1e-8
say(f"linear reference with the source: E_ac/delta_tot^2 = {S1:.5f} (t=7), {S2:.5f} (t=25)  "
    f"[sudden-equivalent fraction sqrt(S1/L1) = {np.sqrt(S1 / L1):.3f}]")
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import clusters as cl   # noqa: E402


def xcl(M, b, xi=1.0):
    c = cl.Clusters(M, b, xi=xi)
    return 1.0 / (c.R_cl(c.N_star) * c.k_eva)


cases = [("fiducial 1e4 g, no puff", 178.0, xcl(1e4, 1e-6)),
         ("fiducial 1e4 g, puff 1.8", 178.0 / 1.8 ** 3, xcl(1e4, 1e-6, 1.8)),
         ("saturated 1e2 g (beta=1e-2)", 178.0, xcl(1e2, 1e-2))]
for lab, Delta, xcl_ in cases:
    tau_over_R = (2.0 / 3.0) * xcl_
    dtot = Delta * (s0 / tau_over_R) ** (1.0 / 3.0)
    e1, e2, snap, b = run_pair(t_off=s0, source_delta=dtot, source_s0=s0)
    prof[f"src_{lab}"] = np.array(snap)
    say(f"{lab:28s} x_cl={xcl_:.3g} delta_tot={dtot:6.2f}  kappa(7R)={e1 / (S1 * dtot ** 2):.4f}  "
        f"kappa(25R)={e2 / (S2 * dtot ** 2):.4f}  injected={b.injected:.3f}")
say(f"\nwall time {time.time() - t0:.0f} s, grid n={N}, dr=R/{N_PER_R}")
with open(os.path.join(OUT, "hydro_scan.txt"), "w", newline="\n") as fh:
    fh.write("\n".join(lines) + "\n")
np.savez(os.path.join(OUT, "hydro_profiles.npz"), **prof)
