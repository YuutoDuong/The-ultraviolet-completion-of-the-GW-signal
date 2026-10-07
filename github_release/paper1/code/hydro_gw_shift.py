"""
Module F, second number: how the non-linear burst changes the GW output beyond its energy.

A strong burst moves the shell's power to longer wavelengths (the N-wave lengthens), and
long sound waves make GWs more efficiently.  For each saved profile (t = 7R after the end of
the release) this computes the GW integral of the shell's velocity spectrum,
    P_v(k) ~ k^3 |u~(k)|^2,   u = sqrt(rho) gamma v  (T^ij = w gamma^2 v^i v^j, w ~ rho),
with acoustic_gw.P_GW, and compares it with kappa_E^2 x the linear template (kappa_E = the
energy efficiency of hydro_scan.txt), i.e. with what a constant kappa gives:
    R_shift = GW[non-linear spectrum] / (kappa_E^2 GW[linear spectrum]).
Top hats use the delta0 = 0.01 profile as the linear template; the release histories need
their own linear template, run here (delta_tot = 1e-4, same grid as hydro_scan.py, ~4 min).
Writes ../results/hydro_gw_shift.txt.
"""
import os
import re
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hydro1d as h          # noqa: E402
import acoustic_gw as ag     # noqa: E402

OUT = os.path.join(HERE, "..", "results")
T_COLL, T_LATE, S0, N_PER_R = 7.0, 25.0, 10.0, 200          # as hydro_scan.py
R_MAX = 1.0 + 1.05 * (T_LATE + S0) + 3.0
N = int(N_PER_R * R_MAX)
KR = np.geomspace(0.01, 40, 1500)
XCL = 1e4                         # any x_cl >> 1; R_shift does not depend on it


def u_of_k(r, rho, v, k=KR):
    dr = np.gradient(r)
    kr = np.outer(k, r)
    j1 = np.where(kr > 1e-6, (np.sin(kr) / np.maximum(kr, 1e-6) - np.cos(kr)) / np.maximum(kr, 1e-6), kr / 3)
    u = np.sqrt(rho) * v / np.sqrt(1.0 - v * v)
    return 4 * np.pi * (j1 * (r ** 2 * u * dr)).sum(axis=1)


def gw(P):
    lP = np.log(np.maximum(P, 1e-300))
    Pv = lambda z: np.where((z / XCL > KR[0]) & (z / XCL < KR[-1]),
                            np.exp(np.interp(np.log(z / XCL), np.log(KR), lP)), 0.0)
    xs = np.geomspace(0.05 * XCL, 60 * XCL, 90)
    g = ag.P_GW(xs, Pv, breaks=[])
    return float(np.trapezoid(g, np.log(xs))), float(xs[np.argmax(g)] / XCL)


def main():
    prof = dict(np.load(os.path.join(OUT, "hydro_profiles.npz")))
    txt = open(os.path.join(OUT, "hydro_scan.txt")).read()
    top = {float(a): float(b) for a, b, c, d in
           re.findall(r"^\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", txt, flags=re.M)}
    src = [(lab.strip(), float(dt), float(k7)) for lab, dt, k7, k25 in
           re.findall(r"^(.+?)\s+x_cl=\S+\s+delta_tot=\s*([0-9.]+)\s+kappa\(7R\)=([0-9.]+)\s+kappa\(25R\)=([0-9.]+)",
                      txt, flags=re.M)]
    lines = ["R_shift = GW[non-linear shell spectrum] / (kappa_E^2 GW[linear template]), t = 7R",
             f"{'case':40s} {'delta':>8s} {'kappa_E':>8s} {'R_shift':>8s} {'kR_peak lin -> NL':>18s}"]
    r0, rho0, v0 = prof["top_0.01"]
    P0 = KR ** 3 * (u_of_k(r0, rho0, v0) / 0.01) ** 2
    I0, k0 = gw(P0)
    for d0 in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        r, rho, v = prof[f"top_{d0}"]
        I, kp = gw(KR ** 3 * u_of_k(r, rho, v) ** 2)
        lines.append(f"{'top hat':40s} {d0:8.2f} {top[d0]:8.4f} {I / (I0 * (d0 ** 2 * top[d0]) ** 2):8.3f} "
                     f"{k0:8.2f} -> {kp:6.2f}")
        print(lines[-1], flush=True)
    # linear template for the release histories
    b = h.Burst(R=1.0, r_max=R_MAX, n=N, source_delta=1e-4, source_s0=S0)
    b.run(S0 + T_COLL)
    rho, v = h.prim(b.E, b.S)
    P1 = KR ** 3 * (u_of_k(b.r, rho, v) / 1e-4) ** 2
    I1, k1 = gw(P1)
    for lab, dt, k7 in src:
        r, rho, v = prof[f"src_{lab}"]
        I, kp = gw(KR ** 3 * u_of_k(r, rho, v) ** 2)
        lines.append(f"{'release: ' + lab:40s} {dt:8.2f} {k7:8.4f} {I / (I1 * (dt ** 2 * k7) ** 2):8.3f} "
                     f"{k1:8.2f} -> {kp:6.2f}")
        print(lines[-1], flush=True)
    with open(os.path.join(OUT, "hydro_gw_shift.txt"), "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
