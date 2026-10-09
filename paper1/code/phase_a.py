"""
Phase A -- the go/no-go numbers (COMPUTE_PLAN.md).

For each benchmark (M_in, beta_f) and population (monochromatic, Choptuik) this
computes the radiation density spectrum handed over at evaporation, channel by
channel, and the acoustic GW spectrum each channel sources (acoustic_gw.py):

  LIN-NL   linear spectrum cut at k_NL                 (2605.21474 'conservative')
  LIN-UV   linear spectrum cut at k_PBH                (2012.08151 bound; 2605.21474 'upper')
  NL-A     halo model, uniform clusters, no substructure   (this work)
  NL-B     NL-A continued as P ~ k^(9/5) above its peak  (stable clustering, n = 0)
  SHOT     shot noise of individual endpoints (incoherent; suddenness.py Eq. 3)
  NL-A+F, NL-B+F   the cluster channels with the non-linear burst efficiency of
                   module F (burst_efficiency.py; hydro=True)
  ...-sh           NL-A+F, NL-B+F and LIN-NL with shock-limited sound (sound_lifetime.py)

each with and without diffusion damping (damping.py).  Run on Windows Python with one
thread at low priority; ~1-3 minutes.  Writes phase_a_results.csv, phase_a_spectra.npz,
phase_a_spectra.png and a printed table.
"""

import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import cosmology as co          # noqa: E402
import clusters as cl           # noqa: E402
import suddenness as sd         # noqa: E402
import damping as dm            # noqa: E402
import acoustic_gw as ag        # noqa: E402
import burst_efficiency as be   # noqa: E402
import sound_lifetime as sl     # noqa: E402

OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

BENCH = [
    (1e2, 1.1e-6 * (1e2 / 1e4) ** (-17 / 24), "BBN-bound"),
    (1e4, 1.1e-6, "BBN-bound"),
    (1e6, 1.1e-6 * (1e6 / 1e4) ** (-17 / 24), "BBN-bound"),
    (1e8, 1.1e-6 * (1e8 / 1e4) ** (-17 / 24), "BBN-bound"),
    (1e4, 1e-6, "Stage-1 fiducial"),
    (1e0, 30 * co.beta_crit(1e0), "30 beta_c"),
    (1e4, 30 * co.beta_crit(1e4), "30 beta_c"),
]

# approximate minima of power-law-integrated sensitivity curves, Omega h^2, [f_lo, f_hi] Hz
# PLACEHOLDERS until the real curves are downloaded (COMPUTE_PLAN week 5)
DETECTORS = {
    "LISA": (1e-13, 1e-4, 1e-1),
    "DECIGO": (1e-16, 1e-2, 10.0),
    "BBO": (1e-17, 1e-2, 10.0),
    "ET": (1e-12, 3.0, 1e3),
    "CE": (1e-12, 5.0, 2e3),
}
CHOPTUIK = sd.choptuik()      # t_eff fixed so that x = k/k_eva maps as for M_in

# NL-B slope.  Stable clustering for a white-noise seed (n = 0): xi ~ r^-gamma with
# gamma = 3(3+n)/(5+n) = 9/5, so the dimensionless power grows as k^(9/5) above the peak.
# Same result from the clumps themselves: a clump of N holes forms when sigma(N) ~ D/sqrt(N)
# reaches delta_c, so N ~ a_f^2, its density stays ~ a_f^-3 (no expansion), hence R ~ a_f^(5/3),
# N ~ R^(6/5), density contrast ~ R^(-9/5) and k^3 N ~ k^(9/5).  (Was 6/5 before 2026-10-05:
# the N ~ R^(6/5) exponent had been used for the power; see RECHECK P1-15.)
NLB_SLOPE = 9.0 / 5.0


def channel_spectra(M, beta, pop, xi=1.0, hydro=False):
    """
    hydro=True applies the module F burst efficiency (burst_efficiency.py) as kappa_gw,
    the factor that reproduces the GW integral; hydro="energy" applies the energy
    efficiency (for sound-energy budgets).  It enters as kappa of each cluster mass inside
    the one-halo integral (clusters.P_halo), and for NL-B's substructure continuation as
    kappa of sub-clumps of density Delta_eff (x/x_pk)^(9/5) and size R_cl x_pk/x (stable
    clustering, NLB_SLOPE).  Linear and 2-halo parts are unchanged.
    """
    c = cl.Clusters(M, beta, xi=xi)
    s = c.sc
    keva = c.k_eva
    xPBH = c.k_PBH / keva
    # S_coh, S_inc as functions of x = q/k_eva.  pop: "mono", "choptuik", or any object
    # with S_coh(x) and S_inc2(x) (e.g. suddenness.LogNormalMF for the width scan)
    lx = np.linspace(np.log(1e-2), np.log(10 * xPBH), 900)
    xg = np.exp(lx)
    if isinstance(pop, str) and pop == "mono":
        Scoh2 = lambda x: sd.S_mono(x) ** 2
        Sinc2 = Scoh2
        shot_ratio = 1.0
    else:
        mf = CHOPTUIK if isinstance(pop, str) else pop
        if hasattr(mf, "tabulate"):
            mf.tabulate(xg[0], xg[-1])
        Scoh2 = lambda x: mf.S_coh(np.atleast_1d(x)) ** 2
        Sinc2 = lambda x: mf.S_inc2(np.atleast_1d(x))
        shot_ratio = mf.shot_ratio             # true number density, suddenness Eq. (3)
    lS_coh = np.log(np.maximum(Scoh2(xg), 1e-300))      # S_coh = 0 where phi is unresolvable
    lS_inc = np.log(np.maximum(Sinc2(xg), 1e-300))
    k = xg * keva
    Plin = c.P_lin(k)
    p1, p2 = c.P_halo(k)
    # 2-halo exclusion: suppress P_2h above k_* (halos do not overlap)
    p2x = p2 * np.exp(-(xg * keva / c.k_star) ** 2)
    PA = np.minimum(Plin, p1 + p2x)
    # NL-B: continue as x^(9/5) beyond the peak of P_A
    ipk = np.argmax(np.where(xg < xPBH, PA, 0.0))
    PB = np.where(xg > xg[ipk], PA[ipk] * (xg / xg[ipk]) ** NLB_SLOPE, PA)
    if hydro:
        # NL-A: kappa per cluster mass; NL-B: the same reduction up to the peak of P_A,
        # then sub-clumps of density Delta_eff r^(9/5) and size R_cl/r, r = x/x_pk
        p1h, _ = c.P_halo(k, hydro=hydro)
        PAh = np.minimum(Plin, p1h + p2x)
        red = PAh / PA
        r = np.maximum(xg / xg[ipk], 1.0)
        ksub = be.kappa_cluster(c.Delta / c.xi ** 3 * r ** NLB_SLOPE, r / c.R_cl_H(c.N_star),
                                gw=(hydro is True))
        PB = PB * np.where(xg > xg[ipk], red[ipk] * ksub / ksub[0], red)
        PA = PAh
    PB = np.minimum(PB, Plin)
    Pshot = c.P_S(k) * shot_ratio
    # k_NL from P_lin = 1
    xNL = float(np.exp(np.interp(0.0, np.log(Plin), lx)))

    def make(Parr, cut=None):
        lP = np.log(np.maximum(Parr, 1e-300))

        def Pdr(z):
            z = np.asarray(z, dtype=float)
            out = np.exp(np.interp(np.log(z), lx, lP) + np.interp(np.log(z), lx, lS_coh))
            if cut is not None:
                out = np.where(z <= cut, out, 0.0)
            return np.where((z > xg[0]) & (z < xg[-1]), out, 0.0)
        return Pdr

    def make_shot(cut):
        lP = np.log(Pshot)

        def Pdr(z):
            z = np.asarray(z, dtype=float)
            out = np.exp(np.interp(np.log(z), lx, lP) + np.interp(np.log(z), lx, lS_inc))
            return np.where((z > xg[0]) & (z <= cut), out, 0.0)
        return Pdr

    ch = {
        "LIN-NL": (make(Plin, cut=xNL), [xNL]),
        "LIN-UV": (make(Plin, cut=xPBH), [xPBH]),
        "NL-A": (make(PA, cut=xPBH), [xPBH]),
        "NL-B": (make(PB, cut=xPBH), [xPBH]),
        "SHOT": (make_shot(cut=xPBH), [xPBH]),
    }
    info = dict(c=c, s=s, xNL=xNL, xPBH=xPBH, xstar=c.k_star / keva,
                xcl=1.0 / (c.R_cl(c.N_star) * keva),
                kappa_cl=float(c.kappa_burst(c.N_star, gw=False)),
                kappa_gw=float(c.kappa_burst(c.N_star, gw=True)),
                Pdr_peak_A=float(np.max(PA * np.exp(lS_coh))),
                x_peak_A=float(xg[np.argmax(PA * np.exp(lS_coh))]))
    return ch, info


def run_benchmark(M, beta, pop, xi=1.0, nx=90, hydro=False, only=None, damped_only=False,
                  lifetime=False):
    """
    only: channel names to compute (default all); damped_only skips the undamped spectra.
    lifetime=True: shock-limited sound (sound_lifetime.py).  Each channel's source switches off
    at y_end = sqrt(1 + 2 H tau_sh), tau_sh = d_* / U_f, with U_f from that channel's energy
    spectrum (module F energy efficiency for the cluster channels).  Not applied to SHOT.
    info['Htau'] then holds H tau_sh per channel.
    """
    ch, info = channel_spectra(M, beta, pop, xi, hydro=hydro)
    s = info["s"]
    T = s["T_evap"]
    xD = float(dm.kD_over_aH(T))
    p = float(dm.damping_exponent(T))
    ups = ag.Upsilon(p)
    xs = np.exp(np.linspace(np.log(0.3), np.log(2.5 * info["xPBH"]), nx))
    f = xs * s["f_eva"]
    if lifetime:
        che = ch if not hydro or hydro == "energy" else channel_spectra(M, beta, pop, xi, hydro="energy")[0]
        L = sl.d_star(info["c"])
        info["Htau"] = {}
    res = {}
    for name, (Pdr, br) in ch.items():
        if only is not None and name not in only:
            continue
        Pv = lambda z, Pdr=Pdr: ag.PV_OVER_PDELTA * Pdr(z)
        y_end, ups_n = None, ups
        if lifetime and name != "SHOT":
            Ht = sl.H_tau_sh(L, sl.rms_velocity(che[name][0], info["xPBH"]))
            info["Htau"][name] = Ht
            y_end = float(sl.y_end(Ht))
            ups_n = ag.Upsilon(p, y_end=y_end)
        und = None if damped_only else ag.P_GW(xs, Pv, breaks=br, y_end=y_end)
        # explosion-time spread for the incoherent channel: none (mono), ~1 Hubble time
        # (Choptuik), ~2 sigma (narrow log-normal)
        HT = 0.0 if (isinstance(pop, str) and pop == "mono") else (1.0 if isinstance(pop, str) else pop.HT)
        mode = "cont" if (name == "SHOT" and HT > 0) else "once"
        dmp = ag.P_GW(xs, Pv, x_D=xD, ups=ups_n, mode=mode, HT=HT, breaks=br, y_end=y_end)
        res[name] = (None if und is None else ag.today_h2(und, T), ag.today_h2(dmp, T))
    return xs, f, res, info, xD


def main():
    t0 = time.time()
    rows, store = [], {}
    hdr = ("M_g,beta,label,pop,xi,T_evap,y_evap,x_D,x_NL,x_star,x_cl,x_PBH,N_star,"
           "channel,peak_h2_undamped,f_peak_undamped,peak_h2_damped,f_peak_damped")
    for (M, beta, lab) in BENCH:
        for pop in ("mono", "choptuik"):
            for xi in ((1.0, 1.8) if pop == "mono" else (1.0,)):
                xs, f, res, info, xD = run_benchmark(M, beta, pop, xi)
                # module F: cluster channels with the non-linear burst efficiency
                _, _, resF, _, _ = run_benchmark(M, beta, pop, xi, hydro=True, only=("NL-A", "NL-B"))
                res.update({n + "+F": v for n, v in resF.items()})
                if pop == "mono":
                    # shock-limited sound (sound_lifetime.py): cluster channels and the k_NL cut
                    _, _, resS, infS, _ = run_benchmark(M, beta, pop, xi, hydro=True, lifetime=True,
                                                        only=("NL-A", "NL-B", "LIN-NL"))
                    res.update({(n if n.startswith("LIN") else n + "+F") + "-sh": v for n, v in resS.items()})
                    print("   H tau_sh: " + ", ".join(f"{n} {v:.2e}" for n, v in infS["Htau"].items()))
                s = info["s"]
                key = f"{M:.0e}_{beta:.2e}_{pop}_xi{xi}"
                store[key + "_f"] = f
                print(f"\n{lab:16s} M={M:.0e} g beta={beta:.2e} {pop:8s} xi={xi}:  "
                      f"T_evap={s['T_evap']:.2e} GeV  y={s['y_evap']:.2e}  x_D={xD:.2e}  "
                      f"x_NL={info['xNL']:.2e}  x_cl={info['xcl']:.2e}  x_PBH={info['xPBH']:.2e}  "
                      f"max Pdr(NL-A)={info['Pdr_peak_A']:.2e} at x={info['x_peak_A']:.2e}  "
                      f"kappa_cl={info['kappa_cl']:.3f} (GW-equivalent {info['kappa_gw']:.3f})")
                for name, (u, d) in res.items():
                    store[key + "_" + name + "_u"] = u
                    store[key + "_" + name + "_d"] = d
                    iu, id_ = np.argmax(u), np.argmax(d)
                    print(f"   {name:7s} peak h2: undamped {u[iu]:.2e} @ {f[iu]:.2e} Hz | "
                          f"damped {d[id_]:.2e} @ {f[id_]:.2e} Hz")
                    rows.append(f"{M:.3e},{beta:.4e},{lab},{pop},{xi},{s['T_evap']:.4e},"
                                f"{s['y_evap']:.4e},{xD:.4e},{info['xNL']:.4e},{info['xstar']:.4e},"
                                f"{info['xcl']:.4e},{info['xPBH']:.4e},{info['c'].N_star:.4e},{name},"
                                f"{u[iu]:.4e},{f[iu]:.4e},{d[id_]:.4e},{f[id_]:.4e}")
    with open(os.path.join(OUT, "phase_a_results.csv"), "w", newline="\n") as fh:
        fh.write(hdr + "\n" + "\n".join(rows) + "\n")
    np.savez(os.path.join(OUT, "phase_a_spectra.npz"), **store)
    print(f"\nwall time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
