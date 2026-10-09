"""
Phase A validation suite (modules B, C, D, E, E2, F), in the style of validate.py
(Stage 0) and validate_stage1.py.  Every check prints PASS/FAIL with the number it
compared.  P8 and F5-F7 read results written by width_scan.py, hydro_scan.py and
hydro_gw_shift.py.  Run:  python validate_phase_a.py      (~1.5 minutes, one thread)
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import cosmology as co          # noqa: E402
import clusters as cl           # noqa: E402
import suddenness as sd         # noqa: E402
import damping as dm            # noqa: E402
import acoustic_gw as ag        # noqa: E402

results = []
BAR = "=" * 96


def check(label, value, lo, hi, fmt="{:.4g}"):
    ok = bool(np.isfinite(value) and lo <= value <= hi)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<62s} {fmt.format(value):>12s}   "
          f"in [{fmt.format(lo)}, {fmt.format(hi)}]")
    results.append(ok)


def slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


# --------------------------------------------------------------------------- D
print(BAR + "\nMODULE D  damping.py\n" + BAR)
Tdec = dm.neutrino_decoupling_T()
check("D1 neutrino decoupling Gamma_nu = H  [MeV]", Tdec * 1e3, 1.0, 2.0)
T = 1e5
w = (4 / 3) * dm.rho_rad(T)
analytic = np.sqrt(1.0 / ((2 / 3) * (dm.C_ETA * T ** 3 / w) * dm.hubble_rad(T)))
check("D2 EW-plasma k_D/aH at 1e5 GeV vs closed form (ratio)", float(dm.kD_over_aH(T)) / analytic,
      0.999, 1.001)
check("D3 damping exponent at 10 MeV (neutrinos) ", float(dm.damping_exponent(1e-2)), 5, 5)
check("D3 damping exponent at 1e5 GeV (plasma)", float(dm.damping_exponent(1e5)), 1, 1)
Ts = np.geomspace(5e-3, 1.0, 20)
check("D4 neutrino regime: d ln(k_D/aH)/d ln T  (expect ~3/2 + g* drift)",
      slope(Ts, dm.kD_over_aH(Ts)), 1.3, 2.2)

# --------------------------------------------------------------------------- C
print(BAR + "\nMODULE C  suddenness.py\n" + BAR)
for wt in (100.0, 1e3, 1e4):
    check(f"C1 single hole: numeric FT / Gamma(1/3)/3 (w tau)^-1/3 at w tau={wt:.0e}",
          sd.S_mono_numeric(wt, n=2000001) / sd.single_hole_S(wt), 0.99, 1.01)
x = np.geomspace(1e2, 1e5, 7)
# exact linear acoustics vs the decoupling-time estimate: expect ~15% above (Phase A note)
check("C2 mono S(x) / (sqrt(2/3) x)^(-1/3)  [Inomata 2020 via 2605.21474 App.]",
      float(np.mean(sd.S_mono(x) / (np.sqrt(2 / 3) * x) ** (-1 / 3))), 1.0, 1.25)
mf = sd.choptuik()
xs = np.geomspace(1e3, 1e5, 9)
Sc = mf.S_coh(xs)
check("C3 Choptuik coherent slope d ln S/d ln k  (universal -4/3)", slope(xs, Sc), -1.36, -1.30)
check("C4 Choptuik S_coh x^(4/3) / kappa_0 (2605.21474: 2.3)", float(np.mean(Sc * xs ** (4 / 3))) / 2.3,
      0.85, 1.25)
Si = np.sqrt(mf.S_inc2(xs))
check("C5 Choptuik shot-noise slope (single-hole -1/3)", slope(xs, Si), -0.34, -0.32)
w = mf.omega(xs)
check("C5 S_inc^2 = 0.797 (w tau(<M0>))^(-2/3) exactly",
      float(np.max(np.abs(mf.S_inc2(xs) / (sd.G13 ** 2 * (w * mf.tau_mean_num) ** (-2 / 3)) - 1))), 0.0, 1e-9,
      fmt="{:.2e}")
x0 = np.array([100.0])
s0 = sd.S_coh_from_stage0(mf, 100.0, n=6001)
check("C6 Stage-0 route / characteristic-function route at x=100", s0 / float(mf.S_coh(x0)[0]), 0.8, 1.25)
ln01 = sd.lognormal(0.1)
xl = np.array([1.0, 2.0])     # omega sigma_tau << 1: the population still goes off together
check("C7 log-normal sigma=0.1 at x<=2: S_coh / S_mono (coherent limit)",
      float(np.max(np.abs(ln01.S_coh(xl) / sd.S_mono(xl) - 1))), 0.0, 0.3, fmt="{:.3f}")
xh = np.geomspace(3e3, 3e4, 5)
check("C7 log-normal sigma=0.1 far above k_univ: universal slope (-4/3)", slope(xh, ln01.S_coh(xh)), -1.40, -1.27)

# --------------------------------------------------------------------------- B
print(BAR + "\nMODULE B  clusters.py\n" + BAR)
check("B1 T_S(0) = 1/5", float(cl.T_S(0.0)), 0.2 - 1e-12, 0.2 + 1e-12)
check("B1 kappa^2 T_S(kappa) -> C = 9/8 at kappa = 1e6", float(1e12 * cl.T_S(1e6)), 1.124, 1.126)
c = cl.Clusters(1e4, 1e-6)
D = 1.5 * c.sc["y_evap"]
check("B2 sigma^2(N) N / D^2 at N = 100 (2412.01890 Eq. sigmaM2, mu = 1)", float(c.sigma2(100.0)) * 100 / D ** 2,
      0.98, 1.0)
check("B3 N_* / (D/delta_c)^2  (transfer function lowers it slightly)", c.N_star / (D / cl.DELTA_C) ** 2, 0.6, 1.0)
lnN = np.linspace(np.log(1e-6), np.log(c.N_star) + 6 * np.log(10), 4000)
N = np.exp(lnN)
check("B4 Press-Schechter: int dF = 1", float(np.trapezoid(c.dF_dlnN(N), lnN)), 0.999, 1.001)
check("B4 Press-Schechter: int dF b = 1", float(np.trapezoid(c.dF_dlnN(N) * c.bias(N), lnN)), 0.99, 1.01)
check("B4 mass-weighted <N>/N_*  (white noise: 1)", float(np.trapezoid(c.dF_dlnN(N) * N, lnN)) / c.N_star,
      0.85, 1.05)
Rrs, sv = cl.holst_R_and_sigma_v(1e9, 1e-20, 1.0)
check("B5 2412.01890 Eq. R_cl prefactor (they quote ~1e6) [log10]", np.log10(Rrs), 5.5, 6.5)
check("B5 2412.01890 Eq. sigma_v prefactor (they quote ~1e-3) [log10]", np.log10(sv), -3.5, -2.5)
check("B6 adiabatic puff-up factor for clusters formed at evaporation", cl.puff_factor(1.0), 1.5, 2.2)
check("B7 cluster virial velocity at the fiducial point (non-relativistic)", float(c.sigma_v(c.N_star)),
      1e-5, 1e-2, fmt="{:.2e}")

# --------------------------------------------------------------------------- E
print(BAR + "\nMODULE E  acoustic_gw.py\n" + BAR)
check("E1 P_v / P_delta = c_s^2/(2 Gamma^2) = 3/32", ag.PV_OVER_PDELTA, 3 / 32 - 1e-12, 3 / 32 + 1e-12)
U1 = ag.Upsilon(1.0)
check("E2 Upsilon(X -> 0) = 1  (p = 1)", float(U1(1e-9)), 0.9999, 1.0001)
check("E2 Upsilon(X) * 2X -> 1 at X = 1e6", float(U1(1e6) * 2e6), 0.99, 1.01)
for n in (2.0, 7 / 3, 3.0):
    xx = np.array([1.0, 10.0, 100.0])
    r = ag.P_GW(xx, lambda z, n=n: z ** n) / (ag.J_powerlaw(n) * xx ** (2 * n - 1))
    check(f"E3 power law P_v = z^{n:.3f}: quadrature / closed form", float(np.max(np.abs(r - 1))), 0, 1e-3,
          fmt="{:.2e}")
q0, sg = 50.0, 0.02
xg = np.linspace(40, 75, 701)
pk = xg[np.argmax(ag.P_GW(xg, lambda z: np.exp(-np.log(z / q0) ** 2 / (2 * sg ** 2)), n=400))]
check("E4 narrow velocity peak q0: GW peak / (2 c_s q0)", pk / (2 * ag.CS * q0), 0.995, 1.005)
Av = ag.PV_OVER_PDELTA * (4 / 9) * (2 / (3 * np.pi)) * (2 / 3) ** (-1 / 3)
ours = ag.J_powerlaw(7 / 3) * Av ** 2
theirs = ag.CS ** (7 / 3) * (ag.CS ** 2 - 1) ** 2 / (576 * 6 ** (1 / 3) * np.pi)
theta_uv_low = 1.1671   # 2409.12125 Eq. (E7) at k << k_uv, i.e. int ds (1-s^2)^2/(1-s^2/3)^(5/3)
# 2409.12125 below Fig. 4: the full numerics are 2x the resonant formula; E2 confirms it
check("E5 linear mono power law: ours / [2 x Omega_res x Theta_uv]  (2409.12125)",
      ours / (2 * theirs * theta_uv_low), 0.99, 1.01)

# --------------------------------------------------------------------------- E2
print(BAR + "\nMODULE E2  sigw_kernel.py  (independent induced-GW kernel)\n" + BAR)
import sigw_kernel as sk   # noqa: E402
from scipy.integrate import quad   # noqa: E402
for q, cc in ((0.3, 5.0), (-0.7, 2.0), (1e-3, 50.0)):
    Cq, Sq = sk._CS_int(np.array([q]), cc)
    Cn = quad(lambda t: 1.0 / (t + cc), 0, np.inf, weight="cos", wvar=abs(q))[0]
    Sn = np.sign(q) * quad(lambda t: 1.0 / (t + cc), 0, np.inf, weight="sin", wvar=abs(q))[0]
    check(f"E2-1 closed-form C,S vs oscillatory quadrature, q={q:+.0e} c={cc:g} (max rel err)",
          max(abs(Cq[0] / Cn - 1), abs(Sq[0] / Sn - 1)), 0, 1e-6, fmt="{:.1e}")
x_uv = 1e4
TSf = lambda kap: 1.0 / (5.0 + kap ** 2)
PPhi = lambda z: TSf(z) ** 2 * (2 / (3 * np.pi)) * (z / x_uv) ** 3 * (np.sqrt(2 / 3) * z) ** (-2 / 3)
peak = ag.CS ** (7 / 3) * (ag.CS ** 2 - 1) ** 2 / (576 * 6 ** (1 / 3) * np.pi) * x_uv ** (17 / 3 - 8)
xr = np.array([0.3, 0.9])
res = sk.omega_gw(xr * x_uv, PPhi, x_uv=x_uv, resonant_only=True)
full = sk.omega_gw(xr * x_uv, PPhi, x_uv=x_uv)
th = [quad(lambda s: (s * s - 1) ** 2 / (1 - s * s / 3) ** (5 / 3), -s0, s0)[0]
      for s0 in (1.0, 2 / 0.9 - np.sqrt(3))]
for i, r in enumerate(xr):
    check(f"E2-2 resonant-only kernel / 2409.12125 Eq. (4.5) at k/k_uv={r}",
          res[i] / (peak * r ** (11 / 3) * th[i]), 0.99, 1.01)
    check(f"E2-3 full kernel / resonant-only at k/k_uv={r}  (2409.12125: 2)", full[i] / res[i], 1.98, 2.02)
Pd = lambda z: (4 / 9) * np.asarray(z) ** 4 * PPhi(z)
ssm = ag.P_GW(xr * x_uv, lambda z: ag.PV_OVER_PDELTA * np.where(z <= x_uv, Pd(z), 0.0), breaks=[x_uv], n=400)
for i, r in enumerate(xr):
    check(f"E2-4 sound-shell / full induced-GW kernel, power law, k/k_uv={r}", ssm[i] / full[i], 0.99, 1.01)

# --------------------------------------------------------------------------- integration
print(BAR + "\nINTEGRATION  phase_a.py channels\n" + BAR)
import phase_a as pa   # noqa: E402
for M in (1e2, 1e4):
    b = 1.1e-6 * (M / 1e4) ** (-17 / 24)
    xs_, f_, res, info, xD = pa.run_benchmark(M, b, "mono", nx=60)
    lin_uv = res["LIN-UV"][0].max()
    check(f"P1 LIN-UV peak at the 2012.08151 bound reproduces Delta N_eff level, M={M:.0e}",
          lin_uv, 1e-6, 3e-5, fmt="{:.2e}")
    check(f"P2 NL-A / LIN-NL (clusters above conservative cut), M={M:.0e}",
          res["NL-A"][1].max() / res["LIN-NL"][1].max(), 10.0, 1e3, fmt="{:.2e}")
    check(f"P3 NL-A / LIN-UV (linear extrapolation overshoots), M={M:.0e} [log10]",
          np.log10(res["NL-A"][1].max() / lin_uv), -12, -5, fmt="{:.2f}")
    xs_, f_, resc, infoc, xD = pa.run_benchmark(M, b, "choptuik", nx=60)
    check(f"P4 Choptuik SHOT / LIN-NL (shot noise escapes suppression), M={M:.0e} [log10]",
          np.log10(resc["SHOT"][1].max() / resc["LIN-NL"][1].max()), 2.0, 8.0, fmt="{:.2f}")

ch, info = pa.channel_spectra(1e4, 1e-6, "mono")
Pdr, br = ch["NL-A"]
xp = np.array([info["x_peak_A"]])
r = (ag.P_GW(xp, lambda z: ag.PV_OVER_PDELTA * Pdr(z), breaks=br, n=400)
     / sk.omega_gw(xp, sk.P_Phi_from_Pdelta(Pdr), x_uv=br[0]))[0]
check("P5 cluster channel at its peak: sound-shell / full induced-GW kernel", r, 0.99, 1.01)

mf = pa.CHOPTUIK
check("P6 Choptuik shot_ratio = <M>/M_mw (true number density of holes)", mf.shot_ratio, 0.92, 0.94)
check("P7 M_mw = (3 t_eff)^(1/3): benchmark M_in is the mass-weighted mean [rel. diff]",
      abs(mf.M_mw / (3 * mf.t_eff) ** (1 / 3) - 1), 0.0, 1e-9, fmt="{:.1e}")
trunc = os.path.join(pa.OUT, "width_trunc_check.txt")
if os.path.exists(trunc):
    rat = [float(l.split()[-1]) for l in open(trunc).read().splitlines()[2:] if l.strip()]
    check("P8 width scan: 5-sigma truncation vs none, worst NL-A ratio (sigma<=3e-3)",
          max(abs(np.array(rat) - 1)), 0.0, 0.02, fmt="{:.4f}")
else:
    print("  [FAIL] P8 needs results/width_trunc_check.txt: run width_scan.py first")
    results.append(False)

# --------------------------------------------------------------------------- F
print(BAR + "\nMODULE F  hydro1d.py, burst_efficiency.py  (non-linear cluster burst)\n" + BAR)
import re                        # noqa: E402
import hydro1d as h              # noqa: E402
import burst_efficiency as be    # noqa: E402
b0 = h.Burst(R=1.0, r_max=10.0, n=400).run(2.0)
rho0, v0 = h.prim(b0.E, b0.S)
check("F1 uniform fluid at rest stays static, max(|v|, |rho-1|)",
      float(max(np.max(np.abs(v0)), np.max(np.abs(rho0 - 1)))), 0.0, 1e-12, fmt="{:.1e}")
RMAX, NF = 11.35, 1700                                    # 150 cells per R
lin = h.Burst(R=1.0, r_max=RMAX, n=NF, delta0=1e-4).run(7.0)
check("F2 linear top hat: E_ac(t=7R) / exact delta0^2 V/8 (numerical dissipation)",
      lin.acoustic_energy() / h.linear_acoustic_energy(1e-4), 0.85, 1.0)
bs = h.Burst(R=1.0, r_max=10.0, n=1200, source_delta=0.5, source_s0=2.0).run(4.0)
check("F3 energy conservation with the Hawking source [rel. error]",
      abs(bs.excess_energy() / (bs.injected * np.sum(bs.shape * bs.V)) - 1), 0.0, 1e-6, fmt="{:.1e}")
b3 = h.Burst(R=1.0, r_max=RMAX, n=NF, delta0=3.0).run(7.0)
k3 = (b3.acoustic_energy() / 9.0) / (lin.acoustic_energy() / 1e-8)
check("F4 kappa(delta0=3) at t=7R, 150 cells/R (scan at 200: 0.427)", k3, 0.39, 0.47)
txt = open(os.path.join(pa.OUT, "hydro_scan.txt")).read()
tab = np.array([[float(v) for v in m] for m in
                re.findall(r"^\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", txt, flags=re.M)])
check("F5 fit kappa(delta) vs hydro_scan top hats (t=25R), worst rel. deviation",
      float(np.max(np.abs(be.kappa_tophat(tab[:, 0]) / np.minimum(tab[:, 2], 1.0) - 1))), 0.0, 0.04)
src = [(float(a), float(b)) for a, b in
       re.findall(r"delta_tot=\s*([0-9.]+)\s+kappa\(7R\)=[0-9.]+\s+kappa\(25R\)=([0-9.]+)", txt)]
check("F6 release histories follow the fit at 0.21 delta_tot, worst rel. deviation",
      max(abs(be.kappa_tophat(be.A_SRC * d) / k - 1) for d, k in src), 0.0, 0.04)
sh = open(os.path.join(pa.OUT, "hydro_gw_shift.txt")).read()
gs = [(float(t[-6]), float(t[-4])) for t in (l.split() for l in sh.splitlines() if l.startswith("release:"))]
check("F7 GW shift fit (1+delta_tot/8.1)^0.43 vs the three release runs, worst rel. dev.",
      max(abs(be.gw_shift(d) / R - 1) for d, R in gs), 0.0, 0.02)
cs = [cl.Clusters(1e4, 1e-6), cl.Clusters(1e4, 1e-6, xi=1.8), cl.Clusters(1e2, 1e-2)]
check("F8 Clusters.kappa_burst(N_*) vs the hydro release runs (same delta_tot), worst rel.",
      max(abs(float(c.kappa_burst(c.N_star, gw=False)) / k - 1) for c, (d, k) in zip(cs, src)), 0.0, 0.06)
_, f_, r0_, _, _ = pa.run_benchmark(1e4, 1e-6, "mono", nx=60, only=("NL-A",), damped_only=True)
_, f_, r1_, _, _ = pa.run_benchmark(1e4, 1e-6, "mono", nx=60, only=("NL-A",), damped_only=True, hydro=True)
check("F9 cluster channel with module F / without, fiducial 1e4 g (integral)",
      np.trapezoid(r1_["NL-A"][1], np.log(f_)) / np.trapezoid(r0_["NL-A"][1], np.log(f_)), 0.2, 0.4)

# --------------------------------------------------------------------------- G, H
print(BAR + "\nMODULE L  sound_lifetime.py  (shock-limited sound)\n" + BAR)
import sound_lifetime as sl     # noqa: E402
import phase_a as pa_l          # noqa: E402
for ye in (1.01, 2.0):
    check(f"L1 Upsilon(X->0) with cutoff y_end={ye}: / (1 - 1/y_end)",
          float(ag.Upsilon(5.0, y_end=ye)(1e-12)) / (1 - 1 / ye), 0.999, 1.001)
Xs = np.geomspace(1e-3, 1e3, 7)
check("L2 Upsilon with y_end = 1e6 vs no cutoff, worst rel. deviation (X = 1e-3..1e3)",
      float(np.max(np.abs(ag.Upsilon(5.0, y_end=1e6)(Xs) / ag.Upsilon(5.0)(Xs) - 1))), 0, 1e-3)
check("L3 upsilon_sh(H tau = 1e-4) / 1e-4  (short-lived source: ~ H tau)",
      float(sl.upsilon_sh(1e-4)) / 1e-4, 0.999, 1.0)
zz = np.geomspace(1.0, 1e3, 2)
pl = lambda z: 1e-3 * (np.asarray(z) / 10.0) ** 3 * (np.asarray(z) < 100.0)
xg = np.array([30.0, 100.0])
check("L4 P_GW with y_end=1.5, no damping / P_GW x (1 - 1/1.5)  (exact)",
      float(np.max(np.abs(ag.P_GW(xg, pl, breaks=[100.0], y_end=1.5) /
                          (ag.P_GW(xg, pl, breaks=[100.0]) * (1 - 1 / 1.5)) - 1))), 0, 1e-12, fmt="{:.1e}")
chE, infE = pa_l.channel_spectra(1e4, 1.1e-6, "mono", 1.0, hydro="energy")
Uf = sl.rms_velocity(chE["NL-A"][0], infE["xPBH"])
check("L5 Gamma U_f^2 of NL-A+F (energy) on the bound at 1e4 g vs Table III K = 0.0099",
      (4 / 3) * Uf ** 2, 0.0090, 0.0105)
cE = infE["c"]
check("L6 d_* k_eva x xi x_cl / (4 pi Delta/3)^(1/3)  (identity)",
      sl.d_star(cE) * cE.xi * infE["xcl"] / (4 * np.pi * cE.Delta / 3) ** (1 / 3), 0.999999, 1.000001)
check("L7 H tau_sh = d_*/U_f, NL-A+F on the bound at 1e4 g (sound shocks early)",
      sl.H_tau_sh(sl.d_star(cE), Uf), 2e-3, 1e-2, fmt="{:.2e}")

print(BAR + "\nMODULES G, H  detectors.py (Schmitz PLI curves), relics.py (Holst et al. relics)\n" + BAR)
import detectors as dt   # noqa: E402
import relics as rl      # noqa: E402
check("G1 BBO PLI minimum h2 Omega (SNR 1, 1 yr; Schmitz Fig.)", float(np.min(dt.CURVES["BBO"][1])), 5e-18, 1.5e-17,
      fmt="{:.2e}")
check("G2 LISA PLI minimum h2 Omega (SNR 1, 1 yr)", float(np.min(dt.CURVES["LISA"][1])), 1e-14, 3e-14, fmt="{:.2e}")
fpl = np.geomspace(1e-3, 1e1, 400)
check("G3 a power law at 10x the BBO curve's tangent point gives R = 10",
      dt.detection_ratio("BBO", fpl, 10 * dt.omega_pli("BBO", fpl)), 9.99, 10.01)
check("H1 relic grid: their benchmark (1e6 g, t_i = 1e-30 s) is excluded [log10 L]",
      float(np.log10(rl.limit_integral(1e6, np.sqrt(rl.T_I_COEF * co.HBAR / (1e-30 * np.sqrt(2) * co.H_form(1e6)))))),
      0.0, 10.0, fmt="{:.2f}")
check("H2 t_i(M, beta) growth-matched to H_eq: coefficient (2/3)^1.5 (2 sqrt2/3)", rl.T_I_COEF, 0.512, 0.514)
check("H3 merged-relic limit at 1e6 g: beta_relic", rl.beta_relic(1e6), 3e-5, 1e-4, fmt="{:.2e}")
check("H4 no relic limit up to beta = 1 at 1e3 g [max log10 L]",
      float(np.log10(np.max(rl.limit_integral(1e3, np.geomspace(co.beta_crit(1e3), 1, 200))))), -40, 0.0, fmt="{:.2f}")

print(BAR)
print(f"PHASE A VALIDATION: {sum(results)}/{len(results)} checks passed")
print(BAR)
