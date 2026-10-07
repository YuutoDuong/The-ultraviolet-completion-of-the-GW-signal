"""
Stage 1 validation: every scale in cosmology.py, checked against something
external to it -- a textbook limit, a standard quoted number, or an analytic
asymptote derived by hand.

Run:  python validate_stage1.py
"""

import numpy as np
import cosmology as co

TOL = 0.03          # default: 3 % relative
results = []


def check(label, measured, predicted, tol=TOL, absolute=False):
    if absolute:
        err = abs(measured - predicted)
    else:
        err = abs(measured - predicted) / abs(predicted) if predicted else np.inf
    ok = bool(np.isfinite(err) and err < tol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<52s} got {measured:12.5g}   "
          f"want {predicted:12.5g}   err {err:.2e}")
    results.append(ok)


BAR = "=" * 108
FID_M, FID_B = 1.0e4, 1.0e-6          # fiducial point used throughout

print(BAR)
print("TEST 1-2  Single-hole normalisation against standard quoted values")
print(BAR)
# tau(1e10 g) = 410 s and T_BH(1e13 g) = 1.06 GeV are the two numbers every
# PBH paper agrees on; they fix G g_H and the 30720 pi in dM/dt.
check("lifetime(1e10 g) / s", co.lifetime_seconds(1e10), 410.0, tol=0.05)
check("T_BH(1e13 g) / GeV", co.T_hawking(1e13), 1.06, tol=0.02)
check("tau scales as M^3", co.lifetime_seconds(2e10) / co.lifetime_seconds(1e10),
      8.0)
check("T_BH scales as 1/M", co.T_hawking(2e13) / co.T_hawking(1e13), 0.5)

print()
print(BAR)
print("TEST 3  Exact matter+radiation background reduces to the known limits")
print(BAR)
for y, want, era in ((1e-6, 0.5, "RD"), (1e-4, 0.5, "RD"),
                     (1e8, 2.0 / 3.0, "MD"), (1e12, 2.0 / 3.0, "MD")):
    check(f"H t at y={y:.0e}  ({era})", co.F_of_y(y) * co.H_over_Heq(y), want)

print()
print(BAR)
print("TEST 4  Background is self-consistent with the formation time")
print(BAR)
# F(y) evaluated at y = beta_f must reproduce t_f = 1/(2 H_f) identically:
# the RD leg of the exact solution has to land on the formation epoch that
# H_f was defined from.  Any factor error in H_eq shows up here.
for M, b in ((1e4, 1e-6), (1e2, 1e-8), (1e7, 1e-10)):
    Hf = co.H_form(M)
    Heq = np.sqrt(2.0) * b ** 2 * Hf
    check(f"t_f from F(beta_f)  M={M:.0e} g, beta={b:.0e}",
          co.F_of_y(b) / Heq, 1.0 / (2.0 * Hf), tol=1e-6)

print()
print(BAR)
print("TEST 5-7  Poisson spectrum normalisation and the comoving scale ratios")
print(BAR)
# The 2/(3 pi) in P_S(k) is NOT an input: it follows from shot noise plus the
# definition (4pi/3) dbar^3 n = 1 that k_PBH is built on.  If these disagree,
# k_PBH and P_S are using two different mean separations.
check("P_S coefficient from shot noise", co.poisson_amplitude_coefficient(),
      2.0 / (3.0 * np.pi), tol=1e-12)
for M, b in ((1e4, 1e-6), (1e6, 1e-9)):
    s = co.scales(M, b)
    check(f"k_PBH/k_f = (beta/gamma)^(1/3)   M={M:.0e}",
          s["k_PBH"] / s["k_f"], (b / co.GAMMA) ** (1.0 / 3.0), tol=1e-12)
    check(f"k_eq/k_f  = sqrt2 beta          M={M:.0e}",
          s["k_eq"] / s["k_f"], np.sqrt(2.0) * b, tol=1e-12)

print()
print(BAR)
print("TEST 8  Critical abundance for PBH domination")
print(BAR)
# beta_crit is a closed form; feed it back through the numerical root find and
# demand y_evap = 1 exactly.
for M in (1e0, 1e3, 1e6, 1e8):
    bc = co.beta_crit(M)
    check(f"y_evap(beta_c) = 1   M={M:.0e} g", co.y_evap_of(M, bc), 1.0, tol=1e-8)
# and the hand-derived scaling  beta_c = 0.2232 m_pl / M_in
for M in (1e0, 1e4, 1e8):
    check(f"beta_c * M / m_pl   M={M:.0e} g",
          co.beta_crit(M) * M * co.GRAM / co.M_PL_NR, 0.22321, tol=1e-3)
print(f"\n  => beta_c = {co.beta_crit(1.0):.3e} (1 g / M_in)        "
      f"[RECHECK item 1: compare against the paper you cite]")

print()
print(BAR)
print("TEST 9  Adiabatic non-linear scale is independent of (M_in, beta_f)")
print(BAR)
want = np.sqrt(5.0 / (2.0 * np.sqrt(co.A_S)))
for M, b in ((1e2, 1e-8), (1e4, 1e-6), (1e7, 1e-11), (1e8, 1e-12)):
    s = co.scales(M, b)
    check(f"k_NL^adia / k_eva   M={M:.0e} g, beta={b:.0e}",
          s["k_NL_adia"] / s["k_eva"], want, tol=1e-12)
print(f"\n  => k_NL^adia = {want:.1f} k_eva, from sqrt(A_s) = {np.sqrt(co.A_S):.3e}")

print()
print(BAR)
print("TEST 10  Frequency machinery against the textbook horizon-mode value")
print(BAR)


def f_horizon_today(T, gs, gss):
    """Independent implementation: f_0 of the mode k = aH at temperature T."""
    H = np.pi / 3.0 * np.sqrt(gs / 10.0) * T ** 2 / co.M_PL
    a = (co.GS0 / gss) ** (1.0 / 3.0) * co.T0 / T
    return H * a * co.GEV_TO_HZ / (2.0 * np.pi)


check("f_0(k=aH) at T = 1 GeV, g_* = 100  [Hz]",
      f_horizon_today(1.0, 100.0, 100.0), 2.63e-8, tol=0.02)
check("f_0 scales linearly with T",
      f_horizon_today(10.0, 100.0, 100.0) / f_horizon_today(1.0, 100.0, 100.0),
      10.0, tol=1e-12)
# now the same quantity through the module's own chain (k_eva = y H_evap,
# a_evap = y, a_evap/a_0 from entropy conservation)
for M, b in ((1e4, 1e-6), (1e6, 1e-9)):
    s = co.scales(M, b)
    gs = co.g_star(s["T_evap"])
    check(f"f_eva via scales() vs direct   M={M:.0e} g",
          s["f_eva"], f_horizon_today(s["T_evap"], gs, gs), tol=0.02)

print()
print(BAR)
print("TEST 11  F(y): the exact factorisation, and its exact inverse")
print(BAR)


def _F_naive(y):
    """The textbook form, kept only as the thing to be beaten."""
    return (2.0 * np.sqrt(2.0) / 3.0) * (np.sqrt(1.0 + y) * (y - 2.0) + 2.0)


# (a) (u-1)^2 (u+2) with u = sqrt(1+y) is an identity, so the two must agree
#     wherever the naive form still has digits left -- y ~ 0.1 and above.
for y in (1e-1, 1.0, 1e2, 1e6):
    check(f"factored / naive at y={y:.0e}",
          float(co.F_of_y(y)) / _F_naive(y), 1.0, tol=1e-12)

# (b) Below that the naive form's error GROWS as y falls, like
#     ulp(2) / ((3/4) y^2).  Truncation error would shrink -- this is the
#     cancellation signature, the bug class that broke Stage 0's Gamma.
print("  ---- naive-form relative error, measured against the factored form:")
prev, grows = 0.0, True
for y in (1e-3, 1e-4, 1e-5, 1e-6):
    rel = abs(_F_naive(y) / float(co.F_of_y(y)) - 1.0)
    print(f"       y={y:.0e}   measured {rel:.2e}   "
          f"ulp(2)/((3/4)y^2) = {2.0 ** -52 * 2.0 / (0.75 * y ** 2):.2e}")
    grows, prev = grows and rel > prev, rel
print(f"  [{'PASS' if grows else 'FAIL'}] naive-form error grows as y falls  "
      "(cancellation, not truncation)")
results.append(grows)
dead = abs(_F_naive(1e-9) / float(co.F_of_y(1e-9)) - 1.0) > 0.1
print(f"  [{'PASS' if dead else 'FAIL'}] naive form has lost the answer "
      "entirely by y=1e-9")
results.append(dead)

# (c) The analytic inverse must reproduce a brute-force root find, and must
#     round-trip across the cos/cosh branch boundary at y = 3.  With the
#     half-angle form the round trip is exact to machine precision over the
#     full 24 decades in y, so the tolerance here is 1e-11, not a fudge.
for M, b in ((1e4, 1e-6), (1e2, 1e-6), (1e8, 1e-13)):
    check(f"y_of_F vs brentq   M={M:.0e} g, beta={b:.0e}",
          co.y_evap_of(M, b), co.y_evap_brentq(M, b), tol=1e-10)
for y in (1e-12, 1e-9, 1e-6, 1e-3, 1.0, 3.0, 1e3, 1e6, 1e12):
    check(f"y_of_F(F(y)) / y at y={y:.0e}",
          float(co.y_of_F(co.F_of_y(y))) / y, 1.0, tol=1e-11)

# (d) The RD limit is approached with the right expansion coefficients.
#     Multiplying (1 - y/3 + 3y^2/16) by (1 + y/2 - y^2/8),
#         H t = (1/2) (1 + y/6 - 5 y^2/48) + O(y^3).
#     Pinning those coefficients makes this a test rather than a restatement
#     of H t -> 1/2.
for y in (1e-3, 1e-5, 1e-7):
    check(f"H t = (1/2)(1 + y/6 - 5y^2/48) at y={y:.0e}",
          float(co.F_of_y(y)) * co.H_over_Heq(y),
          0.5 * (1.0 + y / 6.0 - 5.0 * y ** 2 / 48.0), tol=1e-8)

print()
print(BAR)
print("TEST 12  Non-linear Poisson decades match the hand-derived form")
print(BAR)
# exact:  R = [ D+(y) / sqrt(3 pi/2) ]^(2/3),  D+ = 1 + 3y/2.  Dropping the
# "1 +" is exactly the error the first version of this test made.
for M, b in ((1e4, 1e-6), (1e6, 1e-10), (1e8, 1e-13)):
    s = co.scales(M, b)
    y = s["y_evap"]
    check(f"R_nl exact   M={M:.0e} g, y={y:.2e}", s["R_nl"],
          ((1.0 + 1.5 * y) / np.sqrt(1.5 * np.pi)) ** (2.0 / 3.0), tol=1e-12)
# the asymptote R -> (0.691 y)^(2/3) is claimed only for y >> 1, and its 1/y
# approach is itself the check -- the leading correction is 0.44/y.
for M, b in ((1e4, 1e-6), (1e2, 1e-5), (1e0, 1e-4)):
    s = co.scales(M, b)
    y = s["y_evap"]
    check(f"   -> (0.691 y)^(2/3)   y={y:.2e}",
          s["R_nl"] / (0.6910 * y) ** (2.0 / 3.0), 1.0, tol=max(1e-3, 2.0 / y))

print()
print(BAR)
print("TEST 13  Domination flag, then the hierarchy where it applies")
print(BAR)
# The hierarchy statement is only meaningful for beta_f > beta_c.  Below it
# there is no eMD phase at all, k_eq > k_eva, and k_NL^iso runs above k_PBH
# because the Poisson modes never grow -- correct physics, wrong test point.
for M, b, want in ((1e2, 1e-8, False), (1e2, 1e-6, True),
                   (1e4, 1e-6, True), (1e7, 1e-11, True)):
    s = co.scales(M, b)
    got = s["dominates"]
    ok = (got == want) and (got == (b > s["beta_c"]))
    print(f"  [{'PASS' if ok else 'FAIL'}] M={M:.0e} g, beta={b:.0e}: "
          f"dominates={got!s:<5s} beta_c={s['beta_c']:.2e}  "
          f"y_evap={s['y_evap']:.3g}")
    results.append(ok)
for M, b in ((1e2, 1e-6), (1e4, 1e-6), (1e7, 1e-11)):
    s = co.scales(M, b)
    order = [("k_eva", s["k_eva"]), ("k_eq", s["k_eq"]),
             ("k_NL_iso", s["k_NL_iso"]), ("k_PBH", s["k_PBH"]),
             ("k_f", s["k_f"])]
    vals = [v for _, v in order]
    ok = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    print(f"  [{'PASS' if ok else 'FAIL'}] M={M:.0e} g, beta={b:.0e}   "
          + " < ".join(n for n, _ in order))
    results.append(ok)

print()
print(BAR)
print("TEST 14-15  Boundaries of the viable parameter region")
print(BAR)
# BBN: find M where T_evap drops to 4 MeV, at a beta_f safely in the dominated
# regime.  Scan rather than solve so the answer is not circular.
Ms = np.logspace(7.0, 9.5, 400)
Tev = np.array([co.scales(M, 1e3 * co.beta_crit(M))["T_evap"] for M in Ms])
M_bbn = float(np.interp(-np.log10(co.T_BBN), -np.log10(Tev), Ms))
check("M_in at T_evap = 4 MeV  [g]", M_bbn, 5.0e8, tol=0.35)
check("M_in minimum from H_f < H_inf  [g]", co.M_min_inflation(), 0.44, tol=0.05)

print()
print(BAR)
print("TEST 16  Fiducial point lands where the literature puts it")
print(BAR)
s = co.scales(FID_M, FID_B)
check("M=1e4 g: f_eva in the LISA band  [Hz]", s["f_eva"], 7.3e-4, tol=0.10)
check("M=1e4 g: T_evap  [GeV]", s["T_evap"], 2.8e4, tol=0.10)

print()
print(BAR)
print("TEST 17  Vectorised and scalar paths are literally the same code")
print(BAR)
Mg = np.array([1e1, 1e3, 1e5, 1e7])
Bg = np.array([[1e-5], [1e-8], [1e-11]])
A = co.scales_array(Mg, Bg)
worst, worst_key, dom_ok = 0.0, "", True
for i, b in enumerate(Bg[:, 0]):
    for j, M in enumerate(Mg):
        sc = co.scales(M, b)
        dom_ok = dom_ok and bool(A["dominates"][i, j]) == sc["dominates"]
        for key in ("y_evap", "k_PBH", "k_NL_iso", "k_eva", "f_eva", "f_PBH",
                    "R_nl", "T_evap", "beta_c", "T_f"):
            d = abs(A[key][i, j] / sc[key] - 1.0)
            if d > worst:
                worst, worst_key = d, f"{key} at M={M:.0e}, beta={b:.0e}"
check(f"worst of 12 points x 10 keys ({worst_key})", worst, 0.0,
      tol=1e-12, absolute=True)
print(f"  [{'PASS' if dom_ok else 'FAIL'}] domination flag agrees at all "
      "12 points")
results.append(dom_ok)

print()
print(BAR)
print("DIAGNOSTIC  the Stage 1 numbers, fiducial M_in = 1e4 g, beta_f = 1e-6")
print(BAR)
s = co.scales(FID_M, FID_B)
print(f"  T_BH                          {s['T_BH']:.3e} GeV")
print(f"  tau                           {s['tau_s']:.3e} s")
print(f"  beta_c                        {s['beta_c']:.3e}   (beta_f/beta_c = "
      f"{FID_B / s['beta_c']:.1f})")
print(f"  y_evap = a_evap/a_eq          {s['y_evap']:.3e}   "
      f"({s['N_efolds_eMD']:.1f} e-folds of eMD)")
print(f"  T_evap                        {s['T_evap']:.3e} GeV")
print(f"  Poisson delta at k_PBH, t_f   {np.sqrt(2 / (3 * np.pi)):.4f}"
      "        <-- O(1) before any growth")
print(f"  decades of non-linear Poisson {s['decades_nl']:.2f}")
print()
print(f"  {'scale':<12s} {'k / k_eva':>12s} {'f today [Hz]':>15s}")
for nm, kk, ff in (("k_eq", s["k_eq"], s["f_eq"]),
                   ("k_eva", s["k_eva"], s["f_eva"]),
                   ("k_univ", s["k_univ"], s["f_univ"]),
                   ("k_NL^adia", s["k_NL_adia"], s["f_NL_adia"]),
                   ("k_NL^iso", s["k_NL_iso"], s["f_NL_iso"]),
                   ("k_PBH", s["k_PBH"], s["f_PBH"])):
    print(f"  {nm:<12s} {kk / s['k_eva']:>12.3e} {ff:>15.3e}")
print()
print(f"  The k_UV bracket spans f = {s['f_NL_iso']:.2e} Hz to "
      f"{s['f_PBH']:.2e} Hz  ({s['decades_nl']:.2f} decades).")

print()
print(BAR)
print(f"RESULT: {sum(results)}/{len(results)} checks passed")
print(BAR)
