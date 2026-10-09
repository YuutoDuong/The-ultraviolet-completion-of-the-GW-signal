"""
Validation suite: reproduce the published endpoint scalings before extending them.

Targets (He, Ma, Sasaki & Takhistov, arXiv:2606.09804, Figs. 1-2;
         Gouttenoire, Leister & Schwaller, arXiv:2605.21474 Sec. 5):

  f(M,t) -> M^alpha  universal low-mass tail, independent of f0
  n_c    ~ s^1                                     s = t_evap - t
  rho_c  ~ s^(1 + 1/(alpha+1))    [mono: s^(1/(alpha+1))]
  S_Phi  ~ k^(-(alpha+2)/(alpha+1))  [mono: k^(-1/(alpha+1))]
  Omega_GW ~ k^(5 - 4b),  b = -dlnS_Phi/dlnk
     -> k^(-1/3) extended   vs   k^(+11/3) monochromatic
"""

import numpy as np
import evaporation as ev

TOL = 0.03
results = []


def loglog_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan
    return np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0]


def check(label, measured, predicted, tol=TOL):
    err = abs(measured - predicted)
    ok = np.isfinite(err) and err < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<44s} got {measured:+8.4f}   "
          f"want {predicted:+8.4f}   |d| {err:.4f}")
    results.append(bool(ok))


MFS = {
    "Choptuik power law p=3.78": ev.make_powerlaw(),
    "log-normal sigma=0.01": ev.make_lognormal(sigma=0.01),
    "log-normal sigma=0.10": ev.make_lognormal(sigma=0.10),
    "log-normal sigma=0.50": ev.make_lognormal(sigma=0.50),
}

BAR = "=" * 94

print(BAR)
print("TEST 1  Universal low-mass tail of the evolved mass function:  f(M,t) ~ M^alpha")
print(BAR)
for alpha in (2.0, 1.0):
    for name, (f0, M_max) in MFS.items():
        # Sample deep in the depletion tail. A near-monochromatic f0 (sigma=0.01)
        # spans barely a factor 1.05 in mass, so at s ~ 0.2 t_evap the surviving
        # population has not yet been driven onto the universal tail -- see TEST 6.
        s = 1e-5 * ev.lifetime(M_max, alpha)
        Mt = ev.M_top_s(s, alpha)
        M = np.logspace(np.log10(Mt) - 3.0, np.log10(Mt) - 1.0, 40)
        f = np.array([float(ev.f_of_M_s(m, s, f0, M_max, alpha)) for m in M])
        check(f"alpha={alpha:.0f}  {name}", loglog_slope(M, f), alpha)

print()
print(BAR)
print("TEST 2  Endpoint depletion of comoving number and energy density")
print(BAR)
for alpha in (2.0, 1.0, 0.0):
    f0, M_max = ev.make_powerlaw()
    te = ev.lifetime(M_max, alpha)
    s = np.logspace(-8, -5, 20) * te
    n = np.array([ev.n_comoving_s(si, f0, M_max, alpha) for si in s])
    r = np.array([ev.rho_comoving_s(si, f0, M_max, alpha) for si in s])
    rm = np.array([ev.rho_comoving_mono_s(si, alpha) for si in s])
    check(f"alpha={alpha:.0f}  n_c   vs s", loglog_slope(s, n), 1.0)
    check(f"alpha={alpha:.0f}  rho_c vs s", loglog_slope(s, r), 1.0 + 1.0 / (alpha + 1.0))
    check(f"alpha={alpha:.0f}  rho_c vs s  MONOCHROMATIC", loglog_slope(s, rm),
          1.0 / (alpha + 1.0))

print()
print(BAR)
print("TEST 3  Collective decay rate develops the pole  Gamma -> (a+2)/((a+1) s)")
print(BAR)
for alpha in (2.0, 1.0):
    f0, M_max = ev.make_powerlaw()
    te = ev.lifetime(M_max, alpha)
    for frac in (1e-3, 1e-5, 1e-7):
        G = ev.Gamma_collective_s(frac * te, f0, M_max, alpha)
        check(f"alpha={alpha:.0f}  s/t_evap={frac:.0e}   Gamma*s",
              G * frac * te, (alpha + 2.0) / (alpha + 1.0))

print()
print(BAR)
print("TEST 4  Suppression factor of the Newtonian potential  S_Phi(k), asymptotic k")
print(BAR)
# The universal scaling is an ENDPOINT result: it holds once the mode decouples
# deep in the depletion tail. Narrow mass functions reach that tail only at high
# k (see TEST 6), so the slope is fitted over the top decade of a wide grid.
kk = np.logspace(2.0, 7.0, 40)        # k / k_eva
ktop = kk[kk >= 1e6]
for alpha in (2.0, 1.0):
    for name, (f0, M_max) in MFS.items():
        S = np.array([ev.S_Phi(k, f0, M_max, alpha) for k in ktop])
        check(f"alpha={alpha:.0f}  {name:<26s} -dlnS/dlnk",
              -loglog_slope(ktop, S), (alpha + 2.0) / (alpha + 1.0))
f0, M_max = ev.make_powerlaw()
for alpha in (2.0, 1.0):
    S = np.array([ev.S_Phi(k, f0, M_max, alpha, monochromatic=True) for k in ktop])
    check(f"alpha={alpha:.0f}  MONOCHROMATIC{'':<14s} -dlnS/dlnk",
          -loglog_slope(ktop, S), 1.0 / (alpha + 1.0))

print()
print(BAR)
print("TEST 5  Implied high-frequency GW spectral index   Omega_GW ~ k^(5-4b)")
print(BAR)
alpha = 2.0
f0, M_max = ev.make_powerlaw()
b_ext = -loglog_slope(kk, np.array([ev.S_Phi(k, f0, M_max, alpha) for k in kk]))
b_mon = -loglog_slope(kk, np.array([ev.S_Phi(k, f0, M_max, alpha,
                                             monochromatic=True) for k in kk]))
check("extended mass function   n_GW", 5.0 - 4.0 * b_ext, -1.0 / 3.0)
check("monochromatic            n_GW", 5.0 - 4.0 * b_mon, 11.0 / 3.0)
print(f"\n  Poltergeist suppression at k/k_eva = 1e3:  "
      f"Omega_ext / Omega_mono = {(1e3) ** (-4 * (b_ext - b_mon)):.2e}")

print()
print(BAR)
print("TEST 6  DIAGNOSTIC  onset scale of the universal regime vs mass-function width")
print(BAR)
print("  The k^-4/3 law is an endpoint result. How far out in k must a mode sit")
print("  before it feels the depletion tail? Below k_univ the monochromatic")
print("  approximation is still numerically adequate, however wrong in principle.")
print()
print(f"  {'mass function':<26s} {'M_max/M_c':>10s} {'k_univ/k_eva':>14s}")
alpha = 2.0
target_slope = (alpha + 2.0) / (alpha + 1.0)
kgrid = np.logspace(1.0, 8.0, 71)
for sig in (0.01, 0.05, 0.10, 0.30, 0.50, 1.00):
    f0, M_max = ev.make_lognormal(sigma=sig)
    S = np.array([ev.S_Phi(k, f0, M_max, alpha) for k in kgrid])
    k_univ = np.nan
    for i in range(len(kgrid) - 6):
        sl = -loglog_slope(kgrid[i:i + 6], S[i:i + 6])
        if np.isfinite(sl) and abs(sl - target_slope) < 0.05 * target_slope:
            k_univ = kgrid[i]
            break
    print(f"  log-normal sigma={sig:<9.2f} {M_max:>10.3f} {k_univ:>14.2e}")
f0, M_max = ev.make_powerlaw()
S = np.array([ev.S_Phi(k, f0, M_max, alpha) for k in kgrid])
k_univ = np.nan
for i in range(len(kgrid) - 6):
    sl = -loglog_slope(kgrid[i:i + 6], S[i:i + 6])
    if np.isfinite(sl) and abs(sl - target_slope) < 0.05 * target_slope:
        k_univ = kgrid[i]
        break
print(f"  {'Choptuik p=3.78':<26s} {M_max:>10.3f} {k_univ:>14.2e}")

print()
print(BAR)
print(f"RESULT: {sum(results)}/{len(results)} checks passed")
print(BAR)
