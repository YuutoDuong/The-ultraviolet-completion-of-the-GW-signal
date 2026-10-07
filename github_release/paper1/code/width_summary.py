"""
Summary of results/width_scan.csv: for each benchmark, the width sigma_1/2 at which the
coherent cluster signal (NL-A+F, integrated) has fallen to half its monochromatic value,
sigma_1/2 * x_cl, and the width sigma_x at which it drops below the shot noise.
Writes ../results/width_summary.txt.
"""
import os
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
rows = list(csv.DictReader(open(os.path.join(OUT, "width_scan.csv"))))


def series(sub, ch):
    pts = sorted((float(r["sigma"]), float(r["int_h2_damped"])) for r in sub
                 if r["channel"] == ch and r["sigma"] not in ("mono", "choptuik"))
    return np.array(pts).T


def crossing(s, y, level):
    """First sigma where y falls below level (log-log interpolation)."""
    below = np.nonzero(y < level)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    t = (np.log(level) - np.log(y[i - 1])) / (np.log(y[i]) - np.log(y[i - 1]))
    return float(np.exp(np.log(s[i - 1]) + t * (np.log(s[i]) - np.log(s[i - 1]))))


lines = [f"{'M_g':>7s} {'point':22s} {'x_cl':>9s} {'mono A+F':>10s} {'sigma_1/2':>10s} {'x_cl*s1/2':>10s} "
         f"{'sigma_x':>9s} {'SHOT mono':>10s} {'SHOT s=0.1':>10s} {'Chop A+F':>10s} {'Chop SHOT':>10s}"]
for M in ("1.0e+02", "1.0e+04", "1.0e+06"):
    for lab in ("2012.08151 bound", "saturated (beta=1e-2)"):
        sub = [r for r in rows if r["M_g"] == M and r["label"] == lab]
        val = lambda ch, sg: float([r for r in sub if r["channel"] == ch and r["sigma"] == sg][0]["int_h2_damped"])
        xcl = float(sub[0]["x_cl"])
        s, yA = series(sub, "NL-A+F")
        _, yS = series(sub, "SHOT")
        mono = val("NL-A+F", "mono")
        s12 = crossing(s, yA, 0.5 * mono)
        sx = crossing(s, yA / yS, 1.0)
        lines.append(f"{float(M):7.0e} {lab:22s} {xcl:9.3e} {mono:10.3e} {s12:10.2e} {xcl * s12:10.3f} "
                     f"{sx:9.2e} {val('SHOT', 'mono'):10.3e} {val('SHOT', '0.1'):10.3e} "
                     f"{val('NL-A+F', 'choptuik'):10.3e} {val('SHOT', 'choptuik'):10.3e}")
print("\n".join(lines))
with open(os.path.join(OUT, "width_summary.txt"), "w", newline="\n") as fh:
    fh.write("\n".join(lines) + "\n")
