"""
Common look of the paper's figures, used by every plot_*.py script and plot_paper_figs.py.

Size.  The figures are drawn at the size at which they are printed (revtex4-2, aps/prd, one
column: \\textwidth = 510 pt = 7.06 in; single-panel figures are included at 0.62\\textwidth),
so the 8-9 pt fonts set here are the sizes in the paper.
Fonts.  Computer Modern, as in the text: set by LaTeX when latex and dvipng are installed
(the PDF then goes through pdflatex, see save()), otherwise by Matplotlib's mathtext with the
same fonts.
Colours.  One colour per channel throughout, from Paul Tol's colour-blind-safe 'vibrant'
scheme.
Cutoffs.  until_cutoff() lets a spectrum with a sharp cutoff end there: on a log scale it
would otherwise fall vertically to the bottom of the plot.
"""
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

WIDE = 510.0 / 72.27          # \textwidth [in]
NARROW = 0.62 * WIDE          # single-panel figures
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

COL = {"LIN-UV": "0.40", "LIN-NL": "#0077BB", "NL-A": "#CC3311", "NL-B": "#EE7733", "SHOT": "#009988"}
COL.update({"NL-A+F": COL["NL-A"], "NL-B+F": COL["NL-B"]})
CYAN, MAGENTA = "#33BBEE", "#EE3377"
DET = "0.62"                  # detector sensitivity curves
LIMIT = 5.6e-6 * 0.3          # Delta N_eff < 0.3 on the integrated Omega_GW h^2 today
USETEX = bool(shutil.which("latex") and shutil.which("dvipng") and shutil.which("pdflatex"))


def use():
    """Apply the paper style (call once, before creating figures)."""
    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.6, "lines.linewidth": 1.3, "patch.linewidth": 0.6,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.2, "ytick.major.size": 3.2, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "legend.frameon": False, "legend.handlelength": 2.4, "legend.borderaxespad": 0.4,
        "legend.labelspacing": 0.3, "legend.handletextpad": 0.5, "legend.columnspacing": 1.2,
        "axes.titlepad": 4.0, "axes.labelpad": 2.5,
        "lines.solid_capstyle": "butt", "lines.dash_capstyle": "butt",
        "savefig.dpi": 300, "pdf.fonttype": 42, "figure.dpi": 100,
        "hatch.linewidth": 0.5,
    })
    if USETEX:
        plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                             "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
                             "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
                             "pgf.preamble": r"\usepackage{amsmath,amssymb}"})
    else:
        plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                             "axes.formatter.use_mathtext": True, "axes.unicode_minus": False})


def save(fig, name):
    """
    Write results/<name>.pdf (vector, for the paper) and results/<name>.png (300 dpi).  With LaTeX
    the PDF is typeset by pdflatex (the pgf backend): Matplotlib's own PDF backend drops the
    minus sign of the TeX math fonts (seen with Matplotlib 3.11), so 10^-6 would print as 10^6.
    """
    fig.savefig(os.path.join(OUT, f"{name}.png"))
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), **({"backend": "pgf"} if USETEX else {}))
    plt.close(fig)


def until_cutoff(x, y, steep=25.0):
    """
    y for a log plot of a spectrum that ends at a sharp cutoff.  Points from the first zero
    on are dropped, and so are the last points before it where the curve falls more steeply
    than d ln y / d ln x = -steep: there the spectrum plunges to zero within a few per cent of
    the cutoff, which on a log scale is a vertical line down to the bottom of the plot.
    """
    x = np.asarray(x, dtype=float)
    y = np.array(y, dtype=float)
    pos = np.nonzero(y > 0)[0]
    if not len(pos):
        return np.full_like(y, np.nan)
    i = pos[-1]
    zero = np.nonzero(y[pos[0]:] <= 0)[0]
    if len(zero):
        i = pos[0] + zero[0] - 1          # last positive point before the first zero
    while i > pos[0] and np.log(y[i] / y[i - 1]) / np.log(x[i] / x[i - 1]) < -steep:
        i -= 1
    out = np.full_like(y, np.nan)
    out[pos[0]:i + 1] = y[pos[0]:i + 1]
    return out


def tex_num(v, digits=1):
    """2.3e-06 -> $2.3\\times10^{-6}$ (mantissa 1 -> $10^{-6}$)."""
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10.0 ** e
    if round(m, digits) == 10.0:
        m, e = 1.0, e + 1
    if abs(m - 1.0) < 0.5 * 10.0 ** -digits:
        return rf"$10^{{{e}}}$"
    return rf"${m:.{digits}f}\times10^{{{e}}}$"
