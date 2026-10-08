# The ultraviolet completion of the GW signal of PBH reheating: code

Code and data products for

> K. Duong, *The ultraviolet completion of the gravitational-wave signal of primordial black
> hole reheating: clustering, shot noise and dissipation*, arXiv: TBU.

Light primordial black holes (PBHs) can dominate the early Universe and reheat it when they
evaporate. The gravitational waves (GWs) induced at evaporation depend on how the PBH density
field is handled on small scales, where linear theory fails. This code replaces the usual
ultraviolet cutoff with the physics that operates there:

- **Clusters.** The PBHs sit in virialized clusters, described with a halo model (`clusters.py`).
- **The hand-over.** The PBH mass is handed over to radiation according to exact linear acoustics
  (`suddenness.py`, `evaporation.py`).
- **The burst.** The non-linear burst of each cluster is simulated with 1D relativistic
  hydrodynamics (`hydro1d.py`, `burst_efficiency.py`).
- **Damping.** Neutrino and plasma diffusion damp the sound (`damping.py`).
- **Lifetime.** The sound field steepens into shocks after a short time (`sound_lifetime.py`).

The GWs of the resulting sound field are computed with the sound-shell model
(`acoustic_gw.py`). It is checked against the full second-order induced-GW kernel
(`sigw_kernel.py`) and agrees to 0.1%.

## Layout

```
cosmology.py            background, scales, Poisson seed, linear growth (Stage 1)
evaporation.py          endpoint dynamics of extended mass functions (Stage 0)
validate.py             Stage-0 checks of evaporation.py
validate_stage1.py      Stage-1 checks of cosmology.py (71 checks)
paper1/code/            everything specific to the paper (one module per task, see below)
paper1/results/         every table, data product and figure of the paper, as shipped
papers/data/            third-party data used by the code (CC BY 4.0, see NOTICE.md)
```

The scripts find each other and the data through relative paths, so keep this layout.

## Requirements

Python 3.12 or later and three packages:

```
pip install -r requirements.txt
```

Tested with Python 3.14.3, NumPy 2.5.1, SciPy 1.18.1 and Matplotlib 3.11.0. NumPy 2 is
required: the code uses `np.trapezoid`. Everything runs on one CPU thread. Set
`OMP_NUM_THREADS=1` if your NumPy would otherwise use several.

The figures set their text with LaTeX, as in the paper, when `latex`, `pdflatex` and
`dvipng` are installed; without them Matplotlib's own Computer Modern fonts are used
(`figstyle.py`).

## Checking the code

Run from `paper1/code/`:

```
python validate_phase_a.py       # 80 checks of modules B-H and L, about 2 minutes
python ../../validate_stage1.py  # 71 checks of the scales, a few seconds
python ../../validate.py         # 35 checks of the evaporation endpoint, about 4 minutes
```

Every check prints PASS or FAIL together with the number it compared.

## Reproducing the paper

Run from `paper1/code/`. Each script overwrites its outputs in `paper1/results/`.

| Paper | Script(s) | Output in `paper1/results/` |
|---|---|---|
| Fig. 1, Table I (scales) | `plot_paper_figs.py` | `fig1_scales.pdf`, `fig1_scales.txt` |
| Fig. 2 (hand-over suppression) | `plot_paper_figs.py` | `fig2_suppression.pdf` |
| Fig. 3 (spectrum handed over) | `phase_a.py`, then `plot_paper_figs.py` | `fig3_spectra.pdf` |
| Fig. 4, Table II (burst efficiency) | `hydro_scan.py`, `hydro_gw_shift.py`, `hydro_spectrum.py`, then `plot_hydro.py` | `hydro_scan.txt`, `hydro_gw_shift.txt`, `hydro_efficiency.pdf` |
| Fig. 5, Table III (energy budget) | `energy_budget.py`, `plot_paper_figs.py` | `energy_budget.txt`, `fig4_energy.pdf` |
| Fig. 6, Table IV (spectra) | `phase_a.py` (Table IV); `plot_phase_a.py` (Fig. 6, recomputes the spectra on a finer frequency grid) | `phase_a_results.csv`, `phase_a_spectra.pdf` |
| Fig. 7 (saturation) | `beta_scan.py`, then `plot_phase_a.py` | `beta_scan.csv`, `phase_a_beta_scan.pdf` |
| Fig. 8, Table V (mass-function width) | `width_scan.py`, `width_summary.py`, then `plot_width_scan.py` | `width_scan.csv`, `width_summary.txt`, `width_scan.pdf` |
| Fig. 9, Table VI (plane, detectability) | `plane_scan.py`, `detect_map.py`, then `plot_plane.py` | `plane_scan.csv`, `detect_map.csv`, `plane.pdf` |
| Sec. VII C (lifetime of the sound) | `lifetime_summary.py` (after `plane_scan.py`) | `lifetime_summary.txt` |
| Table VII (sound-shell model vs induced-GW kernel) | `e2_compare.py` | `e2_compare.txt` |
| Table VIII (damping scale vs the literature) | `damping_compare.py` | `damping_compare.txt` |
| Sec. IX A (validity) | `validity.py` (prints; saved as `validity.txt`) | `validity.txt` |
| Sec. IX C (binary mergers) | `mergers.py` (prints) | — |

Most steps take a few minutes on one thread; the longest are:
- `width_scan.py`: about 45 minutes;
- `beta_scan.py`: about 20 minutes;
- `plane_scan.py`: about 6 minutes;
- the hydrodynamics scans: several minutes each;
- `plot_phase_a.py` and `plot_plane.py`: about 2 minutes each.

## Modules in `paper1/code/`

| Module | What it does |
|---|---|
| `clusters.py` | Press–Schechter cluster statistics, halo model, puff-up |
| `suddenness.py` | hand-over of the PBH mass to radiation: coherent and incoherent parts |
| `phase_a.py` | the channels of the paper (LIN-NL, LIN-UV, NL-A, NL-B, SHOT, with module F and shock-limited sound) for one point |
| `hydro1d.py`, `burst_efficiency.py` | 1D spherical relativistic hydrodynamics; the burst efficiency in closed form |
| `damping.py` | diffusion damping scale from neutrino and plasma viscosity |
| `acoustic_gw.py` | sound-shell model for the GW spectrum, with damping and a finite source lifetime |
| `sound_lifetime.py` | shock-formation time of the sound field and the resulting suppression |
| `sigw_kernel.py` | full second-order induced-GW kernel of a sudden transition |
| `detectors.py`, `detect_map.py` | power-law-integrated sensitivity curves; detection ratios on the plane |
| `relics.py` | limits from merged relics (from the data of Holst, Krnjaic and Xiao) |
| `mergers.py` | early-binary mergers before evaporation |
| `figstyle.py` | common look of the figures: printed size, fonts, colours, spectra that end at a cutoff |

## Data from other authors

`papers/data/` holds two public datasets under CC BY 4.0, unchanged except that only the
files the code reads are included:

- the merged-relic limit grid of Holst, Krnjaic and Xiao (arXiv:2412.01890), Zenodo
  [10.5281/zenodo.17210834](https://doi.org/10.5281/zenodo.17210834);
- the power-law-integrated sensitivity curves of Schmitz (arXiv:2002.04615), Zenodo
  [10.5281/zenodo.3689582](https://doi.org/10.5281/zenodo.3689582).

See `papers/data/NOTICE.md`. If you use these files, cite the original papers.

## Citation

If you use this code, please cite the paper above (the arXiv number will be added on
posting). Repository:
<https://github.com/YuutoDuong/The-ultraviolet-completion-of-the-GW-signal>; a Zenodo
archive with a DOI will follow.

## License

The code is released under the license in `LICENSE`. The third-party data in `papers/data/`
keep their CC BY 4.0 licenses.

## Contact

Khoa Duong, Phenikaa Institute for Advanced Study, Phenikaa University, Hanoi, Vietnam
(ORCID [0009-0009-9409-3298](https://orcid.org/0009-0009-9409-3298)).
