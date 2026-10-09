# TRUST-BC: A Multicohort Benchmark for Leakage-Free, Calibration-Audited, and Uncertainty-Aware Breast Cancer Prediction Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![TRIPOD+AI](https://img.shields.io/badge/reporting-TRIPOD%2BAI-informational)](supplementary/TRUST-BC_Online_Resource_1.docx)
[![Version](https://img.shields.io/badge/version-1.1.0-green.svg)](CHANGELOG.md)

**TRUST-BC** is an open-source auditing template and benchmark for leakage-free,
calibration-audited and uncertainty-aware machine learning on tabular breast cancer data. It jointly
implements:

- **Leakage-free nested cross-validation** — scaling, PCA and feature selection are fit only
  inside training folds, with an **architecture-matched leaky comparator** to measure the
  leakage effect.
- **Corrected statistical inference** — Nadeau–Bengio corrected tests and confidence
  intervals, Holm–Bonferroni control, and TOST equivalence testing.
- **Distribution-free uncertainty quantification** — split conformal prediction (marginal
  95% coverage), with coverage reported by outcome class and subgroup.
- **Calibration and clinical utility** — Brier score, ECE, calibration slope/intercept,
  calibration-in-the-large, and decision curve analysis.
- **Fixed-horizon registry labelling and survival modelling** — correct 5-year labels for
  SEER and a penalised Weibull AFT branch with Kaplan–Meier stratification.
- **Fairness and interpretability** — subgroup performance by race and age (SEER) and
  Shapley attributions on the held-out lockboxes.

Cohorts: **WBCD** (cytopathology, *n* = 569), **Coimbra** (serum biomarkers, case–control,
*n* = 116) and **SEER** (registry; *N* = 3,270 for 5-year classification, *N* = 4,024 for
survival).

📄 **Manuscript (v1.1):** [`paper/TRUST-BC_manuscript_v1.1.docx`](paper/TRUST-BC_manuscript_v1.1.docx),
submitted to the *Journal of Medical Systems*. Preprint: <https://doi.org/10.5281/zenodo.22926051>
📎 **Online Resource 1 (supplementary material):**
[`supplementary/TRUST-BC_Online_Resource_1.docx`](supplementary/TRUST-BC_Online_Resource_1.docx)
(TRIPOD+AI and TRIPOD+AI for Abstracts checklists, PROBAST+AI self-assessment, participant flow,
Tables S3–S9, extended methods and limitations S10, model card S11, dataset datasheets S12).
Every table in it is also provided as CSV in [`supplementary/tables/`](supplementary/tables/).

> **v1.1 correction.** v1.0 labelled all 616 SEER deaths as 5-year events, including 158
> deaths after month 60. v1.1 uses a fixed-horizon label (458 events). See
> [`CHANGELOG.md`](CHANGELOG.md) and Online Resource 1, Section S8.

---

## Key Results (v1.1)

| Cohort | Unified pipeline (leakage-free) | Leaky baseline | Leakage effect (Nested PCA − Leaky) | Unified ≡ Leaky (TOST)? |
|---|---|---|---|---|
| WBCD | Accuracy 97.08% (95% CI 95.34–98.82) | 96.85% | +0.31 pts (−0.54 to +1.16) | ✅ *p* = 0.0002 |
| Coimbra | Accuracy 71.35% (62.27–80.43) | 62.73% | +0.52 pts (−4.68 to +5.72) | ❌ *p* = 0.906 (not equivalent; see note below) |
| SEER 5-year | AUROC 0.739 (0.709–0.769) | 0.735 | +0.0000 AUROC (−0.0014 to +0.0014) | ❌ *p* = 0.225 |

- Global scaling/PCA leakage produced **negligible optimism** in all three cohorts; differences
  between Unified and Leaky reflect adaptive RF feature selection.
- In Coimbra the Unified pipeline was numerically higher than Leaky (+8.62 points, 95% CI −1.35 to
  +18.59), but the difference was not statistically significant after Holm correction
  (*p*<sub>Holm</sub> = 0.266); the cohort is small (*n* = 116) and the intervals are wide.
- SEER probabilities were **systematically too high** (mean predicted 28.6% vs 14.0% observed;
  calibration-in-the-large −0.99) because of balanced class weighting.
- SEER conformal coverage was 95.3% overall but only **69.9% among deaths** (99.4% among survivors).
- Lockboxes (15%, evaluated once): WBCD accuracy 100% (*n* = 86; a small-sample ceiling effect,
  not leakage), Coimbra 61.1% (*n* = 18), SEER AUROC 0.722 (0.646–0.790; *n* = 491, 69 events).
- Weibull AFT survival branch (test n = 805, 123 deaths): C-index 0.722 (approx. 95% CI 0.675–0.771),
  log-rank *p* = 1.02 × 10⁻⁹.

---

## Repository Structure

```
.
├── src/
│   └── trust_bc_multicohort_benchmark.py   # Full pipeline (all cohorts); SEER_LABEL_MODE switch
├── scripts/                                # v1.1 re-analysis and supplementary analyses
├── notebooks/
│   └── trust_bc_multicohort_benchmark.ipynb  # Colab notebook of the v1.0 run (archival)
├── paper/
│   └── TRUST-BC_manuscript_v1.1.docx       # Submitted manuscript
├── supplementary/
│   ├── TRUST-BC_Online_Resource_1.docx     # Online Resource 1
│   └── tables/                             # All Online Resource tables as CSV/JSON
├── results/
│   ├── WBCD/, Coimbra/                      # v1.0 results (unchanged)
│   ├── SEER/                                # v1.0 SEER results (original label; archival)
│   ├── SEER_v1.1/                           # corrected SEER results
│   ├── figures_v1.1/                        # Figures 2-6 as used in the manuscript
│   └── logs/
├── data/README.md                           # Data provenance and download instructions
├── docs/                                    # Reference re-analysis (Tadj et al. 2026, CC BY)
├── environment/                             # Package freeze of the v1.0 run (Google Colab)
├── requirements.txt                         # v1.0 pipeline (src/)
├── requirements-revision.txt                # v1.1 re-analysis scripts (scripts/)
└── CHANGELOG.md, CITATION.cff, CONTRIBUTING.md, LICENSE
```

---

## Installation

```bash
git clone https://github.com/RezaAbolghasemi/TRUST-BC-Multicohort-Benchmark.git
cd TRUST-BC-Multicohort-Benchmark
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt            # full pipeline in src/ (v1.0 versions)
pip install -r requirements-revision.txt   # v1.1 re-analysis scripts in scripts/ (separate environment)
```

Requires Python ≥ 3.11 (the pinned SciPy and SHAP releases do not install on Python 3.10; the
pins resolve on Python 3.12). The exact v1.0 Colab environment is in
[`environment/full-colab-environment-freeze.txt`](environment/full-colab-environment-freeze.txt).

## Usage

1. **Data.** WBCD loads through scikit-learn and Coimbra downloads from UCI at runtime. SEER
   must be downloaded manually (see [`data/README.md`](data/README.md)).
2. **Full pipeline (all cohorts):**
   ```bash
   python src/trust_bc_multicohort_benchmark.py
   ```
   Uses the corrected SEER label by default. Set `SEER_LABEL_MODE = 'as_reported'` at the top
   of the script to reproduce v1.0.
3. **v1.1 SEER re-analysis and supplementary tables:** see [`scripts/README.md`](scripts/README.md).
4. **Precomputed results:** [`results/`](results/) (v1.0 console log in `results/logs/`).

## Reproducibility Notes

- Seeds fixed globally (`SEED = 42`); single-threaded searches for bit-identical results.
- Preprocessing is fit only inside training folds; sample-hash overlap 0.00% (WBCD, Coimbra)
  and 0.20% (SEER, duplicate covariate patterns, not index-level leakage). The public SEER file
  contains one exact duplicate record, which was retained (*N* = 4,024).
- The 15% lockbox is evaluated once, after model selection, with no retuning.
- With the v1.0 label, the v1.1 re-implementation reproduced all 35 SEER outer-fold scores of
  the four scikit-learn pipelines exactly (scikit-learn 1.8.0).
- Reported following **TRIPOD+AI**; an author self-assessment against **PROBAST+AI** is in
  Online Resource 1 (it is not an independent appraisal).

## Citation

```bibtex
@article{shayegan2026trustbc,
  title   = {TRUST-BC: A Multicohort Benchmark for Leakage-Free, Calibration-Audited, and Uncertainty-Aware Breast Cancer Prediction Models},
  author  = {Shayegan, Mohammad Amin and Abolghasemi, Reza and Salehi, Mohammadreza and Kiarsi, Armaghan},
  journal = {Preprint (Zenodo)},
  year    = {2026},
  doi     = {10.5281/zenodo.22926051}
}
```
See also [`CITATION.cff`](CITATION.cff).

## Authors

- **Mohammad Amin Shayegan**\* — Department of Computer, Shi.C., Islamic Azad University, Shiraz, Iran
- **Reza Abolghasemi**\* — Department of Computer, Shi.C., Islamic Azad University, Shiraz, Iran
- **Mohammadreza Salehi** — Department of Computer, Shi.C., Islamic Azad University, Shiraz, Iran
- **Armaghan Kiarsi** — Faculty of Health, Medicine and Society (Oncology), University of Chester, UK

\*Corresponding authors: `MA.Shayegan@iau.ac.ir`, `Reza.Abolghasemi@iau.ir`

## License

- Code (`src/`, `scripts/`, `notebooks/`) is released under the [MIT License](LICENSE).
- Manuscript and Online Resource (`paper/`, `supplementary/`) are © the authors.
- Third-party data (WBCD, Coimbra, SEER) remain subject to their original licenses — see
  [`data/README.md`](data/README.md). No patient-level data are redistributed; prediction files
  contain only row indices, labels and model outputs.

## Disclaimer

For methodological benchmarking and research only. Not validated for clinical decision-making
or patient care.
