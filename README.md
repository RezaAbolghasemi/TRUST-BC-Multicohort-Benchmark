# TRUST-BC: A Multicohort Benchmark for Leakage-Free, Calibrated, and Uncertainty-Aware Breast Cancer Prediction Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TRIPOD+AI](https://img.shields.io/badge/reporting-TRIPOD%2BAI-informational)](supplementary/TRUST-BC_Supplementary_v1.1.pdf)
[![Version](https://img.shields.io/badge/version-1.1.0-green.svg)](CHANGELOG.md)

**TRUST-BC** is an open-source auditing template and benchmark for leakage-audited,
calibrated and uncertainty-aware machine learning on tabular breast cancer data. It jointly
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

📄 **Paper (v1.1):** [`paper/TRUST-BC_manuscript_v1.1.pdf`](paper/TRUST-BC_manuscript_v1.1.pdf) ·
[Word](paper/TRUST-BC_manuscript_v1.1.docx)
📎 **Supplementary Material:** [`supplementary/TRUST-BC_Supplementary_v1.1.pdf`](supplementary/TRUST-BC_Supplementary_v1.1.pdf)
(TRIPOD+AI checklist, PROBAST+AI self-assessment, participant flow, all supplementary tables)

> **v1.1 correction.** v1.0 labelled all 616 SEER deaths as 5-year events, including 158
> deaths after month 60. v1.1 uses a fixed-horizon label (458 events). See
> [`CHANGELOG.md`](CHANGELOG.md) and Supplementary Section S8.

---

## Key Results (v1.1)

| Cohort | Unified pipeline (leakage-free) | Leaky baseline | Leakage effect (Nested PCA − Leaky) | Unified ≡ Leaky (TOST)? |
|---|---|---|---|---|
| WBCD | Accuracy 97.08% (95% CI 95.34–98.82) | 96.85% | +0.31 pts (−0.54 to +1.16) | ✅ *p* = 0.0002 |
| Coimbra | Accuracy 71.35% (62.27–80.43) | 62.73% | +0.52 pts (−4.68 to +5.72) | ❌ *p* = 0.906 (Unified better) |
| SEER 5-year | AUROC 0.739 (0.709–0.769) | 0.735 | +0.0000 AUROC (−0.0014 to +0.0014) | ❌ *p* = 0.225 |

- Global scaling/PCA leakage produced **negligible optimism** in all three cohorts; differences
  between Unified and Leaky reflect adaptive RF feature selection.
- SEER probabilities were **systematically too high** (mean predicted 28.6% vs 14.0% observed;
  calibration-in-the-large −1.00) because of balanced class weighting.
- SEER conformal coverage was 95.3% overall but only **69.9% among deaths** (99.4% among survivors).
- SEER lockbox (n = 491, 69 events): AUROC 0.722 (0.646–0.790).
- Weibull AFT survival branch (test n = 805, 123 deaths): C-index 0.722 (approx. 95% CI 0.675–0.771),
  log-rank *p* = 1.02 × 10⁻⁹.

---

## Repository Structure

```
.
├── src/
│   └── trust_bc_multicohort_benchmark.py   # Full pipeline (all cohorts); SEER_LABEL_MODE switch
├── scripts/                                # v1.1 re-analysis and TRIPOD+AI supplementary analyses
├── notebooks/
│   └── trust_bc_multicohort_benchmark.ipynb  # Colab notebook of the v1.0 run (archival)
├── paper/
│   ├── TRUST-BC_manuscript_v1.1.pdf / .docx  # Revised manuscript
│   ├── latex/                               # LaTeX source of manuscript and supplement
│   └── archive/                             # v1.0 manuscript
├── supplementary/
│   ├── TRUST-BC_Supplementary_v1.1.pdf / .docx
│   └── tables/                              # All supplementary tables as CSV/JSON
├── results/
│   ├── WBCD/, Coimbra/                      # v1.0 results (unchanged)
│   ├── SEER/                                # v1.0 SEER results (original label; archival)
│   ├── SEER_v1.1/                           # corrected SEER results
│   └── logs/
├── data/README.md                           # Data provenance and download instructions
├── environment/                             # Package freeze of the v1.0 run
├── CHANGELOG.md, CITATION.cff, requirements.txt, LICENSE
```

---

## Installation

```bash
git clone https://github.com/mohammdreza-ui/TRUST-BC-Multicohort-Benchmark.git
cd TRUST-BC-Multicohort-Benchmark
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python ≥ 3.10. The exact v1.0 Colab environment is in
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
  and 0.20% (SEER, duplicate covariate patterns).
- The 15% lockbox is evaluated once, after model selection, with no retuning.
- With the v1.0 label, the v1.1 re-implementation reproduced all 35 SEER outer-fold scores of
  the four scikit-learn pipelines exactly (scikit-learn 1.8.0).
- Reported following **TRIPOD+AI**; an author self-assessment against **PROBAST+AI** is in
  the Supplementary Material (it is not an independent appraisal).

## Citation

```bibtex
@article{shayegan2026trustbc,
  title   = {TRUST-BC: A Multicohort Benchmark for Leakage-Free, Calibrated, and Uncertainty-Aware Breast Cancer Prediction Models},
  author  = {Shayegan, Mohammad Amin and Abolghasemi, Reza and Salehi, Mohammadreza and Kiarsi, Armaghan},
  journal = {Preprint},
  year    = {2026}
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
- Manuscript and supplement (`paper/`, `supplementary/`) are © the authors.
- Third-party data (WBCD, Coimbra, SEER) remain subject to their original licenses — see
  [`data/README.md`](data/README.md). No patient-level data are redistributed; prediction files
  contain only row indices, labels and model outputs.

## Disclaimer

For methodological benchmarking and research only. Not validated for clinical decision-making
or patient care.
