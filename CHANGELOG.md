# Changelog

## v1.1.0 — 2026-10-09 (TRIPOD+AI / PROBAST+AI revision; release matching the submitted manuscript)

### Updated for the submitted manuscript and Online Resource 1
- `supplementary/TRUST-BC_Online_Resource_1.docx` replaces `TRUST-BC_Supplementary_v1.1`. It adds
  Section S10 (extended methods, results and limitations), the model card (Table S10) and the dataset
  datasheets (Table S11).
- Every Online Resource table is now available as CSV in `supplementary/tables/` (added S1, S1b, S2a, S2b,
  S4, S6 coverage, S8, S9, S10, S11). Files not matching Online Resource numbering were renamed:
  `S6_seer_fairness_subgroups.csv` to `Table5_seer_subgroup_performance.csv`,
  `S1_participant_flow_counts.json` to `FigS1_participant_flow_counts.json`,
  `S8_lockbox_additional_CIs.json` to `lockbox_additional_CIs_WBCD_Coimbra.json`
  (`scripts/tripod_supplementary_analysis.py` writes the new names).
- `paper/TRUST-BC_manuscript_v1.1.docx` is the submitted manuscript.
- Title, repository URL (`RezaAbolghasemi/TRUST-BC-Multicohort-Benchmark`) and preprint DOI
  (10.5281/zenodo.22926051) now match the manuscript in `README.md` and `CITATION.cff`.
- README: SEER calibration-in-the-large corrected from −1.00 to −0.99 (pooled out-of-fold value −0.995, as in the
  manuscript and Table 5 data); the Coimbra Unified-vs-Leaky difference is no longer described as "Unified
  better" (+8.62 points, 95% CI −1.35 to +18.59, Holm *p* = 0.266, i.e. not significant); WBCD/Coimbra lockbox
  results added.
- Python requirement corrected to ≥ 3.11: the pinned SciPy 1.16.3 and SHAP 0.52.0 cannot be installed on 3.10, so
  the CI workflow (Python 3.10) could not have passed. CI now uses Python 3.12 and a separate environment for the
  v1.1 scripts.
- Added `requirements-revision.txt` (scikit-learn 1.8.0, NumPy 2.4.4, SciPy 1.17.1, pandas 3.0.2) for `scripts/`.
- Removed references to `paper/latex/`, `paper/archive/` and `supplementary/` LaTeX files that are not in the repository;
  fixed stale paths in `CONTRIBUTING.md`.
- `docs/`: the reference PDF is Tadj et al. (2026), CC BY; renamed accordingly and attribution added.
- Checks run: all scripts compile; `riley_sample_size.py` reproduces Table S9; `wbcd_lockbox_ci.py` reproduces the stored
  WBCD lockbox predictions exactly.

### Fixed
- **SEER 5-year label.** In v1.0 every death was labelled as a 5-year event, including
  158 of 616 deaths that occurred after month 60 (patients alive at the 5-year horizon).
  `src/trust_bc_multicohort_benchmark.py` now uses a fixed-horizon label
  (`SEER_LABEL_MODE = 'fixed_60m'`); set `SEER_LABEL_MODE = 'as_reported'` to reproduce v1.0.
  Corrected cohort: N = 3,270, 458 events (v1.0: 616).
- **Confidence intervals.** Cross-validated CIs now use the Nadeau–Bengio corrected
  variance (the v1.0 percentile bootstrap over correlated folds was too narrow).
- Author list, affiliations and repository links now match the manuscript.
- CI workflow pointed to a non-existent file (`src/trust_bc.py`); fixed.

### Added
- `scripts/`: re-analysis of the SEER classification branch with the corrected label
  (scikit-learn pipelines), survival C-index CI, Riley sample-size calculation,
  WBCD lockbox CIs, and all TRIPOD+AI supplementary analyses.
- `results/SEER_v1.1/`: corrected SEER results (fold scores, out-of-fold and lockbox
  predictions, conformal, DCA, fairness subgroups, figures).
- `supplementary/`: Supplementary Material (PDF, DOCX, LaTeX) with the completed
  TRIPOD+AI and TRIPOD+AI for Abstracts checklists, PROBAST+AI self-assessment,
  participant flow, characteristics and all supplementary tables (CSV).
- `paper/`: revised manuscript v1.1 (PDF, DOCX, LaTeX source).

### Figures
- Figures 2–6 redrawn at final print size in the original v1.0 visual style (seaborn whitegrid,
  same palettes, titles and SHAP beeswarm layout) by `scripts/make_publication_figures.py`
  (`results/figures_v1.1/`). WBCD and Coimbra out-of-fold, conformal and lockbox predictions were
  re-created with identical code and seeds (`scripts/wbcd_oof.py`, `scripts/wbcd_conformal.py`,
  `scripts/coimbra_revision.py`); all 50 fold accuracies, conformal percentages and lockbox metrics
  matched v1.0 exactly (Coimbra file SHA-256 identical to the v1.0 log). Coimbra lockbox bootstrap CIs
  and Coimbra characteristics by partition were added.

### Changed (interpretation)
- An architecture-matched contrast (Nested PCA-only vs Leaky) shows that global
  scaling/PCA leakage produced negligible optimism in all three cohorts; the
  non-equivalence of Unified vs Leaky in Coimbra and SEER reflects adaptive RF feature
  selection. The v1.0 "leakage-driven optimism" claim was withdrawn.

### Reproducibility check
- With `as_reported` labels, the v1.1 re-implementation reproduced all 35 v1.0 SEER
  outer-fold scores of the four scikit-learn pipelines exactly (scikit-learn 1.8.0).

### Not re-run
- XGBoost/LightGBM baselines for the corrected SEER endpoint (v1.0 values are kept in
  `results/SEER/` and Supplementary Table S8).

## v1.0.0 — 2026-09-01
- Initial release (results in `results/WBCD`, `results/Coimbra`, `results/SEER`).
