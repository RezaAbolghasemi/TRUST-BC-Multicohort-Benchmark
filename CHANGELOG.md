# Changelog

## v1.1.0 — 2026-09-27 (TRIPOD+AI / PROBAST+AI revision)

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
