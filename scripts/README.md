# Revision (v1.1) scripts

All scripts run on CPU with scikit-learn, pandas, SciPy, matplotlib and seaborn only.
Install them with `pip install -r requirements-revision.txt` (Python ≥ 3.11); this is a separate environment from the v1.0 pipeline in `src/` (`requirements.txt`).
`<SEER.csv>` is the public SEER extraction described in `data/README.md`.

| Script | Purpose | Output |
|---|---|---|
| `seer_revision_core.py` | Shared data preparation and pipelines (identical to `src/`) | — |
| `seer_revision_cv.py <SEER.csv> <fixed_60m\|as_reported> <out>` | Outer 5×7 nested CV for Leaky, Nested PCA, Nested RF, Unified | fold scores, out-of-fold predictions |
| `seer_revision_eval.py <SEER.csv> fixed_60m <out> [--skip-sensitivity]` | Statistics, calibration, DCA, conformal, lockbox (bootstrap CIs), fairness, permutation Shapley, figures | `SEER_fixed_60m_*` |
| `seer_revision_samplesize.py <SEER.csv> fixed_60m <out>` | Learning curves (B = 200) | sample-size CSV and figure |
| `seer_survival_ci.py <SEER.csv>` | Bootstrap CI for the Weibull AFT C-index | console |
| `wbcd_lockbox_ci.py <out>` | Re-creates WBCD lockbox predictions | `WBCD_lockbox_predictions.csv` |
| `riley_sample_size.py` | Riley et al. minimum sample size | console |
| `wbcd_oof.py <out>` | Re-creates WBCD out-of-fold predictions (reproduces all 50 v1.0 fold accuracies exactly) | `WBCD_oof_predictions.csv` |
| `coimbra_revision.py <dataR2.csv> <out>` | Re-creates Coimbra out-of-fold, conformal and lockbox predictions (identical to v1.0) | `results/Coimbra/revision_v1.1/` |
| `wbcd_conformal.py <out.csv>` | Re-creates the 100 WBCD conformal iterations (identical percentages to v1.0) | per-iteration counts |
| `make_publication_figures.py . <SEER.csv> results/WBCD/revision_v1.1 results/figures_v1.1 results/SEER_v1.1/folds_fixed_60m.pkl results/Coimbra/revision_v1.1 dataR2.csv` | Redraws Figures 2–6 at print size in the original v1.0 style | `results/figures_v1.1/` |
| `tripod_supplementary_analysis.py <SEER.csv> <rev_dir> <out>` | Corrected CIs (S5), contrasts (S7), characteristics (S3, S3b), subgroup performance (main Table 5), flow counts (Fig. S1), WBCD/Coimbra lockbox CIs | `supplementary/tables/` |

Run order to regenerate `results/SEER_v1.1/` (about 45 minutes on one CPU):

```bash
python scripts/seer_revision_cv.py SEER.csv fixed_60m results/SEER_v1.1
python scripts/seer_revision_eval.py SEER.csv fixed_60m results/SEER_v1.1 --skip-sensitivity
python scripts/seer_revision_samplesize.py SEER.csv fixed_60m results/SEER_v1.1
# n_components sensitivity: run seer_revision_eval.py without --skip-sensitivity
python scripts/wbcd_lockbox_ci.py results/SEER_v1.1
python scripts/tripod_supplementary_analysis.py SEER.csv results/SEER_v1.1 supplementary/tables dataR2.csv results/Coimbra/revision_v1.1
# optional reproduction check of v1.0
python scripts/seer_revision_cv.py SEER.csv as_reported results/SEER_v1.1
```

Checks run for the final release (Python 3.12, `requirements-revision.txt`): `riley_sample_size.py`
reproduces Online Resource Table S9 exactly, and `wbcd_lockbox_ci.py` reproduces the stored WBCD lockbox
predictions (identical indices, maximum absolute difference 0.0 in the predicted probabilities).
Tables S1, S1b, S2a, S2b, S4, S8, S9, S10 and S11 in `supplementary/tables/` are transcriptions of the
corresponding Online Resource tables, not script outputs.
