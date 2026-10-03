"""Generate numbers.tex and fair_rows.tex for the manuscript from the v1.1 re-analysis outputs."""
import json, sys, pandas as pd, numpy as np
rev, supp = sys.argv[1], sys.argv[2]
s = json.load(open(f'{rev}/SEER_fixed_60m_summary.json'))
t3 = pd.read_csv(f'{rev}/SEER_fixed_60m_table3.csv').set_index('Pipeline')
cn = pd.read_csv(f'{supp}/S7_leakage_contrasts.csv'); cn = cn[(cn.cohort == 'SEER') & (cn.label_version == 'v1.1')].set_index('contrast')
ex = json.load(open(f'{supp}/S8_lockbox_additional_CIs.json'))
M = {}
f3 = lambda x: f'{x:.3f}'
for key, p in [('Leak', 'Leaky'), ('PCA', 'Nested_PCA'), ('RF', 'Nested_RF'), ('Uni', 'Unified')]:
    r = t3.loc[p]
    M[f'seer{key}AUC'] = f3(r.AUROC_mean); M[f'seer{key}SD'] = f3(r.AUROC_sd)
    M[f'seer{key}CI'] = f'{r.AUROC_CI_low_NB:.3f}--{r.AUROC_CI_high_NB:.3f}'
    M[f'seer{key}BE'] = f'{r.Brier:.3f} / {r.ECE:.3f}'
    M[f'seer{key}ECE'] = f3(r.ECE)
M['seerUniCIs'] = M['seerUniCI']; M['seerUniCI'] = '95\\% CI ' + M['seerUniCIs']
M['seerUniP'] = f"{t3.loc['Unified'].p_vs_Leaky_unadj:.4f}"; M['seerUniTOST'] = f"{t3.loc['Unified'].pTOST_vs_Leaky:.4f}"
M['seerCITL'] = f"{t3.loc['Unified'].CITL:.2f}"; M['seerSlope'] = f"{t3.loc['Unified'].Cal_slope:.2f}"
fair = pd.read_csv(f'{supp}/S6_seer_fairness_subgroups.csv')
M['seerMeanPred'] = f"{fair.iloc[0].mean_predicted_pct:.1f}"
d = cn.loc['Unified-Leaky']; M['seerDiffUL'] = f'{d.mean_diff:+.4f}'; M['seerDiffULci'] = f'{d.CI90_low:+.4f} to {d.CI90_high:+.4f}'
d = cn.loc['NestedPCA-Leaky']; M['seerDiffPL'] = f'{d.mean_diff:+.4f}'; M['seerDiffPLci'] = f'{d.CI95_low:+.4f} to {d.CI95_high:+.4f}'
c = s['conformal']
M['seerConfCov'] = f"{c['Coverage_pct_mean']:.2f}"; M['seerConfSingle'] = f"{c['Singleton_pct']:.2f}"; M['seerConfAmb'] = f"{c['Ambiguous_pct']:.2f}"
M['seerNcal'] = str(c['n_cal']); M['seerNcalEv'] = f"{c['n_cal_events']:.0f}"
L = s['lockbox']
M['seerLbAcc'] = f"{100*L['Accuracy']:.2f}"; M['seerLbAccCI'] = f"{100*L['Accuracy_CI'][0]:.2f}--{100*L['Accuracy_CI'][1]:.2f}"
M['seerLbBal'] = f"{100*L['Balanced_accuracy']:.2f}"
M['seerLbSens'] = f"{100*L['Sensitivity']:.2f}\\% ({L['TP']}/{L['TP']+L['FN']})"; M['seerLbSpec'] = f"{100*L['Specificity']:.2f}\\% ({L['TN']}/{L['TN']+L['FP']})"
M['seerLbAUC'] = f3(L['AUROC']); M['seerLbAUCCI'] = f"{L['AUROC_CI'][0]:.3f}--{L['AUROC_CI'][1]:.3f}"
M['seerLbBrier'] = f3(L['Brier']); M['seerLbBrierCI'] = f"{L['Brier_CI'][0]:.3f}--{L['Brier_CI'][1]:.3f}"
M['seerLbSlope'] = f"{L['Cal_slope']:.2f}"; M['seerLbSlopeCI'] = f"{L['Slope_Int_CI'][0][0]:.2f} to {L['Slope_Int_CI'][1][0]:.2f}"
M['seerLbInt'] = f"{L['Cal_intercept']:.2f}"; M['seerLbIntCI'] = f"{L['Slope_Int_CI'][0][1]:.2f} to {L['Slope_Int_CI'][1][1]:.2f}"
M['seerLbCITL'] = f"{L['CITL']:.2f}"; M['seerLbCITLCI'] = f"{L['CITL_CI'][0]:.2f} to {L['CITL_CI'][1]:.2f}"
M['seerLbCov'] = f"{100*L['Conformal_coverage']:.2f}"; M['seerLbCovCI'] = f"{100*L['Conformal_CI'][0]:.2f}--{100*L['Conformal_CI'][1]:.2f}"
M['wbcdLbBrier'] = f"{ex['WBCD_lockbox']['Brier']:.3f}"; M['wbcdLbBrierCI'] = f"{ex['WBCD_lockbox']['Brier_CI'][0]:.3f}--{ex['WBCD_lockbox']['Brier_CI'][1]:.3f}"
rf = s['arch'].get('rf', 0); M['seerRFfolds'] = str(rf); M['seerRFpct'] = f'{100*rf/35:.0f}'
nice = {'Progesterone Status_Positive': 'progesterone receptor status', 'N Stage': 'N stage', 'Reginol Node Positive': 'number of positive regional nodes',
        'Age': 'age', 'Grade': 'histological grade', 'Regional Node Examined': 'number of nodes examined', 'T Stage': 'T stage', 'Tumor Size': 'tumour size',
        'Estrogen Status_Positive': 'estrogen receptor status'}
top = [nice.get(x, x) for x in s['shapley_top'][:5]]
M['seerShapTop'] = ', '.join(top[:-1]) + ' and ' + top[-1]
dca = pd.read_csv(f'{rev}/SEER_fixed_60m_dca.csv'); m = (dca.nb_model > dca.nb_treat_all + 5e-4) & (dca.nb_model > 5e-4) & (dca.threshold < 0.5)
M['seerDCAlo'] = f'{dca[m].threshold.min():.2f}'; M['seerDCAhi'] = f'{dca[m].threshold.max():.2f}'
cs = pd.read_csv(f'{rev}/SEER_fixed_60m_conformal_subgroups.csv'); cs = cs[cs.variable.isin(['race', 'age'])]
M['seerCovRange'] = f"{cs.coverage_pct.min():.1f}\\% to {cs.coverage_pct.max():.1f}\\%"
c3 = pd.read_csv(f'{supp}/S3_cohort_characteristics.csv'); sm = pd.to_numeric(c3[c3.cohort == 'SEER'].SMD, errors='coerce').abs().max()
M['seerMaxSMD'] = f'$\\le${sm:.2f}'
try:
    nc = pd.read_csv(f'{rev}/SEER_fixed_60m_n_components_sensitivity.csv')
    M['seerNcompRange'] = f'maximal fluctuation {nc.mean_score.max()-nc.mean_score.min():.4f} AUROC'
except Exception:
    M['seerNcompRange'] = 'see Figure~\\ref{fig:dim}'
o, b, w, oth = fair.iloc[0], fair[fair.group == 'Black'].iloc[0], fair[fair.group == 'White'].iloc[0], fair[fair.group == 'Other'].iloc[0]
a1, a3 = fair[fair.group == '<50'].iloc[0], fair[fair.group == '60-69'].iloc[0]
M['seerFairText'] = (f"Discrimination was similar across racial groups (AUROC {w.AUROC:.3f} White, {b.AUROC:.3f} Black, {oth.AUROC:.3f} Other; overlapping CIs) "
    f"but declined with age ({a1.AUROC:.3f} for $<$50 years vs {a3.AUROC:.3f} for 60--69 years). Calibration differed markedly: the model overpredicted risk in all groups, "
    f"but least for Black patients (CITL {b.CITL:.2f}, 95\\% CI {b.CITL_low:.2f} to {b.CITL_high:.2f}; observed risk {b.observed_risk_pct:.1f}\\% vs predicted {b.mean_predicted_pct:.1f}\\%) "
    f"and most for Other-race patients (CITL {oth.CITL:.2f}; observed {oth.observed_risk_pct:.1f}\\% vs predicted {oth.mean_predicted_pct:.1f}\\%). "
    f"Because predicted risks varied much less between groups than observed risks, the model compressed the higher mortality of Black patients relative to other groups, so a single recalibration would not correct all groups equally. "
    f"Sensitivity at the 0.5 threshold was lowest for Other-race ({oth.sensitivity_pct:.1f}\\%) and Black ({b.sensitivity_pct:.1f}\\%) patients, and conformal coverage ranged from {cs.coverage_pct.min():.1f}\\% to {cs.coverage_pct.max():.1f}\\%. "
    f"Subgroup sizes, particularly the {int(b.events)} events among Black and {int(oth.events)} among Other-race patients, make these estimates imprecise.")
import re
neg = lambda v: re.sub(r'(?<![\w.\-])-(?=\d)', '$-$', v)
M = {k: (v if k == 'seerMaxSMD' else neg(v)) for k, v in M.items()}
with open('numbers.tex', 'w') as fh:
    for k, v in M.items():
        fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
rows = []
for r in fair.itertuples():
    cov = '--' if np.isnan(r.conformal_coverage_pct) else f'{r.conformal_coverage_pct:.1f}'
    g = 'Overall' if r.variable == 'Overall' else f"{r.variable}: {r.group.replace('<','$<$')}"
    rows.append(f"{g} & {r.n:,} & {r.events} & {r.observed_risk_pct:.1f} / {r.mean_predicted_pct:.1f} & "
                f"\\makecell[l]{{{r.AUROC:.3f}\\\\({r.AUROC_low:.3f}--{r.AUROC_high:.3f})}} & \\makecell[l]{{{r.CITL:.2f}\\\\({r.CITL_low:.2f} to {r.CITL_high:.2f})}} & {r.cal_slope:.2f} & "
                f"{r.sensitivity_pct:.1f} / {r.specificity_pct:.1f} & {cov}\\\\")
open('fair_rows.tex', 'w').write(neg('\n'.join(rows)))
print(open('numbers.tex').read())
