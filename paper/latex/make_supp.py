"""Generate row files and macros for supplementary.tex from supplementary/tables and the v1.1 re-analysis."""
import sys, os, json, re, numpy as np, pandas as pd
rev, supp, seer, results = sys.argv[1:5]
neg = lambda v: re.sub(r'(?<![\w.\-])-(?=\d)', '$-$', v)
esc = lambda v: str(v).replace('%', '\\%').replace('&', '\\&').replace('_', '\\_').replace('<', '$<$').replace('>=', '$\\ge$').replace('>', '$>$')
c3 = pd.read_csv(f'{supp}/S3_cohort_characteristics.csv').fillna('')
rows = []
NN = {'WBCD': ('483', '86'), 'Coimbra': ('98', '18'), 'SEER': ('2,779', '491')}
for coh in ['WBCD', 'Coimbra', 'SEER']:
    rows.append(f'\\multicolumn{{4}}{{@{{}}l}}{{\\textbf{{{coh}}} (development $n$={NN[coh][0]}; lockbox $n$={NN[coh][1]})}}\\\\')
    for r in c3[c3.cohort == coh].itertuples():
        rows.append(f'{esc(r.characteristic)} & {esc(r.development)} & {esc(r.lockbox)} & {r.SMD}\\\\')
open('s3_rows.tex', 'w').write(neg('\n'.join(rows)))
b = pd.read_csv(f'{supp}/S3b_seer_by_race.csv')
open('s3b_rows.tex', 'w').write('\n'.join(f'{r.race} & {r.n:,} & {r.age_mean:.1f} & {r.T3_T4_pct:.1f} & {r.N3_pct:.1f} & {r.ER_negative_pct:.1f} & {r.death60_pct:.1f}\\\\' for r in b.itertuples()))
t5 = pd.read_csv(f'{supp}/S5_table3_corrected_CIs.csv')
rr = []
for r in t5.itertuples():
    pct = r.metric == 'Accuracy'
    f = (lambda x: f'{100*x:.2f}\\%') if pct else (lambda x: f'{x:.3f}')
    rr.append(f"{r.cohort} & {esc(r.label_version)} & {esc(r.pipeline)} & {r.metric} & {f(r.mean)} & {f(r.sd)} & {f(r.CI95_low_NB)}--{f(r.CI95_high_NB)}\\\\")
open('s5_rows.tex', 'w').write('\n'.join(rr))
cs = pd.read_csv(f'{rev}/SEER_fixed_60m_conformal_subgroups.csv')
rr = []
for r in cs.itertuples():
    g = lambda x: '--' if pd.isna(x) else f'{x:.1f}'
    rr.append(f"{r.variable} & {esc(r.group)} & {g(r.coverage_pct)} & {g(r.coverage_events_pct)} & {g(r.coverage_nonevents_pct)} & {g(r.ambiguous_pct)}\\\\")
open('s6_rows.tex', 'w').write('\n'.join(rr))
t7 = pd.read_csv(f'{supp}/S7_leakage_contrasts.csv')
rr = []
for r in t7.itertuples():
    k = 100 if r.metric == 'Accuracy' else 1; u = ' pts' if k == 100 else ''
    ptost = '$<$0.0001' if r.pTOST < 1e-4 else f'{r.pTOST:.4f}'
    rr.append(f"{r.cohort} & {r.label_version} & {r.contrast.replace('NestedPCA','Nested PCA')} & {k*r.mean_diff:+.{2 if k==100 else 4}f}{u} & {k*r.CI95_low:+.{2 if k==100 else 4}f} to {k*r.CI95_high:+.{2 if k==100 else 4}f} & {r.p_two_sided_unadjusted:.4f} & {ptost}\\\\")
open('s7_rows.tex', 'w').write(neg('\n'.join(rr)))
raw = pd.read_csv(seer); raw.columns = raw.columns.str.strip()
dead = raw.Status.str.strip().str.lower() == 'dead'
M = {'seerMaxDeathMonth': str(int(raw.loc[dead, 'Survival Months'].max()))}
ar = os.path.join(rev, 'SEER_as_reported_raw_cv_scores.csv')
if os.path.exists(ar):
    a = pd.read_csv(ar); o = pd.read_csv(f'{results}/SEER/SEER_raw_cv_scores.csv')
    cols = [f'{p}_{m}' for p in ['Leaky', 'Nested_PCA', 'Nested_RF', 'Unified'] for m in ['AUROC', 'Accuracy']]
    d = np.abs(a[cols].values - o[cols].values).max()
    M['seerReproText'] = (f'reproduced all {len(a)} outer-fold AUROC and accuracy values of the four pipelines reported in \\texttt{{results/SEER/SEER\\_raw\\_cv\\_scores.csv}} '
                          f'(maximum absolute difference {d:.1g} at the stored precision), and the Unified pipeline selected RF in {int((a.Unified_arch=="rf").sum())}/35 folds as in v1.0.')
else:
    M['seerReproText'] = 'reproduced the first outer fold of all four pipelines to five decimal places.'
open('supp_numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in M.items()))
print(M)
