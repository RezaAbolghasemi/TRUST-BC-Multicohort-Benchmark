"""
TRIPOD+AI supplementary analyses for TRUST-BC (revision v1.1).

Computes, from the committed results/ folder (and, where needed, the public datasets):
  * Nadeau-Bengio corrected 95% CIs for every cohort/pipeline (Table 3)
  * architecture-matched leakage contrasts (Nested PCA - Leaky) and Unified - Leaky
  * bootstrap CIs for WBCD lockbox metrics (re-created predictions)
  * Hanley-McNeil approximate CI for the Coimbra lockbox AUROC (no predictions stored)
  * per-patient SEER subgroup (fairness) metrics from out-of-fold predictions
  * cohort characteristics (WBCD, SEER) by development / lockbox partition
  * participant-flow counts
Outputs CSV files to supplementary/tables/ and a LaTeX macro file used by the manuscript.
Usage: python scripts/tripod_supplementary_analysis.py <SEER.csv> <revision_dir> <out_dir> [dataR2.csv] [coimbra_revision_dir]
"""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
sys.path.insert(0, os.path.dirname(__file__))
from seer_revision_core import load_seer, dev_holdout_split

seer_path, rev, out = sys.argv[1], sys.argv[2], sys.argv[3]
coim_path = sys.argv[4] if len(sys.argv) > 4 else None
coim_rev = sys.argv[5] if len(sys.argv) > 5 else os.path.join(os.path.dirname(__file__), '..', 'results', 'Coimbra', 'revision_v1.1')
root = os.path.join(os.path.dirname(__file__), '..')
os.makedirs(out, exist_ok=True)
M = {}  # LaTeX macros

def nb_ci(a, K=5, level=0.95):
    a = np.asarray(a); n = len(a); se = a.std(ddof=1) * np.sqrt(1 / n + 1 / (K - 1))
    t = stats.t.ppf(1 - (1 - level) / 2, n - 1); return a.mean(), a.mean() - t * se, a.mean() + t * se
def tost(a, b, margin, K=5):
    d = np.asarray(a) - np.asarray(b); n = len(d); se = d.std(ddof=1) * np.sqrt(1 / n + 1 / (K - 1))
    return max(stats.t.sf((d.mean() + margin) / se, n - 1), stats.t.cdf((d.mean() - margin) / se, n - 1))
def ptest(a, b, K=5):
    d = np.asarray(a) - np.asarray(b); n = len(d); se = d.std(ddof=1) * np.sqrt(1 / n + 1 / (K - 1))
    return 2 * stats.t.sf(abs(d.mean() / se), n - 1)
def cal_slope_int(y, p):
    lp = logit(np.clip(p, 1e-7, 1 - 1e-7)); m = LogisticRegression(C=1e5).fit(lp.reshape(-1, 1), y)
    return m.coef_[0][0], m.intercept_[0]
def citl(y, p):
    lp = logit(np.clip(p, 1e-7, 1 - 1e-7)); a = 0.0
    for _ in range(100):
        mu = expit(a + lp); step = np.sum(y - mu) / np.sum(mu * (1 - mu)); a += step
        if abs(step) < 1e-10: break
    return a
def boot(y, p, fn, B=2000, seed=42):
    rng = np.random.default_rng(seed); i1 = np.where(y == 1)[0]; i0 = np.where(y == 0)[0]; v = []
    for _ in range(B):
        s = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        try: v.append(fn(y[s], p[s]))
        except Exception: pass
    return np.percentile(np.array(v), [2.5, 97.5], axis=0)

# ---------- Table 3 corrected CIs and contrasts ----------
rows, con = [], []
for c, m, margin in [('WBCD', 'Accuracy', 0.02), ('Coimbra', 'Accuracy', 0.02), ('SEER', 'AUROC', 0.01)]:
    r = pd.read_csv(os.path.join(root, 'results', c, f'{c}_raw_cv_scores.csv'))
    for p in ['Leaky', 'XGBoost', 'LightGBM', 'Nested_PCA', 'Nested_RF', 'Unified']:
        mu, lo, hi = nb_ci(r[f'{p}_{m}'])
        rows.append({'cohort': c, 'label_version': 'v1.0 (as reported)' if c == 'SEER' else 'original', 'pipeline': p, 'metric': m,
                     'mean': mu, 'sd': r[f'{p}_{m}'].std(ddof=1), 'CI95_low_NB': lo, 'CI95_high_NB': hi})
    for name, a, b in [('Unified-Leaky', 'Unified', 'Leaky'), ('NestedPCA-Leaky', 'Nested_PCA', 'Leaky')]:
        d = r[f'{a}_{m}'] - r[f'{b}_{m}']; mu, lo, hi = nb_ci(d); _, lo9, hi9 = nb_ci(d, level=0.90)
        con.append({'cohort': c, 'label_version': 'v1.0' if c == 'SEER' else 'original', 'contrast': name, 'metric': m, 'mean_diff': mu,
                    'CI95_low': lo, 'CI95_high': hi, 'CI90_low': lo9, 'CI90_high': hi9,
                    'p_two_sided_unadjusted': ptest(r[f'{a}_{m}'], r[f'{b}_{m}']), 'pTOST': tost(r[f'{a}_{m}'], r[f'{b}_{m}'], margin)})
fx = pd.read_csv(os.path.join(rev, 'SEER_fixed_60m_raw_cv_scores.csv'))
for p in ['Leaky', 'Nested_PCA', 'Nested_RF', 'Unified']:
    mu, lo, hi = nb_ci(fx[f'{p}_AUROC'])
    rows.append({'cohort': 'SEER', 'label_version': 'v1.1 (fixed 60-month)', 'pipeline': p, 'metric': 'AUROC', 'mean': mu,
                 'sd': fx[f'{p}_AUROC'].std(ddof=1), 'CI95_low_NB': lo, 'CI95_high_NB': hi})
for name, a, b in [('Unified-Leaky', 'Unified', 'Leaky'), ('NestedPCA-Leaky', 'Nested_PCA', 'Leaky')]:
    d = fx[f'{a}_AUROC'] - fx[f'{b}_AUROC']; mu, lo, hi = nb_ci(d); _, lo9, hi9 = nb_ci(d, level=0.90)
    con.append({'cohort': 'SEER', 'label_version': 'v1.1', 'contrast': name, 'metric': 'AUROC', 'mean_diff': mu,
                'CI95_low': lo, 'CI95_high': hi, 'CI90_low': lo9, 'CI90_high': hi9,
                'p_two_sided_unadjusted': ptest(fx[f'{a}_AUROC'], fx[f'{b}_AUROC']), 'pTOST': tost(fx[f'{a}_AUROC'], fx[f'{b}_AUROC'], 0.01)})
t3 = pd.DataFrame(rows); t3.to_csv(os.path.join(out, 'S5_table3_corrected_CIs.csv'), index=False)
cn = pd.DataFrame(con); cn.to_csv(os.path.join(out, 'S7_leakage_contrasts.csv'), index=False)
print(cn.round(4).to_string())

# ---------- WBCD lockbox bootstrap ----------
w = pd.read_csv(os.path.join(rev, 'WBCD_lockbox_predictions.csv'))
yw, pw = w.y.values, w.p.values
wb = {'Brier': brier_score_loss(yw, pw), 'Brier_CI': boot(yw, pw, brier_score_loss).tolist(),
      'AUROC': roc_auc_score(yw, pw), 'AUROC_CI': boot(yw, pw, roc_auc_score).tolist()}
# ---------- Coimbra lockbox (re-created predictions) ----------
cl = pd.read_csv(os.path.join(coim_rev, 'Coimbra_lockbox_predictions.csv'))
yc_, pc_ = cl.y.values, cl.p.values
co = {'AUROC': roc_auc_score(yc_, pc_), 'AUROC_CI': boot(yc_, pc_, roc_auc_score).tolist(),
      'Brier': brier_score_loss(yc_, pc_), 'Brier_CI': boot(yc_, pc_, brier_score_loss).tolist(),
      'slope_intercept': list(cal_slope_int(yc_, pc_)), 'slope_intercept_CI': boot(yc_, pc_, lambda a, b: np.array(cal_slope_int(a, b))).tolist()}
json.dump({'WBCD_lockbox': wb, 'Coimbra_lockbox': co}, open(os.path.join(out, 'S8_lockbox_additional_CIs.json'), 'w'), indent=2)
print('WBCD', wb, 'Coimbra', co)

# ---------- SEER per-patient fairness ----------
oof = pd.read_csv(os.path.join(rev, 'SEER_fixed_60m_oof_predictions.csv'))
pp = oof.groupby('orig_index').agg(y=('y', 'first'), p=('Unified', 'mean'), age=('age', 'first'), race=('race', 'first')).reset_index()
pp['race_grp'] = pp.race.replace({'Other (American Indian/AK Native, Asian/Pacific Islander)': 'Other'})
pp['age_grp'] = pd.cut(pp.age, [0, 49, 59, 200], labels=['<50', '50-59', '60-69']).astype(str)
fr = []
for var, g in [('Overall', None)] + [('Race', x) for x in ['White', 'Black', 'Other']] + [('Age', x) for x in ['<50', '50-59', '60-69']]:
    d = pp if g is None else pp[pp.race_grp == g] if var == 'Race' else pp[pp.age_grp == g]
    y, p = d.y.values, d.p.values
    a = roc_auc_score(y, p); alo, ahi = boot(y, p, roc_auc_score)
    ci = citl(y, p); clo, chi = boot(y, p, citl)
    s, _ = cal_slope_int(y, p)
    pr = p >= 0.5
    fr.append({'variable': var, 'group': 'All' if g is None else g, 'n': len(d), 'events': int(y.sum()), 'observed_risk_pct': 100 * y.mean(),
               'mean_predicted_pct': 100 * p.mean(), 'AUROC': a, 'AUROC_low': alo, 'AUROC_high': ahi, 'CITL': ci, 'CITL_low': clo,
               'CITL_high': chi, 'cal_slope': s, 'sensitivity_pct': 100 * (pr & (y == 1)).sum() / max(1, (y == 1).sum()),
               'specificity_pct': 100 * (~pr & (y == 0)).sum() / max(1, (y == 0).sum())})
fr = pd.DataFrame(fr)
cs = pd.read_csv(os.path.join(rev, 'SEER_fixed_60m_conformal_subgroups.csv'))
cov = {(r.variable, r.group): r.coverage_pct for r in cs.itertuples()}
fr['conformal_coverage_pct'] = [np.nan if v == 'Overall' else cov.get(('race' if v == 'Race' else 'age', {'60-69': '>=60'}.get(g, g)), np.nan)
                                for v, g in zip(fr.variable, fr.group)]
fr.to_csv(os.path.join(out, 'S6_seer_fairness_subgroups.csv'), index=False)
print(fr.round(3).to_string())

# ---------- characteristics ----------
d = load_breast_cancer(); Xw = pd.DataFrame(d.data, columns=d.feature_names); yw_all = 1 - d.target
man = pd.read_csv(os.path.join(root, 'results', 'WBCD', 'WBCD_split_manifest.csv'))
def smd(a, b): return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
ch = []
for c in [x for x in Xw.columns if x.startswith('mean ')]:
    a, b = Xw[man.Split.values == 'Dev'][c], Xw[man.Split.values == 'Holdout'][c]
    ch.append({'cohort': 'WBCD', 'characteristic': c + ', mean (SD)', 'development': f'{a.mean():.3g} ({a.std():.2g})', 'lockbox': f'{b.mean():.3g} ({b.std():.2g})', 'SMD': round(smd(a, b), 2)})
ch.append({'cohort': 'WBCD', 'characteristic': 'Malignant, n (%)', 'development': f'180 ({100*180/483:.1f})', 'lockbox': f'32 ({100*32/86:.1f})', 'SMD': ''})
if coim_path:
    cdf = pd.read_csv(coim_path); cm = pd.read_csv(os.path.join(coim_rev, 'Coimbra_split_check.csv'))
    hold = set(cm[cm.Split == 'Holdout'].Original_Index); sp = np.array(['Holdout' if i in hold else 'Dev' for i in range(len(cdf))])
    for c in [x for x in cdf.columns if x != 'Classification']:
        a, b = cdf[sp == 'Dev'][c], cdf[sp == 'Holdout'][c]
        ch.append({'cohort': 'Coimbra', 'characteristic': c.replace('MCP.1', 'MCP-1') + ', mean (SD)', 'development': f'{a.mean():.3g} ({a.std():.3g})',
                   'lockbox': f'{b.mean():.3g} ({b.std():.3g})', 'SMD': round(smd(a, b), 2)})
    nd, nh = (cdf[sp == 'Dev'].Classification == 2).sum(), (cdf[sp == 'Holdout'].Classification == 2).sum()
    ch.append({'cohort': 'Coimbra', 'characteristic': 'Breast cancer cases, n (%)', 'development': f'{nd} ({100*nd/(sp=="Dev").sum():.1f})',
               'lockbox': f'{nh} ({100*nh/(sp=="Holdout").sum():.1f})', 'SMD': ''})
raw = pd.read_csv(seer_path); raw.columns = raw.columns.str.strip()
X, y, meta = load_seer(seer_path, 'fixed_60m')
keep = ~((raw.Status.str.strip().str.lower() == 'alive') & (raw['Survival Months'] < 60))
R = raw[keep].reset_index(drop=True)
Xd, Xh, yd, yh, idd, idh = dev_holdout_split(X.values, y)
split = np.array(['Dev'] * len(R), dtype=object); split[idh] = 'Holdout'
R['split'] = split; R['y5'] = y
def pct(s, v): return f'{(s == v).sum()} ({100*(s == v).mean():.1f})'
for part in [('Dev', 'development'), ('Holdout', 'lockbox')]:
    pass
cats = [('Age, years, mean (SD)', 'Age', 'num'), ('Race', 'Race', 'cat'), ('Marital status', 'Marital Status', 'cat'), ('T stage', 'T Stage', 'cat'),
        ('N stage', 'N Stage', 'cat'), ('Grade', 'Grade', 'cat'), ('A stage', 'A Stage', 'cat'), ('Tumour size, mm, median (IQR)', 'Tumor Size', 'med'),
        ('Estrogen receptor', 'Estrogen Status', 'cat'), ('Progesterone receptor', 'Progesterone Status', 'cat'),
        ('Regional nodes examined, median (IQR)', 'Regional Node Examined', 'med'), ('Regional nodes positive, median (IQR)', 'Reginol Node Positive', 'med'),
        ('Follow-up, months, median (IQR)', 'Survival Months', 'med')]
for lab, col, kind in cats:
    a, b = R[R.split == 'Dev'][col], R[R.split == 'Holdout'][col]
    if kind == 'num':
        ch.append({'cohort': 'SEER', 'characteristic': lab, 'development': f'{a.mean():.1f} ({a.std():.1f})', 'lockbox': f'{b.mean():.1f} ({b.std():.1f})', 'SMD': round(smd(a, b), 2)})
    elif kind == 'med':
        ch.append({'cohort': 'SEER', 'characteristic': lab, 'development': f'{a.median():.0f} ({a.quantile(.25):.0f}-{a.quantile(.75):.0f})',
                   'lockbox': f'{b.median():.0f} ({b.quantile(.25):.0f}-{b.quantile(.75):.0f})', 'SMD': round(smd(a, b), 2)})
    else:
        for v in sorted(R[col].dropna().unique()):
            vv = v.replace('Other (American Indian/AK Native, Asian/Pacific Islander)', 'Other (AI/AN, API)')
            ch.append({'cohort': 'SEER', 'characteristic': f'{lab}: {vv}, n (%)', 'development': pct(a, v), 'lockbox': pct(b, v), 'SMD': ''})
ch.append({'cohort': 'SEER', 'characteristic': 'Death within 60 months, n (%)', 'development': pct(R[R.split == 'Dev'].y5, 1), 'lockbox': pct(R[R.split == 'Holdout'].y5, 1), 'SMD': ''})
ch.append({'cohort': 'SEER', 'characteristic': 'Missing values, n', 'development': '0', 'lockbox': '0', 'SMD': ''})
pd.DataFrame(ch).to_csv(os.path.join(out, 'S3_cohort_characteristics.csv'), index=False)
# by race (TRIPOD 20b)
g = []
for rc, dd in R.groupby('Race'):
    g.append({'race': rc.replace('Other (American Indian/AK Native, Asian/Pacific Islander)', 'Other'), 'n': len(dd), 'age_mean': dd.Age.mean(),
              'T3_T4_pct': 100 * dd['T Stage'].isin(['T3', 'T4']).mean(), 'N3_pct': 100 * (dd['N Stage'] == 'N3').mean(),
              'ER_negative_pct': 100 * (dd['Estrogen Status'] == 'Negative').mean(), 'death60_pct': 100 * dd.y5.mean()})
pd.DataFrame(g).to_csv(os.path.join(out, 'S3b_seer_by_race.csv'), index=False)
print(pd.DataFrame(g).round(1).to_string())
st = raw.Status.str.strip().str.lower(); mo = raw['Survival Months']
flow = {'source': len(raw), 'deaths_total': int((st == 'dead').sum()), 'deaths_le60': int(((st == 'dead') & (mo <= 60)).sum()),
        'deaths_gt60': int(((st == 'dead') & (mo > 60)).sum()), 'alive_lt60_excluded': int(((st == 'alive') & (mo < 60)).sum()),
        'alive_ge60': int(((st == 'alive') & (mo >= 60)).sum()), 'age_min': int(raw.Age.min()), 'age_max': int(raw.Age.max()),
        'N0': int((raw['N Stage'] == 'N0').sum()), 'dup_rows': int(raw.drop(columns=['Unnamed: 3'], errors='ignore').duplicated().sum()),
        'followup_median': float(mo.median()), 'followup_max': int(mo.max())}
json.dump(flow, open(os.path.join(out, 'S1_participant_flow_counts.json'), 'w'), indent=2)
print(flow)
