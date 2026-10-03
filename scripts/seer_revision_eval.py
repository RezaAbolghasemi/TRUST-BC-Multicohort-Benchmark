"""Stage 2 of the SEER re-analysis (TRUST-BC v1.1): statistics, calibration, DCA,
split conformal prediction, sealed-lockbox evaluation with bootstrap CIs, fairness
(subgroup) analysis, sensitivity analyses, permutation-Shapley attributions and figures.
Usage: python scripts/seer_revision_eval.py <SEER.csv> <label_mode> <out_dir> [--skip-sensitivity]"""
import sys, os, pickle, json, time
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import logit
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, os.path.dirname(__file__))
from seer_revision_core import *
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (roc_auc_score, brier_score_loss, accuracy_score, roc_curve, auc,
                             precision_recall_curve, average_precision_score, confusion_matrix,
                             balanced_accuracy_score)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from collections import Counter

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 12})
ALPHA = 0.05
path, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
skip_sens = '--skip-sensitivity' in sys.argv
os.makedirs(out, exist_ok=True)
T0 = time.time()
def log(*a):
    print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)

# ----- helpers identical to src/ -----
def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & ((p <= bins[i + 1]) if i == n_bins - 1 else (p < bins[i + 1]))
        if m.sum() > 0:
            e += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return e
def cal_slope_int(y, p):
    lp = logit(np.clip(p, 1e-7, 1 - 1e-7))
    m = LogisticRegression(C=1e5, solver='lbfgs').fit(lp.reshape(-1, 1), y)
    return m.coef_[0][0], m.intercept_[0]
def cal_in_large(y, p):
    """Calibration-in-the-large: intercept of logistic model with logit(p) as offset."""
    lp = logit(np.clip(p, 1e-7, 1 - 1e-7)); a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + lp))); g = np.sum(y - mu); h = np.sum(mu * (1 - mu))
        a += g / h
        if abs(g / h) < 1e-10: break
    return a
def net_benefit(y, p, ths):
    n = len(y); r = []
    for pt in ths:
        pr = p >= pt; tp = np.sum(pr & (y == 1)); fp = np.sum(pr & (y == 0))
        r.append(max(tp / n - fp / n * pt / (1 - pt), 0) if pt < 1 else 0.0)
    return np.array(r)
def nb_ci(a, K=5, level=0.95):
    a = np.asarray(a); n = len(a); se = a.std(ddof=1) * np.sqrt(1 / n + 1 / (K - 1))
    t = stats.t.ppf(1 - (1 - level) / 2, n - 1); return a.mean() - t * se, a.mean() + t * se
def nb_test(a, b, K=5):
    d = np.asarray(a) - np.asarray(b); n = len(d); se = d.std(ddof=1) * np.sqrt(1 / n + 1 / (K - 1))
    return 2 * stats.t.sf(abs(d.mean() / se), n - 1)
def tost(a, b, margin, K=5):
    d = np.asarray(a) - np.asarray(b); n = len(d); se = d.std(ddof=1) * np.sqrt(1 / n + 1 / (K - 1))
    return max(stats.t.sf((d.mean() + margin) / se, n - 1), stats.t.cdf((d.mean() - margin) / se, n - 1))
def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z**2 / n; c = (p + z**2 / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h
def boot(y, p, fn, B=2000, seed=SEED):
    rng = np.random.default_rng(seed); i1 = np.where(y == 1)[0]; i0 = np.where(y == 0)[0]; v = []
    for _ in range(B):
        s = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        try: v.append(fn(y[s], p[s]))
        except Exception: pass
    return np.percentile(np.array(v), [2.5, 97.5], axis=0)

# ----- data -----
X, y, meta = load_seer(path, mode)
feat = X.columns.tolist(); Xa = X.values
Xd, Xh, yd, yh, idd, idh = dev_holdout_split(Xa, y)
nc = max(3, min(Xa.shape[1] // 2, 50))
res = pickle.load(open(os.path.join(out, f'folds_{mode}.pkl'), 'rb'))
assert len(res) == 35
summary = {'mode': mode, 'N': int(len(y)), 'events': int(y.sum()), 'dev_n': int(len(yd)), 'dev_events': int(yd.sum()),
           'holdout_n': int(len(yh)), 'holdout_events': int(yh.sum()), 'n_features': len(feat), 'n_components': nc}
log(summary)

# ----- Table 3 -----
P = ['Leaky', 'Nested_PCA', 'Nested_RF', 'Unified']
sc = {p: np.array([r[p + '_AUROC'] for r in res]) for p in P}
acc = {p: np.array([r[p + '_Accuracy'] for r in res]) for p in P}
yo = np.concatenate([r['y'] for r in res])
po = {p: np.concatenate([r[p] for r in res]) for p in P}
rows = []
for p in P:
    lo, hi = nb_ci(sc[p]); s, i = cal_slope_int(yo, po[p])
    rows.append({'Pipeline': p, 'AUROC_mean': sc[p].mean(), 'AUROC_sd': sc[p].std(ddof=1), 'AUROC_CI_low_NB': lo,
                 'AUROC_CI_high_NB': hi, 'Accuracy_mean': acc[p].mean(), 'Brier': brier_score_loss(yo, po[p]),
                 'ECE': ece(yo, po[p]), 'Cal_slope': s, 'Cal_intercept': i, 'CITL': cal_in_large(yo, po[p])})
t3 = pd.DataFrame(rows)
t3['p_vs_Leaky_unadj'] = [np.nan] + [nb_test(sc[p], sc['Leaky']) for p in P[1:]]
t3['pTOST_vs_Leaky'] = [np.nan, tost(sc['Nested_PCA'], sc['Leaky'], 0.01), np.nan, tost(sc['Unified'], sc['Leaky'], 0.01)]
t3.to_csv(os.path.join(out, f'SEER_{mode}_table3.csv'), index=False)
def dci(a, b):
    d = a - b; lo, hi = nb_ci(d); lo90, hi90 = nb_ci(d, level=0.90); return d.mean(), lo, hi, lo90, hi90
diffs = {k: dci(sc[a], sc[b]) for k, (a, b) in {'Unified-Leaky': ('Unified', 'Leaky'), 'NestedPCA-Leaky': ('Nested_PCA', 'Leaky'),
                                                  'Unified-NestedPCA': ('Unified', 'Nested_PCA')}.items()}
pd.DataFrame([{'contrast': k, 'mean_diff': v[0], 'CI95_low': v[1], 'CI95_high': v[2], 'CI90_low': v[3], 'CI90_high': v[4]}
              for k, v in diffs.items()]).to_csv(os.path.join(out, f'SEER_{mode}_contrasts.csv'), index=False)
arch = Counter(r['arch'] for r in res); summary['arch'] = dict(arch)
log(t3.round(4).to_string()); log(diffs)

# ----- DCA on pooled OOF -----
ths = np.linspace(0.01, 0.99, 100)
nbm = net_benefit(yo, po['Unified'], ths); prev = yo.mean()
nba = np.array([max(prev - (1 - prev) * pt / (1 - pt), 0) for pt in ths])
dca = pd.DataFrame({'threshold': ths, 'nb_model': nbm, 'nb_treat_all': nba}); dca.to_csv(os.path.join(out, f'SEER_{mode}_dca.csv'), index=False)
sel = {}
for pt in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    sel[pt] = (float(net_benefit(yo, po['Unified'], [pt])[0]), float(max(prev - (1 - prev) * pt / (1 - pt), 0)))
summary['dca_points'] = sel
better = ths[(nbm > nba + 5e-4) & (nbm > 5e-4) & (ths < 0.5)]; summary['dca_superior_range'] = [float(better.min()), float(better.max())] if len(better) else None
log('DCA', sel, summary['dca_superior_range'])

# ----- fairness on pooled OOF rows (screening only; the reported per-patient subgroup
# analysis with patient-level bootstrap CIs is in tripod_supplementary_analysis.py) -----
oof_idx = np.concatenate([idd[r['test_idx']] for r in res])
m = meta.iloc[oof_idx].reset_index(drop=True)
m['age_grp'] = pd.cut(m['age'], [0, 49, 59, 200], labels=['<50', '50-59', '>=60'])
m['race_grp'] = m['race'].replace({'Other (American Indian/AK Native, Asian/Pacific Islander)': 'Other'})
fair = []
pu = po['Unified']
for col in ['race_grp', 'age_grp']:
    for g in [None] + list(m[col].dropna().unique()):
        mk = np.ones(len(m), bool) if g is None else (m[col] == g).values
        if g is None and col == 'age_grp': continue
        yy, pp = yo[mk], pu[mk]
        a = roc_auc_score(yy, pp); a_lo, a_hi = boot(yy, pp, roc_auc_score, B=1000)
        citl = cal_in_large(yy, pp); c_lo, c_hi = boot(yy, pp, cal_in_large, B=1000)
        sl, _ = cal_slope_int(yy, pp)
        pr = (pp >= 0.5).astype(int); tn, fp, fn, tp = confusion_matrix(yy, pr, labels=[0, 1]).ravel()
        fair.append({'variable': 'Overall' if g is None else col.replace('_grp', ''), 'group': 'All' if g is None else str(g),
                     'n': int(mk.sum()) // 7, 'n_oof_rows': int(mk.sum()), 'events_per_repeat': int(yy.sum()) // 7,
                     'observed_risk': yy.mean(), 'mean_predicted': pp.mean(), 'AUROC': a, 'AUROC_low': a_lo, 'AUROC_high': a_hi,
                     'CITL': citl, 'CITL_low': c_lo, 'CITL_high': c_hi, 'Cal_slope': sl,
                     'Sensitivity': tp / (tp + fn) if tp + fn else np.nan, 'Specificity': tn / (tn + fp) if tn + fp else np.nan})
fair = pd.DataFrame(fair); fair.to_csv(os.path.join(out, f'SEER_{mode}_fairness_oofrows_screening.csv'), index=False)
log(fair.round(3).to_string())

# ----- model selection on full dev, winner's curse, conformal -----
seed_arch = [GridSearchCV(unified_search(nc).estimator, unified_search(nc).param_grid,
                          cv=StratifiedKFold(3, shuffle=True, random_state=s), scoring='roc_auc', n_jobs=1).fit(Xd, yd).best_params_['feature_opt__method']
             for s in range(5)]
summary['winners_curse_seeds'] = dict(Counter(seed_arch))
gs = unified_search(nc).fit(Xd, yd); best = gs.best_estimator_; summary['best_params'] = {k: str(v) for k, v in gs.best_params_.items()}
log('best', summary['best_params'], summary['winners_curse_seeds'])
E, S, F, COV, ncal, ncal_ev = [], [], [], [], [], []
cov_rows = []
for rep in range(100):
    i_all = np.arange(len(yd))
    Xtc, Xte, ytc, yte, itc, ite = train_test_split(Xd, yd, i_all, test_size=0.2, random_state=rep, stratify=yd)
    Xtr, Xca, ytr, yca = train_test_split(Xtc, ytc, test_size=0.25, random_state=rep, stratify=ytc)
    best.fit(Xtr, ytr)
    cp = best.predict_proba(Xca)[np.arange(len(yca)), yca]
    q = np.quantile(1 - cp, np.ceil((len(yca) + 1) * (1 - ALPHA)) / len(yca))
    ps = (1 - best.predict_proba(Xte)) <= q
    covered = ps[np.arange(len(yte)), yte]; sz = ps.sum(1)
    COV.append(covered.mean()); E.append((sz == 0).sum()); S.append((sz == 1).sum()); F.append((sz == 2).sum())
    ncal.append(len(yca)); ncal_ev.append(int(yca.sum()))
    cov_rows.append(pd.DataFrame({'rep': rep, 'orig_index': idd[ite], 'y': yte, 'covered': covered.astype(int), 'size': sz}))
ntest = len(yte)
conf = {'Empty_pct': 100 * np.mean(E) / ntest, 'Singleton_pct': 100 * np.mean(S) / ntest, 'Ambiguous_pct': 100 * np.mean(F) / ntest,
        'Coverage_pct_mean': 100 * np.mean(COV), 'Coverage_pct_sd': 100 * np.std(COV), 'n_test': ntest,
        'n_cal': int(np.mean(ncal)), 'n_cal_events': float(np.mean(ncal_ev)), 'n_proper_train': int(len(ytr)),
        'E_mean': np.mean(E), 'S_mean': np.mean(S), 'F_mean': np.mean(F), 'E_sd': np.std(E), 'S_sd': np.std(S), 'F_sd': np.std(F)}
summary['conformal'] = conf; log('conformal', conf)
cr = pd.concat(cov_rows).merge(meta.reset_index().rename(columns={'index': 'orig_index'}), on='orig_index')
cr['race_grp'] = cr['race'].replace({'Other (American Indian/AK Native, Asian/Pacific Islander)': 'Other'})
cr['age_grp'] = pd.cut(cr['age'], [0, 49, 59, 200], labels=['<50', '50-59', '>=60'])
cg = []
for col in ['race_grp', 'age_grp']:
    for g, d in cr.groupby(col, observed=True):
        cg.append({'variable': col.replace('_grp', ''), 'group': str(g), 'coverage_pct': 100 * d.covered.mean(),
                   'coverage_events_pct': 100 * d[d.y == 1].covered.mean(), 'coverage_nonevents_pct': 100 * d[d.y == 0].covered.mean(),
                   'ambiguous_pct': 100 * (d['size'] == 2).mean()})
cg.append({'variable': 'outcome', 'group': 'event', 'coverage_pct': 100 * cr[cr.y == 1].covered.mean()})
cg.append({'variable': 'outcome', 'group': 'non-event', 'coverage_pct': 100 * cr[cr.y == 0].covered.mean()})
pd.DataFrame(cg).to_csv(os.path.join(out, f'SEER_{mode}_conformal_subgroups.csv'), index=False)
log(pd.DataFrame(cg).round(2).to_string())

# ----- lockbox -----
Xft, Xfc, yft, yfc = train_test_split(Xd, yd, test_size=0.2, stratify=yd, random_state=SEED)
best.fit(Xft, yft)
ph_full = best.predict_proba(Xh); ph = ph_full[:, 1]; pred = best.predict(Xh)
tn, fp, fn, tp = confusion_matrix(yh, pred).ravel()
cal_true = best.predict_proba(Xfc)[np.arange(len(yfc)), yfc]
qh = np.quantile(1 - cal_true, np.ceil((len(yfc) + 1) * (1 - ALPHA)) / len(yfc))
hs = (1 - ph_full) <= qh; hcov = hs[np.arange(len(yh)), yh]
sl, it = cal_slope_int(yh, ph)
lb = {'N': len(yh), 'events': int(yh.sum()), 'Accuracy': accuracy_score(yh, pred), 'Accuracy_CI': wilson(tp + tn, len(yh)),
      'Balanced_accuracy': balanced_accuracy_score(yh, pred), 'Sensitivity': tp / (tp + fn), 'Specificity': tn / (tn + fp),
      'TP': int(tp), 'FN': int(fn), 'TN': int(tn), 'FP': int(fp),
      'AUROC': roc_auc_score(yh, ph), 'AUROC_CI': boot(yh, ph, roc_auc_score),
      'Brier': brier_score_loss(yh, ph), 'Brier_CI': boot(yh, ph, brier_score_loss),
      'Cal_slope': sl, 'Cal_intercept': it, 'CITL': cal_in_large(yh, ph),
      'Slope_Int_CI': boot(yh, ph, lambda a, b: np.array(cal_slope_int(a, b))),
      'CITL_CI': boot(yh, ph, cal_in_large),
      'Conformal_coverage': hcov.mean(), 'Conformal_CI': wilson(hcov.sum(), len(yh)),
      'Ambiguous_pct': 100 * (hs.sum(1) == 2).mean(), 'n_cal_lockbox': len(yfc), 'n_cal_lockbox_events': int(yfc.sum()),
      'n_train_lockbox': len(yft), 'n_train_lockbox_events': int(yft.sum())}
summary['lockbox'] = {k: (np.asarray(v).tolist() if isinstance(v, (tuple, np.ndarray)) else (float(v) if not isinstance(v, int) else v)) for k, v in lb.items()}
pd.DataFrame({'orig_index': idh, 'y': yh, 'p': ph, 'pred': pred, 'covered': hcov.astype(int), 'set_size': hs.sum(1)}).to_csv(
    os.path.join(out, f'SEER_{mode}_lockbox_predictions.csv'), index=False)
log('lockbox', summary['lockbox'])

# ----- permutation Shapley on holdout (same subsample as v1.0 Kernel SHAP) -----
rng = np.random.RandomState(SEED)
bg = resample(Xft, n_samples=50, random_state=SEED, replace=False)
ex_idx = rng.choice(len(Xh), size=100, replace=False); Xe = Xh[ex_idx]
M = 40; nf = Xe.shape[1]; phi = np.zeros((len(Xe), nf)); prs = np.random.default_rng(SEED)
f = lambda Z: best.predict_proba(Z)[:, 1]
for i, x in enumerate(Xe):
    for _ in range(M):
        perm = prs.permutation(nf); Z = np.repeat(bg[None, :, :], nf + 1, axis=0).copy()
        for k in range(nf):
            Z[k + 1:, :, perm[k]] = x[perm[k]]
        v = f(Z.reshape(-1, nf)).reshape(nf + 1, len(bg)).mean(1)
        phi[i, perm] += np.diff(v)
phi /= M
np.save(os.path.join(out, f'SEER_{mode}_shapley_values.npy'), phi)
imp = np.abs(phi).mean(0); order = np.argsort(imp)[::-1][:15]
fig, ax = plt.subplots(figsize=(9, 7))
for r, j in enumerate(order[::-1]):
    v = Xe[:, j]; vn = (v - v.min()) / (v.max() - v.min() + 1e-12)
    ax.scatter(phi[:, j], r + np.random.default_rng(j).uniform(-0.25, 0.25, len(v)), c=vn, cmap='coolwarm', s=14, vmin=0, vmax=1)
ax.set_yticks(range(len(order))); ax.set_yticklabels([feat[j] for j in order[::-1]], fontsize=10)
ax.axvline(0, color='gray', lw=1); ax.set_xlabel('Shapley value (impact on predicted probability)')
sm = plt.cm.ScalarMappable(cmap='coolwarm'); cb = plt.colorbar(sm, ax=ax, ticks=[0, 1]); cb.ax.set_yticklabels(['Low', 'High']); cb.set_label('Feature value')
ax.set_title(f'Permutation Shapley — SEER 5-year (holdout, n=100)', fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(out, f'SEER_{mode}_fig_shapley_summary.png'), dpi=300, bbox_inches='tight'); plt.close(fig)
summary['shapley_top'] = [feat[j] for j in order[:6]]
log('shapley top', summary['shapley_top'])

# ----- figures: main and clinical -----
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
sns.boxplot(data=[sc[p] for p in P], ax=axes[0], palette=['#E74C3C', '#3498DB', '#27AE60', '#9B59B6'])
axes[0].set_xticks(range(4)); axes[0].set_xticklabels([f'{n}\n({sc[p].mean():.3f})' for n, p in zip(['Leaky', 'PCA', 'RF', 'Unified'], P)], fontsize=10)
axes[0].set_ylabel('AUROC', fontweight='bold'); axes[0].set_title('A. Benchmarking Framework (SEER, 5-year)', fontweight='bold', pad=15)
sns.barplot(x=['Empty', 'Certain (Size 1)', 'Uncertain (Size 2)'], y=[np.mean(E), np.mean(S), np.mean(F)], ax=axes[1], palette='Blues')
axes[1].errorbar(x=[0, 1, 2], y=[np.mean(E), np.mean(S), np.mean(F)], yerr=[np.std(E), np.std(S), np.std(F)], fmt='none', c='black', capsize=8, elinewidth=2)
axes[1].set_ylabel('Mean number of patients', fontweight='bold'); axes[1].set_title('B. Conformal Prediction Sets (Dev)', fontweight='bold', pad=15)
axes[2].plot(ths, nbm, label='Unified model (CV probs)', color='blue', linewidth=3)
axes[2].plot(ths, nba, label='Treat all', color='gray', linestyle='--', linewidth=2)
axes[2].plot(ths, np.zeros_like(ths), label='Treat none', color='black', linewidth=2)
axes[2].set_xlim([0, 1]); axes[2].set_ylim([0, max(nbm.max(), nba.max()) + 0.1])
axes[2].set_xlabel('Threshold probability', fontweight='bold'); axes[2].set_ylabel('Net benefit', fontweight='bold')
axes[2].set_title('C. Decision Curve Analysis (Pooled CV)', fontweight='bold', pad=15); axes[2].legend()
plt.tight_layout(pad=3.0); plt.savefig(os.path.join(out, f'SEER_{mode}_fig_main.png'), dpi=300, bbox_inches='tight'); plt.close(fig)

fig2, a2 = plt.subplots(2, 2, figsize=(18, 14))
fpr, tpr, _ = roc_curve(yo, pu)
a2[0, 0].plot(fpr, tpr, color='darkorange', lw=3, label=f'Unified (AUC = {auc(fpr, tpr):.3f})'); a2[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, ls='--')
a2[0, 0].set_title('A. Cross-validated ROC (Pooled OOF)', fontweight='bold'); a2[0, 0].legend(loc='lower right')
pr_, rc_, _ = precision_recall_curve(yo, pu)
a2[0, 1].plot(rc_, pr_, color='purple', lw=3, label=f'Unified (AP = {average_precision_score(yo, pu):.3f})')
a2[0, 1].set_xlim([0, 1]); a2[0, 1].set_ylim([0, 1.05]); a2[0, 1].set_title('B. Cross-validated Precision-Recall', fontweight='bold'); a2[0, 1].legend(loc='upper right')
pt_, pp_ = calibration_curve(yo, pu, n_bins=10, strategy='uniform')
a2[1, 0].plot(pp_, pt_, marker='s', markersize=8, color='green', lw=2, label=f'ECE: {ece(yo, pu):.3f}'); a2[1, 0].plot([0, 1], [0, 1], ls='--', color='gray')
a2[1, 0].set_title('C. Calibration Curve (Reliability Diagram)', fontweight='bold'); a2[1, 0].legend(loc='upper left')
sup = [r.get('rf_support') for r in res if r['arch'] == 'rf' and r.get('rf_support') is not None]
if sup:
    cnt = np.sum(sup, 0); top = np.argsort(cnt)[::-1][:10]
    sns.barplot(x=cnt[top], y=np.array(feat)[top], ax=a2[1, 1], palette='viridis')
    a2[1, 1].set_xlabel(f'Selection frequency (max = {len(sup)})', fontweight='bold')
    a2[1, 1].set_title('D. Feature Selection Stability\n(Analyzed across RF choices)', fontweight='bold')
    summary['rf_selection_counts'] = {feat[j]: int(cnt[j]) for j in top}
plt.tight_layout(pad=4.0); plt.savefig(os.path.join(out, f'SEER_{mode}_fig_clinical.png'), dpi=300, bbox_inches='tight'); plt.close(fig2)
summary['pooled_oof_auc'] = float(auc(fpr, tpr)); summary['pooled_ap'] = float(average_precision_score(yo, pu))
json.dump(summary, open(os.path.join(out, f'SEER_{mode}_summary.json'), 'w'), indent=2, default=float)
log('figures done')

if not skip_sens:
    # n_components sensitivity (identical protocol to src/)
    ns = sorted(set([max(2, nc // 2), nc, min(Xa.shape[1], int(nc * 1.5)), min(Xa.shape[1], nc * 2)]))
    skf = StratifiedKFold(3, shuffle=True, random_state=SEED); rr = []
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    for n_ in ns:
        for meth in ['pca', 'rf']:
            pipe = Pipeline([('scaler', StandardScaler()), ('feature_opt', FeatureMethodSwitcher(method=meth, n_components=n_, random_state=SEED)), ('clf', base_clf())])
            s_ = [roc_auc_score(yd[te], pipe.fit(Xd[tr], yd[tr]).predict_proba(Xd[te])[:, 1]) for tr, te in skf.split(Xd, yd)]
            rr.append({'n_components': n_, 'method': meth.upper(), 'mean_score': np.mean(s_)})
    dn = pd.DataFrame(rr); dn.to_csv(os.path.join(out, f'SEER_{mode}_n_components_sensitivity.csv'), index=False)
    plt.figure(figsize=(7, 5)); sns.lineplot(data=dn, x='n_components', y='mean_score', hue='method', marker='o', linewidth=2)
    plt.title('Sensitivity to n_components (SEER, 5-year)', fontweight='bold'); plt.xlabel('Number of Components', fontweight='bold'); plt.ylabel('Mean 3-Fold ROC_AUC', fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(out, f'SEER_{mode}_fig_n_components_sensitivity.png'), dpi=300, bbox_inches='tight'); plt.close()
    log('ncomp', dn.round(4).to_string())
    # sample-size sensitivity (identical protocol; B=200)
    rows = []
    for frac in (0.30, 0.50, 0.70, 0.85, 1.00):
        n_sub = min(max(25, int(round(len(Xd) * frac))), len(Xd)); accs = []
        for b in range(200):
            rs = SEED + b
            ix = resample(np.arange(len(Xd)), n_samples=n_sub, stratify=yd, random_state=rs, replace=False)
            Xs, ys = Xd[ix], yd[ix]
            Xt1, Xt2, yt1, yt2 = train_test_split(Xs, ys, test_size=0.30, stratify=ys, random_state=rs)
            accs.append(accuracy_score(yt2, best.fit(Xt1, yt1).predict(Xt2)))
        accs = np.array(accs); lo, hi = np.percentile(accs, [2.5, 97.5])
        rows.append({'fraction': frac, 'n_samples': n_sub, 'mean_acc': accs.mean(), 'std_acc': accs.std(ddof=1), 'ci_low': lo, 'ci_high': hi, 'ci_width': hi - lo})
        log('frac', frac, accs.mean())
    ds = pd.DataFrame(rows); ds.to_csv(os.path.join(out, f'SEER_{mode}_sample_size_sensitivity.csv'), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].plot(ds.n_samples, ds.mean_acc, marker='o', color='#3498DB', lw=2); ax[0].fill_between(ds.n_samples, ds.ci_low, ds.ci_high, alpha=.25, color='#3498DB')
    ax[0].set_xlabel('Training sample size', fontweight='bold'); ax[0].set_ylabel('Bootstrap accuracy (mean ± 95% CI)', fontweight='bold'); ax[0].set_title('Learning Curve (SEER, 5-year)', fontweight='bold')
    ax[1].plot(ds.n_samples, ds.ci_width, marker='s', color='#E74C3C', lw=2); ax[1].set_xlabel('Training sample size', fontweight='bold')
    ax[1].set_ylabel('95% CI width (estimation uncertainty)', fontweight='bold'); ax[1].set_title('Estimate Stability vs. Sample Size', fontweight='bold')
    plt.tight_layout(pad=2.5); plt.savefig(os.path.join(out, f'SEER_{mode}_fig_sample_size_sensitivity.png'), dpi=300, bbox_inches='tight'); plt.close(fig)
log('ALL DONE')
