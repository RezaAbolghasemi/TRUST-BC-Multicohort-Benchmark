"""
Print-size redraw of the TRUST-BC figures in the ORIGINAL v1.0 visual style
(seaborn 'whitegrid', original palettes, panel titles, line widths and SHAP-style beeswarm),
with fonts sized for a 16 cm text width so that no down-scaling occurs in the manuscript.

Data: results/<cohort>/*.csv (v1.0), results/SEER_v1.1/* (corrected SEER),
results/WBCD/revision_v1.1/* (re-created WBCD out-of-fold predictions and conformal
iterations, identical to v1.0) and results/Coimbra/revision_v1.1/* (re-created Coimbra
out-of-fold, conformal and lockbox predictions, identical to v1.0).
Usage: python scripts/make_publication_figures.py . <SEER.csv> results/WBCD/revision_v1.1 results/figures_v1.1 [folds_fixed_60m.pkl] [results/Coimbra/revision_v1.1] [dataR2.csv]
"""
import sys, os, json, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import imread
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(__file__))
from seer_revision_core import load_seer, dev_holdout_split, SEED

root, seer_csv, rev_wbcd, out = sys.argv[1:5]
pkl = sys.argv[5] if len(sys.argv) > 5 else os.path.join(root, 'results', 'SEER_v1.1', 'folds_fixed_60m.pkl')
rev_coim = sys.argv[6] if len(sys.argv) > 6 else os.path.join(root, 'results', 'Coimbra', 'revision_v1.1')
coim_csv = sys.argv[7] if len(sys.argv) > 7 else 'dataR2.csv'
os.makedirs(out, exist_ok=True)
R = lambda *p: os.path.join(root, 'results', *p)
W = 6.3  # inches = 16 cm text width

# ---- original v1.0 style (seaborn whitegrid, same palettes), print-size fonts ----
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 6.8,
                     'ytick.labelsize': 6.8, 'legend.fontsize': 6.8, 'savefig.dpi': 600, 'lines.linewidth': 1.4,
                     'axes.titleweight': 'bold', 'axes.labelweight': 'bold'})
BOX_PAL = ['#E74C3C', '#F39C12', '#F1C40F', '#3498DB', '#27AE60', '#9B59B6']
SHAP_CMAP = LinearSegmentedColormap.from_list('shap_red_blue', ['#008bfb', '#7d3fd0', '#ff0052'])

def dca(y, p, ths):
    n = len(y); prev = y.mean()
    nb = [max(((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t), 0) for t in ths]
    return np.array(nb), np.array([max(prev - (1 - prev) * t / (1 - t), 0) for t in ths])

def ece(y, p, n=10):
    b = np.linspace(0, 1, n + 1); e = 0
    for i in range(n):
        m = (p >= b[i]) & ((p <= b[i + 1]) if i == n - 1 else (p < b[i + 1]))
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e

def crop(path, box):
    im = imread(path); h, w = im.shape[:2]
    return im[int(box[2] * h):int(box[3] * h), int(box[0] * w):int(box[1] * w)]

# ---- data ----
raw = {c: pd.read_csv(R(c, f'{c}_raw_cv_scores.csv')) for c in ['WBCD', 'Coimbra']}
seer_fx = pd.read_csv(R('SEER_v1.1', 'SEER_fixed_60m_raw_cv_scores.csv'))
cs_c = pd.read_csv(R('Coimbra', 'Coimbra_conformal_summary.csv')).iloc[0]
cw = pd.read_csv(os.path.join(rev_wbcd, 'WBCD_conformal_iterations.csv'))
cc = pd.read_csv(os.path.join(rev_coim, 'Coimbra_conformal_iterations.csv'))
co = pd.read_csv(os.path.join(rev_coim, 'Coimbra_oof_predictions.csv'))
sj = json.load(open(R('SEER_v1.1', 'SEER_fixed_60m_summary.json')))['conformal']
wo = pd.read_csv(os.path.join(rev_wbcd, 'WBCD_oof_predictions.csv'))
so = pd.read_csv(R('SEER_v1.1', 'SEER_fixed_60m_oof_predictions.csv'))
oof = {'WBCD': (wo.y.values, wo.Unified.values), 'Coimbra': (co.y.values, co.Unified.values), 'SEER': (so.y.values, so.Unified.values)}
ths = np.linspace(0.01, 0.99, 100)

# ================= Figure 2 =================
fig, axes = plt.subplots(3, 3, figsize=(W, 6.6), gridspec_kw={'width_ratios': [1.5, 0.9, 1.0]})
spec = [('WBCD', raw['WBCD'], ['Leaky', 'XGBoost', 'LightGBM', 'Nested_PCA', 'Nested_RF', 'Unified'], 'Accuracy', BOX_PAL),
        ('Coimbra', raw['Coimbra'], ['Leaky', 'XGBoost', 'LightGBM', 'Nested_PCA', 'Nested_RF', 'Unified'], 'Accuracy', BOX_PAL),
        ('SEER 5-year', seer_fx, ['Leaky', 'Nested_PCA', 'Nested_RF', 'Unified'], 'AUROC', [BOX_PAL[0], BOX_PAL[3], BOX_PAL[4], BOX_PAL[5]])]
LAB = {'Leaky': 'Leaky', 'XGBoost': 'XGBoost', 'LightGBM': 'LightGBM', 'Nested_PCA': 'PCA', 'Nested_RF': 'RF', 'Unified': 'Unified'}
for r, (name, df, pipes, met, pal) in enumerate(spec):
    ax = axes[r, 0]
    data = [df[f'{p}_{met}'].values for p in pipes]
    sns.boxplot(data=data, ax=ax, palette=pal, linewidth=0.8, fliersize=2)
    lab = [(f'{np.mean(d):.3f}' if met == 'AUROC' else f'{100*np.mean(d):.1f}%') for d in data]
    ax.set_xticks(range(len(pipes))); ax.set_xticklabels([f'{LAB[p]}\n({l})' for p, l in zip(pipes, lab)], fontsize=5.6)
    ax.set_ylabel(met); ax.set_title(f'A. Benchmarking Framework\n({name})', pad=5)
    # B conformal (mean number of patients, SD error bars)
    ax = axes[r, 1]
    if name == 'WBCD':
        m = [cw['empty'].mean(), cw['singleton'].mean(), cw['ambiguous'].mean()]; sd = [cw['empty'].std(ddof=0), cw['singleton'].std(ddof=0), cw['ambiguous'].std(ddof=0)]
    elif name == 'Coimbra':
        m = [cc['empty'].mean(), cc['singleton'].mean(), cc['ambiguous'].mean()]; sd = [cc['empty'].std(ddof=0), cc['singleton'].std(ddof=0), cc['ambiguous'].std(ddof=0)]
    else:
        m = [sj['E_mean'], sj['S_mean'], sj['F_mean']]; sd = [sj['E_sd'], sj['S_sd'], sj['F_sd']]
    sns.barplot(x=['Empty', 'Certain\n(Size 1)', 'Uncertain\n(Size 2)'], y=m, ax=ax, palette='Blues')
    if sd is not None:
        ax.errorbar(x=[0, 1, 2], y=m, yerr=sd, fmt='none', c='black', capsize=4, elinewidth=1.2)
    ax.set_ylabel('Mean number of patients'); ax.set_title('B. Conformal Prediction\nSets (Dev)', pad=5)
    # C DCA
    ax = axes[r, 2]
    key = {'WBCD': 'WBCD', 'Coimbra': 'Coimbra', 'SEER 5-year': 'SEER'}.get(name)
    if key:
        y, p = oof[key]; nbm, nba = dca(y, p, ths)
        ax.plot(ths, nbm, label='Unified model (CV probs)', color='blue', linewidth=1.8)
        ax.plot(ths, nba, label='Treat all', color='gray', linestyle='--', linewidth=1.2)
        ax.plot(ths, np.zeros_like(ths), label='Treat none', color='black', linewidth=1.2)
        ax.set_xlim([0, 1.0]); ax.set_ylim([0, max(nbm.max(), nba.max()) + 0.1 * max(nbm.max(), nba.max()) + 0.02])
        ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit'); ax.legend(loc='center right' if key == 'WBCD' else 'upper right', frameon=True, fontsize=6.2)
        ax.set_title('C. Decision Curve Analysis\n(Pooled CV)', pad=5)
plt.tight_layout(h_pad=1.3, w_pad=0.8)
plt.savefig(os.path.join(out, 'fig2_benchmark.png'), bbox_inches='tight'); plt.close(fig)

# ================= Figure 3 =================
dW = load_breast_cancer(); wnames = list(dW.feature_names)
Xs, ys, meta = load_seer(seer_csv, 'fixed_60m'); snames = Xs.columns.tolist()
wres = pickle.load(open(os.path.join(rev_wbcd, 'wbcd_oof.pkl'), 'rb'))
sres = pickle.load(open(pkl, 'rb'))
cres = pickle.load(open(os.path.join(rev_coim, 'coimbra_oof.pkl'), 'rb'))
cdf = pd.read_csv(coim_csv); cnames = [c for c in cdf.columns if c != 'Classification']
sup = {'WBCD': ([r['support'] for r in wres if r['support'] is not None], wnames),
       'Coimbra': ([r['support'] for r in cres if r['support'] is not None], cnames),
       'SEER': ([r['rf_support'] for r in sres if r.get('rf_support') is not None], snames)}
def clinical(name, key, fname):
    fig, a = plt.subplots(2, 2, figsize=(W, 4.45))
    y, p = oof[key]
    f, t, _ = roc_curve(y, p)
    a[0, 0].plot(f, t, color='darkorange', lw=1.8, label=f'Unified (AUC = {auc(f, t):.3f})')
    a[0, 0].plot([0, 1], [0, 1], color='navy', lw=1.2, linestyle='--')
    a[0, 0].set_xlim([0, 1]); a[0, 0].set_ylim([0, 1.05]); a[0, 0].set_title('A. Cross-validated ROC (Pooled OOF)'); a[0, 0].legend(loc='lower right', frameon=True)
    a[0, 0].set_xlabel('False positive rate'); a[0, 0].set_ylabel('True positive rate')
    pr, rc, _ = precision_recall_curve(y, p)
    a[0, 1].plot(rc, pr, color='purple', lw=1.8, label=f'Unified (AP = {average_precision_score(y, p):.3f})')
    a[0, 1].set_xlim([0, 1]); a[0, 1].set_ylim([0, 1.05]); a[0, 1].set_title('B. Cross-validated Precision-Recall'); a[0, 1].legend(loc='lower left', frameon=True)
    a[0, 1].set_xlabel('Recall'); a[0, 1].set_ylabel('Precision')
    pt, pp = calibration_curve(y, p, n_bins=10, strategy='uniform')
    a[1, 0].plot(pp, pt, marker='s', markersize=4, color='green', linewidth=1.4, label=f'ECE: {ece(y, p):.3f}')
    a[1, 0].plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    a[1, 0].set_title('C. Calibration Curve (Reliability Diagram)'); a[1, 0].legend(loc='upper left', frameon=True)
    a[1, 0].set_xlabel('Mean predicted probability'); a[1, 0].set_ylabel('Fraction of positives')
    s, names = sup[key]
    if len(s):
        cnt = np.sum(s, 0); top = [j for j in np.argsort(cnt)[::-1][:10] if cnt[j] > 0]
        sns.barplot(x=cnt[top], y=np.array(names)[top], ax=a[1, 1], palette='viridis')
        a[1, 1].set_xlabel(f'Selection frequency (max = {len(s)})'); a[1, 1].tick_params(axis='y', labelsize=6.3)
        a[1, 1].set_title('D. Feature Selection Stability\n(Analyzed across RF choices)')
    fig.suptitle(name, fontsize=9.5, fontweight='bold')
    plt.tight_layout(h_pad=1.2); plt.savefig(os.path.join(out, fname), bbox_inches='tight'); plt.close(fig)
clinical('WBCD', 'WBCD', 'fig3a_clinical_wbcd.png')
clinical('Coimbra', 'Coimbra', 'fig3b_clinical_coimbra.png')
clinical('SEER (5-year, corrected label)', 'SEER', 'fig3c_clinical_seer.png')

# ================= Figure 4: SHAP-style beeswarm (shap.summary_plot layout) =================
def shap_beeswarm(ax, phi, X, names, title, xlabel, max_display=15, row_height=0.4):
    imp = np.abs(phi).mean(0); order = [j for j in np.argsort(imp)[::-1][:max_display] if imp[j] > 1e-12][::-1]
    ax.axvline(0, color='#999999', zorder=-1, lw=0.8)
    for pos, j in enumerate(order):
        ax.axhline(pos, color='#cccccc', lw=0.5, dashes=(1, 5), zorder=-1)
        sh = phi[:, j]; v = X[:, j].astype(float); N = len(sh)
        rs = np.random.RandomState(0); inds = np.arange(N); rs.shuffle(inds); sh_, v_ = sh[inds], v[inds]
        nbins = 100
        quant = np.round(nbins * (sh_ - sh_.min()) / (sh_.max() - sh_.min() + 1e-8))
        order_q = np.argsort(quant + rs.randn(N) * 1e-6)
        layer, last = 0, -1; yy = np.zeros(N)
        for i in order_q:
            if quant[i] != last: layer = 0
            yy[i] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1); layer += 1; last = quant[i]
        yy *= 0.9 * (row_height / np.max(yy + 1))
        vmin, vmax = np.nanpercentile(v_, 5), np.nanpercentile(v_, 95)
        if vmin == vmax: vmin, vmax = np.nanmin(v_), np.nanmax(v_)
        ax.scatter(sh_, pos + yy, c=np.clip(v_, vmin, vmax), cmap=SHAP_CMAP, vmin=vmin, vmax=vmax, s=6, lw=0, alpha=1, rasterized=True, zorder=3)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[j] for j in order], fontsize=6.8)
    ax.set_ylim(-1, len(order)); ax.grid(False)
    for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y', length=0); ax.set_xlabel(xlabel, fontweight='normal'); ax.set_title(title, pad=6)
Xw, yw = dW.data, 1 - dW.target
_, Xwh, _, _ = train_test_split(Xw, yw, test_size=0.15, stratify=yw, random_state=SEED)
phiW = np.load(R('WBCD', 'WBCD_shap_values.npy'))
Xsd, Xsh, ysd, ysh, _, _ = dev_holdout_split(Xs.values, ys)
ex = np.random.RandomState(SEED).choice(len(Xsh), size=100, replace=False)
phiS = np.load(R('SEER_v1.1', 'SEER_fixed_60m_shapley_values.npy'))
fig = plt.figure(figsize=(W, 8.6))
gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.025], height_ratios=[1.35, 0.8, 0.75], hspace=0.42, wspace=0.03)
a1 = fig.add_subplot(gs[0, 0]); shap_beeswarm(a1, phiW, Xwh, wnames, 'SHAP Summary — WBCD (True Holdout, n=86)', 'SHAP value (impact on model output)')
yc = np.where(cdf['Classification'] == 2, 1, 0); Xc = cdf[cnames].values
_, Xch, _, _ = train_test_split(Xc, yc, test_size=0.15, stratify=yc, random_state=SEED)
phiC = np.load(R('Coimbra', 'Coimbra_shap_values.npy'))
a2 = fig.add_subplot(gs[1, 0]); shap_beeswarm(a2, phiC, Xch, cnames, 'SHAP Summary — Coimbra (True Holdout, n=18)', 'SHAP value (impact on model output)')
a3 = fig.add_subplot(gs[2, 0]); shap_beeswarm(a3, phiS, Xsh[ex], snames, 'Shapley Summary — SEER 5-year (True Holdout, n=100)', 'Shapley value (impact on model output)')
for g in (gs[0, 1], gs[1, 1], gs[2, 1]):
    cax = fig.add_subplot(g); cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SHAP_CMAP), cax=cax, ticks=[0, 1])
    cb.ax.set_yticklabels(['Low', 'High'], fontsize=6.5); cb.set_label('Feature value', fontsize=6.8, labelpad=-4); cb.outline.set_visible(False)
plt.savefig(os.path.join(out, 'fig4_shapley.png'), bbox_inches='tight'); plt.close(fig)

# ================= Figure 5 =================
ss = [('WBCD', pd.read_csv(R('WBCD', 'WBCD_sample_size_sensitivity.csv'))), ('Coimbra', pd.read_csv(R('Coimbra', 'Coimbra_sample_size_sensitivity.csv'))),
      ('SEER 5-year', pd.read_csv(R('SEER_v1.1', 'SEER_fixed_60m_sample_size_sensitivity.csv')))]
fig, axes = plt.subplots(3, 2, figsize=(W, 6.9))
for r, (name, d) in enumerate(ss):
    a, b = axes[r]
    a.plot(d.n_samples, d.mean_acc, marker='o', markersize=4, color='#3498DB', linewidth=1.6)
    a.fill_between(d.n_samples, d.ci_low, d.ci_high, alpha=0.25, color='#3498DB')
    a.set_xlabel('Training sample size'); a.set_ylabel('Bootstrap accuracy (mean ± 95% CI)'); a.set_title(f'Learning Curve ({name})')
    b.plot(d.n_samples, d.ci_width, marker='s', markersize=4, color='#E74C3C', linewidth=1.6)
    b.set_xlabel('Training sample size'); b.set_ylabel('95% CI width (estimation uncertainty)'); b.set_title('Estimate Stability vs. Sample Size')
plt.tight_layout(h_pad=1.3, w_pad=1.5); plt.savefig(os.path.join(out, 'fig5_sample_size.png'), bbox_inches='tight'); plt.close(fig)

# ================= Figure 6 =================
nc = [('WBCD', pd.read_csv(R('WBCD', 'WBCD_n_components_sensitivity.csv')), 'ACCURACY'),
      ('Coimbra', pd.read_csv(R('Coimbra', 'Coimbra_n_components_sensitivity.csv')), 'ACCURACY'),
      ('SEER 5-year', pd.read_csv(R('SEER_v1.1', 'SEER_fixed_60m_n_components_sensitivity.csv')), 'ROC_AUC')]
fig, axes = plt.subplots(1, 3, figsize=(W, 2.35))
for a, (name, d, lab) in zip(axes, nc):
    sns.lineplot(data=d, x='n_components', y='mean_score', hue='method', marker='o', linewidth=1.5, markersize=4, ax=a)
    a.set_title(f'Sensitivity to n_components ({name})', fontsize=7.2); a.set_xlabel('Number of Components'); a.set_ylabel(f'Mean 3-Fold {lab}')
    a.legend(title='method', fontsize=6.3, title_fontsize=6.3, frameon=True)
plt.tight_layout(w_pad=1.2); plt.savefig(os.path.join(out, 'fig6_dimensionality.png'), bbox_inches='tight'); plt.close(fig)
print('written', out)
