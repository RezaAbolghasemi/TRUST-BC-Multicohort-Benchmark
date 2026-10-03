"""Sample-size sensitivity (bootstrapped learning curves, B=200) for the corrected SEER
5-year endpoint, using the Unified configuration selected on the development set.
Identical protocol to sample_size_sensitivity() in src/.
Usage: python scripts/seer_revision_samplesize.py <SEER.csv> <label_mode> <out_dir>"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from seer_revision_core import *
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13})
path, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
X, y, meta = load_seer(path, mode); Xa = X.values
Xd, Xh, yd, yh, idd, idh = dev_holdout_split(Xa, y)
nc = max(3, min(Xa.shape[1] // 2, 50))
best = unified_search(nc).fit(Xd, yd).best_estimator_
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
    print(frac, accs.mean(), flush=True)
ds = pd.DataFrame(rows); ds.to_csv(os.path.join(out, f'SEER_{mode}_sample_size_sensitivity.csv'), index=False)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].plot(ds.n_samples, ds.mean_acc, marker='o', color='#3498DB', lw=2); ax[0].fill_between(ds.n_samples, ds.ci_low, ds.ci_high, alpha=.25, color='#3498DB')
ax[0].set_xlabel('Training sample size', fontweight='bold'); ax[0].set_ylabel('Bootstrap accuracy (mean ± 95% CI)', fontweight='bold'); ax[0].set_title('Learning Curve (SEER, 5-year)', fontweight='bold')
ax[1].plot(ds.n_samples, ds.ci_width, marker='s', color='#E74C3C', lw=2); ax[1].set_xlabel('Training sample size', fontweight='bold')
ax[1].set_ylabel('95% CI width (estimation uncertainty)', fontweight='bold'); ax[1].set_title('Estimate Stability vs. Sample Size', fontweight='bold')
plt.tight_layout(pad=2.5); plt.savefig(os.path.join(out, f'SEER_{mode}_fig_sample_size_sensitivity.png'), dpi=300, bbox_inches='tight'); plt.close(fig)
print('done')
