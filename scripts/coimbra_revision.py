"""Re-create Coimbra out-of-fold predictions (Unified, 5x10 outer CV), the 100 split-conformal
iterations and the lockbox predictions with identical code and seeds to src/, from dataR2.csv.
Usage: python scripts/coimbra_revision.py <dataR2.csv> <out_dir>"""
import sys, os, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from seer_revision_core import unified_search, SEED
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score
path, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
df = pd.read_csv(path)
y = np.where(df['Classification'] == 2, 1, 0); Xdf = df.drop(columns=['Classification']); X = Xdf.values
idx = np.arange(len(y))
Xd, Xh, yd, yh, idd, idh = train_test_split(X, y, idx, test_size=0.15, stratify=y, random_state=SEED)
nc = max(3, min(X.shape[1] // 2, 50))
# outer CV
res = []
for i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED).split(Xd, yd)):
    g = unified_search(nc, 'accuracy', StratifiedKFold(3, shuffle=True, random_state=SEED)).fit(Xd[tr], yd[tr])
    fo = g.best_estimator_.named_steps['feature_opt']; arch = g.best_params_['feature_opt__method']
    res.append({'fold': i + 1, 'te': te, 'y': yd[te], 'p': g.predict_proba(Xd[te])[:, 1], 'acc': accuracy_score(yd[te], g.predict(Xd[te])),
                'arch': arch, 'support': fo.transformer_.get_support().astype(int) if arch == 'rf' else None})
pickle.dump(res, open(os.path.join(out, 'coimbra_oof.pkl'), 'wb'))
pd.DataFrame([{'Fold_Iteration': r['fold'], 'Unified_Accuracy': r['acc'], 'arch': r['arch']} for r in res]).to_csv(os.path.join(out, 'Coimbra_unified_fold_check.csv'), index=False)
pd.concat([pd.DataFrame({'fold': r['fold'], 'dev_pos': r['te'], 'orig_index': idd[r['te']], 'y': r['y'], 'Unified': r['p']}) for r in res]).to_csv(os.path.join(out, 'Coimbra_oof_predictions.csv'), index=False)
# conformal
best = unified_search(nc, 'accuracy').fit(Xd, yd).best_estimator_
rows = []
for rep in range(100):
    Xtc, Xte, ytc, yte = train_test_split(Xd, yd, test_size=0.2, random_state=rep, stratify=yd)
    Xtr, Xca, ytr, yca = train_test_split(Xtc, ytc, test_size=0.25, random_state=rep, stratify=ytc)
    best.fit(Xtr, ytr)
    cp = best.predict_proba(Xca)[np.arange(len(yca)), yca]
    q = np.quantile(1 - cp, np.ceil((len(yca) + 1) * 0.95) / len(yca))
    ps = (1 - best.predict_proba(Xte)) <= q; sz = ps.sum(1)
    rows.append({'rep': rep, 'empty': (sz == 0).sum(), 'singleton': (sz == 1).sum(), 'ambiguous': (sz == 2).sum(),
                 'coverage': ps[np.arange(len(yte)), yte].mean(), 'n_test': len(yte)})
cf = pd.DataFrame(rows); cf.to_csv(os.path.join(out, 'Coimbra_conformal_iterations.csv'), index=False)
# lockbox
Xft, Xfc, yft, yfc = train_test_split(Xd, yd, test_size=0.2, stratify=yd, random_state=SEED)
best.fit(Xft, yft); p = best.predict_proba(Xh)[:, 1]
pf = best.predict_proba(Xh); cal = best.predict_proba(Xfc)[np.arange(len(yfc)), yfc]
qh = np.quantile(1 - cal, np.ceil((len(yfc) + 1) * 0.95) / len(yfc)); hs = (1 - pf) <= qh
pd.DataFrame({'orig_index': idh, 'y': yh, 'p': p, 'pred': best.predict(Xh), 'covered': hs[np.arange(len(yh)), yh].astype(int)}).to_csv(os.path.join(out, 'Coimbra_lockbox_predictions.csv'), index=False)
pd.DataFrame({'Original_Index': np.r_[idd, idh], 'Split': ['Dev'] * len(idd) + ['Holdout'] * len(idh)}).to_csv(os.path.join(out, 'Coimbra_split_check.csv'), index=False)
print('conformal %', (100 * cf[['empty', 'singleton', 'ambiguous']].mean() / cf.n_test.mean()).round(2).to_dict(), 'cov', round(100 * cf.coverage.mean(), 2))
print('lockbox acc', accuracy_score(yh, best.predict(Xh)), 'cov', hs[np.arange(len(yh)), yh].mean())
