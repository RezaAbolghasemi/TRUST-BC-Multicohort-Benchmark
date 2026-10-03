"""Re-create WBCD out-of-fold predictions of the Unified pipeline (5x10 outer CV, identical
code and seeds to src/), used to redraw the WBCD ROC/PR/calibration/DCA panels.
Usage: python scripts/wbcd_oof.py <out_dir>"""
import sys, os, pickle, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from seer_revision_core import unified_search, SEED
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
d = load_breast_cancer(); X = d.data; y = 1 - d.target
Xd, Xh, yd, yh = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
nc = max(3, min(X.shape[1] // 2, 50))
rows, oof = [], []
ck = os.path.join(out, 'wbcd_oof.pkl'); res = pickle.load(open(ck, 'rb')) if os.path.exists(ck) else []
for i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED).split(Xd, yd)):
    if i < len(res): continue
    g = unified_search(nc, 'accuracy', StratifiedKFold(3, shuffle=True, random_state=SEED)).fit(Xd[tr], yd[tr])
    fo = g.best_estimator_.named_steps['feature_opt']
    res.append({'fold': i + 1, 'te': te, 'y': yd[te], 'p': g.predict_proba(Xd[te])[:, 1],
                'acc': accuracy_score(yd[te], g.predict(Xd[te])), 'arch': g.best_params_['feature_opt__method'],
                'support': fo.transformer_.get_support().astype(int) if g.best_params_['feature_opt__method'] == 'rf' else None})
    pickle.dump(res, open(ck, 'wb')); print(i, round(res[-1]['acc'], 5), res[-1]['arch'], flush=True)
pd.DataFrame([{'Fold_Iteration': r['fold'], 'Unified_Accuracy': r['acc'], 'arch': r['arch']} for r in res]).to_csv(os.path.join(out, 'WBCD_unified_fold_check.csv'), index=False)
pd.concat([pd.DataFrame({'fold': r['fold'], 'dev_pos': r['te'], 'y': r['y'], 'Unified': r['p']}) for r in res]).to_csv(os.path.join(out, 'WBCD_oof_predictions.csv'), index=False)
