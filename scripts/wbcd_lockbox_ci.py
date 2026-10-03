"""Re-create the WBCD lockbox predictions (scikit-learn only) and bootstrap CIs."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from seer_revision_core import unified_search, SEED
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
d = load_breast_cancer(); X = d.data; y = 1 - d.target
idx = np.arange(len(y))
Xd, Xh, yd, yh, idd, idh = train_test_split(X, y, idx, test_size=0.15, stratify=y, random_state=SEED)
nc = max(3, min(X.shape[1] // 2, 50))
best = unified_search(nc, 'accuracy').fit(Xd, yd).best_estimator_
Xtr, Xcal, ytr, ycal = train_test_split(Xd, yd, test_size=0.2, stratify=yd, random_state=SEED)
best.fit(Xtr, ytr)
p = best.predict_proba(Xh)[:, 1]
out = sys.argv[1] if len(sys.argv) > 1 else '.'
pd.DataFrame({'orig_index': idh, 'y': yh, 'p': p}).to_csv(os.path.join(out, 'WBCD_lockbox_predictions.csv'), index=False)
print('dev', len(yd), yd.sum(), 'hold', len(yh), yh.sum(), 'cal', len(ycal), ycal.sum(), 'train', len(ytr), ytr.sum())
print('acc', accuracy_score(yh, best.predict(Xh)), 'auc', roc_auc_score(yh, p), 'brier', round(brier_score_loss(yh, p), 4))
