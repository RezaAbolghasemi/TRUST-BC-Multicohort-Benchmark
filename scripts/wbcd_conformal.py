"""Re-create the WBCD split-conformal iterations (identical protocol and seeds to src/) to
recover the per-iteration set counts (SD error bars in Figure 2)."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from seer_revision_core import unified_search, SEED
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
d = load_breast_cancer(); X = d.data; y = 1 - d.target
Xd, Xh, yd, yh = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
nc = max(3, min(X.shape[1] // 2, 50))
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
df = pd.DataFrame(rows); df.to_csv(sys.argv[1], index=False)
print((100 * df[['empty', 'singleton', 'ambiguous']].mean() / df.n_test.mean()).round(2).to_dict(), round(100 * df.coverage.mean(), 2))
