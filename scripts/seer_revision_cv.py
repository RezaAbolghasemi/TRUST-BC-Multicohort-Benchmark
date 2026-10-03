"""Stage 1 of the SEER re-analysis: outer 5x7 repeated nested CV for Leaky, Nested PCA,
Nested RF and Unified. Saves fold scores and out-of-fold predictions.
Usage: python scripts/seer_revision_cv.py <SEER.csv> <label_mode> <out_dir>"""
import sys, os, pickle, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from seer_revision_core import *

path, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
X, y, meta = load_seer(path, mode)
Xa = X.values
Xd, Xh, yd, yh, idd, idh = dev_holdout_split(Xa, y)
nc = max(3, min(Xa.shape[1] // 2, 50))
XL = leaky_features(Xd, nc)
splits = outer_splits(Xd, yd)
ck = os.path.join(out, f'folds_{mode}.pkl')
res = pickle.load(open(ck, 'rb')) if os.path.exists(ck) else []
for i, (tr, te) in enumerate(splits):
    if i < len(res):
        continue
    t = time.time()
    res.append(run_outer_fold(i, tr, te, Xd, yd, XL, nc))
    pickle.dump(res, open(ck, 'wb'))
    print(mode, i, round(time.time() - t, 1), round(res[-1]['Unified_AUROC'], 4), flush=True)
rows = [{'Fold_Iteration': r['fold'] + 1, **{k: r[k] for k in r if k.endswith('_AUROC') or k.endswith('_Accuracy')},
         'Unified_arch': r['arch'], 'Unified_C': r['best_C']} for r in res]
pd.DataFrame(rows).to_csv(os.path.join(out, f'SEER_{mode}_raw_cv_scores.csv'), index=False)
oof = []
for r in res:
    d = pd.DataFrame({'fold': r['fold'] + 1, 'dev_pos': r['test_idx'], 'orig_index': idd[r['test_idx']], 'y': r['y']})
    for k in ['Leaky', 'Nested_PCA', 'Nested_RF', 'Unified']:
        d[k] = r[k]
    oof.append(d)
oof = pd.concat(oof)
oof = oof.merge(meta.reset_index().rename(columns={'index': 'orig_index'}), on='orig_index', how='left')
oof.to_csv(os.path.join(out, f'SEER_{mode}_oof_predictions.csv'), index=False)
print('done', mode)
