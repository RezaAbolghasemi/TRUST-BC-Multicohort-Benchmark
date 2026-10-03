"""Re-implementation of the penalized Weibull AFT survival branch (lifelines-style
parameterisation: log-scale linear in standardized covariates, intercept-only shape,
L2 penalty 0.1 on the mean negative log-likelihood) used to obtain a bootstrap CI for
the test-set C-index. Reports the reproduced point estimate for comparison with v1.0."""
import sys, numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
path = sys.argv[1]
raw = pd.read_csv(path, low_memory=False); raw.columns = raw.columns.str.strip()
raw = raw.drop(columns=['Unnamed: 3', '6th Stage'], errors='ignore')
maps = {'T Stage': {'T0': 0, 'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4}, 'N Stage': {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3},
        'Grade': {'Well differentiated; Grade I': 1, 'Moderately differentiated; Grade II': 2,
                  'Poorly differentiated; Grade III': 3, 'Undifferentiated; anaplastic; Grade IV': 4}}
for c, m in maps.items(): raw[c] = raw[c].map(m)
raw['Status'] = raw['Status'].astype(str).str.strip().str.lower()
df = raw.copy(); df['event'] = (df['Status'] == 'dead').astype(int); df = df.drop(columns=['Status'])
df = pd.get_dummies(df, drop_first=True).astype(float)
tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df['event'])
T = 'Survival Months'
cov = [c for c in df.columns if c not in (T, 'event')]
sd = tr[cov].std(0).values
def fit(d):
    X = d[cov].values / sd; t = d[T].values; e = d['event'].values; n = len(t)
    def nll(w):
        b0, b, lr = w[0], w[1:-1], w[-1]; rho = np.exp(lr); lam = np.exp(b0 + X @ b)
        z = (t / lam) ** rho
        ll = e * (np.log(rho) - np.log(lam) + (rho - 1) * (np.log(t) - np.log(lam))) - z
        return -ll.sum() / n + 0.1 * 0.5 * np.sum(b ** 2)
    w0 = np.r_[np.log(t.mean()), np.zeros(X.shape[1]), 0.0]
    return minimize(nll, w0, method='L-BFGS-B').x
def cindex(t, s, e):
    """Harrell's C: higher predicted survival s should mean longer time."""
    t, s, e = map(np.asarray, (t, s, e)); num = 0.0; den = 0.0
    for i in np.where(e == 1)[0]:
        m = t > t[i]; den += m.sum(); num += (s[m] > s[i]).sum() + 0.5 * (s[m] == s[i]).sum()
    return num / den
w = fit(tr)
pred = te[cov].values / sd @ w[1:-1]
c = cindex(te[T].values, pred, te['event'].values)
rng = np.random.default_rng(42); bs = []
tt, ee = te[T].values, te['event'].values
for _ in range(2000):
    i = rng.integers(0, len(te), len(te)); bs.append(cindex(tt[i], pred[i], ee[i]))
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f'n_test={len(te)} events_test={int(ee.sum())} n_train={len(tr)} events_train={int(tr.event.sum())}')
print(f'C_reimpl={c:.4f} bootstrap_SE={np.std(bs):.4f} CI=({lo:.3f},{hi:.3f})')
