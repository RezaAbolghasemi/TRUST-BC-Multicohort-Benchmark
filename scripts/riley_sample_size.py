"""Minimum sample size for developing a binary-outcome prediction model (Riley et al.,
BMJ 2020;368:m441), with the anticipated Cox-Snell R2 derived from an anticipated
C statistic by the simulation approach of Riley, Van Calster & Collins (Stat Med 2021).
Criteria: (i) expected uniform shrinkage >= 0.90; (ii) optimism in Nagelkerke R2 <= 0.05;
(iii) overall risk estimated within +/-0.05."""
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

def r2cs_from_c(c, prev, n=1_000_000, seed=1):
    rng = np.random.default_rng(seed)
    mu = np.sqrt(2) * norm.ppf(c)
    y = rng.random(n) < prev
    lp = np.where(y, rng.normal(mu, 1, n), rng.normal(0, 1, n)).reshape(-1, 1)
    m = LogisticRegression(C=1e6).fit(lp, y)
    p = m.predict_proba(lp)[:, 1]
    ll1 = np.sum(y * np.log(p) + (~y) * np.log(1 - p))
    ll0 = n * (prev * np.log(prev) + (1 - prev) * np.log(1 - prev))
    return 1 - np.exp(-2 * (ll1 - ll0) / n)

def riley(c, prev, params, shrink=0.9, delta=0.05):
    r2 = r2cs_from_c(c, prev)
    max_r2 = 1 - (prev ** prev * (1 - prev) ** (1 - prev)) ** 2
    n1 = params / ((shrink - 1) * np.log(1 - r2 / shrink))
    s2 = r2 / (r2 + delta * max_r2)
    n2 = params / ((s2 - 1) * np.log(1 - r2 / s2))
    n3 = (1.96 / 0.05) ** 2 * prev * (1 - prev)
    return r2, int(np.ceil(n1)), int(np.ceil(n2)), int(np.ceil(n3))

if __name__ == '__main__':
    import sys
    for name, c, prev, p in [('WBCD', 0.99, 212 / 569, 30), ('Coimbra', 0.79, 64 / 116, 9), ('SEER', float(sys.argv[1]) if len(sys.argv) > 1 else 0.73, 458 / 3270, 16)]:
        r2, n1, n2, n3 = riley(c, prev, p)
        print(f'{name}: C={c} prev={prev:.3f} params={p} R2cs={r2:.3f} n_shrinkage={n1} n_optimism={n2} n_overall_risk={n3} -> required={max(n1, n2, n3)}')
