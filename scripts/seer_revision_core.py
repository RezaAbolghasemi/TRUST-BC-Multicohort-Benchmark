"""
Shared core for the SEER endpoint-correction re-analysis (TRUST-BC revision v1.1).

Replicates, line for line, the data preparation and the four scikit-learn
pipelines of src/trust_bc_multicohort_benchmark.py (Leaky, Nested PCA, Nested RF,
Unified) so that the SEER classification branch can be re-run with a corrected
fixed-horizon 5-year label. XGBoost/LightGBM baselines are not part of this module.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = 42
HORIZON = 60


class FeatureMethodSwitcher(BaseEstimator, TransformerMixin):
    """Identical to the class in src/trust_bc_multicohort_benchmark.py."""
    def __init__(self, method='pca', n_components=10, random_state=42):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.method == 'pca':
            self.transformer_ = PCA(n_components=self.n_components, random_state=self.random_state)
        else:
            self.transformer_ = SelectFromModel(
                RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=self.random_state),
                max_features=self.n_components, threshold=-np.inf)
        self.transformer_.fit(X, y)
        return self

    def transform(self, X):
        return self.transformer_.transform(X)


def load_seer(path, label_mode='fixed_60m'):
    """
    label_mode:
      'as_reported' : label = 1 if Status == dead at ANY time (v1.0 behaviour; deaths
                      after month 60 counted as 5-year events).
      'fixed_60m'   : label = 1 only if death occurred at or before month 60;
                      deaths after month 60 are 5-year survivors (label 0).
    In both modes, patients alive with < 60 months of follow-up are excluded.
    Returns X (encoded DataFrame), y (np.array), meta (DataFrame with race, age,
    survival months, status for the retained rows, in the same order).
    """
    raw = pd.read_csv(path, low_memory=False)
    raw.columns = raw.columns.str.strip()
    raw = raw.drop(columns=['Unnamed: 3'], errors='ignore')
    target_col = 'Status'
    time_col = [c for c in raw.columns if 'survival months' in c.lower()][0]
    raw = raw.drop(columns=['6th Stage'], errors='ignore')
    maps = {
        'T Stage': {'T0': 0, 'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4},
        'N Stage': {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3},
        'Grade': {'Well differentiated; Grade I': 1, 'Moderately differentiated; Grade II': 2,
                  'Poorly differentiated; Grade III': 3, 'Undifferentiated; anaplastic; Grade IV': 4}}
    for c, m in maps.items():
        if c in raw.columns:
            raw[c] = raw[c].map(m)
    raw[target_col] = raw[target_col].astype(str).str.strip().str.lower()
    raw = raw[raw[target_col].isin(['alive', 'dead'])]
    raw[time_col] = pd.to_numeric(raw[time_col], errors='coerce')
    raw = raw.dropna(subset=[target_col, time_col]).reset_index(drop=True)

    valid = (raw[target_col] == 'dead') | ((raw[target_col] == 'alive') & (raw[time_col] >= HORIZON))
    df = raw[valid].reset_index(drop=True)
    if label_mode == 'as_reported':
        y = np.where(df[target_col] == 'dead', 1, 0)
    elif label_mode == 'fixed_60m':
        y = np.where((df[target_col] == 'dead') & (df[time_col] <= HORIZON), 1, 0)
    else:
        raise ValueError(label_mode)
    meta = df[['Age', 'Race', time_col, target_col]].copy()
    meta.columns = ['age', 'race', 'survival_months', 'status']
    X = df.drop(columns=[time_col, 'patient_id', 'id', target_col], errors='ignore')
    num = X.select_dtypes(include=np.number).columns
    X[num] = X[num].fillna(X[num].median())
    cat = X.select_dtypes(exclude=np.number).columns
    X[cat] = X[cat].fillna('Unknown')
    X = pd.get_dummies(X, drop_first=True).astype(float)
    return X, y, meta


def dev_holdout_split(X_arr, y):
    idx = np.arange(len(y))
    return train_test_split(X_arr, y, idx, test_size=0.15, stratify=y, random_state=SEED)


def base_clf():
    return VotingClassifier([
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)),
        ('svm', SVC(kernel='linear', class_weight='balanced', probability=True, random_state=SEED))],
        voting='soft')


def unified_search(n_components, scoring='roc_auc', cv=None):
    cv = cv or StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('feature_opt', FeatureMethodSwitcher(n_components=n_components, random_state=SEED)),
                     ('clf', base_clf())])
    grid = {'feature_opt__method': ['pca', 'rf'], 'clf__svm__C': [0.1, 1.0, 10.0]}
    return GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=1)


def run_outer_fold(fold_id, train_idx, test_idx, X_dev, y_dev, X_leaky, n_components, scoring='roc_auc'):
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    X_tr, X_te, y_tr, y_te = X_dev[train_idx], X_dev[test_idx], y_dev[train_idx], y_dev[test_idx]
    out = {'fold': fold_id, 'test_idx': test_idx, 'y': y_te}
    leaky = VotingClassifier([
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)),
        ('svm', SVC(kernel='linear', class_weight='balanced', C=1.0, probability=True, random_state=SEED))],
        voting='soft').fit(X_leaky[train_idx], y_tr)
    out['Leaky'] = leaky.predict_proba(X_leaky[test_idx])[:, 1]
    out['Leaky_pred'] = leaky.predict(X_leaky[test_idx])
    pca = GridSearchCV(Pipeline([('scaler', StandardScaler()),
                                 ('feature_opt', PCA(n_components=n_components, random_state=SEED)),
                                 ('clf', base_clf())]),
                       {'clf__svm__C': [0.1, 1.0, 10.0]}, cv=inner_cv, scoring=scoring, n_jobs=1).fit(X_tr, y_tr)
    out['Nested_PCA'] = pca.predict_proba(X_te)[:, 1]; out['Nested_PCA_pred'] = pca.predict(X_te)
    rf = GridSearchCV(Pipeline([('scaler', StandardScaler()),
                                ('feature_opt', SelectFromModel(
                                    RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED),
                                    max_features=n_components, threshold=-np.inf)),
                                ('clf', base_clf())]),
                      {'clf__svm__C': [0.1, 1.0, 10.0]}, cv=inner_cv, scoring=scoring, n_jobs=1).fit(X_tr, y_tr)
    out['Nested_RF'] = rf.predict_proba(X_te)[:, 1]; out['Nested_RF_pred'] = rf.predict(X_te)
    uni = unified_search(n_components, scoring, inner_cv).fit(X_tr, y_tr)
    out['Unified'] = uni.predict_proba(X_te)[:, 1]; out['Unified_pred'] = uni.predict(X_te)
    out['arch'] = uni.best_params_['feature_opt__method']
    out['best_C'] = uni.best_params_['clf__svm__C']
    fo = uni.best_estimator_.named_steps['feature_opt']
    out['rf_support'] = fo.transformer_.get_support().astype(int) if out['arch'] == 'rf' else None
    for k in ['Leaky', 'Nested_PCA', 'Nested_RF', 'Unified']:
        out[k + '_AUROC'] = roc_auc_score(y_te, out[k])
        out[k + '_Accuracy'] = accuracy_score(y_te, out[k + '_pred'])
    return out


def outer_splits(X_dev, y_dev, n_repeats=7):
    return list(RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED).split(X_dev, y_dev))


def leaky_features(X_dev, n_components):
    return PCA(n_components=n_components, random_state=SEED).fit_transform(StandardScaler().fit_transform(X_dev))
