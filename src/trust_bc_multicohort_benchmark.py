#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRUST-BC: A Multicohort Benchmark for Leakage-Free, Calibration-Audited, and Uncertainty-Aware Breast Cancer Prediction Models

This script implements the full evaluation pipeline described in our manuscript.
It provides nested cross-validation with leakage prevention, split conformal prediction,
calibration assessment, decision curve analysis, sensitivity analyses, SHAP interpretability,
and an optional Weibull AFT survival branch for the SEER cohort.

The framework is benchmarked on three breast cancer datasets:
    - WBCD (Wisconsin Breast Cancer, cytomorphological)
    - Coimbra (serum metabolic biomarkers)
    - SEER (population-based registry, with censoring-aware filtering)

Reported following TRIPOD+AI; an author self-assessment against PROBAST+AI is in
Online Resource 1 (supplementary/). For details, see the accompanying paper.

Version 1.1: the SEER classification label is now a fixed-horizon 5-year label
(see SEER_LABEL_MODE). Set SEER_LABEL_MODE = 'as_reported' to reproduce v1.0.

Authors: Mohammad Amin Shayegan, Reza Abolghasemi, Mohammadreza Salehi, Armaghan Kiarsi
"""

import os
import glob
import time
import hashlib
import urllib.request
from io import BytesIO
from collections import Counter
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
from scipy.special import logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    GridSearchCV,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
import shap

# Optional imports (graceful fallback if packages missing)
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from lifelines import WeibullAFTFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')


# -----------------------------------------------------------
# Global configuration
# -----------------------------------------------------------

SEED = 42
# SEER classification label: 'fixed_60m' (v1.1, correct 5-year label) or 'as_reported' (v1.0)
SEER_LABEL_MODE = 'fixed_60m'
np.random.seed(SEED)

ALPHA = 0.05
N_CONFORMAL_REPEATS = 100
N_BOOTSTRAP_SENSITIVITY = 200

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
})


# -------------------------------------------------------------------------
# Helper classes and utilities
# -------------------------------------------------------------------------

class FeatureMethodSwitcher(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that selects either PCA or Random Forest
    based feature importance for dimensionality reduction.

    This dynamic selector is used inside nested CV to prevent leakage.
    """
    def __init__(self, method: str = 'pca', n_components: int = 10,
                 random_state: int = 42):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.method == 'pca':
            self.transformer_ = PCA(
                n_components=self.n_components,
                random_state=self.random_state
            )
        else:
            base_rf = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=self.random_state
            )
            self.transformer_ = SelectFromModel(
                base_rf,
                max_features=self.n_components,
                threshold=-np.inf
            )
        self.transformer_.fit(X, y)
        return self

    def transform(self, X):
        return self.transformer_.transform(X)


def corrected_resampled_t_test(scores_a: np.ndarray, scores_b: np.ndarray,
                               n_splits: int, n_repeats: int) -> Tuple[float, float]:
    """
    Nadeau-Bengio corrected resampled t-test for paired comparisons.
    Returns (t_stat, p_value).
    """
    diff = np.array(scores_a) - np.array(scores_b)
    n = len(diff)
    mean_diff = np.mean(diff)
    var_diff = np.var(diff, ddof=1)
    correction = 1.0 / (n_splits - 1)
    var_corrected = var_diff * (1.0 / n + correction)
    if var_corrected == 0:
        return 0.0, 1.0
    t_stat = mean_diff / np.sqrt(var_corrected)
    df = n - 1
    p_value = t.sf(np.abs(t_stat), df) * 2
    return t_stat, p_value


def corrected_resampled_tost(scores_a: np.ndarray, scores_b: np.ndarray,
                             margin: float, n_splits: int) -> float:
    """
    Two One-Sided Tests (TOST) for equivalence using corrected variance.
    Returns the TOST p-value.
    """
    diff = np.array(scores_a) - np.array(scores_b)
    n = len(diff)
    mean_diff = np.mean(diff)
    var_diff = np.var(diff, ddof=1)
    correction = 1.0 / (n_splits - 1)
    var_corrected = var_diff * (1.0 / n + correction)
    if var_corrected == 0:
        return 0.0 if abs(mean_diff) < margin else 1.0

    se = np.sqrt(var_corrected)
    df = n - 1
    t1 = (mean_diff - (-margin)) / se
    p1 = t.sf(t1, df)
    t2 = (mean_diff - margin) / se
    p2 = t.cdf(t2, df)
    return max(p1, p2)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    # ECE over uniform probability bins
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / len(y_prob)) * abs(
                y_true[mask].mean() - y_prob[mask].mean()
            )
    return ece


def calculate_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    # logistic recalibration -> (slope, intercept); optimal is slope=1, intercept=0
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    logit_prob = logit(y_prob_clipped)
    cal_lr = LogisticRegression(C=1e5, solver='lbfgs')
    cal_lr.fit(logit_prob.reshape(-1, 1), y_true)
    return cal_lr.coef_[0][0], cal_lr.intercept_[0]


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray,
                thresholds: np.ndarray) -> List[float]:
    n = len(y_true)
    nb = []
    for pt in thresholds:
        preds = (y_prob >= pt).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        if pt < 1.0:
            nb.append((tp / n) - (fp / n) * (pt / (1 - pt)))
        else:
            nb.append(0.0)
    return [max(v, 0) for v in nb]


def quantify_row_overlap(X_train: np.ndarray, X_test: np.ndarray) -> float:
    train_hashes = {hash(tuple(row)) for row in X_train}
    test_hashes = {hash(tuple(row)) for row in X_test}
    overlap = train_hashes.intersection(test_hashes)
    if len(test_hashes) == 0:
        return 0.0
    return len(overlap) / len(test_hashes)


# --- Sensitivity analyses ---

def sample_size_sensitivity(X_dev: np.ndarray, y_dev: np.ndarray,
                            dataset_name: str, base_pipeline: Pipeline,
                            output_dir: str,
                            fractions: Tuple[float, ...] = (0.30, 0.50, 0.70, 0.85, 1.00),
                            n_bootstrap: int = 200, test_size: float = 0.30) -> None:
    """
    Generate learning curves by bootstrapping subsamples of the development set.
    Saves results and plots.
    """
    print("    [Sensitivity] Sample-size learning curves...")
    n_total = len(X_dev)
    rows = []

    for frac in fractions:
        n_sub = max(25, int(round(n_total * frac)))
        n_sub = min(n_sub, n_total)
        accs = []
        for b in range(n_bootstrap):
            rs = SEED + b
            idx = resample(
                np.arange(n_total),
                n_samples=n_sub,
                stratify=y_dev,
                random_state=rs,
                replace=False
            )
            X_sub, y_sub = X_dev[idx], y_dev[idx]
            if len(np.unique(y_sub)) < 2:
                continue
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_sub, y_sub,
                    test_size=test_size,
                    stratify=y_sub,
                    random_state=rs
                )
                model = base_pipeline
                model.fit(X_tr, y_tr)
                accs.append(accuracy_score(y_te, model.predict(X_te)))
            except Exception:
                continue
        if len(accs) < 10:
            continue
        accs = np.array(accs)
        ci_low, ci_high = np.percentile(accs, [2.5, 97.5])
        rows.append({
            "fraction": frac,
            "n_samples": n_sub,
            "mean_acc": accs.mean(),
            "std_acc": accs.std(ddof=1),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_width": ci_high - ci_low,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f"{dataset_name}_sample_size_sensitivity.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(df["n_samples"], df["mean_acc"], marker="o", color="#3498DB", linewidth=2)
    axes[0].fill_between(
        df["n_samples"],
        df["ci_low"],
        df["ci_high"],
        alpha=0.25,
        color="#3498DB"
    )
    axes[0].set_xlabel("Training sample size", fontweight="bold")
    axes[0].set_ylabel("Bootstrap accuracy (mean ± 95% CI)", fontweight="bold")
    axes[0].set_title(f"Learning Curve ({dataset_name})", fontweight="bold")

    axes[1].plot(df["n_samples"], df["ci_width"], marker="s", color="#E74C3C", linewidth=2)
    axes[1].set_xlabel("Training sample size", fontweight="bold")
    axes[1].set_ylabel("95% CI width (estimation uncertainty)", fontweight="bold")
    axes[1].set_title("Estimate Stability vs. Sample Size", fontweight="bold")

    plt.tight_layout(pad=2.5)
    plt.savefig(
        os.path.join(output_dir, f"{dataset_name}_fig_sample_size_sensitivity.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


def n_components_sensitivity(X_dev: np.ndarray, y_dev: np.ndarray,
                             dataset_name: str, output_dir: str,
                             n_features: int, default_n: int,
                             cv_scoring: str = 'roc_auc') -> None:
    """
    Evaluate performance sensitivity to the number of retained components
    for both PCA and Random-Forest-based selection.
    """
    print("    [Sensitivity] Dimensionality reduction (n_components) analysis...")
    test_ns = sorted(list(set([
        max(2, default_n // 2),
        default_n,
        min(n_features, int(default_n * 1.5)),
        min(n_features, default_n * 2)
    ])))

    results = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    base_clf = VotingClassifier(
        [
            ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)),
            ('svm', SVC(kernel='linear', class_weight='balanced', probability=True, random_state=SEED))
        ],
        voting='soft'
    )

    for n_comp in test_ns:
        for method in ['pca', 'rf']:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_opt', FeatureMethodSwitcher(method=method, n_components=n_comp, random_state=SEED)),
                ('clf', base_clf)
            ])
            scores = []
            for train_idx, test_idx in skf.split(X_dev, y_dev):
                pipe.fit(X_dev[train_idx], y_dev[train_idx])
                if cv_scoring == 'roc_auc':
                    score = roc_auc_score(
                        y_dev[test_idx],
                        pipe.predict_proba(X_dev[test_idx])[:, 1]
                    )
                else:
                    score = accuracy_score(
                        y_dev[test_idx],
                        pipe.predict(X_dev[test_idx])
                    )
                scores.append(score)
            results.append({
                'n_components': n_comp,
                'method': method.upper(),
                'mean_score': np.mean(scores)
            })

    df_sens = pd.DataFrame(results)
    df_sens.to_csv(os.path.join(output_dir, f"{dataset_name}_n_components_sensitivity.csv"), index=False)

    plt.figure(figsize=(7, 5))
    sns.lineplot(
        data=df_sens,
        x='n_components',
        y='mean_score',
        hue='method',
        marker='o',
        linewidth=2
    )
    plt.title(f"Sensitivity to n_components ({dataset_name})", fontweight='bold')
    plt.xlabel("Number of Components", fontweight='bold')
    plt.ylabel(f"Mean 3-Fold {cv_scoring.upper()}", fontweight='bold')
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{dataset_name}_fig_n_components_sensitivity.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def shap_interpretability(fitted_model: Any, X_train_bg: np.ndarray,
                          X_explain: np.ndarray, feature_names: List[str],
                          dataset_name: str, output_dir: str,
                          n_background: int = 50, n_explain_max: int = 100,
                          max_display: int = 15, random_state: int = SEED) -> None:
    """
    Generate SHAP summary plots on the independent holdout set using KernelExplainer.
    """
    print("    [Interpretability] Computing SHAP values...")
    rng = np.random.RandomState(random_state)
    background = shap.sample(
        X_train_bg,
        min(n_background, len(X_train_bg)),
        random_state=random_state
    )
    n_exp = min(n_explain_max, len(X_explain))
    if len(X_explain) > n_exp:
        idx = rng.choice(len(X_explain), size=n_exp, replace=False)
        X_expl_sub = X_explain[idx]
    else:
        X_expl_sub = X_explain

    predict_fn = lambda x: fitted_model.predict_proba(x)[:, 1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_expl_sub, nsamples=100)

    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values,
        X_expl_sub,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.title(f"SHAP Summary — {dataset_name} (True Holdout, n={n_exp})", fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{dataset_name}_fig_shap_summary.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)
    np.save(os.path.join(output_dir, f"{dataset_name}_shap_values.npy"), shap_values)


#####################################################################
# Main pipeline function
#####################################################################

def run_trust_bc_pipeline(X: pd.DataFrame, y: pd.Series,
                          dataset_name: str,
                          base_output_dir: str,
                          n_splits: int = 5,
                          n_repeats: int = 10,
                          cv_scoring: str = 'accuracy',
                          tost_margin: float = 0.02) -> None:
    """
    Execute the complete TRUST-BC evaluation for a single dataset.
    This function orchestrates data splitting, nested CV, conformal prediction,
    sensitivity analyses, lockbox evaluation, SHAP, and figure generation.
    """
    start_time = time.time()
    output_dir = os.path.join(base_output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(X, pd.DataFrame):
        X_arr = X.values
        feature_names = X.columns.tolist()
    else:
        X_arr = np.array(X)
        feature_names = [f"Feat_{i}" for i in range(X_arr.shape[1])]
    y_arr = np.array(y)

    print("\n" + "=" * 90)
    print(f" TRUST-BC Evaluation: {dataset_name} ".center(90, '*'))
    print("=" * 90)

    # ---------------------------------------------------------------------
    # 1. Deterministic split into development (85%) and holdout (15%)
    # ---------------------------------------------------------------------
    indices = np.arange(len(y_arr))
    X_dev, X_holdout, y_dev, y_holdout, idx_dev, idx_holdout = train_test_split(
        X_arr, y_arr, indices,
        test_size=0.15,
        stratify=y_arr,
        random_state=SEED
    )

    manifest_df = pd.DataFrame({'Original_Index': indices, 'Split': 'None'})
    manifest_df.loc[idx_dev, 'Split'] = 'Dev'
    manifest_df.loc[idx_holdout, 'Split'] = 'Holdout'
    manifest_df.to_csv(os.path.join(output_dir, f"{dataset_name}_split_manifest.csv"), index=False)

    n_features = X_dev.shape[1]
    n_components = max(3, min(n_features // 2, 50))
    print(f"[*] Dev: {len(X_dev)} | Holdout: {len(X_holdout)} | Features: {n_features} -> {n_components}")
    print(f"[*] Split manifest saved.")

    # ---------------------------------------------------------------------
    # 2. Setup cross-validation schemes
    # ---------------------------------------------------------------------
    outer_cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=SEED
    )
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    # ----------------------------------------------------------------------
    # 3. Leaky baseline: scale + PCA on the entire development set before splitting
    # ----------------------------------------------------------------------
    scaler_leaky = StandardScaler()
    pca_leaky = PCA(n_components=n_components, random_state=SEED)
    X_leaky = pca_leaky.fit_transform(scaler_leaky.fit_transform(X_dev))

    # --------------------------------------------------------------------
    # 4. Data structures for storing CV results
    # --------------------------------------------------------------------
    scores = {
        'leaky_acc': [], 'leaky_auc': [],
        'xgb_acc': [], 'xgb_auc': [],
        'lgbm_acc': [], 'lgbm_auc': [],
        'pca_acc': [], 'pca_auc': [],
        'rf_acc': [], 'rf_auc': [],
        'unified_acc': [], 'unified_auc': [],
    }
    oof_y_true = []
    oof_prob = []
    # Dictionary to collect OOF probabilities for each pipeline (for calibration per pipeline)
    oof_prob_per_pipeline = {
        'Leaky': [], 'XGBoost': [], 'LightGBM': [],
        'Nested_PCA': [], 'Nested_RF': [], 'Unified': []
    }
    architecture_choices = []
    overlap_ratios = []
    feat_counts_rf = np.zeros(n_features)

    base_clf = VotingClassifier(
        [
            ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)),
            ('svm', SVC(kernel='linear', class_weight='balanced', probability=True, random_state=SEED))
        ],
        voting='soft'
    )

    unified_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_opt', FeatureMethodSwitcher(n_components=n_components, random_state=SEED)),
        ('clf', base_clf)
    ])
    unified_grid = {
        'feature_opt__method': ['pca', 'rf'],
        'clf__svm__C': [0.1, 1.0, 10.0]
    }

    if XGBClassifier is not None:
        xgb_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(eval_metric='logloss', random_state=SEED))
        ])
    else:
        xgb_pipe = None
        scores['xgb_acc'] = []
        scores['xgb_auc'] = []

    if LGBMClassifier is not None:
        lgbm_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LGBMClassifier(random_state=SEED, verbose=-1))
        ])
    else:
        lgbm_pipe = None
        scores['lgbm_acc'] = []
        scores['lgbm_auc'] = []

    if XGBClassifier is None or LGBMClassifier is None:
        print("[!] XGBoost or LightGBM not installed; skipping those baselines.")

    xgb_grid = {
        'clf__n_estimators': [50, 100],
        'clf__learning_rate': [0.01, 0.1],
        'clf__max_depth': [3, 5]
    }
    lgbm_grid = {
        'clf__n_estimators': [50, 100],
        'clf__learning_rate': [0.01, 0.1],
        'clf__max_depth': [3, 5]
    }

    # ----------------------------------------------------------------------
    # 5. Outer cross-validation loop
    # ----------------------------------------------------------------------
    print(f"[1] Running leakage-free nested CV ({n_splits}x{n_repeats} folds)...")
    for train_idx, test_idx in outer_cv.split(X_dev, y_dev):
        X_train, X_test = X_dev[train_idx], X_dev[test_idx]
        y_train, y_test = y_dev[train_idx], y_dev[test_idx]

        overlap_ratios.append(quantify_row_overlap(X_train, X_test))

        pos_weight_fold = float(np.sum(y_train == 0) / np.sum(y_train == 1))

        X_train_l, X_test_l = X_leaky[train_idx], X_leaky[test_idx]
        leaky_clf = VotingClassifier(
            [
                ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)),
                ('svm', SVC(kernel='linear', class_weight='balanced', C=1.0, probability=True, random_state=SEED))
            ],
            voting='soft'
        )
        leaky_clf.fit(X_train_l, y_train)
        leaky_probs = leaky_clf.predict_proba(X_test_l)[:, 1]
        scores['leaky_acc'].append(accuracy_score(y_test, leaky_clf.predict(X_test_l)))
        scores['leaky_auc'].append(roc_auc_score(y_test, leaky_probs))
        oof_prob_per_pipeline['Leaky'].extend(leaky_probs)

        # n_jobs pinned to 1 here (and in every other GridSearchCV call below) so
        # that reruns on different machines/core counts reproduce identical numbers
        if xgb_pipe is not None:
            if dataset_name == "SEER":
                xgb_grid['clf__scale_pos_weight'] = [1, pos_weight_fold]
            xgb_search = GridSearchCV(
                xgb_pipe, xgb_grid,
                cv=inner_cv, scoring=cv_scoring, n_jobs=1
            ).fit(X_train, y_train)
            xgb_probs = xgb_search.predict_proba(X_test)[:, 1]
            scores['xgb_acc'].append(accuracy_score(y_test, xgb_search.predict(X_test)))
            scores['xgb_auc'].append(roc_auc_score(y_test, xgb_probs))
            oof_prob_per_pipeline['XGBoost'].extend(xgb_probs)

        if lgbm_pipe is not None:
            if dataset_name == "SEER":
                lgbm_grid['clf__scale_pos_weight'] = [1, pos_weight_fold]
            lgbm_search = GridSearchCV(
                lgbm_pipe, lgbm_grid,
                cv=inner_cv, scoring=cv_scoring, n_jobs=1
            ).fit(X_train, y_train)
            lgbm_probs = lgbm_search.predict_proba(X_test)[:, 1]
            scores['lgbm_acc'].append(accuracy_score(y_test, lgbm_search.predict(X_test)))
            scores['lgbm_auc'].append(roc_auc_score(y_test, lgbm_probs))
            oof_prob_per_pipeline['LightGBM'].extend(lgbm_probs)

        pca_search = GridSearchCV(
            Pipeline([
                ('scaler', StandardScaler()),
                ('feature_opt', PCA(n_components=n_components, random_state=SEED)),
                ('clf', base_clf)
            ]),
            {'clf__svm__C': [0.1, 1.0, 10.0]},
            cv=inner_cv, scoring=cv_scoring, n_jobs=1
        ).fit(X_train, y_train)
        pca_probs = pca_search.predict_proba(X_test)[:, 1]
        scores['pca_acc'].append(accuracy_score(y_test, pca_search.predict(X_test)))
        scores['pca_auc'].append(roc_auc_score(y_test, pca_probs))
        oof_prob_per_pipeline['Nested_PCA'].extend(pca_probs)

        rf_search = GridSearchCV(
            Pipeline([
                ('scaler', StandardScaler()),
                ('feature_opt', SelectFromModel(
                    RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED),
                    max_features=n_components,
                    threshold=-np.inf
                )),
                ('clf', base_clf)
            ]),
            {'clf__svm__C': [0.1, 1.0, 10.0]},
            cv=inner_cv, scoring=cv_scoring, n_jobs=1
        ).fit(X_train, y_train)
        rf_probs = rf_search.predict_proba(X_test)[:, 1]
        scores['rf_acc'].append(accuracy_score(y_test, rf_search.predict(X_test)))
        scores['rf_auc'].append(roc_auc_score(y_test, rf_probs))
        oof_prob_per_pipeline['Nested_RF'].extend(rf_probs)

        unified_search = GridSearchCV(
            unified_pipe, unified_grid,
            cv=inner_cv, scoring=cv_scoring, n_jobs=1
        ).fit(X_train, y_train)
        unified_probs = unified_search.predict_proba(X_test)[:, 1]
        scores['unified_acc'].append(accuracy_score(y_test, unified_search.predict(X_test)))
        scores['unified_auc'].append(roc_auc_score(y_test, unified_probs))

        oof_y_true.extend(y_test)
        oof_prob.extend(unified_probs)
        oof_prob_per_pipeline['Unified'].extend(unified_probs)

        chosen = unified_search.best_params_['feature_opt__method']
        architecture_choices.append(chosen)
        if chosen == 'rf':
            feat_counts_rf += unified_search.best_estimator_.named_steps['feature_opt'].transformer_.get_support().astype(int)

    # -----------------------------------------------------------------------
    # 6. Aggregate results and statistical testing
    # -----------------------------------------------------------------------
    test_uni = scores['unified_auc'] if cv_scoring == 'roc_auc' else scores['unified_acc']
    test_leaky = scores['leaky_auc'] if cv_scoring == 'roc_auc' else scores['leaky_acc']
    test_xgb = scores['xgb_auc'] if cv_scoring == 'roc_auc' else scores['xgb_acc']
    test_lgbm = scores['lgbm_auc'] if cv_scoring == 'roc_auc' else scores['lgbm_acc']
    test_pca = scores['pca_auc'] if cv_scoring == 'roc_auc' else scores['pca_acc']
    test_rf = scores['rf_auc'] if cv_scoring == 'roc_auc' else scores['rf_acc']

    df_raw = pd.DataFrame({
        'Fold_Iteration': range(1, len(scores['leaky_acc']) + 1),
        'Leaky_Accuracy': scores['leaky_acc'],
        'Leaky_AUROC': scores['leaky_auc'],
        'XGBoost_Accuracy': scores['xgb_acc'] if scores['xgb_acc'] else [np.nan]*len(scores['leaky_acc']),
        'XGBoost_AUROC': scores['xgb_auc'] if scores['xgb_auc'] else [np.nan]*len(scores['leaky_acc']),
        'LightGBM_Accuracy': scores['lgbm_acc'] if scores['lgbm_acc'] else [np.nan]*len(scores['leaky_acc']),
        'LightGBM_AUROC': scores['lgbm_auc'] if scores['lgbm_auc'] else [np.nan]*len(scores['leaky_acc']),
        'Nested_PCA_Accuracy': scores['pca_acc'],
        'Nested_PCA_AUROC': scores['pca_auc'],
        'Nested_RF_Accuracy': scores['rf_acc'],
        'Nested_RF_AUROC': scores['rf_auc'],
        'Unified_Accuracy': scores['unified_acc'],
        'Unified_AUROC': scores['unified_auc'],
    })
    df_raw.to_csv(os.path.join(output_dir, f'{dataset_name}_raw_cv_scores.csv'), index=False)

    pipeline_scores = {
        'Leaky': test_leaky,
        'XGBoost': test_xgb,
        'LightGBM': test_lgbm,
        'Nested_PCA': test_pca,
        'Nested_RF': test_rf,
        'Unified': test_uni,
    }

    # bootstrap CI for all six pipelines (Table 3), not just the unified one
    boot_ci = {}
    for name, arr in pipeline_scores.items():
        if len(arr) == n_repeats * n_splits:
            boot_means_i = [
                np.mean(resample(np.array(arr).reshape(n_repeats, n_splits), random_state=i))
                for i in range(2000)
            ]
            boot_ci[name] = np.percentile(boot_means_i, [2.5, 97.5])
        else:
            boot_ci[name] = (np.nan, np.nan)

    ci_low, ci_high = boot_ci['Unified']
    ci_display = f"{ci_low*100:.1f}% - {ci_high*100:.1f}%" if cv_scoring == 'accuracy' else f"{ci_low:.3f} - {ci_high:.3f}"

    _, p_xgb = corrected_resampled_t_test(test_leaky, test_xgb, n_splits, n_repeats)
    _, p_lgbm = corrected_resampled_t_test(test_leaky, test_lgbm, n_splits, n_repeats)
    _, p_uni_xgb = corrected_resampled_t_test(test_uni, test_xgb, n_splits, n_repeats)
    _, p_uni_lgbm = corrected_resampled_t_test(test_uni, test_lgbm, n_splits, n_repeats)
    _, p_pca_rf = corrected_resampled_t_test(test_pca, test_rf, n_splits, n_repeats)
    _, p_uni_leaky = corrected_resampled_t_test(test_uni, test_leaky, n_splits, n_repeats)

    # Holm-Bonferroni family = the 5 pre-specified comparisons from Section 2.5.
    # PCA vs RF is a descriptive ablation, not part of the formal family, so it
    # stays out of the correction and is reported below as a raw p-value.
    pvals = [p_xgb, p_lgbm, p_uni_xgb, p_uni_lgbm, p_uni_leaky]
    _, pvals_corr, _, _ = multipletests(pvals, method='holm')
    (p_xgb_corr, p_lgbm_corr, p_uni_xgb_corr,
     p_uni_lgbm_corr, p_uni_leaky_corr) = pvals_corr
    p_pca_rf_corr = np.nan

    p_tost_unified = corrected_resampled_tost(
        test_uni, test_leaky,
        margin=tost_margin,
        n_splits=n_splits
    )

    oof_y_true_np = np.array(oof_y_true)
    oof_prob_np = np.array(oof_prob)
    cv_brier = brier_score_loss(oof_y_true_np, oof_prob_np)
    cv_ece = expected_calibration_error(oof_y_true_np, oof_prob_np)
    cv_cal_slope, cv_cal_intercept = calculate_calibration_metrics(oof_y_true_np, oof_prob_np)

    # Compute Brier and ECE for each pipeline individually (saved to CSV)
    calib_rows = []
    for name, probs in oof_prob_per_pipeline.items():
        probs_arr = np.array(probs)
        if len(probs_arr) == len(oof_y_true_np):
            b = brier_score_loss(oof_y_true_np, probs_arr)
            e = expected_calibration_error(oof_y_true_np, probs_arr)
            calib_rows.append({'Pipeline': name, 'Brier': round(b, 3), 'ECE': round(e, 3)})
    df_calib_per_pipeline = pd.DataFrame(calib_rows)
    df_calib_per_pipeline.to_csv(
        os.path.join(output_dir, f"{dataset_name}_calibration_per_pipeline.csv"), index=False
    )
    print(f"\n -> Per-pipeline calibration:\n{df_calib_per_pipeline.to_string(index=False)}")

    # Table 3 summary (mean/std, CI, Holm p-values, TOST, Brier, ECE) per pipeline
    calib_lookup = {row['Pipeline']: row for row in calib_rows}
    adj_p_vs_leaky = {
        'Leaky': np.nan, 'XGBoost': p_xgb_corr, 'LightGBM': p_lgbm_corr,
        'Nested_PCA': np.nan, 'Nested_RF': np.nan, 'Unified': p_uni_leaky_corr,
    }
    adj_p_vs_unified = {
        'Leaky': np.nan, 'XGBoost': p_uni_xgb_corr, 'LightGBM': p_uni_lgbm_corr,
        'Nested_PCA': np.nan, 'Nested_RF': np.nan, 'Unified': np.nan,
    }
    p_tost_col = {
        'Leaky': np.nan, 'XGBoost': np.nan, 'LightGBM': np.nan,
        'Nested_PCA': np.nan, 'Nested_RF': np.nan, 'Unified': p_tost_unified,
    }

    table3_rows = []
    for name in ['Leaky', 'XGBoost', 'LightGBM', 'Nested_PCA', 'Nested_RF', 'Unified']:
        arr = pipeline_scores[name]
        ci_lo, ci_hi = boot_ci[name]
        calib = calib_lookup.get(name, {})
        table3_rows.append({
            'Pipeline': name,
            'Primary_Metric_Mean': round(np.mean(arr), 4) if len(arr) else None,
            'Primary_Metric_Std': round(np.std(arr, ddof=1), 4) if len(arr) else None,
            'CI_95_low': round(ci_lo, 4) if not np.isnan(ci_lo) else None,
            'CI_95_high': round(ci_hi, 4) if not np.isnan(ci_hi) else None,
            'Adj_p_vs_Leaky': round(adj_p_vs_leaky[name], 4) if not np.isnan(adj_p_vs_leaky[name]) else None,
            'Adj_p_vs_Unified': round(adj_p_vs_unified[name], 4) if not np.isnan(adj_p_vs_unified[name]) else None,
            'pTOST_vs_Leaky': round(p_tost_col[name], 4) if not np.isnan(p_tost_col[name]) else None,
            'Brier': calib.get('Brier'),
            'ECE': calib.get('ECE'),
        })
    df_table3 = pd.DataFrame(table3_rows)
    df_table3.to_csv(os.path.join(output_dir, f"{dataset_name}_table3_summary.csv"), index=False)
    print(f"\n -> Table 3 summary:\n{df_table3.to_string(index=False)}")


    # --------------------------------------
    # 7. Print summary
    # --------------------------------------
    print(f"\n -> Baseline Row Overlap (Hash): {np.mean(overlap_ratios)*100:.2f}%")
    print("    Note: PCA leakage is parametric (global component estimation), not row duplication.")
    print(f"\n -> Leaky baseline:          Acc={np.mean(scores['leaky_acc'])*100:.2f}%, AUC={np.mean(scores['leaky_auc']):.3f}")
    if xgb_pipe is not None:
        print(f" -> Tuned XGBoost:           Acc={np.mean(scores['xgb_acc'])*100:.2f}%, AUC={np.mean(scores['xgb_auc']):.3f} (Holm-p vs Leaky={p_xgb_corr:.3f})")
    else:
        print(" -> Tuned XGBoost:           Not installed")
    if lgbm_pipe is not None:
        print(f" -> Tuned LightGBM:          Acc={np.mean(scores['lgbm_acc'])*100:.2f}%, AUC={np.mean(scores['lgbm_auc']):.3f} (Holm-p vs Leaky={p_lgbm_corr:.3f})")
    else:
        print(" -> Tuned LightGBM:          Not installed")
    print(f" -> Nested PCA-only:         Acc={np.mean(scores['pca_acc'])*100:.2f}%, AUC={np.mean(scores['pca_auc']):.3f}")
    print(f" -> Nested RF-only:          Acc={np.mean(scores['rf_acc'])*100:.2f}%, AUC={np.mean(scores['rf_auc']):.3f} (PCA vs RF p={p_pca_rf:.3f}, descriptive only)")
    print(f" -> Unified pipeline:        Acc={np.mean(scores['unified_acc'])*100:.2f}%, AUC={np.mean(scores['unified_auc']):.3f} [95% Block-Boot CI ({cv_scoring}): {ci_display}]")
    print(f"    (Superiority vs Leaky, Holm p): {p_uni_leaky_corr:.4f}")
    print(f"    (TOST Equivalence vs Leaky, margin={tost_margin*100}%): {p_tost_unified:.4f}")
    if xgb_pipe is not None:
        print(f"    (Superiority vs XGBoost, Holm p): {p_uni_xgb_corr:.4f}")
    if lgbm_pipe is not None:
        print(f"    (Superiority vs LightGBM, Holm p): {p_uni_lgbm_corr:.4f}")
    print(f" -> Pooled Calibration:      Brier={cv_brier:.3f}, ECE={cv_ece:.3f} | Slope={cv_cal_slope:.3f}, Intercept={cv_cal_intercept:.3f}")

    arch_dist = Counter(architecture_choices)
    print(f" -> Architecture Selection Frequency (Unified): {dict(arch_dist)}")

    # Save architecture selection frequencies for the unified pipeline
    df_arch = pd.DataFrame([
        {'method': k, 'count': v, 'total_folds': len(architecture_choices),
         'frequency_pct': round(100 * v / len(architecture_choices), 2)}
        for k, v in arch_dist.items()
    ])
    df_arch.to_csv(os.path.join(output_dir, f"{dataset_name}_architecture_selection.csv"), index=False)

    # ----------------------------------------------------------------------
    # 8. Conformal prediction (split conformal with 100 random calibrations)
    # ----------------------------------------------------------------------
    print(f"\n[2] Split conformal prediction (coverage target = {100*(1-ALPHA)}%, {N_CONFORMAL_REPEATS} repetitions)...")

    seed_archs = []
    for seed in range(5):
        gs = GridSearchCV(
            unified_pipe, unified_grid,
            cv=StratifiedKFold(3, shuffle=True, random_state=seed),
            scoring=cv_scoring, n_jobs=1
        ).fit(X_dev, y_dev)
        seed_archs.append(gs.best_params_['feature_opt__method'])
    print(f"    Winner's curse check (5 seeds): {dict(Counter(seed_archs))}")

    best_cp_model = GridSearchCV(
        unified_pipe, unified_grid,
        cv=inner_cv, scoring=cv_scoring, n_jobs=1
    ).fit(X_dev, y_dev).best_estimator_

    empty_counts, singleton_counts, full_counts, coverage_rates = [], [], [], []

    for rep in range(N_CONFORMAL_REPEATS):
        X_train_cal, X_test_cp, y_train_cal, y_test_cp = train_test_split(
            X_dev, y_dev,
            test_size=0.2,
            random_state=rep,
            stratify=y_dev
        )
        X_train_cp, X_cal_cp, y_train_cp, y_cal_cp = train_test_split(
            X_train_cal, y_train_cal,
            test_size=0.25,
            random_state=rep,
            stratify=y_train_cal
        )

        best_cp_model.fit(X_train_cp, y_train_cp)
        cal_probs = best_cp_model.predict_proba(X_cal_cp)[np.arange(len(y_cal_cp)), y_cal_cp]
        q_hat = np.quantile(
            1 - cal_probs,
            np.ceil((len(y_cal_cp) + 1) * (1 - ALPHA)) / len(y_cal_cp)
        )
        pred_sets = (1 - best_cp_model.predict_proba(X_test_cp)) <= q_hat

        coverage_rates.append(pred_sets[np.arange(len(y_test_cp)), y_test_cp].mean())
        sizes = pred_sets.sum(axis=1)
        empty_counts.append((sizes == 0).sum())
        singleton_counts.append((sizes == 1).sum())
        full_counts.append((sizes == 2).sum())

    print(f" -> Set sizes (mean): Empty={np.mean(empty_counts):.1f}, Certain={np.mean(singleton_counts):.1f}, Uncertain={np.mean(full_counts):.1f}")
    print(f" -> Empirical coverage: {np.mean(coverage_rates)*100:.2f}% (target {100*(1-ALPHA)}%)")

    # Compute percentages of empty/singleton/ambiguous sets and save summary
    n_test_cp = len(y_test_cp)   # constant because test_size=0.2 each iteration
    pct_empty = 100 * np.mean(empty_counts) / n_test_cp
    pct_singleton = 100 * np.mean(singleton_counts) / n_test_cp
    pct_ambiguous = 100 * np.mean(full_counts) / n_test_cp

    print(f" -> Empty sets:     {pct_empty:.2f}%")
    print(f" -> Singleton sets: {pct_singleton:.2f}%")
    print(f" -> Ambiguous sets: {pct_ambiguous:.2f}%")

    df_conformal_summary = pd.DataFrame([{
        'Dataset': dataset_name,
        'Empty_pct': round(pct_empty, 2),
        'Singleton_pct': round(pct_singleton, 2),
        'Ambiguous_pct': round(pct_ambiguous, 2),
        'Coverage_pct_mean': round(100 * np.mean(coverage_rates), 2),
        'Coverage_pct_std': round(100 * np.std(coverage_rates), 2),
        'N_test_approx': n_test_cp,
    }])
    df_conformal_summary.to_csv(
        os.path.join(output_dir, f"{dataset_name}_conformal_summary.csv"), index=False
    )

    # ------------------------------------------------------
    # 9. Sensitivity analyses (sample size and n_components)
    # ------------------------------------------------------
    print("\n[3] Sensitivity analyses...")
    sample_size_sensitivity(
        X_dev, y_dev, dataset_name, best_cp_model, output_dir,
        n_bootstrap=N_BOOTSTRAP_SENSITIVITY
    )
    n_components_sensitivity(
        X_dev, y_dev, dataset_name, output_dir,
        n_features, n_components, cv_scoring=cv_scoring
    )

    # ---------------------------------------------------
    # 10. Final evaluation on the true holdout lockbox
    # ---------------------------------------------------
    print("\n[4] Evaluating on independent holdout lockbox (15%)...")
    X_final_train, X_final_cal, y_final_train, y_final_cal = train_test_split(
        X_dev, y_dev,
        test_size=0.2,
        stratify=y_dev,
        random_state=SEED
    )
    best_cp_model.fit(X_final_train, y_final_train)

    holdout_preds = best_cp_model.predict(X_holdout)
    holdout_probs_full = best_cp_model.predict_proba(X_holdout)
    holdout_probs = holdout_probs_full[:, 1]

    bal_acc = balanced_accuracy_score(y_holdout, holdout_preds)
    tn, fp, fn, tp = confusion_matrix(y_holdout, holdout_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    acc_ci_low, acc_ci_high = proportion_confint(
        np.sum(y_holdout == holdout_preds),
        len(y_holdout),
        method='wilson'
    )

    cal_true_final = best_cp_model.predict_proba(X_final_cal)[np.arange(len(y_final_cal)), y_final_cal]
    q_hat_holdout = np.quantile(
        1 - cal_true_final,
        np.ceil((len(y_final_cal) + 1) * (1 - ALPHA)) / len(y_final_cal)
    )
    holdout_pred_sets = (1 - holdout_probs_full) <= q_hat_holdout
    cov_ci_low, cov_ci_high = proportion_confint(
        np.sum(holdout_pred_sets[np.arange(len(y_holdout)), y_holdout]),
        len(y_holdout),
        method='wilson'
    )

    ho_cal_slope, ho_cal_intercept = calculate_calibration_metrics(y_holdout, holdout_probs)

    print(f" -> Holdout Accuracy:         {accuracy_score(y_holdout, holdout_preds)*100:.2f}% [95% CI: {acc_ci_low*100:.1f}% - {acc_ci_high*100:.1f}%]")
    print(f" -> Holdout Balanced Acc:     {bal_acc*100:.2f}%")
    print(f" -> Holdout Sensitivity:      {sensitivity*100:.2f}%")
    print(f" -> Holdout Specificity:      {specificity*100:.2f}%")
    print(f" -> Holdout AUROC:            {roc_auc_score(y_holdout, holdout_probs):.3f}")
    print(f" -> Holdout Brier:            {brier_score_loss(y_holdout, holdout_probs):.3f}")
    print(f" -> Holdout Calibration:      Slope={ho_cal_slope:.3f}, Intercept={ho_cal_intercept:.3f}")
    print(f" -> Holdout CP Coverage:      {holdout_pred_sets[np.arange(len(y_holdout)), y_holdout].mean()*100:.2f}% [95% CI: {cov_ci_low*100:.1f}% - {cov_ci_high*100:.1f}%]")
    print(f" -> Holdout Confusion:        [TN={tn}, FP={fp}] | [FN={fn}, TP={tp}]")

    # Save holdout lockbox metrics (Table 4 in the manuscript)
    df_holdout_summary = pd.DataFrame([{
        'Dataset': dataset_name,
        'N_holdout': len(y_holdout),
        'Accuracy_pct': round(accuracy_score(y_holdout, holdout_preds) * 100, 2),
        'Accuracy_CI_low_pct': round(acc_ci_low * 100, 2),
        'Accuracy_CI_high_pct': round(acc_ci_high * 100, 2),
        'Balanced_Accuracy_pct': round(bal_acc * 100, 2),
        'Sensitivity_pct': round(sensitivity * 100, 2),
        'Specificity_pct': round(specificity * 100, 2),
        'TP': int(tp), 'FN': int(fn), 'TN': int(tn), 'FP': int(fp),
        'AUROC': round(roc_auc_score(y_holdout, holdout_probs), 3),
        'Brier': round(brier_score_loss(y_holdout, holdout_probs), 3),
        'Calibration_Slope': round(ho_cal_slope, 3),
        'Calibration_Intercept': round(ho_cal_intercept, 3),
        'Conformal_Coverage_pct': round(
            holdout_pred_sets[np.arange(len(y_holdout)), y_holdout].mean() * 100, 2
        ),
        'Conformal_Coverage_CI_low_pct': round(cov_ci_low * 100, 2),
        'Conformal_Coverage_CI_high_pct': round(cov_ci_high * 100, 2),
    }])
    df_holdout_summary.to_csv(
        os.path.join(output_dir, f"{dataset_name}_holdout_lockbox_metrics.csv"), index=False
    )

    # ----------------------------------------------------------------------
    # 11. SHAP interpretability on holdout
    # ----------------------------------------------------------------------
    print("\n[5] SHAP interpretability on holdout...")
    shap_interpretability(
        best_cp_model,
        X_final_train,
        X_holdout,
        feature_names,
        dataset_name,
        output_dir
    )

    # -----------------------------------------------------------------
    # 12. Generate main result figures
    # -----------------------------------------------------------------
    print("\n[6] Generating main figures...")

    thresholds = np.linspace(0.01, 0.99, 100)
    nb_model = net_benefit(oof_y_true_np, oof_prob_np, thresholds)
    prevalence = np.mean(oof_y_true_np)
    nb_treat_all = [max(prevalence - (1 - prevalence) * (pt / (1 - pt)), 0) for pt in thresholds]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    plot_data = [
        test_leaky, test_xgb, test_lgbm,
        test_pca, test_rf, test_uni
    ]
    metric_label = 'AUROC' if cv_scoring == 'roc_auc' else 'Accuracy'
    sns.boxplot(
        data=plot_data,
        ax=axes[0],
        palette=['#E74C3C', '#F39C12', '#F1C40F', '#3498DB', '#27AE60', '#9B59B6']
    )
    labels = []
    for arr in plot_data:
        mean_val = np.mean(arr)
        if cv_scoring == 'roc_auc':
            labels.append(f'{mean_val:.3f}')
        else:
            labels.append(f'{mean_val*100:.1f}%')
    axes[0].set_xticklabels([
        f'Leaky\n({labels[0]})',
        f'XGBoost\n({labels[1]})',
        f'LightGBM\n({labels[2]})',
        f'PCA\n({labels[3]})',
        f'RF\n({labels[4]})',
        f'Unified\n({labels[5]})'
    ], fontsize=10)
    axes[0].set_ylabel(metric_label, fontweight='bold')
    axes[0].set_title(f'A. Benchmarking Framework ({dataset_name})', fontweight='bold', pad=15)

    sns.barplot(
        x=['Empty', 'Certain (Size 1)', 'Uncertain (Size 2)'],
        y=[np.mean(empty_counts), np.mean(singleton_counts), np.mean(full_counts)],
        ax=axes[1],
        palette='Blues'
    )
    axes[1].errorbar(
        x=[0, 1, 2],
        y=[np.mean(empty_counts), np.mean(singleton_counts), np.mean(full_counts)],
        yerr=[np.std(empty_counts), np.std(singleton_counts), np.std(full_counts)],
        fmt='none', c='black', capsize=8, elinewidth=2
    )
    axes[1].set_ylabel('Mean number of patients', fontweight='bold')
    axes[1].set_title('B. Conformal Prediction Sets (Dev)', fontweight='bold', pad=15)

    axes[2].plot(thresholds, nb_model, label='Unified model (CV probs)', color='blue', linewidth=3)
    axes[2].plot(thresholds, nb_treat_all, label='Treat all', color='gray', linestyle='--', linewidth=2)
    axes[2].plot(thresholds, np.zeros_like(thresholds), label='Treat none', color='black', linewidth=2)
    axes[2].set_xlim([0, 1.0])
    y_max = max(max(nb_model), max(nb_treat_all)) + 0.1
    axes[2].set_ylim([0, y_max])
    axes[2].set_xlabel('Threshold probability', fontweight='bold')
    axes[2].set_ylabel('Net benefit', fontweight='bold')
    axes[2].set_title('C. Decision Curve Analysis (Pooled CV)', fontweight='bold', pad=15)
    axes[2].legend()

    plt.tight_layout(pad=3.0)
    plt.savefig(
        os.path.join(output_dir, f'{dataset_name}_fig_main.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 14))

    fpr, tpr, _ = roc_curve(oof_y_true_np, oof_prob_np)
    axes2[0, 0].plot(fpr, tpr, color='darkorange', lw=3, label=f'Unified (AUC = {auc(fpr, tpr):.3f})')
    axes2[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes2[0, 0].set_xlim([0.0, 1.0])
    axes2[0, 0].set_ylim([0.0, 1.05])
    axes2[0, 0].set_title('A. Cross-validated ROC (Pooled OOF)', fontweight='bold')
    axes2[0, 0].legend(loc='lower right')

    precision, recall, _ = precision_recall_curve(oof_y_true_np, oof_prob_np)
    axes2[0, 1].plot(
        recall, precision,
        color='purple', lw=3,
        label=f'Unified (AP = {average_precision_score(oof_y_true_np, oof_prob_np):.3f})'
    )
    axes2[0, 1].set_xlim([0.0, 1.0])
    axes2[0, 1].set_ylim([0.0, 1.05])
    axes2[0, 1].set_title('B. Cross-validated Precision-Recall', fontweight='bold')
    axes2[0, 1].legend(loc='lower left')

    prob_true, prob_pred = calibration_curve(oof_y_true_np, oof_prob_np, n_bins=10, strategy='uniform')
    axes2[1, 0].plot(prob_pred, prob_true, marker='s', markersize=8, color='green', linewidth=2, label=f'ECE: {cv_ece:.3f}')
    axes2[1, 0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes2[1, 0].set_title('C. Calibration Curve (Reliability Diagram)', fontweight='bold')
    axes2[1, 0].legend(loc='upper left')

    rf_chosen_times = arch_dist.get('rf', 0)
    if rf_chosen_times > 0:
        top_idx = np.argsort(feat_counts_rf)[::-1][:10]
        sns.barplot(
            x=feat_counts_rf[top_idx],
            y=np.array(feature_names)[top_idx],
            ax=axes2[1, 1],
            palette='viridis'
        )
        axes2[1, 1].set_xlabel(f'Selection frequency (max = {rf_chosen_times})', fontweight='bold')
        axes2[1, 1].set_title('D. Feature Selection Stability\n(Analyzed across RF choices)', fontweight='bold')
    else:
        axes2[1, 1].text(0.5, 0.5, 'RF was never selected by the unified pipeline', ha='center', va='center', fontsize=12)
        axes2[1, 1].set_title('D. Feature Selection Stability', fontweight='bold')

    plt.tight_layout(pad=4.0)
    plt.savefig(
        os.path.join(output_dir, f'{dataset_name}_fig_clinical.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig2)

    print(f" Finished {dataset_name} in {time.time() - start_time:.2f} seconds.")


# --- Survival analysis for SEER (optional) ---

def run_survival_analysis(raw_seer_df, time_col, target_col, output_dir):
    """Run the Weibull AFT survival analysis on the SEER cohort."""
    if not LIFELINES_AVAILABLE:
        print("[!] lifelines not installed; skipping survival branch.")
        return

    print("\n" + "=" * 70)
    print("[Survival Branch] Running penalized Weibull AFT model...")
    print("=" * 70)

    df_surv = raw_seer_df.copy()
    df_surv['event'] = np.where(df_surv[target_col] == 'dead', 1, 0)
    df_surv = df_surv.drop(columns=[target_col, 'patient_id', 'id'], errors='ignore')

    num_cols = df_surv.select_dtypes(include=np.number).columns
    df_surv[num_cols] = df_surv[num_cols].fillna(df_surv[num_cols].median())
    cat_cols = df_surv.select_dtypes(exclude=np.number).columns
    df_surv[cat_cols] = df_surv[cat_cols].fillna('Unknown')
    df_surv = pd.get_dummies(df_surv, drop_first=True)

    df_surv_train, df_surv_test = train_test_split(
        df_surv,
        test_size=0.2,
        random_state=SEED,
        stratify=df_surv['event']
    )

    aft = WeibullAFTFitter(penalizer=0.1)
    aft.fit(df_surv_train, duration_col=time_col, event_col='event')

    test_preds = aft.predict_expectation(df_surv_test)
    if np.isinf(test_preds).sum() > 0:
        test_preds = test_preds.replace([np.inf, -np.inf], test_preds[~np.isinf(test_preds)].max() * 1.5)

    c_index = concordance_index(
        df_surv_test[time_col],
        test_preds,
        df_surv_test['event']
    )
    print(f" -> AFT C-index on test set: {c_index:.3f}")

    train_preds = aft.predict_expectation(df_surv_train)
    if np.isinf(train_preds).sum() > 0:
        train_preds = train_preds.replace([np.inf, -np.inf], train_preds[~np.isinf(train_preds)].max() * 1.5)
    median_train = train_preds.median()
    high_risk_mask = test_preds < median_train

    fig_km, ax_km = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=df_surv_test[~high_risk_mask][time_col],
        event_observed=df_surv_test[~high_risk_mask]['event'],
        label='Low Risk (Predicted >= median)'
    )
    kmf.plot_survival_function(ax=ax_km, color='green', ci_show=True)
    kmf.fit(
        durations=df_surv_test[high_risk_mask][time_col],
        event_observed=df_surv_test[high_risk_mask]['event'],
        label='High Risk (Predicted < median)'
    )
    kmf.plot_survival_function(ax=ax_km, color='red', ci_show=True)

    lr_test = logrank_test(
        df_surv_test[high_risk_mask][time_col],
        df_surv_test[~high_risk_mask][time_col],
        event_observed_A=df_surv_test[high_risk_mask]['event'],
        event_observed_B=df_surv_test[~high_risk_mask]['event']
    )
    ax_km.set_title("Kaplan-Meier Risk Stratification (Test Set)", fontweight='bold')
    ax_km.set_xlabel("Survival Time (Months)", fontweight='bold')
    ax_km.set_ylabel("Survival Probability", fontweight='bold')
    ax_km.text(
        0.05, 0.05,
        f"Log-Rank p = {lr_test.p_value:.1e}",
        transform=ax_km.transAxes,
        bbox=dict(facecolor='white', alpha=0.8),
        fontweight='bold'
    )
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "SEER_AFT_KM_Stratification.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig_km)
    print(f" -> Kaplan-Meier log-rank p-value: {lr_test.p_value:.2e}")


# ---------------------------------------------------------------------------
# Main execution: load data and run the pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("results", f"run_{run_timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"[*] Global output directory: {base_output_dir}")

    # WBCD (Wisconsin Breast Cancer)
    print("\n>>> Loading WBCD from scikit-learn...")
    wbcd = load_breast_cancer()
    X_wbcd = pd.DataFrame(wbcd.data, columns=wbcd.feature_names)
    # scikit-learn target: 0=malignant, 1=benign -> we want 1=malignant
    run_trust_bc_pipeline(
        X_wbcd,
        pd.Series(1 - wbcd.target),
        "WBCD",
        base_output_dir,
        cv_scoring='accuracy',
        tost_margin=0.02
    )

    # Coimbra (metabolic biomarkers)
    print("\n>>> Loading Coimbra from UCI...")
    try:
        coimbra_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00451/dataR2.csv"
        raw_data = urllib.request.urlopen(coimbra_url).read()
        sha_hash = hashlib.sha256(raw_data).hexdigest()
        print(f"    Coimbra SHA-256: {sha_hash}")
        coimbra_df = pd.read_csv(BytesIO(raw_data))
        # Classification: 1=Healthy, 2=Cancer -> map to 0/1
        y_coimbra = pd.Series(np.where(coimbra_df['Classification'] == 2, 1, 0))
        run_trust_bc_pipeline(
            coimbra_df.drop(columns=['Classification']),
            y_coimbra,
            "Coimbra",
            base_output_dir,
            cv_scoring='accuracy',
            tost_margin=0.02
        )
    except Exception as e:
        print(f"Failed to load Coimbra: {e}")

    # SEER (population registry)
    print("\n>>> Loading SEER dataset...")
    seer_candidates = glob.glob("*SEER*.csv") or glob.glob("*seer*.csv")
    seer_path = seer_candidates[0] if seer_candidates else "SEER.csv"

    if not os.path.exists(seer_path):
        try:
            from google.colab import files
            print("\n[Upload SEER file]")
            uploaded = files.upload()
            if uploaded:
                seer_path = list(uploaded.keys())[0]
        except ImportError:
            pass

    if os.path.exists(seer_path):
        try:
            raw_seer_df = pd.read_csv(seer_path, low_memory=False)
            raw_seer_df.columns = raw_seer_df.columns.str.strip()
            raw_seer_df = raw_seer_df.drop(columns=['Unnamed: 3'], errors='ignore')

            target_col = 'Status'
            time_cols = [col for col in raw_seer_df.columns if 'survival months' in col.lower()]

            if target_col in raw_seer_df.columns and len(time_cols) > 0:
                time_col_exact = time_cols[0]

                print("    Dropping '6th Stage' to reduce collinearity (preserving T/N stage).")
                raw_seer_df = raw_seer_df.drop(columns=['6th Stage'], errors='ignore')

                ordinal_maps = {
                    'T Stage': {'T0': 0, 'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4},
                    'N Stage': {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3},
                    'Grade': {
                        'Well differentiated; Grade I': 1,
                        'Moderately differentiated; Grade II': 2,
                        'Poorly differentiated; Grade III': 3,
                        'Undifferentiated; anaplastic; Grade IV': 4
                    }
                }
                for col, mapping in ordinal_maps.items():
                    if col in raw_seer_df.columns:
                        raw_seer_df[col] = raw_seer_df[col].map(mapping)

                raw_seer_df[target_col] = raw_seer_df[target_col].astype(str).str.strip().str.lower()
                raw_seer_df = raw_seer_df[raw_seer_df[target_col].isin(['alive', 'dead'])]
                raw_seer_df[time_col_exact] = pd.to_numeric(raw_seer_df[time_col_exact], errors='coerce')
                raw_seer_df = raw_seer_df.dropna(subset=[target_col, time_col_exact]).reset_index(drop=True)

                if LIFELINES_AVAILABLE:
                    run_survival_analysis(
                        raw_seer_df,
                        time_col_exact,
                        target_col,
                        os.path.join(base_output_dir, "SEER")
                    )
                else:
                    print("[!] lifelines not installed; skipping survival branch.")

                print("\n" + "=" * 70)
                print("[Classification Branch] Filtering right-censored patients...")
                print("=" * 70)

                valid_mask = (raw_seer_df[target_col] == 'dead') | (
                    (raw_seer_df[target_col] == 'alive') & (raw_seer_df[time_col_exact] >= 60)
                )
                excluded = (~valid_mask).sum()
                print(f"    Excluded censored <60mo: {excluded} of {len(raw_seer_df)} patients.")
                print(f"    Final classification cohort: {valid_mask.sum()} patients.")

                seer_class_df = raw_seer_df[valid_mask].reset_index(drop=True)
                # v1.1 fix: fixed-horizon 5-year label. In v1.0 every death was labelled 1,
                # including 158 deaths that occurred after month 60 (patients alive at 5 years).
                # Set SEER_LABEL_MODE = 'as_reported' to reproduce the v1.0 results exactly.
                if SEER_LABEL_MODE == 'as_reported':
                    y_seer = pd.Series(np.where(seer_class_df[target_col] == 'dead', 1, 0))
                else:
                    y_seer = pd.Series(np.where((seer_class_df[target_col] == 'dead') &
                                                (seer_class_df[time_col_exact] <= 60), 1, 0))
                n_late = int(((seer_class_df[target_col] == 'dead') & (seer_class_df[time_col_exact] > 60)).sum())
                print(f"    Label mode: {SEER_LABEL_MODE}; deaths after month 60 (5-year survivors): {n_late}")
                print(f"    5-year events: {int(y_seer.sum())}")

                leakage_cols = [time_col_exact, 'patient_id', 'id']
                seer_class_df = seer_class_df.drop(columns=leakage_cols + [target_col], errors='ignore')

                num_cols = seer_class_df.select_dtypes(include=np.number).columns
                seer_class_df[num_cols] = seer_class_df[num_cols].fillna(seer_class_df[num_cols].median())
                cat_cols = seer_class_df.select_dtypes(exclude=np.number).columns
                seer_class_df[cat_cols] = seer_class_df[cat_cols].fillna('Unknown')

                # T/N Stage and Grade already mapped above; encode the rest (Race, Marital Status, etc.)
                n_raw_predictors = seer_class_df.shape[1]
                seer_class_df = pd.get_dummies(seer_class_df, drop_first=True)
                print(f"    Encoded {n_raw_predictors} raw predictors into {seer_class_df.shape[1]} "
                      f"numeric columns (one-hot encoding of categorical variables).")

                run_trust_bc_pipeline(
                    seer_class_df,
                    y_seer,
                    "SEER",
                    base_output_dir,
                    n_repeats=7,
                    cv_scoring='roc_auc',
                    tost_margin=0.01
                )

            else:
                print(f"[!] Target column '{target_col}' or survival time column not found.")

        except Exception as e:
            print(f"Failed to process SEER: {e}")

    else:
        print("[!] SEER file not found; skipping SEER evaluation.")

    try:
        import subprocess
        with open(os.path.join(base_output_dir, "requirements.txt"), "w") as f:
            subprocess.run(['pip', 'freeze'], stdout=f)
        print("\n[*] Environment packages saved to requirements.txt")
    except Exception:
        pass

    print("\n=== TRUST-BC pipeline completed successfully ===")