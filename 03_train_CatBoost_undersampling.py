#!/usr/bin/env python3
"""
03_train_and_select_catboost_gpu_2026.py

Model training + model selection using CatBoost with heavy resampling,
and a direct comparison between:

  (A) a Bayesian-optimized CatBoost model (BayesSearchCV over hyperparameters)
  (B) an “out-of-the-box” CatBoost model (CatBoost defaults, with only a few safe settings)

The script trains BOTH candidates, evaluates BOTH, then picks ONE winner and saves ONLY:
  OUT_DIR/best_model.cbm

This file is deliberately “single-script pipeline style”:
- it reads prepared train/test CSVs
- it does an internal split inside train.csv into train/validation
- it searches hyperparameters only on the internal training portion
- it compares optimized vs out-of-the-box based on validation performance
- it evaluates both on the independent test.csv for reporting
- it writes JSON + a human-readable TXT report

Plain meaning summary (“this means…”):
- train.csv is not directly the final training set: we split it again into X_tr and X_val.
  This means we keep some data aside (X_val) to choose between models without touching test.csv.
- BayesSearchCV runs only on X_tr (after balancing with undersampling).
  This means the hyperparameter search is not allowed to “peek” at validation or test.
- Both final candidates (best-bayes and ootb) are trained on X_tr_r (balanced X_tr).
  This means both candidates get the same training signal for a fair comparison.
- The winner is chosen using repeated balanced evaluation on X_val.
  This means we compare models under a balanced class distribution (reduces bias from imbalance).
- The test set is never used to choose the winner—only to report final performance.
  This means the final numbers on test are a more honest estimate of generalization.

UPDATED (requested):
- Save ONLY ONE model as best_model.cbm (winner is either bayes_best or ootb)
- Keep everything else as close as possible

IMPORTANT IMPLEMENTATION DETAIL:
- CatBoost can treat object/string columns as categorical, but you must tell it which columns.
  This means we detect categorical columns (object/string/category), convert them to string,
  and pass their indices to CatBoost as cat_features, preventing common dtype-related crashes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Bayesian optimization:
# BayesSearchCV tries different hyperparameter settings using Bayesian optimization
# rather than a grid or random search. This means it attempts to be more efficient
# by learning which hyperparameters tend to work well as it goes.
from skopt import BayesSearchCV
from skopt.space import Integer, Real


# =========================
# CONFIG (EDIT THESE)
# =========================

TARGET = "isBugPresent"
RANDOM_STATE = 42

# These CSVs are produced by the earlier split script.
# train.csv and test.csv here already represent the outer split of your pipeline:
# - train.csv is used for training AND internal validation split
# - test.csv is held out until the end for final evaluation
TRAIN_CSV = Path(f"data_{TARGET}") / f"splits_{str(RANDOM_STATE)}" / "train.csv"
TEST_CSV  = Path(f"data_{TARGET}") / f"splits_{str(RANDOM_STATE)}" / "test.csv"

# All outputs of this training step are written here.
OUT_DIR = Path(f"data_{TARGET}") / f"models_{str(RANDOM_STATE)}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Manual features (recommended). If None, infer = all columns except EXCLUDE_COLS.
# This means:
# - If you specify FEATURES explicitly, only those columns are used as model inputs.
# - If FEATURES is None, we auto-include everything except the excluded columns.
FEATURES: Optional[List[str]] = None

# Columns to exclude from auto-feature inference.
# Right now, only TARGET is excluded. The commented lines below show typical metadata
# you might exclude if those columns exist. This means you control whether identifiers
# are allowed as features.
EXCLUDE_COLS = [
    TARGET,
#    "hash", "date", "fileId", "fileName",
#    "source_file", "source_project",
#    "authorId",
]

# Internal validation fraction from train.csv:
# train.csv -> split into (X_tr, X_val)
# This means:
# - X_tr is the portion used to fit models / do Bayesian search
# - X_val is the portion used to compare Bayes-optimized vs out-of-the-box
VAL_SIZE = 0.2
STRATIFY = True  # means keep class ratio similar in X_tr and X_val

# Heavy resampling setup:
# We use RandomUnderSampler to balance classes by removing majority samples.
# This means:
# - The model sees a balanced training set (less biased toward majority class).
# - Balanced repeated evaluation approximates “how the model behaves when classes are equal”.
N_REPEATS = 1 #HERE VERY IMPORTANT TO INCREASE FOR PROPER RUNS
SAMPLING_STRATEGY = 1.0   # 1.0 => fully balanced (minority == majority)

# Selection metric (for Bayes vs OOTB comparison) on VALIDATION.
# We pick the winner by balanced F1 on repeatedly balanced validation samples.
# This means:
# - We focus on F1, not accuracy, which is more meaningful for imbalance.
SELECTION_METRIC = "val_balanced_f1"

# CatBoost GPU settings (note: in your build_model/build_ootb_model GPU is commented out).
# This means:
# - BayesSearchCV base estimator is set to GPU here, but the final training calls use
#   build_model/build_ootb_model where GPU is currently commented out.
# If you want GPU everywhere, uncomment task_type/devices inside build_model/build_ootb_model.
CATBOOST_GPU_DEVICE = "0"
VERBOSE = 1

# OOTB model settings:
# These are additional parameters layered on top of CatBoost defaults.
# This means:
# - Leave empty to use “true defaults”
# - Add safe settings here if you need them (e.g., class_weights, loss_function, etc.)
OOTB_EXTRA_PARAMS: Dict = {}

# Bayesian search setup:
# BAYES_N_ITER controls how many hyperparameter candidates BayesSearchCV evaluates.
# BAYES_CV is the internal cross-validation folds inside BayesSearchCV.
# This means:
# - BayesSearchCV itself splits X_tr_r into folds to score candidates.
# - That internal CV is separate from our outer validation split X_val.
BAYES_N_ITER = 2 #HERE VERY IMPORTANT TO INCREASE FOR PROPER RUNS
BAYES_CV = 3
BAYES_SCORING = "f1"
BAYES_N_JOBS = 1  # IMPORTANT: avoid GPU fights / avoid over-parallelizing CatBoost training

# Search space (adjust freely):
# These ranges define what BayesSearchCV is allowed to try.
# This means:
# - Larger ranges -> potentially better models but search becomes harder.
# - More iterations -> better chance to find good params but more runtime.
BAYES_SEARCH_SPACE = {
    "depth": Integer(4, 12),
    "iterations": Integer(500, 8000),
    "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
    "l2_leaf_reg": Real(1.0, 20.0, prior="log-uniform"),
    "border_count": Integer(32, 255),
}

# Early stopping:
# During the “final” training of the two candidates, we provide eval_set=(X_val, y_val).
# This means:
# - CatBoost can stop early if it stops improving on the validation set.
EARLY_STOPPING_ROUNDS = 100

# NEW: skip Bayesian hyperparameter tuning and run ONLY out-of-the-box CatBoost.
# This means:
# - BayesSearchCV is not executed at all.
# - The script still produces the same reports, but “bayes_best” is treated as the ootb run.
SKIP_BAYES_OPTIMIZATION = True


# =========================
# Helpers
# =========================

def infer_features(df: pd.DataFrame) -> List[str]:
    """
    Infer feature columns as: all columns except those listed in EXCLUDE_COLS.

    This means:
    - If EXCLUDE_COLS contains only TARGET, then every other column becomes a feature.
    - If you uncomment the metadata/id columns in EXCLUDE_COLS, those are removed from features.
    """
    return [c for c in df.columns if c not in set(EXCLUDE_COLS)]


def ensure_binary_target(y: pd.Series) -> pd.Series:
    """
    Convert a boolean-like target to integer 0/1.

    This means:
    - bool dtype becomes {False->0, True->1}
    - string-like variants ("true"/"false"/"0"/"1") are mapped to {0,1}
    - otherwise, we leave the series as-is (but later cast to int in main)

    Purpose:
    - CatBoost and sklearn metrics expect numeric labels for binary classification.
    """
    if y.dtype == bool:
        return y.astype(int)
    if y.dtype == object:
        lowered = y.astype(str).str.strip().str.lower()
        if set(lowered.dropna().unique()).issubset({"true", "false", "0", "1"}):
            return lowered.map({"true": 1, "false": 0, "1": 1, "0": 0}).astype("Int64")
    return y


def get_cat_feature_indices(X: pd.DataFrame) -> List[int]:
    """
    Return integer indices of categorical columns, identified by pandas dtypes:
    object, string, category.

    This means:
    - We do not guess categories from numeric columns.
    - Only columns that are explicitly non-numeric (or category dtype) are treated as categorical.

    CatBoost expects categorical columns to be specified via indices if you pass a numpy/pandas matrix.
    """
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return [X.columns.get_loc(c) for c in cat_cols]


def coerce_categoricals_to_str(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical-like columns to string.

    This means:
    - Any object/string/category column is cast to str.
    - Missing values become 'nan' string after cast, which CatBoost can still handle as category tokens.

    Why:
    - Mixed object columns can contain numbers + strings; explicit str conversion avoids dtype conflicts.
    - This is a common fix for CatBoost errors that occur when pandas object columns are inconsistent.
    """
    X2 = X.copy()
    cat_cols = X2.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in cat_cols:
        X2[c] = X2[c].astype(str)
    return X2


def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, float]:
    """
    Compute a small set of standard binary classification metrics.

    This means:
    - acc: standard accuracy
    - bacc: balanced accuracy (average recall across classes)
    - precision/recall/f1: computed for positive class (label 1)
    - auc: ROC-AUC based on predicted probabilities (if possible)

    ROC-AUC may fail if only one class is present in y_true.
    This means we catch exceptions and store NaN.
    """
    out: Dict[str, float] = {}
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["bacc"] = float(balanced_accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    if y_proba is not None:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out["auc"] = float("nan")
    else:
        out["auc"] = float("nan")
    return out


def eval_on(dfX: pd.DataFrame, dfy: pd.Series, model: CatBoostClassifier) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate a model on a dataset (X,y).

    This means:
    - We use model.predict_proba to get probabilities for class 1.
    - We threshold at 0.5 to produce predicted labels.
    - We compute metrics and a 2x2 confusion matrix with labels [0,1].

    Output:
    - metrics dict
    - confusion matrix (numpy array shape (2,2))
    """
    proba = model.predict_proba(dfX)[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = metrics_binary(dfy.to_numpy(), pred, proba)
    cm = confusion_matrix(dfy.to_numpy(), pred, labels=[0, 1])
    return m, cm


def undersample(X: pd.DataFrame, y: pd.Series, seed: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Randomly undersample the majority class to achieve the desired class ratio.

    This means:
    - If SAMPLING_STRATEGY == 1.0, the result has equal numbers of class 0 and class 1.
    - If SAMPLING_STRATEGY < 1.0, the minority will be smaller than the majority, but still closer.

    We preserve DataFrame/Series types when possible so column names remain intact.
    """
    rus = RandomUnderSampler(sampling_strategy=SAMPLING_STRATEGY, random_state=seed)
    Xr, yr = rus.fit_resample(X, y)
    if not isinstance(Xr, pd.DataFrame):
        Xr = pd.DataFrame(Xr, columns=X.columns)
    if not isinstance(yr, pd.Series):
        yr = pd.Series(yr, name=y.name)
    return Xr, yr


def build_model(params: Dict, seed: int, cat_feature_indices: List[int]) -> CatBoostClassifier:
    """
    Build a CatBoost model with specified hyperparameters (params).

    This means:
    - params comes from Bayesian optimization (best_params).
    - we always set loss_function and eval_metric explicitly.
    - we pass cat_features indices so CatBoost knows which columns are categorical.

    NOTE:
    - GPU settings are currently commented out in this function.
      This means final training here will run on CPU unless you uncomment them.
    """
    return CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        #task_type="GPU", #GPU
        #devices=CATBOOST_GPU_DEVICE,
        allow_writing_files=False,
        verbose=VERBOSE,
        cat_features=cat_feature_indices if len(cat_feature_indices) > 0 else None,
    )


def build_ootb_model(seed: int, cat_feature_indices: List[int]) -> CatBoostClassifier:
    """
    Build an “out-of-the-box” CatBoost model (defaults + minimal safe overrides).

    This means:
    - We do not set depth/learning_rate/etc. (unless you add them via OOTB_EXTRA_PARAMS).
    - We still pass cat_features indices to avoid categorical dtype errors.
    - We set random_seed for reproducibility.

    NOTE:
    - GPU settings are currently commented out here too.
      This means ootb training here will run on CPU unless you uncomment them.
    """
    return CatBoostClassifier(
        #task_type="GPU", #GPU
        #devices=CATBOOST_GPU_DEVICE,
        allow_writing_files=False,
        verbose=VERBOSE,
        random_seed=seed,
        cat_features=cat_feature_indices if len(cat_feature_indices) > 0 else None,
        **OOTB_EXTRA_PARAMS,
    )


def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute per-key mean across a list of metric dictionaries.

    This means:
    - If a key is missing in some dict, it is treated as NaN for that dict.
    - We use nanmean so NaNs do not break aggregation.
    """
    keys = sorted({k for d in dicts for k in d.keys()})
    return {k: float(np.nanmean([d.get(k, np.nan) for d in dicts])) for k in keys}


def sum_confusions(cms: List[np.ndarray]) -> List[List[int]]:
    """
    Sum confusion matrices over multiple runs.

    This means:
    - if you run repeated evaluation N_REPEATS times, this returns the element-wise sum
      of all N confusion matrices (counts add up).
    """
    s = np.sum(np.stack(cms, axis=0), axis=0)
    return s.astype(int).tolist()


def repeated_eval_balanced(X: pd.DataFrame, y: pd.Series, model: CatBoostClassifier, seed0: int) -> Dict:
    """
    Repeated balanced evaluation:
    - For each repetition:
        1) undersample (X,y) to get a balanced dataset
        2) evaluate model on that balanced dataset
    - Aggregate by:
        - mean of metrics across repetitions
        - sum of confusion matrices across repetitions

    This means:
    - We simulate performance under balanced class distributions.
    - We reduce variance by averaging across multiple random undersamples.
    - The “metrics_mean” is what we use to compare models on validation.
    """
    mets = []
    cms = []
    for r in range(N_REPEATS):
        seed = seed0 + r
        Xb, yb = undersample(X, y, seed=seed + 20_000)
        m, cm = eval_on(Xb, yb, model)
        mets.append(m)
        cms.append(cm)
    return {
        "metrics_mean": mean_dict(mets),
        "confusion_matrix_sum": sum_confusions(cms),
        "n_repeats": N_REPEATS,
        "sampling_strategy": SAMPLING_STRATEGY,
    }


# =========================
# Main
# =========================

def main() -> None:
    """
    Main execution flow.

    High-level phases (what happens in which order):
    1) Load train.csv and test.csv
    2) Choose feature columns
    3) Prepare X_all/y_all from train.csv
    4) Convert categorical features, detect cat_feature_indices
    5) Split train.csv into X_tr (train split) and X_val (validation split)
       This means:
       - X_tr is used to train models / tune hyperparameters
       - X_val is used to decide between bayes_best and ootb
    6) Undersample X_tr once -> X_tr_r (balanced training set)
       This means both candidate models are trained on a balanced version of the same X_tr.
    7) Bayesian optimization (BayesSearchCV) on X_tr_r
       This means the hyperparameters are chosen only from training data (not val/test).
    8) Train two models:
       - best_bayes_model (with best_params)
       - ootb_model (defaults)
       Both are trained on X_tr_r and early-stopped using X_val.
       This means both models are comparable and can stop early based on validation.
    9) Evaluate both on:
       - validation (raw + repeated balanced)
       - test (raw + repeated balanced)
    10) Pick the winner based on validation balanced F1 and save ONLY one model:
       OUT_DIR/best_model.cbm
    11) Write JSON reports + a human-readable report.txt
    """
    if not TRAIN_CSV.exists():
        raise SystemExit(f"Missing train file: {TRAIN_CSV.resolve()}")
    if not TEST_CSV.exists():
        raise SystemExit(f"Missing test file:  {TEST_CSV.resolve()}")

    train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
    test_df  = pd.read_csv(TEST_CSV, low_memory=False)

    if TARGET not in train_df.columns or TARGET not in test_df.columns:
        raise SystemExit(f"TARGET column '{TARGET}' missing from train/test.")

    # Determine features:
    # - If FEATURES is None, infer from EXCLUDE_COLS.
    # This means your feature set is controlled either manually (FEATURES) or automatically (EXCLUDE_COLS).
    feats = FEATURES if FEATURES is not None else infer_features(train_df)

    # Ensure requested columns exist.
    missing_cols = [c for c in feats + [TARGET] if c not in train_df.columns]
    if missing_cols:
        raise SystemExit("Missing columns in train:\n" + "\n".join(missing_cols))
    missing_cols_test = [c for c in feats + [TARGET] if c not in test_df.columns]
    if missing_cols_test:
        raise SystemExit("Missing columns in test:\n" + "\n".join(missing_cols_test))

    # Prepare X/y from train.csv:
    # This means train.csv is the “outer training set” produced by the earlier pipeline step.
    y_all = ensure_binary_target(train_df[TARGET]).astype(int)
    X_all = train_df[feats].copy()

    # Ensure categorical columns are safe for CatBoost:
    # This means we avoid object dtype ambiguity by casting to string and explicitly flagging categorical indices.
    X_all = coerce_categoricals_to_str(X_all)
    cat_feature_indices = get_cat_feature_indices(X_all)

    # Internal split of train.csv into training/validation:
    # This means we keep test.csv untouched until final evaluation.
    strat = y_all if STRATIFY else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=strat,
        shuffle=True,
    )

    # -----------------------------
    # 1) BAYESIAN OPTIMIZATION on X_tr only (undersampled)
    # -----------------------------
    # Undersample X_tr ONCE before BayesSearchCV.
    # This means the hyperparameter search runs on a balanced training set.
    X_tr_r, y_tr_r = undersample(X_tr, y_tr, seed=RANDOM_STATE)

    # Base estimator for BayesSearchCV:
    # This means BayesSearchCV will clone this estimator and set different hyperparameters from the search space.
    base_for_bayes = CatBoostClassifier(
        #task_type="GPU",
        #devices=CATBOOST_GPU_DEVICE,
        allow_writing_files=False,
        verbose=VERBOSE,
        random_seed=RANDOM_STATE,
        loss_function="Logloss",
        eval_metric="AUC",
        cat_features=cat_feature_indices if len(cat_feature_indices) > 0 else None,
    )

    # BayesSearchCV configuration:
    # This means it will:
    # - try BAYES_N_ITER different hyperparameter settings
    # - evaluate each setting with BAYES_CV-fold CV on X_tr_r
    # - optimize according to BAYES_SCORING (here: f1)
    if SKIP_BAYES_OPTIMIZATION:
        print("[BAYES] skipped (SKIP_BAYES_OPTIMIZATION=True).")
        best_params = {}
    else:
        bayes = BayesSearchCV(
            estimator=base_for_bayes,
            search_spaces=BAYES_SEARCH_SPACE,
            n_iter=BAYES_N_ITER,
            cv=BAYES_CV,
            scoring=BAYES_SCORING,
            n_jobs=BAYES_N_JOBS,
            random_state=RANDOM_STATE,
            verbose=VERBOSE,
            refit=True,
        )

        print("[BAYES] starting BayesSearchCV...")
        bayes.fit(X_tr_r, y_tr_r)
        print("[BAYES] done.")
        best_params = bayes.best_params_

    # -----------------------------
    # 2) Train BOTH models on X_tr only (not X_val)
    # -----------------------------
    # best_bayes_model:
    # This means we create a CatBoost model with the best hyperparameters found by BayesSearchCV.
    best_bayes_model = build_model(best_params, seed=RANDOM_STATE, cat_feature_indices=cat_feature_indices)
    best_bayes_model.fit(
        X_tr_r, y_tr_r,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    # ootb_model:
    # This means we train CatBoost mostly with defaults (plus categorical handling and reproducibility settings).
    ootb_model = build_ootb_model(seed=RANDOM_STATE, cat_feature_indices=cat_feature_indices)
    ootb_model.fit(
        X_tr_r, y_tr_r,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    if SKIP_BAYES_OPTIMIZATION:
        best_bayes_model = ootb_model
        best_params = {"note": "BayesSearchCV skipped; bayes_best treated as ootb."}

    # -----------------------------
    # 3) Evaluate BOTH on validation (raw + balanced repeated)
    # -----------------------------
    # Raw validation:
    # This means we evaluate on the untouched validation distribution (potentially imbalanced).
    bayes_val_raw_metrics, bayes_val_raw_cm = eval_on(X_val, y_val, best_bayes_model)
    ootb_val_raw_metrics, ootb_val_raw_cm = eval_on(X_val, y_val, ootb_model)

    # Balanced repeated validation:
    # This means we repeatedly undersample X_val to a balanced set and average metrics.
    bayes_val_bal = repeated_eval_balanced(X_val, y_val, best_bayes_model, seed0=RANDOM_STATE + 1000)
    ootb_val_bal  = repeated_eval_balanced(X_val, y_val, ootb_model, seed0=RANDOM_STATE + 2000)

    # Decide winner by selection metric on validation balanced mean.
    # This means model selection is based on balanced F1 (as configured).
    metric_key = "f1" if SELECTION_METRIC == "val_balanced_f1" else None
    if metric_key is None:
        raise SystemExit("This script currently expects SELECTION_METRIC='val_balanced_f1'.")

    bayes_score = float(bayes_val_bal["metrics_mean"].get(metric_key, np.nan))
    ootb_score  = float(ootb_val_bal["metrics_mean"].get(metric_key, np.nan))

    # Winner decision:
    # This means:
    # - If bayes_score is finite and >= ootb_score, Bayes model wins.
    # - Otherwise, ootb model wins.
    winner = "bayes_best" if (np.isfinite(bayes_score) and bayes_score >= ootb_score) else "ootb"

    # pick final model (ONLY ONE gets saved as best_model.cbm)
    if winner == "bayes_best":
        best_model = best_bayes_model
        best_model_kind = "bayes_best"
        best_model_params = best_params
    else:
        best_model = ootb_model
        best_model_kind = "ootb"
        best_model_params = {"note": "CatBoost default params + GPU settings", **OOTB_EXTRA_PARAMS}

    # -----------------------------
    # 4) Evaluate BOTH on test (raw + balanced repeated)
    # -----------------------------
    # Prepare test features:
    # This means test.csv stays independent and is only used for reporting after model selection.
    y_test = ensure_binary_target(test_df[TARGET]).astype(int)
    X_test = test_df[feats].copy()
    X_test = coerce_categoricals_to_str(X_test)

    # Raw test evaluation (untouched distribution).
    bayes_test_raw_metrics, bayes_test_raw_cm = eval_on(X_test, y_test, best_bayes_model)
    ootb_test_raw_metrics,  ootb_test_raw_cm  = eval_on(X_test, y_test, ootb_model)

    # Balanced repeated test evaluation:
    # This means we estimate how each model behaves if classes were balanced at test time.
    bayes_test_bal = repeated_eval_balanced(X_test, y_test, best_bayes_model, seed0=RANDOM_STATE + 3000)
    ootb_test_bal  = repeated_eval_balanced(X_test, y_test, ootb_model, seed0=RANDOM_STATE + 4000)

    # -----------------------------
    # 5) Save ONLY best model + reports
    # -----------------------------
    # Save features metadata:
    # This means downstream evaluation scripts can reuse the exact feature list.
    (OUT_DIR / "features.json").write_text(json.dumps({"target": TARGET, "features": feats}, indent=2), encoding="utf-8")

    # Save only the winner model:
    # This means there is exactly one canonical model artifact produced by this script.
    best_model_path = OUT_DIR / "best_model.cbm"
    best_model.save_model(str(best_model_path))

    # training_report.json:
    # This means you get a machine-readable summary of:
    # - what data splits were used
    # - what BayesSearchCV found
    # - validation performance for both candidates
    # - which model won and which model was saved
    training_report = {
        "target": TARGET,
        "features": feats,
        "random_state": RANDOM_STATE,
        "val_size": VAL_SIZE,
        "sampling_strategy": SAMPLING_STRATEGY,
        "n_repeats": N_REPEATS,
        "cat_feature_indices": cat_feature_indices,
        "bayes": {
            "search_space": {k: str(v) for k, v in BAYES_SEARCH_SPACE.items()},
            "n_iter": BAYES_N_ITER,
            "cv": BAYES_CV,
            "scoring": BAYES_SCORING,
            "best_params": best_params,
        },
        "validation": {
            "bayes_best": {
                "raw": {"metrics": bayes_val_raw_metrics, "confusion_matrix": bayes_val_raw_cm.astype(int).tolist()},
                "balanced_repeated": bayes_val_bal,
                "selection_score": bayes_score,
            },
            "ootb": {
                "raw": {"metrics": ootb_val_raw_metrics, "confusion_matrix": ootb_val_raw_cm.astype(int).tolist()},
                "balanced_repeated": ootb_val_bal,
                "selection_score": ootb_score,
            },
            "winner_by": SELECTION_METRIC,
            "winner": winner,
            "best_model_saved_as": str(best_model_path.resolve()),
            "best_model_kind": best_model_kind,
            "best_model_params": best_model_params,
        },
    }
    (OUT_DIR / "training_report.json").write_text(json.dumps(training_report, indent=2), encoding="utf-8")

    # test_report.json:
    # This means you get a machine-readable summary of test performance for both candidates.
    # Even though only one model is saved, we still report both for comparison.
    test_report = {
        "target": TARGET,
        "features": feats,
        "best_model": {
            "path": str(best_model_path.resolve()),
            "kind": best_model_kind,
            "params": best_model_params,
        },
        "test": {
            "bayes_best": {
                "raw": {"metrics": bayes_test_raw_metrics, "confusion_matrix": bayes_test_raw_cm.astype(int).tolist(), "rows": int(len(X_test))},
                "balanced_repeated": bayes_test_bal,
            },
            "ootb": {
                "raw": {"metrics": ootb_test_raw_metrics, "confusion_matrix": ootb_test_raw_cm.astype(int).tolist(), "rows": int(len(X_test))},
                "balanced_repeated": ootb_test_bal,
            },
        },
    }
    (OUT_DIR / "test_report.json").write_text(json.dumps(test_report, indent=2), encoding="utf-8")

    # Human-readable report:
    # This means you can read results quickly without opening JSON.
    def fmt_metrics(d: Dict[str, float]) -> str:
        keys = ["acc", "bacc", "precision", "recall", "f1", "auc"]
        return "\n".join([f"    {k}: {d.get(k, float('nan')):.6g}" for k in keys])

    report_lines = []
    report_lines.append(f"TARGET: {TARGET}")
    report_lines.append(f"RANDOM_STATE: {RANDOM_STATE}")
    report_lines.append(f"TRAIN_CSV: {TRAIN_CSV.resolve()}")
    report_lines.append(f"TEST_CSV:  {TEST_CSV.resolve()}")
    report_lines.append(f"N_FEATURES: {len(feats)}")
    report_lines.append(f"CAT_FEATURES: {len(cat_feature_indices)} (indices: {cat_feature_indices})")
    report_lines.append("")
    report_lines.append("=== WHAT THIS RUN DID (PLAIN MEANING) ===")
    report_lines.append("  - Split train.csv into X_tr and X_val.")
    report_lines.append("    This means we used X_tr for tuning/training and X_val for model selection.")
    report_lines.append("  - Ran BayesSearchCV on balanced X_tr (X_tr_r).")
    report_lines.append("    This means hyperparameters were chosen without using validation or test.")
    report_lines.append("  - Trained two models on X_tr_r: (1) Bayes-best (2) out-of-the-box.")
    report_lines.append("    This means both candidates were trained on the same balanced training data.")
    report_lines.append("  - Selected the winner using repeated balanced evaluation on X_val.")
    report_lines.append("    This means we compared models under a balanced class distribution.")
    report_lines.append("  - Evaluated both candidates on the held-out test.csv for reporting.")
    report_lines.append("    This means test performance was not used to choose the winner.")
    report_lines.append("")
    report_lines.append("=== BAYES SEARCH ===")
    report_lines.append(f"  n_iter: {BAYES_N_ITER}, cv: {BAYES_CV}, scoring: {BAYES_SCORING}, n_jobs: {BAYES_N_JOBS}")
    report_lines.append(f"  best_params: {best_params}")
    report_lines.append("")
    report_lines.append("=== VALIDATION (RAW) ===")
    report_lines.append("  [Bayes-best]")
    report_lines.append(fmt_metrics(bayes_val_raw_metrics))
    report_lines.append("  [OOTB]")
    report_lines.append(fmt_metrics(ootb_val_raw_metrics))
    report_lines.append("")
    report_lines.append("=== VALIDATION (BALANCED REPEATED, MEAN METRICS) ===")
    report_lines.append(f"  [Bayes-best] f1_mean: {bayes_score:.6g}")
    report_lines.append("\n".join([f"    {k}: {v:.6g}" for k, v in bayes_val_bal["metrics_mean"].items()]))
    report_lines.append(f"  [OOTB]      f1_mean: {ootb_score:.6g}")
    report_lines.append("\n".join([f"    {k}: {v:.6g}" for k, v in ootb_val_bal["metrics_mean"].items()]))
    report_lines.append("")
    report_lines.append(f"WINNER by {SELECTION_METRIC}: {winner}")
    report_lines.append(f"BEST MODEL SAVED: {best_model_path.resolve()}")
    report_lines.append(f"BEST MODEL TYPE:  {best_model_kind}")
    report_lines.append("")
    report_lines.append("=== TEST (RAW) ===")
    report_lines.append("  [Bayes-best]")
    report_lines.append(fmt_metrics(bayes_test_raw_metrics))
    report_lines.append("  [OOTB]")
    report_lines.append(fmt_metrics(ootb_test_raw_metrics))
    report_lines.append("")
    report_lines.append("=== TEST (BALANCED REPEATED, MEAN METRICS) ===")
    report_lines.append("  [Bayes-best]")
    report_lines.append("\n".join([f"    {k}: {v:.6g}" for k, v in bayes_test_bal["metrics_mean"].items()]))
    report_lines.append("  [OOTB]")
    report_lines.append("\n".join([f"    {k}: {v:.6g}" for k, v in ootb_test_bal["metrics_mean"].items()]))

    report_path = OUT_DIR / "report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[OK] Saved best model only: {best_model_path}")
    print(f"[OK] Training report: {OUT_DIR / 'training_report.json'}")
    print(f"[OK] Test report:     {OUT_DIR / 'test_report.json'}")
    print(f"[OK] TXT report:     {report_path}")


if __name__ == "__main__":
    main()
