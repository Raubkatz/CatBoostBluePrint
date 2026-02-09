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
- Both final candidates (best-bayes and ootb) are trained on an undersampled fraction of X_tr.
  This means both candidates get the same training signal for a fair comparison.
- The winner is chosen using balanced evaluation on X_val via repeated balanced excerpts.
  This means we compare models under a balanced class distribution (reduces bias from imbalance).
- The test set is never used to choose the winner—only to report final performance.
  This means the final numbers on test are a more honest estimate of generalization.

UPDATED (requested workflow change):
- We do N_REPEATS different internal train/validation splits (each is a fresh 80/20 split of train.csv).
- For each split:
  - we undersample X_tr to a balanced dataset using TRAIN_MINORITY_FRACTION of the minority class
  - we train ootb, and optionally a hyperparameter-optimized model
  - we evaluate each candidate on X_val using N_VAL_EXCERPTS balanced excerpts and average the score
  - we keep the better candidate for that split
- Across all splits, we keep the single best model overall.
- If N_REPEATS == 1, this whole procedure happens exactly once.
- Once the best model is found, we do not retrain again; we just save that best model.

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
from skopt import BayesSearchCV
from skopt.space import Integer, Real


# =========================
# CONFIG (EDIT THESE)
# =========================

TARGET = "isBugPresent"
RANDOM_STATE = 42

TRAIN_CSV = Path(f"data_{TARGET}") / f"splits_{str(RANDOM_STATE)}" / "train.csv"
TEST_CSV  = Path(f"data_{TARGET}") / f"splits_{str(RANDOM_STATE)}" / "test.csv"

OUT_DIR = Path(f"data_{TARGET}") / f"models_{str(RANDOM_STATE)}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES: Optional[List[str]] = None

EXCLUDE_COLS = [
    TARGET,
#    "hash", "date", "fileId", "fileName",
#    "source_file", "source_project",
#    "authorId",
]

VAL_SIZE = 0.2
STRATIFY = True

# Repeats:
# - N_REPEATS controls how many different internal 80/20 splits we do (each split trains + evaluates models).
# - If N_REPEATS == 1, we do exactly one split and one selection pass.
N_REPEATS = 1  # IMPORTANT: increase for multiple independent splits

# Training undersampling:
# We train on a balanced subset where we keep TRAIN_MINORITY_FRACTION of the minority class
# and the same number from the majority class.
TRAIN_MINORITY_FRACTION = 0.9

# Validation evaluation:
# For each split, we evaluate candidates on X_val using N_VAL_EXCERPTS balanced excerpts.
# Each excerpt is balanced and built using VAL_MINORITY_FRACTION of the minority class in X_val.
N_VAL_EXCERPTS = 100
VAL_MINORITY_FRACTION = 0.9

# Selection metric:
# We select the better model within each split based on mean balanced F1 across val excerpts.
SELECTION_METRIC = "val_balanced_f1"

CATBOOST_GPU_DEVICE = "0"
VERBOSE = 0

OOTB_EXTRA_PARAMS: Dict = {}

#BAYES_N_ITER = 10  # IMPORTANT: increase for proper tuning
BAYES_N_ITER = 4  # IMPORTANT: increase for proper tuning

BAYES_CV = 3
BAYES_SCORING = "f1"
BAYES_N_JOBS = 1

BAYES_SEARCH_SPACE = {
    "depth": Integer(4, 12),
    "iterations": Integer(500, 8000),
    "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
    "l2_leaf_reg": Real(1.0, 20.0, prior="log-uniform"),
    "border_count": Integer(32, 255),
}

EARLY_STOPPING_ROUNDS = 100

# NEW: skip Bayesian hyperparameter tuning and run ONLY out-of-the-box CatBoost.
SKIP_BAYES_OPTIMIZATION = True

# =========================
# Helpers
# =========================

def infer_features(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in set(EXCLUDE_COLS)]


def ensure_binary_target(y: pd.Series) -> pd.Series:
    if y.dtype == bool:
        return y.astype(int)
    if y.dtype == object:
        lowered = y.astype(str).str.strip().str.lower()
        if set(lowered.dropna().unique()).issubset({"true", "false", "0", "1"}):
            return lowered.map({"true": 1, "false": 0, "1": 1, "0": 0}).astype("Int64")
    return y


def get_cat_feature_indices(X: pd.DataFrame) -> List[int]:
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return [X.columns.get_loc(c) for c in cat_cols]


def coerce_categoricals_to_str(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy()
    cat_cols = X2.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in cat_cols:
        X2[c] = X2[c].astype(str)
    return X2


def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, float]:
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
    proba = model.predict_proba(dfX)[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = metrics_binary(dfy.to_numpy(), pred, proba)
    cm = confusion_matrix(dfy.to_numpy(), pred, labels=[0, 1])
    return m, cm


def undersample_balanced_fraction(
    X: pd.DataFrame,
    y: pd.Series,
    fraction_minority: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed=seed)

    X_array = np.asarray(X)
    y_array = np.asarray(y)

    classes = np.unique(y_array)
    if len(classes) != 2:
        raise ValueError("undersample_balanced_fraction supports only binary targets.")

    counts = {cls: int(np.sum(y_array == cls)) for cls in classes}
    minority_class = min(counts, key=counts.get)
    majority_class = max(counts, key=counts.get)

    minority_idx = np.where(y_array == minority_class)[0]
    majority_idx = np.where(y_array == majority_class)[0]

    n_min = counts[minority_class]
    n_pick = int(round(n_min * fraction_minority))
    if n_pick <= 0:
        raise ValueError(f"fraction_minority too small -> n_pick={n_pick}")

    if n_pick > len(minority_idx) or n_pick > len(majority_idx):
        raise ValueError("Not enough samples to build balanced subset at requested fraction.")

    picked_min = rng.choice(minority_idx, size=n_pick, replace=False)
    picked_maj = rng.choice(majority_idx, size=n_pick, replace=False)

    idx = np.concatenate([picked_min, picked_maj])
    rng.shuffle(idx)

    Xb = X_array[idx]
    yb = y_array[idx]

    Xb = pd.DataFrame(Xb, columns=X.columns)
    yb = pd.Series(yb, name=y.name)
    return Xb, yb


def build_model(params: Dict, seed: int, cat_feature_indices: List[int]) -> CatBoostClassifier:
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
    keys = sorted({k for d in dicts for k in d.keys()})
    return {k: float(np.nanmean([d.get(k, np.nan) for d in dicts])) for k in keys}


def sum_confusions(cms: List[np.ndarray]) -> List[List[int]]:
    s = np.sum(np.stack(cms, axis=0), axis=0)
    return s.astype(int).tolist()


def evaluate_on_val_excerpts(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model: CatBoostClassifier,
    seed0: int,
) -> Dict:
    mets: List[Dict[str, float]] = []
    cms: List[np.ndarray] = []

    for k in range(N_VAL_EXCERPTS):
        seed = seed0 + k
        Xb, yb = undersample_balanced_fraction(
            X_val, y_val,
            fraction_minority=VAL_MINORITY_FRACTION,
            seed=seed + 50_000,
        )
        m, cm = eval_on(Xb, yb, model)
        mets.append(m)
        cms.append(cm)

    return {
        "metrics_mean": mean_dict(mets),
        "confusion_matrix_sum": sum_confusions(cms),
        "n_excerpts": N_VAL_EXCERPTS,
        "val_minority_fraction": VAL_MINORITY_FRACTION,
    }


def repeated_eval_balanced(X: pd.DataFrame, y: pd.Series, model: CatBoostClassifier, seed0: int) -> Dict:
    """
    Retained for reporting consistency (test-time balanced evaluation).
    If N_REPEATS == 1, it runs exactly once.
    """
    mets = []
    cms = []
    for r in range(N_REPEATS):
        seed = seed0 + r
        rus = RandomUnderSampler(sampling_strategy=1.0, random_state=seed + 20_000)
        Xr, yr = rus.fit_resample(X, y)
        if not isinstance(Xr, pd.DataFrame):
            Xr = pd.DataFrame(Xr, columns=X.columns)
        if not isinstance(yr, pd.Series):
            yr = pd.Series(yr, name=y.name)
        m, cm = eval_on(Xr, yr, model)
        mets.append(m)
        cms.append(cm)
    return {
        "metrics_mean": mean_dict(mets),
        "confusion_matrix_sum": sum_confusions(cms),
        "n_repeats": N_REPEATS,
        "sampling_strategy": 1.0,
    }


# =========================
# Main
# =========================

def main() -> None:
    if not TRAIN_CSV.exists():
        raise SystemExit(f"Missing train file: {TRAIN_CSV.resolve()}")
    if not TEST_CSV.exists():
        raise SystemExit(f"Missing test file:  {TEST_CSV.resolve()}")

    train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
    test_df  = pd.read_csv(TEST_CSV, low_memory=False)

    if TARGET not in train_df.columns or TARGET not in test_df.columns:
        raise SystemExit(f"TARGET column '{TARGET}' missing from train/test.")

    feats = FEATURES if FEATURES is not None else infer_features(train_df)

    missing_cols = [c for c in feats + [TARGET] if c not in train_df.columns]
    if missing_cols:
        raise SystemExit("Missing columns in train:\n" + "\n".join(missing_cols))
    missing_cols_test = [c for c in feats + [TARGET] if c not in test_df.columns]
    if missing_cols_test:
        raise SystemExit("Missing columns in test:\n" + "\n".join(missing_cols_test))

    y_all = ensure_binary_target(train_df[TARGET]).astype(int)
    X_all = train_df[feats].copy()

    X_all = coerce_categoricals_to_str(X_all)
    cat_feature_indices = get_cat_feature_indices(X_all)

    best_overall_model: Optional[CatBoostClassifier] = None
    best_overall_kind: Optional[str] = None
    best_overall_params: Dict = {}
    best_overall_split_index: Optional[int] = None
    best_overall_val_score: float = float("-inf")

    per_split_summaries: List[Dict] = []

    for split_i in range(N_REPEATS):
        split_seed = RANDOM_STATE + split_i

        strat = y_all if STRATIFY else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all,
            test_size=VAL_SIZE,
            random_state=split_seed,
            stratify=strat,
            shuffle=True,
        )

        # NEW: also build one balanced validation set used for CatBoost eval_set (early stopping)
        X_val_r, y_val_r = undersample_balanced_fraction(
            X_val, y_val,
            fraction_minority=VAL_MINORITY_FRACTION,
            seed=split_seed + 20_000,
        )

        X_tr_r, y_tr_r = undersample_balanced_fraction(
            X_tr, y_tr,
            fraction_minority=TRAIN_MINORITY_FRACTION,
            seed=split_seed + 10_000,
        )

        ootb_model = build_ootb_model(seed=split_seed, cat_feature_indices=cat_feature_indices)
        ootb_model.fit(
            X_tr_r, y_tr_r,
            eval_set=(X_val_r, y_val_r),
            use_best_model=True,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )

        if SKIP_BAYES_OPTIMIZATION:
            best_params = {"note": "BayesSearchCV skipped; bayes_best treated as ootb."}
            best_bayes_model = ootb_model
        else:
            base_for_bayes = CatBoostClassifier(
                #task_type="GPU",
                #devices=CATBOOST_GPU_DEVICE,
                allow_writing_files=False,
                verbose=VERBOSE,
                random_seed=split_seed,
                loss_function="Logloss",
                eval_metric="AUC",
                cat_features=cat_feature_indices if len(cat_feature_indices) > 0 else None,
            )

            bayes = BayesSearchCV(
                estimator=base_for_bayes,
                search_spaces=BAYES_SEARCH_SPACE,
                n_iter=BAYES_N_ITER,
                cv=BAYES_CV,
                scoring=BAYES_SCORING,
                n_jobs=BAYES_N_JOBS,
                random_state=split_seed,
                verbose=VERBOSE + 1,
                refit=True,
            )

            print(f"[BAYES][split={split_i}] starting BayesSearchCV...")
            bayes.fit(X_tr_r, y_tr_r)
            print(f"[BAYES][split={split_i}] done.")
            best_params = bayes.best_params_

            best_bayes_model = build_model(best_params, seed=split_seed, cat_feature_indices=cat_feature_indices)
            best_bayes_model.fit(
                X_tr_r, y_tr_r,
                eval_set=(X_val_r, y_val_r),
                use_best_model=True,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            )

        # IMPORTANT: validation metrics are computed ONLY on balanced datasets
        bayes_val_raw_metrics, bayes_val_raw_cm = eval_on(X_val_r, y_val_r, best_bayes_model)
        ootb_val_raw_metrics, ootb_val_raw_cm = eval_on(X_val_r, y_val_r, ootb_model)

        bayes_val_bal = evaluate_on_val_excerpts(X_val, y_val, best_bayes_model, seed0=split_seed + 1000)
        ootb_val_bal  = evaluate_on_val_excerpts(X_val, y_val, ootb_model,       seed0=split_seed + 2000)

        metric_key = "f1" if SELECTION_METRIC == "val_balanced_f1" else None
        if metric_key is None:
            raise SystemExit("This script currently expects SELECTION_METRIC='val_balanced_f1'.")

        bayes_score = float(bayes_val_bal["metrics_mean"].get(metric_key, np.nan))
        ootb_score  = float(ootb_val_bal["metrics_mean"].get(metric_key, np.nan))

        winner = "bayes_best" if (np.isfinite(bayes_score) and bayes_score >= ootb_score) else "ootb"

        if winner == "bayes_best":
            split_best_model = best_bayes_model
            split_best_kind = "bayes_best"
            split_best_params = best_params
            split_best_score = bayes_score
        else:
            split_best_model = ootb_model
            split_best_kind = "ootb"
            split_best_params = {"note": "CatBoost default params + GPU settings", **OOTB_EXTRA_PARAMS}
            split_best_score = ootb_score

        if np.isfinite(split_best_score) and split_best_score > best_overall_val_score:
            best_overall_model = split_best_model
            best_overall_kind = split_best_kind
            best_overall_params = split_best_params
            best_overall_split_index = split_i
            best_overall_val_score = split_best_score

        per_split_summaries.append({
            "split_index": split_i,
            "split_seed": split_seed,
            "train_minority_fraction": TRAIN_MINORITY_FRACTION,
            "val_minority_fraction": VAL_MINORITY_FRACTION,
            "n_val_excerpts": N_VAL_EXCERPTS,
            "validation": {
                "bayes_best": {
                    "raw": {"metrics": bayes_val_raw_metrics, "confusion_matrix": bayes_val_raw_cm.astype(int).tolist()},
                    "balanced_excerpts": bayes_val_bal,
                    "selection_score": bayes_score,
                },
                "ootb": {
                    "raw": {"metrics": ootb_val_raw_metrics, "confusion_matrix": ootb_val_raw_cm.astype(int).tolist()},
                    "balanced_excerpts": ootb_val_bal,
                    "selection_score": ootb_score,
                },
                "winner_by": SELECTION_METRIC,
                "winner": winner,
                "early_stopping_eval_set": "balanced_X_val",
            },
            "split_best": {
                "kind": split_best_kind,
                "params": split_best_params,
                "val_score": split_best_score,
            },
        })

        print(
            f"[SPLIT {split_i}] winner={winner} "
            f"(bayes_f1={bayes_score:.6g}, ootb_f1={ootb_score:.6g}) "
            f"=> split_best={split_best_kind}, split_best_f1={split_best_score:.6g}"
        )

    if best_overall_model is None or best_overall_kind is None or best_overall_split_index is None:
        raise SystemExit("No valid best model found (all selection scores were NaN).")

    y_test = ensure_binary_target(test_df[TARGET]).astype(int)
    X_test = test_df[feats].copy()
    X_test = coerce_categoricals_to_str(X_test)

    # IMPORTANT: test metrics are computed ONLY on balanced datasets
    X_test_r, y_test_r = undersample_balanced_fraction(
        X_test, y_test,
        fraction_minority=VAL_MINORITY_FRACTION,
        seed=RANDOM_STATE + 60_000,
    )
    test_raw_metrics, test_raw_cm = eval_on(X_test_r, y_test_r, best_overall_model)
    test_bal = repeated_eval_balanced(X_test, y_test, best_overall_model, seed0=RANDOM_STATE + 3000)

    (OUT_DIR / "features.json").write_text(
        json.dumps({"target": TARGET, "features": feats}, indent=2),
        encoding="utf-8",
    )

    best_model_path = OUT_DIR / "best_model.cbm"
    best_overall_model.save_model(str(best_model_path))

    training_report = {
        "target": TARGET,
        "features": feats,
        "random_state": RANDOM_STATE,
        "val_size": VAL_SIZE,
        "n_repeats": N_REPEATS,
        "train_minority_fraction": TRAIN_MINORITY_FRACTION,
        "val_minority_fraction": VAL_MINORITY_FRACTION,
        "n_val_excerpts": N_VAL_EXCERPTS,
        "cat_feature_indices": cat_feature_indices,
        "bayes": {
            "skipped": SKIP_BAYES_OPTIMIZATION,
            "search_space": {k: str(v) for k, v in BAYES_SEARCH_SPACE.items()},
            "n_iter": BAYES_N_ITER,
            "cv": BAYES_CV,
            "scoring": BAYES_SCORING,
        },
        "selection": {
            "metric": SELECTION_METRIC,
            "best_overall_split_index": best_overall_split_index,
            "best_overall_val_score": float(best_overall_val_score),
            "best_model_kind": best_overall_kind,
            "best_model_params": best_overall_params,
            "best_model_saved_as": str(best_model_path.resolve()),
        },
        "splits": per_split_summaries,
    }
    (OUT_DIR / "training_report.json").write_text(json.dumps(training_report, indent=2), encoding="utf-8")

    test_report = {
        "target": TARGET,
        "features": feats,
        "best_model": {
            "path": str(best_model_path.resolve()),
            "kind": best_overall_kind,
            "params": best_overall_params,
            "best_overall_split_index": int(best_overall_split_index),
            "best_overall_val_score": float(best_overall_val_score),
        },
        "test": {
            "raw_balanced": {
                "metrics": test_raw_metrics,
                "confusion_matrix": test_raw_cm.astype(int).tolist(),
                "rows": int(len(X_test_r)),
                "minority_fraction": VAL_MINORITY_FRACTION,
            },
            "balanced_repeated": test_bal,
        },
    }
    (OUT_DIR / "test_report.json").write_text(json.dumps(test_report, indent=2), encoding="utf-8")

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
    report_lines.append("  - Repeated N_REPEATS internal 80/20 splits of train.csv.")
    report_lines.append("    This means each split trains + validates fresh candidate models.")
    report_lines.append("  - For each split, trained on a balanced subset of X_tr using TRAIN_MINORITY_FRACTION.")
    report_lines.append("    This means training always uses a balanced dataset.")
    report_lines.append("  - Used a balanced X_val subset as CatBoost eval_set for early stopping.")
    report_lines.append("    This means validation inside CatBoost never used the full imbalanced validation set.")
    report_lines.append("  - Compared ootb vs bayes_best on X_val using balanced validation excerpts only.")
    report_lines.append("    This means selection is based only on balanced validation performance.")
    report_lines.append("  - Evaluated on test using balanced subsets only for reporting.")
    report_lines.append("")
    report_lines.append("=== BEST OVERALL MODEL (BY VALIDATION) ===")
    report_lines.append(f"  best_overall_split_index: {best_overall_split_index}")
    report_lines.append(f"  best_overall_val_score (balanced f1 mean): {best_overall_val_score:.6g}")
    report_lines.append(f"  best_model_kind: {best_overall_kind}")
    report_lines.append(f"  best_model_path: {best_model_path.resolve()}")
    report_lines.append("")
    report_lines.append("=== TEST (RAW, BALANCED SUBSET) ===")
    report_lines.append(fmt_metrics(test_raw_metrics))
    report_lines.append("")
    report_lines.append("=== TEST (BALANCED REPEATED, MEAN METRICS) ===")
    report_lines.append("\n".join([f"    {k}: {v:.6g}" for k, v in test_bal["metrics_mean"].items()]))

    report_path = OUT_DIR / "report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[OK] Saved best model only: {best_model_path}")
    print(f"[OK] Training report: {OUT_DIR / 'training_report.json'}")
    print(f"[OK] Test report:     {OUT_DIR / 'test_report.json'}")
    print(f"[OK] TXT report:     {report_path}")


if __name__ == "__main__":
    main()
