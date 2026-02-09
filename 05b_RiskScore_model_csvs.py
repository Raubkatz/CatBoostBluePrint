#!/usr/bin/env python3
"""
06_apply_best_model_to_regarded_csvs_2026_riskscore.py

Risk score workflow (CatBoostRegressor).

Same behavior as the classifier batch script, but:
- Loads the regressor from: data_{TARGET}/models_{RANDOM_STATE}_riskscore/best_model.cbm
- Loads features from:      data_{TARGET}/models_{RANDOM_STATE}_riskscore/features.json
- Predicts a continuous risk score per row.
- Clips risk score to [0, 1].
- Translates risk score -> class using threshold 0.5.

For each CSV:
- Append TWO prediction columns:
    {TARGET}_predicted_risk_score   (clipped to [0,1])
    {TARGET}_predicted             (translated class from risk score)
- Save as: *_predicted.csv
- If TARGET exists:
    (A) Evaluate on full CSV (raw)
    (B) Evaluate with 100x balanced undersampling (50/50 by true labels)
  Save per-file TXT report: *_report.txt

CSV ordering requirement:
- At the end of the table, keep these side-by-side (in this order):
    {TARGET}                       (if present)
    {TARGET}_predicted
    {TARGET}_predicted_risk_score
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# =============================================================================
# CONFIG (EDIT THESE)
# =============================================================================

TARGET = "isBugPresent"
RANDOM_STATE = 42

MODEL_SUFFIX = "_riskscore"

MODEL_PATH = Path(f"data_{TARGET}") / f"models_{str(RANDOM_STATE)}{MODEL_SUFFIX}" / "best_model.cbm"
FEATURES_JSON_PATH = Path(f"data_{TARGET}") / f"models_{str(RANDOM_STATE)}{MODEL_SUFFIX}" / "features.json"

# Folder that contains the CSVs you want to analyze
REGARDED_DIR = Path("to_be_tested")

# Threshold for converting risk score -> class
THRESHOLD = 0.5

# Balanced undersampling evaluation
N_ITERATIONS_BALANCED = 100
BALANCED_MINORITY_FRACTION = 0.9  # fraction of minority kept each iteration


# =============================================================================
# Helpers
# =============================================================================

def load_features(features_json_path: Path) -> List[str]:
    with open(features_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "features" not in payload or "target" not in payload:
        raise ValueError(f"Invalid features.json structure: {features_json_path}")

    if payload["target"] != TARGET:
        raise ValueError(
            f"features.json target mismatch: expected '{TARGET}', got '{payload['target']}'"
        )

    return payload["features"]


def coerce_categoricals_to_str(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy()
    cat_cols = X2.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in cat_cols:
        X2[c] = X2[c].astype(str)
    return X2


def custom_equal_under_sampler(X, y, fraction=0.8, random_state=None):
    rng = np.random.default_rng(seed=random_state)

    X_array = np.asarray(X)
    y_array = np.asarray(y)

    unique_classes = np.unique(y_array)
    if len(unique_classes) != 2:
        raise ValueError("This function supports only binary classification.")

    class_counts = {cls: np.sum(y_array == cls) for cls in unique_classes}
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)

    minority_indices = np.where(y_array == minority_class)[0]
    majority_indices = np.where(y_array == majority_class)[0]

    num_minority = class_counts[minority_class]
    num_to_pick_minority = int(round(num_minority * fraction))
    if num_to_pick_minority <= 0:
        raise ValueError(f"fraction too small -> num_to_pick_minority={num_to_pick_minority}")

    if num_to_pick_minority > len(minority_indices) or num_to_pick_minority > len(majority_indices):
        raise ValueError("Not enough samples to build a balanced subset at requested fraction.")

    minority_sampled = rng.choice(minority_indices, size=num_to_pick_minority, replace=False)
    majority_sampled = rng.choice(majority_indices, size=num_to_pick_minority, replace=False)

    combined_indices = np.concatenate([minority_sampled, majority_sampled])
    rng.shuffle(combined_indices)

    return X_array[combined_indices], y_array[combined_indices]


def dict_to_cr_string(agg_dict: Dict) -> str:
    lines = []
    lines.append("               precision    recall  f1-score   support")
    possible_labels = ["0", "1", "2", "3", "accuracy", "macro avg", "weighted avg"]
    for lbl in possible_labels:
        if lbl in agg_dict:
            if lbl == "accuracy":
                val_str = f"{agg_dict[lbl]:.4f}"
                lines.append(f"   accuracy                           {val_str}")
            else:
                precision = agg_dict[lbl]["precision"]
                recall = agg_dict[lbl]["recall"]
                f1 = agg_dict[lbl]["f1-score"]
                support = agg_dict[lbl]["support"]
                lines.append(
                    f"       {lbl:>2}         {precision:>9.4f}   {recall:>7.4f}   {f1:>8.4f}   {int(support):>6}"
                )
    return "\n".join(lines)


def risk_to_class(risk_scores: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    risk_scores = np.asarray(risk_scores, dtype=float)
    return (risk_scores >= threshold).astype(int)


def clip_risk_scores(risk_scores: np.ndarray) -> np.ndarray:
    # Required behavior: <0 -> 0, >1 -> 1
    rs = np.asarray(risk_scores, dtype=float)
    rs = np.where(rs < 0.0, 0.0, rs)
    rs = np.where(rs > 1.0, 1.0, rs)
    return rs


def balanced_eval_100x(
    model: CatBoostRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    fraction: float,
    n_iterations: int,
) -> Tuple[str, np.ndarray, np.ndarray, float]:
    """
    Returns:
      - averaged classification-report-like string
      - summed confusion matrix over iterations
      - relative confusion matrix (row-normalized)
      - mean accuracy over iterations
    """
    classes = np.unique(np.asarray(y))
    if len(classes) != 2:
        raise ValueError("Balanced eval expects binary y with exactly 2 classes.")
    labels = [0, 1]

    acc_scores: List[float] = []
    avg_conf_matrix = np.zeros((2, 2), dtype=float)
    aggregated_metrics: Dict = {}

    for i in range(n_iterations):
        X_under, y_under = custom_equal_under_sampler(X, y, fraction=fraction, random_state=i + 10_000)

        risk_under = model.predict(X_under)
        risk_under = clip_risk_scores(risk_under)
        y_pred_under = risk_to_class(risk_under, threshold=THRESHOLD)

        acc_scores.append(accuracy_score(y_under, y_pred_under))
        avg_conf_matrix += confusion_matrix(y_under, y_pred_under, labels=labels)

        metrics_dict = classification_report(y_under, y_pred_under, output_dict=True, zero_division=0)

        for label, scores in metrics_dict.items():
            if isinstance(scores, dict):
                if label not in aggregated_metrics:
                    aggregated_metrics[label] = {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1-score": 0.0,
                        "support": 0
                    }
                aggregated_metrics[label]["precision"] += scores["precision"]
                aggregated_metrics[label]["recall"]    += scores["recall"]
                aggregated_metrics[label]["f1-score"]  += scores["f1-score"]
                if "support" in scores:
                    aggregated_metrics[label]["support"] += scores["support"]
            else:
                if label not in aggregated_metrics:
                    aggregated_metrics[label] = 0.0
                aggregated_metrics[label] += scores

    for label, scores in aggregated_metrics.items():
        if isinstance(scores, dict):
            scores["precision"] /= n_iterations
            scores["recall"]    /= n_iterations
            scores["f1-score"]  /= n_iterations
            scores["support"]   = scores["support"] / n_iterations
        else:
            aggregated_metrics[label] /= n_iterations

    avg_class_report_str = dict_to_cr_string(aggregated_metrics)

    denom = avg_conf_matrix.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    relative_conf_matrix = avg_conf_matrix / denom

    return avg_class_report_str, avg_conf_matrix, relative_conf_matrix, float(np.mean(acc_scores))


def reorder_end_columns(
    df: pd.DataFrame,
    cols_at_end_in_order: List[str],
) -> pd.DataFrame:
    """
    Move specified columns to the end of the dataframe, keeping their given order.
    Any of those columns that don't exist are ignored.
    """
    present = [c for c in cols_at_end_in_order if c in df.columns]
    if not present:
        return df
    remaining = [c for c in df.columns if c not in set(present)]
    return df[remaining + present]


def process_one_csv(
    model: CatBoostRegressor,
    selected_features: List[str],
    input_path: Path,
) -> None:
    df = pd.read_csv(input_path, low_memory=False)

    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{input_path.name}] Missing required feature columns.\n"
            + "\n".join(missing)
        )

    X = df[selected_features].copy()
    X = coerce_categoricals_to_str(X)

    # Predict risk score -> clip -> translate to class
    risk = model.predict(X)
    risk = clip_risk_scores(risk)
    y_pred = risk_to_class(risk, threshold=THRESHOLD)

    # Column naming: proba column becomes risk score column
    pred_risk_col = f"{TARGET}_predicted_risk_score"
    pred_class_col = f"{TARGET}_predicted"

    df[pred_risk_col] = risk.astype(float)
    df[pred_class_col] = y_pred.astype(int)

    # Reorder at end: true label, predicted class, risk score (side-by-side)
    df = reorder_end_columns(df, [TARGET, pred_class_col, pred_risk_col])

    out_csv = input_path.with_name(input_path.stem + "_predicted_risk_score.csv")
    df.to_csv(out_csv, index=False)

    # Build report
    report_txt = []
    report_txt.append(f"FILE: {input_path.name}")
    report_txt.append(f"MODEL: {MODEL_PATH.resolve()}")
    report_txt.append(f"MODEL_TYPE: CatBoostRegressor (risk score)")
    report_txt.append(f"TARGET: {TARGET}")
    report_txt.append(f"RANDOM_STATE: {RANDOM_STATE}")
    report_txt.append(f"THRESHOLD (risk->class): {THRESHOLD}")
    report_txt.append("RISK_SCORE_CLIPPING: <0 -> 0, >1 -> 1")
    report_txt.append(f"N_ROWS: {len(df)}")
    report_txt.append(f"N_FEATURES: {len(selected_features)}")
    report_txt.append(f"SAVED_PREDICTIONS_CSV: {out_csv.name}")
    report_txt.append("")

    if TARGET in df.columns:
        y_true = df[TARGET].astype(int).to_numpy()

        full_report = classification_report(y_true, y_pred, zero_division=0)
        full_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        report_txt.append("=== FULL CSV (RAW, IMBALANCED) ===")
        report_txt.append("Confusion matrix (labels [0,1]):")
        report_txt.append(np.array2string(full_cm, separator=", "))
        report_txt.append("")
        report_txt.append("Classification report:")
        report_txt.append(full_report)
        report_txt.append("")

        avg_cr, cm_sum, cm_rel, acc_mean = balanced_eval_100x(
            model=model,
            X=X,
            y=pd.Series(y_true),
            fraction=BALANCED_MINORITY_FRACTION,
            n_iterations=N_ITERATIONS_BALANCED,
        )

        report_txt.append("=== BALANCED UNDERSAMPLED (100x) ===")
        report_txt.append(f"minority_fraction_kept: {BALANCED_MINORITY_FRACTION}")
        report_txt.append(f"n_iterations: {N_ITERATIONS_BALANCED}")
        report_txt.append(f"mean_accuracy: {acc_mean:.6f}")
        report_txt.append("")
        report_txt.append("Summed confusion matrix over iterations (labels [0,1]):")
        report_txt.append(np.array2string(cm_sum, separator=", "))
        report_txt.append("")
        report_txt.append("Relative confusion matrix (row-normalized):")
        report_txt.append(np.array2string(cm_rel, separator=", "))
        report_txt.append("")
        report_txt.append("Averaged classification report:")
        report_txt.append(avg_cr)
        report_txt.append("")
    else:
        report_txt.append("=== NO GROUND TRUTH AVAILABLE ===")
        report_txt.append(f"Column '{TARGET}' not present in CSV. Only predictions were generated.")
        report_txt.append("")

    out_report = input_path.with_name(input_path.stem + "_report.txt")
    out_report.write_text("\n".join(report_txt), encoding="utf-8")

    print(f"[OK] {input_path.name}")
    print(f"     -> {out_csv.name}")
    print(f"     -> {out_report.name}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")
    if not FEATURES_JSON_PATH.exists():
        raise FileNotFoundError(f"features.json not found: {FEATURES_JSON_PATH.resolve()}")
    if not REGARDED_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {REGARDED_DIR.resolve()}")

    csvs = sorted(REGARDED_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in: {REGARDED_DIR.resolve()}")

    print(f"[LOAD] Model: {MODEL_PATH}")
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))

    print(f"[LOAD] Features: {FEATURES_JSON_PATH}")
    selected_features = load_features(FEATURES_JSON_PATH)
    print(f"[INFO] n_features={len(selected_features)}")

    print(f"[INFO] regarded_dir={REGARDED_DIR.resolve()}")
    print(f"[INFO] n_csv_files={len(csvs)}")

    n_ok = 0
    n_fail = 0

    for p in csvs:
        try:
            process_one_csv(model, selected_features, p)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            fail_report = p.with_name(p.stem + "_report.txt")
            fail_report.write_text(
                "\n".join([
                    f"FILE: {p.name}",
                    "STATUS: FAILED",
                    f"ERROR: {repr(e)}",
                ]) + "\n",
                encoding="utf-8",
            )
            print(f"[FAIL] {p.name} -> wrote {fail_report.name}")

    print(f"[DONE] ok={n_ok} fail={n_fail} total={len(csvs)}")


if __name__ == "__main__":
    main()