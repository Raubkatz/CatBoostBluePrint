#!/usr/bin/env python3
"""
06_apply_best_model_to_project_csv_2026.py

Apply the trained “best_model.cbm” (winner from the 2026 pipeline) to ONE project CSV.

What this script does (precise + plain meaning):

1) Load the trained model artifact:
   - data_{TARGET}/models_{RANDOM_STATE}/best_model.cbm
   This means: we are NOT training anything here; we only reuse the already trained model.

2) Load the exact feature list used during training:
   - data_{TARGET}/models_{RANDOM_STATE}/features.json
   This means: we will select only those columns needed for prediction, in the exact same order.

3) Load ONE project CSV from the folder:
   - to_be_tested/
   This means: you point to a single file, and we process exactly that file.

4) Prepare the input matrix X by selecting the required features:
   - If a required feature is missing, we fail early with a clear error.
   - If categorical-like columns exist (object/string/category), we cast them to string.
   This means: we match CatBoost’s expectations and avoid dtype-related crashes.

5) Predict for every row and append a new column:
   - {TARGET}_predicted
   This means: we keep ALL original columns (including unused ones), and only append one prediction column.

6) Print a single-row prediction (user-chosen index):
   This means: you can quickly sanity-check a particular sample.

7) Save the updated CSV to:
   - tested/
   with the same base filename as the input.
   This means: input file name is preserved; output contains an extra prediction column.

Example run:
  python 06_apply_best_model_to_project_csv_2026.py

You edit CONFIG below to set:
- TARGET, RANDOM_STATE
- Which file to load from to_be_tested/
- Which row index to print
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pandas as pd
from catboost import CatBoostClassifier


# =============================================================================
# CONFIG (EDIT THESE)
# =============================================================================

TARGET = "isBugPresent"
RANDOM_STATE = 42

# Where the trained model lives (produced by 03_train_and_select_catboost_gpu_2026.py)
MODEL_PATH = Path(f"data_{TARGET}") / f"models_{str(RANDOM_STATE)}" / "best_model.cbm"
FEATURES_JSON_PATH = Path(f"data_{TARGET}") / f"models_{str(RANDOM_STATE)}" / "features.json"

# Input/output folders for “new” project files you want to run inference on
TO_BE_TESTED_DIR = Path("to_be_tested")
TESTED_DIR = Path("tested")

# Choose exactly one file to run (set to a filename inside to_be_tested/)
# Example: PROJECT_FILE = "my_project.csv"
PROJECT_FILE = None  # <-- set to something, or keep None to auto-pick the first CSV found

# Print prediction for one particular row index
# This means: we will print that row’s predicted label and (if available) probability.
PRINT_ROW_INDEX = 0

# Prediction threshold (only used for converting probabilities -> class)
# This means: probability >= threshold => predicted class 1 else 0.
THRESHOLD = 0.5


# =============================================================================
# Helpers
# =============================================================================

def load_features(features_json_path: Path) -> List[str]:
    """
    Load {"target": ..., "features": [...]} from features.json.

    Plain meaning:
    - This file is the single source of truth for feature order.
    - We must use the same order at inference time as during training.
    """
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
    """
    Cast categorical-like columns (object/string/category) to string.

    Plain meaning:
    - If a column contains mixed Python types (common in CSVs), it becomes object dtype.
    - CatBoost expects categorical columns to be consistently represented.
    - Converting to string is a robust, simple approach for inference.
    """
    X2 = X.copy()
    cat_cols = X2.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in cat_cols:
        X2[c] = X2[c].astype(str)
    return X2


def pick_project_file(to_be_tested_dir: Path, project_file: str | None) -> Path:
    """
    Pick a single input file.

    Plain meaning:
    - If PROJECT_FILE is set, we use exactly that.
    - If PROJECT_FILE is None, we pick the first *.csv found in to_be_tested/.
    """
    if project_file is not None:
        p = to_be_tested_dir / project_file
        if not p.exists():
            raise FileNotFoundError(f"Requested PROJECT_FILE not found: {p.resolve()}")
        return p

    csvs = sorted(to_be_tested_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in: {to_be_tested_dir.resolve()}")
    return csvs[0]


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # -------------------------
    # 0) Validate paths
    # -------------------------
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")
    if not FEATURES_JSON_PATH.exists():
        raise FileNotFoundError(f"features.json not found: {FEATURES_JSON_PATH.resolve()}")

    if not TO_BE_TESTED_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {TO_BE_TESTED_DIR.resolve()}")

    TESTED_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) Load model
    # -------------------------
    print(f"[LOAD] Model: {MODEL_PATH}")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    # -------------------------
    # 2) Load features (source of truth)
    # -------------------------
    print(f"[LOAD] Features: {FEATURES_JSON_PATH}")
    selected_features = load_features(FEATURES_JSON_PATH)
    print(f"[INFO] n_features={len(selected_features)}")

    # -------------------------
    # 3) Load one project CSV
    # -------------------------
    input_path = pick_project_file(TO_BE_TESTED_DIR, PROJECT_FILE)
    print(f"[LOAD] Project CSV: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    # -------------------------
    # 4) Select required features
    # -------------------------
    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required feature columns in the input CSV.\n"
            "This means the model cannot run because the training features are not present.\n"
            "Missing columns:\n" + "\n".join(missing)
        )

    # X is ONLY the columns needed for inference, in the right order.
    # Plain meaning:
    # - We keep full df unchanged, but create X for prediction.
    X = df[selected_features].copy()
    X = coerce_categoricals_to_str(X)

    # -------------------------
    # 5) Predict for all rows
    # -------------------------
    # We compute both:
    # - proba (if available): probability for class 1
    # - label: 0/1 by threshold
    #
    # Plain meaning:
    # - The model outputs probabilities; we convert to a discrete decision.
    try:
        proba = model.predict_proba(X)[:, 1]
        y_pred = (proba >= THRESHOLD).astype(int)
        has_proba = True
    except Exception:
        # Fallback: model might only support predict() for some reason.
        # Plain meaning: we still produce a predicted label column.
        y_pred = model.predict(X).astype(int)
        proba = None
        has_proba = False

    pred_col = f"{TARGET}_predicted"
    df[pred_col] = y_pred

    # Optionally also store probability column (useful for analysis).
    # This means: you can later choose a different threshold without re-running inference.
    if has_proba:
        proba_col = f"{TARGET}_predicted_proba"
        df[proba_col] = proba

    # -------------------------
    # 6) Print one sample prediction
    # -------------------------
    idx = PRINT_ROW_INDEX
    if idx < 0 or idx >= len(df):
        raise IndexError(
            f"PRINT_ROW_INDEX out of bounds: {idx} (dataset has {len(df)} rows)"
        )

    print("\n[SAMPLE PREDICTION]")
    print(f"  row_index: {idx}")
    print(f"  {pred_col}: {int(df.loc[idx, pred_col])}")
    if has_proba:
        print(f"  {TARGET}_predicted_proba: {float(df.loc[idx, proba_col]):.6f}")
    print("  (This means: this is the model’s predicted label for that specific row.)\n")

    # -------------------------
    # 7) Save full table with appended predictions
    # -------------------------
    output_path = TESTED_DIR / input_path.name
    df.to_csv(output_path, index=False)

    print(f"[OK] Saved predictions to: {output_path.resolve()}")
    print(f"[OK] Appended columns: {pred_col}" + (f", {proba_col}" if has_proba else ""))


if __name__ == "__main__":
    main()
