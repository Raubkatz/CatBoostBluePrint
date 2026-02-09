#!/usr/bin/env python3
"""
02_train_test_split_2026.py

Loads the cleaned dataset from:
  data_{TARGET}/cleaned_dataset.csv

Performs a train/test split and writes:
  data_{TARGET}/splits/train.csv
  data_{TARGET}/splits/test.csv

Supports:
- Stratified split on TARGET (recommended for classification)
- Optional group-aware split (e.g., by source_project) using GroupShuffleSplit
- Optional "repo holdout" mode (test contains only unseen projects)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit


# =========================
# CONFIG (EDIT THESE)
# =========================

TARGET = "isBugPresent"

CLEANED_CSV = Path(f"data_{TARGET}") / "cleaned_dataset.csv"


FEATURES: Optional[List[str]] = None

EXCLUDE_COLS = [
    TARGET,
    "hash", "date", "fileId", "fileName",
    "source_file", "source_project",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

OUT_DIR = Path(f"data_{TARGET}") / f"splits_{str(RANDOM_STATE)}"

STRATIFY = True

GROUP_COL: Optional[str] = None  # set to None to disable
REQUIRE_GROUP_HOLDOUT = False


# =========================
# Helpers
# =========================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def infer_features(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in set(EXCLUDE_COLS)]


def main() -> None:
    if not CLEANED_CSV.exists():
        raise SystemExit(f"Cleaned dataset not found: {CLEANED_CSV.resolve()}")

    ensure_dir(OUT_DIR)

    df = pd.read_csv(CLEANED_CSV, low_memory=False)

    if TARGET not in df.columns:
        raise SystemExit(f"TARGET column '{TARGET}' missing from dataset.")

    features = FEATURES if FEATURES is not None else infer_features(df)

    missing = [c for c in ([TARGET] + features) if c not in df.columns]
    if missing:
        raise SystemExit("Missing columns:\n" + "\n".join(missing))

    # Drop rows with missing target
    df = df.loc[df[TARGET].notna()].copy()

    # IMPORTANT: do NOT mutate GROUP_COL (global) inside main()
    group_col = GROUP_COL

    # Validate group column if requested
    if group_col is not None and group_col not in df.columns:
        if REQUIRE_GROUP_HOLDOUT:
            raise SystemExit(
                f"GROUP_COL='{group_col}' requested but missing from dataset.\n"
                f"Set GROUP_COL=None or ensure it's present."
            )
        group_col = None

    # Split
    if group_col is None:
        strat = df[TARGET] if STRATIFY else None
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=strat,
            shuffle=True,
        )
        split_mode = "train_test_split" + ("_stratified" if STRATIFY else "")
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        groups = df[group_col].astype(str)
        train_idx, test_idx = next(gss.split(df, groups=groups))
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        split_mode = f"GroupShuffleSplit({group_col})"

    # Save
    train_path = OUT_DIR / "train.csv"
    test_path = OUT_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Report
    report = {
        "cleaned_csv": str(CLEANED_CSV.resolve()),
        "out_dir": str(OUT_DIR.resolve()),
        "target": TARGET,
        "n_features": len(features),
        "features": features,
        "split_mode": split_mode,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "target_counts_total": df[TARGET].value_counts(dropna=False).to_dict(),
        "target_counts_train": train_df[TARGET].value_counts(dropna=False).to_dict(),
        "target_counts_test": test_df[TARGET].value_counts(dropna=False).to_dict(),
    }

    if group_col is not None:
        report["group_col"] = group_col
        report["n_groups_total"] = int(df[group_col].nunique(dropna=False))
        report["n_groups_train"] = int(train_df[group_col].nunique(dropna=False))
        report["n_groups_test"] = int(test_df[group_col].nunique(dropna=False))
        report["group_overlap"] = int(
            len(set(train_df[group_col].astype(str)) & set(test_df[group_col].astype(str)))
        )

    import json
    report_path = OUT_DIR / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] Split mode: {split_mode}")
    print(f"[OK] Train: {train_path} ({len(train_df)} rows)")
    print(f"[OK] Test:  {test_path} ({len(test_df)} rows)")
    print(f"[OK] Report: {report_path}")


if __name__ == "__main__":
    main()