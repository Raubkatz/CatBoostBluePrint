#!/usr/bin/env python3
"""
01_sanity_stats_and_clean_2026.py

Reads the merged dataset CSV, lets you manually select:
  - FEATURES: list of feature columns to use
  - TARGET:   class/label column to predict (default: isBugPresent)

Then it:
  1) Runs dataset sanity checks (missingness, dtypes, target distribution, negatives)
  2) Cleans data according to NEGATIVE_HANDLING:
       - "discard"           -> drop rows with any negative numeric feature
       - "replace_with_nan"  -> set negative numeric feature values to NaN
       - "none"              -> do nothing
  3) Writes a cleaned CSV into folder: data_{TARGET}/cleaned_dataset.csv
  4) Exports paper-ready LaTeX feature stats tables (overall + per project if available)

Folder layout:
  data_{TARGET}/
    cleaned_dataset.csv
    sanity_report.txt
    sanity_report.json
    stats_tables/
      stats_overall.tex
      stats_by_source_project.tex
      stats_by_source_project.zip

Usage:
  python 01_sanity_stats_and_clean_2026.py

Adjust paths/FEATURES/TARGET at the top.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


# =========================
# CONFIG (EDIT THESE)
# =========================

# Input merged dataset (from your merge script)
INPUT_CSV = Path("merged_csv") / "merged_dataset_2026.csv"

# Target / class column (what you want to predict)
TARGET = "isBugPresent" #class, c_j

# Feature columns (manually choose!)
FEATURES: List[str] = [ #f_i
    # Example subset — REPLACE with what you want:
    "age", "authors", "totalAuthors", "totalLoc",
    "added", "removed", "replaced", "modified",
    "relativeAdded", "relativeRemoved", "relativeReplaced", "relativeModified",
    "hcpf1", "hcpf2", "hcm1", "hcm2",
    "nrOfFunctions", "ccSum", "ccAvg", "ccStd",
    "scatterSingleScore", "scatterAvg15", "scatterSum15", "scatterMax15", "scatterMed15",
    "scatterTimeAvg", "scatterTimeSum", "scatterTimeMax", "scatterTimeMed",
    "dok0.004FileBusFactor0.5", "dok0.004FileBusFactor0.6", "dok0.004FileBusFactor0.7",
    "dok0.004FileBusFactor0.8", "dok0.004FileBusFactor0.9", "dok0.004FileBusFactor1.0",
    "dok0.004ProjectBusFactor0.5",
    "ccSum", "ccAvg", "ccMed", "ccMin", "ccMax", "ccStd",
    "deltaLocToLast", "deltaLocToPrev", "deltaLocToNext",
    "deltaLocAddedToPrev", "deltaLocRemovedToPrev", "deltaLocReplacedToPrev", "deltaLocModifiedToPrev",
    "deltaLocAddedToNext", "deltaLocRemovedToNext", "deltaLocReplacedToNext", "deltaLocModifiedToNext",
]

# How to handle negative values in numeric feature columns
# Options: "discard" | "replace_with_nan" | "none"
NEGATIVE_HANDLING = "replace_with_nan"

# If true: drop rows with missing TARGET
DROP_MISSING_TARGET = True

# Optional: drop rows with ALL feature values missing (after cleaning)
DROP_ALL_FEATURES_MISSING = True

# Optional group column created by merge script (recommended)
PROJECT_COL = "source_project"  # set to None if you don't have it


# =========================
# Helpers
# =========================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_numeric_series(s: pd.Series) -> pd.Series:
    """
    Robust numeric conversion:
    - handles comma decimal separators
    - strips whitespace
    - converts non-parsable values to NaN
    """
    if pd.api.types.is_numeric_dtype(s):
        return s
    s2 = s.astype(str).str.strip()
    # Replace comma-decimals when it looks like a number with comma
    # e.g. "12,34" -> "12.34"
    s2 = s2.str.replace(r"(?<=\d),(?=\d)", ".", regex=True)
    return pd.to_numeric(s2, errors="coerce")


def compute_feature_stats(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Paper-ready descriptive stats for features:
    count, missing, mean, std, min, 25%, median, 75%, max
    """
    rows = []
    for f in features:
        col = df[f]
        num = to_numeric_series(col)
        missing = int(num.isna().sum())
        count = int(num.notna().sum())
        desc = num.describe(percentiles=[0.25, 0.5, 0.75])
        rows.append({
            "feature": f,
            "count": count,
            "missing": missing,
            "mean": float(desc.get("mean", np.nan)),
            "std": float(desc.get("std", np.nan)),
            "min": float(desc.get("min", np.nan)),
            "p25": float(desc.get("25%", np.nan)),
            "median": float(desc.get("50%", np.nan)),
            "p75": float(desc.get("75%", np.nan)),
            "max": float(desc.get("max", np.nan)),
        })
    out = pd.DataFrame(rows).set_index("feature")
    return out


def latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """
    Produces a minimal LaTeX table string.
    """
    # Keep it readable in LaTeX: round floats
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_float_dtype(df2[c]):
            df2[c] = df2[c].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    tex = df2.to_latex(escape=True)
    # Wrap with table environment
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"{tex}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


def negative_handling(
    df: pd.DataFrame,
    features: List[str],
    mode: str
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply negative handling to numeric feature columns.
    Returns cleaned df and a dict of negative counts per feature (pre-clean).
    """
    mode = mode.lower().strip()
    neg_counts: Dict[str, int] = {}

    # Work on a copy
    out = df.copy()

    for f in features:
        if f not in out.columns:
            continue
        num = to_numeric_series(out[f])
        neg_mask = num < 0
        neg_counts[f] = int(neg_mask.sum())
        # write back numeric-converted column so downstream is consistent
        out[f] = num

        if mode == "replace_with_nan":
            out.loc[neg_mask, f] = np.nan

    if mode == "discard":
        # discard rows where ANY feature is negative (based on numeric-converted columns)
        any_neg = np.zeros(len(out), dtype=bool)
        for f in features:
            if f in out.columns:
                col = out[f]
                if not pd.api.types.is_numeric_dtype(col):
                    col = to_numeric_series(col)
                    out[f] = col
                any_neg |= (col < 0).fillna(False).to_numpy()
        out = out.loc[~any_neg].copy()

    elif mode in ("replace_with_nan", "none"):
        pass
    else:
        raise ValueError(f"Unknown NEGATIVE_HANDLING mode: {mode}")

    return out, neg_counts


# =========================
# Main
# =========================

def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found: {INPUT_CSV.resolve()}")

    # Output folder named after class
    out_root = Path(f"data_{TARGET}")
    ensure_dir(out_root)
    stats_dir = out_root / "stats_tables"
    ensure_dir(stats_dir)

    df = pd.read_csv(INPUT_CSV, low_memory=False)

    # Basic column checks
    missing_cols = [c for c in ([TARGET] + FEATURES) if c not in df.columns]
    if missing_cols:
        raise SystemExit(
            "These requested columns are missing from the dataset:\n"
            + "\n".join(missing_cols)
        )

    # Sanity: target distribution (before cleaning)
    target_series = df[TARGET]
    target_counts = target_series.value_counts(dropna=False).to_dict()

    # Apply negative handling + numeric conversion on features
    df_clean, neg_counts = negative_handling(df, FEATURES, NEGATIVE_HANDLING)

    # Drop missing target if requested
    if DROP_MISSING_TARGET:
        df_clean = df_clean.loc[df_clean[TARGET].notna()].copy()

    # Optionally drop rows where all features are missing after cleaning
    if DROP_ALL_FEATURES_MISSING:
        df_clean = df_clean.loc[~df_clean[FEATURES].isna().all(axis=1)].copy()

    # Recompute target distribution after cleaning
    target_counts_after = df_clean[TARGET].value_counts(dropna=False).to_dict()

    # Missingness summary (after cleaning)
    missingness = {
        "TARGET_missing": int(df_clean[TARGET].isna().sum()),
        "FEATURE_missing_total": int(df_clean[FEATURES].isna().sum().sum()),
        "FEATURE_missing_by_col": df_clean[FEATURES].isna().sum().astype(int).to_dict(),
    }

    # Feature stats (overall)
    stats_overall = compute_feature_stats(df_clean, FEATURES)
    (stats_dir / "stats_overall.tex").write_text(
        latex_table(stats_overall, caption="Feature statistics (overall).", label="tab:stats_overall"),
        encoding="utf-8"
    )

    # Feature stats by project (if available)
    by_project_tex_path = stats_dir / "stats_by_source_project.tex"
    zip_path = stats_dir / "stats_by_source_project.zip"

    by_project_tables: List[Tuple[str, str]] = []
    if PROJECT_COL and (PROJECT_COL in df_clean.columns):
        pieces = []
        for project, g in df_clean.groupby(PROJECT_COL, dropna=False):
            project_name = str(project)
            st = compute_feature_stats(g, FEATURES)
            tex = latex_table(
                st,
                caption=f"Feature statistics for project: {project_name}.",
                label=f"tab:stats_{project_name}".replace(" ", "_").replace(".", "_").replace("-", "_")
            )
            pieces.append(f"% ===== {project_name} =====\n{tex}\n")
            by_project_tables.append((project_name, tex))

        by_project_tex_path.write_text("".join(pieces), encoding="utf-8")

        # Also zip individual per-project .tex files (nice for paper workflow)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for project_name, tex in by_project_tables:
                safe = (
                    project_name.replace("/", "_")
                    .replace("\\", "_")
                    .replace(" ", "_")
                    .replace(":", "_")
                )
                zf.writestr(f"stats_{safe}.tex", tex)

    # Save cleaned CSV
    cleaned_csv_path = out_root / "cleaned_dataset.csv"
    df_clean.to_csv(cleaned_csv_path, index=False)

    # Write reports
    report_txt = out_root / "sanity_report.txt"
    report_json = out_root / "sanity_report.json"

    lines = []
    lines.append(f"INPUT: {INPUT_CSV.resolve()}")
    lines.append(f"OUTPUT_FOLDER: {out_root.resolve()}")
    lines.append(f"OUTPUT_CSV: {cleaned_csv_path.resolve()}")
    lines.append("")
    lines.append(f"TARGET: {TARGET}")
    lines.append(f"NUM_FEATURES: {len(FEATURES)}")
    lines.append(f"NEGATIVE_HANDLING: {NEGATIVE_HANDLING}")
    lines.append(f"DROP_MISSING_TARGET: {DROP_MISSING_TARGET}")
    lines.append(f"DROP_ALL_FEATURES_MISSING: {DROP_ALL_FEATURES_MISSING}")
    lines.append("")
    lines.append(f"ROWS_BEFORE: {len(df)}")
    lines.append(f"ROWS_AFTER:  {len(df_clean)}")
    lines.append("")
    lines.append("TARGET_COUNTS_BEFORE:")
    for k, v in target_counts.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("TARGET_COUNTS_AFTER:")
    for k, v in target_counts_after.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("NEGATIVE_COUNTS_PER_FEATURE (BEFORE NEGATIVE HANDLING):")
    for f in FEATURES:
        lines.append(f"  {f}: {neg_counts.get(f, 0)}")
    lines.append("")
    lines.append("MISSINGNESS_AFTER_CLEANING:")
    lines.append(f"  TARGET_missing: {missingness['TARGET_missing']}")
    lines.append(f"  FEATURE_missing_total: {missingness['FEATURE_missing_total']}")
    lines.append("  FEATURE_missing_by_col:")
    for f, v in missingness["FEATURE_missing_by_col"].items():
        lines.append(f"    {f}: {v}")

    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report_payload = {
        "input_csv": str(INPUT_CSV.resolve()),
        "output_folder": str(out_root.resolve()),
        "cleaned_csv": str(cleaned_csv_path.resolve()),
        "target": TARGET,
        "features": FEATURES,
        "negative_handling": NEGATIVE_HANDLING,
        "drop_missing_target": DROP_MISSING_TARGET,
        "drop_all_features_missing": DROP_ALL_FEATURES_MISSING,
        "rows_before": int(len(df)),
        "rows_after": int(len(df_clean)),
        "target_counts_before": {str(k): int(v) for k, v in target_counts.items()},
        "target_counts_after": {str(k): int(v) for k, v in target_counts_after.items()},
        "negative_counts_per_feature": {k: int(v) for k, v in neg_counts.items()},
        "missingness_after_cleaning": missingness,
        "latex_tables": {
            "overall": str((stats_dir / "stats_overall.tex").resolve()),
            "by_project_combined": str(by_project_tex_path.resolve()) if by_project_tex_path.exists() else None,
            "by_project_zip": str(zip_path.resolve()) if zip_path.exists() else None,
        },
    }
    report_json.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print(f"[OK] Cleaned dataset saved to: {cleaned_csv_path}")
    print(f"[OK] Sanity report: {report_txt}")
    print(f"[OK] Sanity JSON:   {report_json}")
    print(f"[OK] LaTeX stats:   {stats_dir / 'stats_overall.tex'}")
    if by_project_tex_path.exists():
        print(f"[OK] LaTeX by-project (combined): {by_project_tex_path}")
        print(f"[OK] LaTeX by-project (zip):      {zip_path}")
    else:
        print("[INFO] No per-project tables written (PROJECT_COL missing or disabled).")


if __name__ == "__main__":
    main()
