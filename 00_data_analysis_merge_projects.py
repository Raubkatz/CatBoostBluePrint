#!/usr/bin/env python3
"""
00_merge_projects_to_csv_2026.py

Merge all per-project CSVs from ./nu_files_2026 into one merged dataset in ./merged_csv.

Features:
- Validates/aligns schema against an expected column list (optional but recommended)
- Adds provenance columns: source_file, source_project
- Handles different delimiters via auto-detection (best-effort)
- Writes a single merged CSV (+ optional Parquet if pyarrow is installed)
- Produces a small merge report (rows per file, missing/extra columns)

Usage:
  python 00_merge_projects_to_csv_2026.py

Optional:
  python 00_merge_projects_to_csv_2026.py --input_dir nu_files_2026 --out_dir merged_csv
  python 00_merge_projects_to_csv_2026.py --no_schema_check
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pandas as pd


EXPECTED_COLUMNS: List[str] = [
    "hash","date","fileId","fileName",
    "dok0.004Values","dok0.004AuthorPreCommit","dok0.004AuthorPostCommit","dok0.004Sum",
    "dok0.004Avg","dok0.004Median","dok0.004Std","dok0.004Min","dok0.004Max",
    "dok0.004RelativeAvg","dok0.004RelativeMedian","dok0.004RelativeMin","dok0.004RelativeMax",
    "dok0.004FileBusFactor0.5","dok0.004FileBusFactor0.6","dok0.004FileBusFactor0.7",
    "dok0.004FileBusFactor0.8","dok0.004FileBusFactor0.9","dok0.004FileBusFactor1.0",
    "dok0.004ProjectBusFactor0.5","dok0.004ProjectBusFactor0.6","dok0.004ProjectBusFactor0.7",
    "dok0.004ProjectBusFactor0.8","dok0.004ProjectBusFactor0.9","dok0.004ProjectBusFactor1.0",
    "isBugPresent","isBugfix","isRefactor","totalBugfix","totalRefactors",
    "revision","age","authors","totalAuthors","authorId",
    "totalLoc","locFirst","locLast","locLast-First",
    "deltaTotalAddedToLast","deltaTotalRemovedToLast","deltaTotalReplacedToLast",
    "added","removed","replaced","modified",
    "relativeAdded","relativeRemoved","relativeReplaced","relativeModified",
    "deltaLocToLast","deltaLocToPrev","deltaLocToNext",
    "deltaLocAddedToPrev","deltaLocRemovedToPrev","deltaLocReplacedToPrev","deltaLocModifiedToPrev",
    "deltaLocAddedToNext","deltaLocRemovedToNext","deltaLocReplacedToNext","deltaLocModifiedToNext",
    "hcpf1","hcpf2","hcm1","hcm2","edhcm","ldhcm","lgdhcm",
    "riskScore","nrOfFunctions",
    "ccSum","ccAvg","ccMed","ccMin","ccMax","ccStd",
    "isKillCommit",
    "scatterSingleScore","scatterAvg15","scatterSum15","scatterMax15","scatterMed15",
    "scatterTimeAvg","scatterTimeSum","scatterTimeMax","scatterTimeMed",
]


def sniff_delimiter(path: Path, sample_bytes: int = 64_000) -> str:
    """
    Best-effort delimiter detection. Falls back to comma.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            sample = f.read(sample_bytes)
        # csv.Sniffer can be fragile; keep it defensive.
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ","


def project_name_from_file(path: Path) -> str:
    """
    Extracts a project identifier from filename.
    Adjust if you encode project name differently.
    """
    return path.stem


def read_csv_flex(path: Path) -> pd.DataFrame:
    delim = sniff_delimiter(path)
    # Try fast path first; fall back to python engine if needed.
    try:
        return pd.read_csv(path, sep=delim, low_memory=False, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, sep=delim, low_memory=False, encoding="utf-8", engine="python")


def align_schema(
    df: pd.DataFrame,
    expected_cols: List[str],
    file_tag: str,
    strict: bool = True
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Align df columns to expected schema:
      - add missing expected columns as NA
      - optionally drop extra columns (strict=True keeps ONLY expected + provenance)
    Returns aligned df and a report dict.
    """
    current = list(df.columns)
    missing = [c for c in expected_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_cols]

    for c in missing:
        df[c] = pd.NA

    if strict:
        df = df[expected_cols + [c for c in df.columns if c in ("source_file", "source_project")]]

    report = {
        "file": [file_tag],
        "missing_cols": missing,
        "extra_cols": extra,
        "original_col_count": [len(current)],
        "final_col_count": [len(df.columns)],
    }
    return df, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="nu_files_2026",
                    help="Folder containing per-project CSV files.")
    ap.add_argument("--out_dir", type=str, default="merged_csv",
                    help="Output folder for merged dataset.")
    ap.add_argument("--output_name", type=str, default="merged_dataset_2026.csv",
                    help="Output merged CSV filename.")
    ap.add_argument("--pattern", type=str, default="*.csv",
                    help="Glob pattern for input files.")
    ap.add_argument("--no_schema_check", action="store_true",
                    help="If set, do not align/validate against EXPECTED_COLUMNS.")
    ap.add_argument("--keep_extra_cols", action="store_true",
                    help="If set, keep extra columns when schema_check is enabled.")
    ap.add_argument("--write_parquet", action="store_true",
                    help="Also write Parquet (requires pyarrow).")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.pattern} in {in_dir.resolve()}")

    merged_chunks: List[pd.DataFrame] = []
    file_reports: List[dict] = []

    for p in files:
        df = read_csv_flex(p)

        # Add provenance
        df["source_file"] = p.name
        df["source_project"] = project_name_from_file(p)

        # Schema alignment (optional)
        if not args.no_schema_check:
            df, rep = align_schema(
                df,
                EXPECTED_COLUMNS,
                file_tag=p.name,
                strict=(not args.keep_extra_cols),
            )
            rep["rows"] = [int(len(df))]
            file_reports.append(rep)
        else:
            file_reports.append({
                "file": [p.name],
                "rows": [int(len(df))],
                "missing_cols": [[]],
                "extra_cols": [[]],
                "original_col_count": [int(len(df.columns))],
                "final_col_count": [int(len(df.columns))],
            })

        merged_chunks.append(df)

    merged = pd.concat(merged_chunks, ignore_index=True)

    out_csv = out_dir / args.output_name
    merged.to_csv(out_csv, index=False)

    # Write a quick merge report
    report_path = out_dir / "merge_report.json"
    # Flatten report lists into scalars for readability
    normalized = []
    for r in file_reports:
        normalized.append({
            "file": r["file"][0],
            "rows": r["rows"][0],
            "missing_cols": r["missing_cols"],
            "extra_cols": r["extra_cols"],
            "original_col_count": r["original_col_count"][0],
            "final_col_count": r["final_col_count"][0],
        })
    summary = {
        "input_dir": str(in_dir.resolve()),
        "out_csv": str(out_csv.resolve()),
        "file_count": len(files),
        "total_rows": int(len(merged)),
        "schema_check": (not args.no_schema_check),
        "kept_extra_cols": bool(args.keep_extra_cols),
        "per_file": normalized,
        "final_columns": list(merged.columns),
    }
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.write_parquet:
        try:
            out_parquet = out_dir / (Path(args.output_name).stem + ".parquet")
            merged.to_parquet(out_parquet, index=False)
        except Exception as e:
            print(f"[WARN] Could not write parquet: {e}")

    print(f"[OK] Merged {len(files)} files -> {out_csv} ({len(merged)} rows)")
    print(f"[OK] Report -> {report_path}")


if __name__ == "__main__":
    main()
