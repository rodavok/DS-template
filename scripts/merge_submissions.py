"""
Merge two submission files, with the specialist overwriting the main.

Useful for training a specialist model on a subset of data (e.g. a specific category)
and blending it back into the global submission.

Usage:
    python scripts/merge_submissions.py <main.csv> <specialist.csv> <output.csv>

    # With custom column names:
    python scripts/merge_submissions.py main.csv specialist.csv merged.csv \\
        --id-col row_id --pred-col prediction

Example:
    python scripts/merge_submissions.py \\
        submissions/submission_main.csv \\
        submissions/submission_specialist.csv \\
        submissions/submission_merged.csv
"""

import sys
import os
import argparse
import pandas as pd


def merge(main_path, specialist_path, output_path, id_col='row_id', pred_col='prediction'):
    main = pd.read_csv(main_path)
    specialist = pd.read_csv(specialist_path)

    if id_col not in main.columns:
        raise ValueError(f"ID column '{id_col}' not found in {main_path}. "
                         f"Available: {list(main.columns)}")

    n_specialist = len(specialist)
    overlap = main[id_col].isin(specialist[id_col]).sum()

    print(f"Main submission:       {len(main):,} rows")
    print(f"Specialist submission: {n_specialist:,} rows")
    print(f"Rows to overwrite:     {overlap:,}")

    merged = main.set_index(id_col)
    merged.update(specialist.set_index(id_col))
    merged = merged.reset_index()[[id_col, pred_col]]

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Merged submission saved to {output_path}")
    print(f"  Predictions range: [{merged[pred_col].min():.4f}, {merged[pred_col].max():.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two submission files")
    parser.add_argument("main", help="Main submission CSV")
    parser.add_argument("specialist", help="Specialist submission CSV (overwrites main)")
    parser.add_argument("output", help="Output CSV path")
    parser.add_argument("--id-col", default="row_id", help="ID column name (default: row_id)")
    parser.add_argument("--pred-col", default="prediction", help="Prediction column name (default: prediction)")
    args = parser.parse_args()

    merge(args.main, args.specialist, args.output, id_col=args.id_col, pred_col=args.pred_col)
