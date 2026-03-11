"""
Retroactively log derived metrics to existing MLflow runs.

Computes and logs:
  - cv_fold_trend   : cv_rmse_fold_last - cv_rmse_fold_1 (positive = degrades over time)
  - test_pred_mean  : mean of prediction column from submission CSV
  - test_pred_std   : std  of prediction column from submission CSV
  - test_pred_iqr   : IQR  of prediction column from submission CSV

Run this after adding new metric logging to model.py to backfill old runs.

Usage:
    python scripts/backfill_metrics.py
    python scripts/backfill_metrics.py --pred-col total_bid  # if your predictions column is named differently
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import mlflow

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}")

SUBMISSIONS_DIR = os.path.join(PROJECT_ROOT, "submissions")

parser = argparse.ArgumentParser()
parser.add_argument("--pred-col", default="prediction",
                    help="Name of the prediction column in submission CSVs (default: prediction)")
args = parser.parse_args()
PRED_COL = args.pred_col

# Build map: run_id_short -> submission file path
sub_map = {}
for path in glob.glob(os.path.join(SUBMISSIONS_DIR, "submission_*.csv")):
    basename = os.path.basename(path)
    short_id = basename.replace("submission_", "").replace(".csv", "")
    sub_map[short_id] = path

client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()

trend_updated = 0
pred_updated = 0
skipped = 0

for exp in experiments:
    runs = client.search_runs(exp.experiment_id, max_results=1000)
    print(f"\n{exp.name}: {len(runs)} runs")

    for run in runs:
        rid = run.info.run_id
        short = rid[:8]
        metrics = run.data.metrics
        updated = False

        # --- cv_fold_trend ---
        if "cv_fold_trend" not in metrics:
            fold_scores = {}
            for k, v in metrics.items():
                if k.startswith("cv_rmse_fold_"):
                    try:
                        idx = int(k.split("_")[-1])
                        fold_scores[idx] = v
                    except ValueError:
                        pass

            if fold_scores and len(fold_scores) >= 2:
                first = fold_scores[min(fold_scores)]
                last = fold_scores[max(fold_scores)]
                client.log_metric(rid, "cv_fold_trend", last - first)
                trend_updated += 1
                updated = True

        # --- test prediction distribution from submission CSV ---
        if "test_pred_mean" not in metrics and short in sub_map:
            try:
                df = pd.read_csv(sub_map[short])
                if PRED_COL in df.columns:
                    preds = df[PRED_COL].values
                    client.log_metric(rid, "test_pred_mean", float(np.mean(preds)))
                    client.log_metric(rid, "test_pred_std", float(np.std(preds)))
                    client.log_metric(rid, "test_pred_iqr",
                                      float(np.percentile(preds, 75) - np.percentile(preds, 25)))
                    pred_updated += 1
                    updated = True
                else:
                    print(f"  Warning: '{PRED_COL}' not found in {sub_map[short]}")
            except Exception as e:
                print(f"  Warning: could not process {sub_map[short]}: {e}")

        if not updated:
            skipped += 1

print(f"\nDone.")
print(f"  cv_fold_trend logged:        {trend_updated} runs")
print(f"  test_pred_* logged:          {pred_updated} runs")
print(f"  Runs with nothing to update: {skipped}")
