"""
OOF Error Analysis — analyze where the model fails.

Loads oof_predictions.csv from MLflow runs and plots residual distributions,
error by prediction percentile, and feature importance.

Skips runs with time_based_cv=False (random CV — leaks future data).

Usage:
    python scripts/error_analysis.py                        # aggregate all valid runs
    python scripts/error_analysis.py --run-id <prefix>      # filter to a specific run
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # non-interactive backend; remove if running in Jupyter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import mlflow

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_all_oof_from_mlflow(run_id_filter=None):
    """
    Load oof_predictions.csv from all valid MLflow runs.
    Skips runs with time_based_cv=False.
    Averages pred per row across runs if multiple runs cover the same row.
    Returns None if no artifacts found.
    """
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    all_dfs = []
    skipped_random_cv = 0

    print("Scanning MLflow runs for OOF artifacts...")
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)
        for run in runs:
            run_id = run.info.run_id

            if run_id_filter and not run_id.startswith(run_id_filter):
                continue

            if run.data.params.get('time_based_cv', 'True') == 'False':
                skipped_random_cv += 1
                continue

            artifact_paths = [a.path for a in client.list_artifacts(run_id)]
            if 'oof_predictions.csv' not in artifact_paths:
                continue

            try:
                with tempfile.TemporaryDirectory() as tmp:
                    local_path = client.download_artifacts(run_id, 'oof_predictions.csv', tmp)
                    df = pd.read_csv(local_path)
                df['_run_id'] = run_id[:8]
                all_dfs.append(df)
                print(f"  Loaded run {run_id[:8]} ({len(df):,} rows, exp={exp.name})")
            except Exception as e:
                print(f"  Skipping run {run_id[:8]}: {e}")

    if skipped_random_cv:
        print(f"  (Skipped {skipped_random_cv} run(s) with random CV)")

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal OOF rows loaded: {len(combined):,} (from {len(all_dfs)} run(s))")

    # Average predictions across runs (if multiple runs cover the same rows)
    if 'actual' not in combined.columns and 'pred' not in combined.columns:
        print("Warning: OOF CSV must have 'actual' and 'pred' columns.")
        return combined

    if len(all_dfs) > 1:
        # Group by row index within each run and average across runs
        print("Averaging predictions across runs...")
        combined['_row_idx'] = combined.groupby('_run_id').cumcount()
        avg = combined.groupby('_row_idx').agg({'actual': 'first', 'pred': 'mean'}).reset_index(drop=True)
        return avg

    return combined[['actual', 'pred']].copy()


def plot_residuals(oof_df, output_dir):
    """Plot residual distribution and error by percentile bucket."""
    oof_df = oof_df.copy()
    oof_df['residual'] = oof_df['actual'] - oof_df['pred']
    oof_df['abs_error'] = oof_df['residual'].abs()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('OOF Error Analysis', fontsize=14, fontweight='bold')

    # 1. Residual histogram
    ax = axes[0, 0]
    ax.hist(oof_df['residual'], bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(oof_df['residual'].mean(), color='orange', linestyle='--', linewidth=1.5,
               label=f"Mean: {oof_df['residual'].mean():+.4f}")
    ax.set_title('Residual Distribution')
    ax.set_xlabel('actual - pred')
    ax.set_ylabel('Count')
    ax.legend()

    # 2. Predicted vs actual
    ax = axes[0, 1]
    ax.scatter(oof_df['pred'], oof_df['actual'], alpha=0.3, s=5, color='steelblue')
    lo = min(oof_df['pred'].min(), oof_df['actual'].min())
    hi = max(oof_df['pred'].max(), oof_df['actual'].max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_title('Predicted vs Actual')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.legend()

    # 3. MAE by prediction percentile bucket
    ax = axes[1, 0]
    oof_df['pred_bucket'] = pd.qcut(oof_df['pred'], q=10, labels=False, duplicates='drop')
    bucket_mae = oof_df.groupby('pred_bucket')['abs_error'].mean()
    bucket_labels = [f"Q{int(b)+1}" for b in bucket_mae.index]
    ax.bar(bucket_labels, bucket_mae.values, color='steelblue', edgecolor='white')
    ax.set_title('Mean Absolute Error by Prediction Decile')
    ax.set_xlabel('Prediction decile (Q1=lowest, Q10=highest)')
    ax.set_ylabel('Mean |actual - pred|')
    ax.tick_params(axis='x', rotation=45)

    # 4. Cumulative error (largest errors)
    ax = axes[1, 1]
    sorted_errors = oof_df['abs_error'].sort_values(ascending=False).reset_index(drop=True)
    pct = (sorted_errors.index + 1) / len(sorted_errors) * 100
    ax.plot(pct, sorted_errors.cumsum() / sorted_errors.sum(), color='steelblue')
    ax.set_title('Cumulative Error Contribution')
    ax.set_xlabel('Top X% of worst predictions')
    ax.set_ylabel('Fraction of total abs error')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50% of total error')
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'oof_error_analysis.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # Summary stats
    print("\n--- Error Summary ---")
    print(f"  RMSE:       {np.sqrt(np.mean(oof_df['residual']**2)):.4f}")
    print(f"  MAE:        {oof_df['abs_error'].mean():.4f}")
    print(f"  Mean bias:  {oof_df['residual'].mean():+.4f}")
    print(f"  Worst 1%:   MAE = {oof_df.nlargest(int(len(oof_df)*0.01), 'abs_error')['abs_error'].mean():.4f}")
    print(f"  Worst 5%:   MAE = {oof_df.nlargest(int(len(oof_df)*0.05), 'abs_error')['abs_error'].mean():.4f}")


def plot_feature_importance_from_mlflow(output_dir):
    """Load and plot feature importance from the best MLflow run."""
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()

    best_run = None
    best_rmse = float('inf')

    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="metrics.cv_rmse_mean > 0",
            order_by=["metrics.cv_rmse_mean ASC"],
            max_results=1
        )
        for run in runs:
            rmse = run.data.metrics.get('cv_rmse_mean', float('inf'))
            if rmse < best_rmse and run.data.params.get('time_based_cv', 'True') != 'False':
                best_rmse = rmse
                best_run = run

    if best_run is None:
        print("No valid runs found for feature importance.")
        return

    print(f"\nBest run: {best_run.info.run_id[:8]} (CV RMSE: {best_rmse:.4f})")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            local_path = client.download_artifacts(best_run.info.run_id, 'feature_importance.csv', tmp)
            fi = pd.read_csv(local_path)

        top_n = min(30, len(fi))
        fi_top = fi.head(top_n)

        fig, ax = plt.subplots(figsize=(10, top_n * 0.35 + 1))
        ax.barh(range(top_n), fi_top['importance'].values, color='steelblue')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(fi_top['feature'].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f'Top {top_n} Feature Importance (run {best_run.info.run_id[:8]})')
        ax.set_xlabel('Importance')
        plt.tight_layout()

        out_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Could not plot feature importance: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', type=str, default=None,
                        help='Filter to a specific run (prefix or full ID)')
    args = parser.parse_args()

    oof_df = load_all_oof_from_mlflow(run_id_filter=args.run_id)

    if oof_df is not None and 'actual' in oof_df.columns and 'pred' in oof_df.columns:
        plot_residuals(oof_df, OUTPUT_DIR)
    else:
        print("\nNo valid OOF data found. Run the model first to generate oof_predictions.csv.")

    plot_feature_importance_from_mlflow(OUTPUT_DIR)
