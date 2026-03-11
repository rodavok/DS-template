# Project Guidelines

## Setup for a New Competition

1. Copy `src/model.py` and edit the two stubs:
   - `load_and_prepare_data()` — load your CSVs from `data/`, apply row-level pre-processing
   - `prepare_features()` — feature engineering per fold (no leakage)
2. Update `DEFAULT_CONFIG` with your feature flags
3. Set `MLFLOW_EXPERIMENT` in `.env` (or edit the constant in `src/model.py`)
4. Put your data files in `data/`
5. Run: `python src/model.py --model lightgbm`

## Running Scripts

Do NOT run training scripts directly. Present the command to the user.

For single runs:
```bash
python src/model.py --model lightgbm
python src/model.py --model stacking --tune
```

For sweeps, always present as a nohup command:
```bash
nohup python scripts/sweep_template.py --trials 60 > logs/sweep.log 2>&1 &
```

## CLI Arguments (`src/model.py`)

| Argument | Type | Description |
|----------|------|-------------|
| `--model` | choice | Model: `lightgbm`, `xgboost`, `catboost`, `ridge`, `elasticnet`, `randomforest`, `stacking` |
| `--stack-models` | list | Base models for stacking |
| `--tune` | flag | Enable hyperparameter tuning via RandomizedSearchCV |
| `--tune-iterations` | int | Number of tuning iterations (default: 30) |
| `--random-cv` | flag | Use random KFold instead of time-based CV |
| `--cv-folds` | int | Number of CV folds (default: 5) |
| `--gpu` | flag | Enable GPU acceleration |
| `--recency-weight` | float | Exponential decay for recency weighting (0=uniform) |
| `--bias-correction` | flag | Shift predictions by mean OOF residual |
| `--clip-target` | float | Remove training samples above this target percentile (e.g. 0.99) |
| `--train-from` | str | ISO date — only train on data from this date onward |

## MLflow Tracking

All runs are tracked under the experiment set in `MLFLOW_EXPERIMENT` (default: `kaggle-competition`).

View the UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Logged metrics per run:**
- `cv_rmse_mean`, `cv_rmse_std`
- `cv_rmse_fold_1` through `cv_rmse_fold_N`
- `cv_mae_mean`, `cv_mae_std`, `cv_mae_fold_N`
- `cv_fold_trend` (last fold minus first fold; positive = degrades over time)
- `oof_bias` (mean OOF residual)
- `test_pred_mean`, `test_pred_std`, `test_pred_iqr`

**CV validity:**
- Ignore runs with `time_based_cv=False` (`--random-cv`) if your data has a time axis

## ntfy Mobile Notifications

Set `NTFY_TOPIC` in `.env`. Sweep scripts send a notification when complete.
Subscribe to your topic in the ntfy app or at https://ntfy.sh/<your-topic>.

## Key Files

| File | Purpose |
|------|---------|
| `src/model.py` | Main pipeline — **edit the two stubs here** |
| `src/utils.py` | Generic utilities (encoding, target encoding, date features) |
| `scripts/sweep_template.py` | Optuna sweep template — copy and customize per sweep |
| `scripts/merge_submissions.py` | Blend two submission files |
| `scripts/error_analysis.py` | OOF residual analysis + feature importance plots |
| `scripts/backfill_metrics.py` | Retroactively log metrics to existing MLflow runs |
