# DS-template

A reusable ML pipeline for Kaggle and other tabular data competitions. Supports LightGBM, XGBoost, CatBoost, Ridge, ElasticNet, RandomForest, and stacking ensembles with MLflow tracking built in.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env: set NTFY_TOPIC and optionally MLFLOW_EXPERIMENT

# 3. Add your data
#    Place CSV files in data/

# 4. Implement the two stubs in src/model.py:
#    - load_and_prepare_data()  — load CSVs, row-level pre-processing
#    - prepare_features()       — feature engineering per fold (no leakage)

# 5. Run
python src/model.py --model lightgbm
```

## Project Structure

```
├── src/
│   ├── model.py          # Main pipeline — edit the two stubs here
│   └── utils.py          # Encoding, target encoding, date features
├── scripts/
│   ├── sweep_template.py     # Optuna sweep — copy and customize per sweep
│   ├── merge_submissions.py  # Blend two submission files
│   ├── error_analysis.py     # OOF residual analysis + feature importance plots
│   └── backfill_metrics.py   # Retroactively log metrics to MLflow runs
├── data/                 # Put your CSVs here (gitignored)
├── notebooks/            # Exploratory notebooks
├── models/               # Saved model artifacts (gitignored)
├── submissions/          # Generated submission files (gitignored)
├── logs/                 # Sweep logs
└── .env.example          # Environment variable template
```

## CLI Reference

```bash
python src/model.py --model <model> [options]
```

| Argument | Description |
|----------|-------------|
| `--model` | `lightgbm`, `xgboost`, `catboost`, `ridge`, `elasticnet`, `randomforest`, `stacking` |
| `--stack-models` | Base models for stacking (e.g. `--stack-models lightgbm xgboost`) |
| `--tune` | Enable hyperparameter tuning via RandomizedSearchCV |
| `--tune-iterations` | Number of tuning iterations (default: 30) |
| `--cv-folds` | Number of CV folds (default: 5) |
| `--random-cv` | Use random KFold instead of time-based CV |
| `--recency-weight` | Exponential decay weight for recent samples (0 = uniform) |
| `--bias-correction` | Shift predictions by mean OOF residual |
| `--clip-target` | Remove training samples above this target percentile (e.g. `0.99`) |
| `--train-from` | ISO date — only train on data from this date onward |
| `--gpu` | Enable GPU acceleration |

**Hyperparameter sweeps** (run in background):
```bash
nohup python scripts/sweep_template.py --trials 60 > logs/sweep.log 2>&1 &
```

## MLflow Tracking

All runs are logged automatically. Launch the UI with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Logged per run: `cv_rmse_mean/std`, per-fold RMSE/MAE, `cv_fold_trend`, `oof_bias`, and test prediction statistics.

## Notifications

Set `NTFY_TOPIC` in `.env` to receive a mobile push notification when a sweep completes. Subscribe at [ntfy.sh](https://ntfy.sh) or in the ntfy app.

## Dependencies

Python 3.10+. Key packages: `lightgbm`, `xgboost`, `catboost`, `scikit-learn`, `optuna`, `mlflow`, `pandas`, `numpy`.

Install all with:
```bash
pip install -r requirements.txt
```
