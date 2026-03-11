"""
Optuna Sweep Template

Searches over feature flags and model configurations using Optuna.
Tracks results in MLflow. Sends mobile notifications via ntfy when done.

Usage:
    python scripts/sweep_template.py --trials 60

To run in the background (recommended for long sweeps):
    nohup python scripts/sweep_template.py --trials 60 > logs/sweep.log 2>&1 &

Customize:
  1. Set STUDY_NAME to a descriptive name for this sweep
  2. Define COMBO_KEYS — the config keys that define a unique trial
  3. Define the search space in objective()
  4. Optionally pin model params in PINNED_MODEL_PARAMS
"""

import os
import sys
import urllib.request

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
from model import main, DEFAULT_CONFIG, _PROJECT_ROOT

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# CUSTOMIZE THESE
# =============================================================================

STUDY_NAME = "my-sweep"   # Name for this sweep (also used as MLflow experiment name)

# Keys that uniquely identify a trial combo (used for duplicate detection).
# List all axes you're searching over.
COMBO_KEYS = [
    "model",
    "recency_weight",
    "clip_target_upper",
    "use_bias_correction",
    # Add your feature flag keys here, e.g.:
    # "use_target_encoding",
    # "train_from",
]

# Optional: pin per-model hyperparameters (loaded from tuned_params.json by default).
# Set to {} to use tuned or default params automatically.
PINNED_MODEL_PARAMS = {}


# =============================================================================
# INFRASTRUCTURE (no need to modify this)
# =============================================================================

TRACKING_URI = f"sqlite:///{os.path.join(_PROJECT_ROOT, 'mlflow.db')}"


def make_combo(params: dict) -> tuple:
    return tuple(params[k] for k in COMBO_KEYS)


def load_existing_combos() -> set:
    """Load previously run combos from this sweep's MLflow experiment."""
    existing = set()
    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)

    exp = client.get_experiment_by_name(STUDY_NAME)
    if exp is None:
        return existing

    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=10000)
    for run in runs:
        p = run.data.params
        if COMBO_KEYS[0] not in p:
            continue
        try:
            combo_values = {}
            for k in COMBO_KEYS:
                raw = p.get(k, "None")
                # Try numeric conversion
                if raw in ("None", "none", ""):
                    combo_values[k] = None
                else:
                    try:
                        combo_values[k] = float(raw) if '.' in raw else int(raw)
                    except (ValueError, TypeError):
                        # Handle booleans
                        if raw in ("True", "False"):
                            combo_values[k] = raw == "True"
                        else:
                            combo_values[k] = raw
            existing.add(make_combo(combo_values))
        except Exception:
            continue

    return existing


_seen_combos: set = set()

mlflc = MLflowCallback(
    tracking_uri=TRACKING_URI,
    metric_name="cv_rmse_mean",
    mlflow_kwargs={"experiment_name": STUDY_NAME},
)


def notify(title, message):
    """Send a push notification via ntfy.sh. Set NTFY_TOPIC in .env to enable."""
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print("[ntfy] NTFY_TOPIC not set, skipping notification")
        return
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode(),
            headers={"Title": title},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[ntfy] notification failed: {e}")


# =============================================================================
# OBJECTIVE: DEFINE YOUR SEARCH SPACE HERE
# =============================================================================

def objective(trial):
    # --- Model selection ---
    model = trial.suggest_categorical("model", ["lightgbm", "xgboost", "catboost"])

    # --- Generic axes (keep or remove as needed) ---
    recency = trial.suggest_categorical("recency_weight", [0.0, 0.5, 1.0, 2.0])
    clip = trial.suggest_categorical("clip_target_upper", [None, 0.99, 0.975, 0.95])
    bias_correction = trial.suggest_categorical("use_bias_correction", [True, False])

    # --- Add your competition-specific feature flag axes below ---
    # Example:
    # use_target_encoding = trial.suggest_categorical("use_target_encoding", [True, False])
    # train_from = trial.suggest_categorical("train_from", [None, "2022-01-01", "2023-01-01"])

    # --- Duplicate detection ---
    combo = make_combo({
        "model": model,
        "recency_weight": recency,
        "clip_target_upper": clip,
        "use_bias_correction": bias_correction,
        # Add your flags here too, e.g.:
        # "use_target_encoding": use_target_encoding,
    })

    if combo in _seen_combos:
        raise optuna.exceptions.TrialPruned()
    _seen_combos.add(combo)

    config = DEFAULT_CONFIG.copy()
    config.update({
        "model": model,
        "recency_weight": recency,
        "clip_target_upper": clip,
        "use_bias_correction": bias_correction,
        # Add your flags here:
        # "use_target_encoding": use_target_encoding,
        "tune": False,
    })

    if PINNED_MODEL_PARAMS:
        config["stacking_model_params"] = PINNED_MODEL_PARAMS

    return main(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=f"Optuna sweep: {STUDY_NAME}")
    parser.add_argument("--trials", type=int, default=60,
                        help="Number of Optuna trials (default: 60)")
    parser.add_argument("--sampler", choices=["random", "tpe"], default="random",
                        help="Optuna sampler (default: random)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(STUDY_NAME)

    print(f"Loading existing combos from '{STUDY_NAME}' experiment...")
    _seen_combos.update(load_existing_combos())
    print(f"  {len(_seen_combos)} existing combos loaded — these will be skipped")
    print()

    sampler = optuna.samplers.RandomSampler() if args.sampler == "random" \
        else optuna.samplers.TPESampler()

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        sampler=sampler,
    )

    print(f"Starting sweep: up to {args.trials} trials ({args.sampler} sampler)")
    print()

    study.optimize(objective, n_trials=args.trials)

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print(f"Trials completed: {completed}")
    print(f"Trials pruned (duplicates): {pruned}")

    if study.best_trial is not None:
        print(f"Best CV RMSE: {study.best_value:.4f}")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        best_params_str = ", ".join(f"{k}={v}" for k, v in study.best_params.items())
        notify(
            title=f"{STUDY_NAME} complete",
            message=f"Best RMSE: {study.best_value:.4f}\n{best_params_str}",
        )
    else:
        print("No trials completed (all pruned as duplicates).")
        notify(
            title=f"{STUDY_NAME} complete",
            message=f"No new trials — all {pruned} pruned as duplicates.",
        )
