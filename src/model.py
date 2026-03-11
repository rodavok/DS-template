"""
Generic Kaggle ML Pipeline

Supports LightGBM, XGBoost, CatBoost, Ridge, ElasticNet, RandomForest, and Stacking.

To use for a new competition:
  1. Implement load_and_prepare_data() with your CSV loading and pre-processing
  2. Implement prepare_features() with your feature engineering
  3. Update DEFAULT_CONFIG with your feature flags
  4. Run: python src/model.py --model lightgbm
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, KFold
from scipy.stats import randint, uniform
import warnings
import mlflow

warnings.filterwarnings('ignore')

# Resolve project root from this file's location so it works on any machine
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(_PROJECT_ROOT, 'mlflow.db')}")

# Set this to the name of your competition/project
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "kaggle-competition")

TUNED_PARAMS_FILE = os.path.join(_PROJECT_ROOT, "tuned_params.json")


def _fix_mlflow_paths():
    """
    Patch all absolute paths in mlflow.db that point to a different machine.
    This happens when mlflow.db is synced between machines with different usernames/paths.
    Updates experiment artifact_location and all run artifact_uris.
    """
    import sqlite3

    db_path = os.path.join(_PROJECT_ROOT, "mlflow.db")
    if not os.path.exists(db_path):
        return

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT experiment_id, artifact_location FROM experiments").fetchall()

    for exp_id, artifact_location in rows:
        suffix = os.path.join("mlruns", str(exp_id))
        if artifact_location.endswith(suffix):
            old_root = artifact_location[: -len(suffix)].rstrip("/")
            if old_root != _PROJECT_ROOT:
                conn.execute(
                    "UPDATE experiments SET artifact_location = replace(artifact_location, ?, ?) "
                    "WHERE artifact_location LIKE ?",
                    (old_root, _PROJECT_ROOT, f"{old_root}%"),
                )
                runs_updated = conn.execute(
                    "UPDATE runs SET artifact_uri = replace(artifact_uri, ?, ?) "
                    "WHERE artifact_uri LIKE ?",
                    (old_root, _PROJECT_ROOT, f"{old_root}%"),
                ).rowcount
                conn.commit()
                print(f"[mlflow] Repointed paths: {old_root} -> {_PROJECT_ROOT} ({runs_updated} runs updated)")
                break

    conn.close()


_fix_mlflow_paths()
mlflow.set_experiment(MLFLOW_EXPERIMENT)


def load_tuned_params():
    """Load previously tuned parameters from disk."""
    if os.path.exists(TUNED_PARAMS_FILE):
        with open(TUNED_PARAMS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_tuned_params(model_name, params):
    """Save tuned parameters for a model to disk."""
    all_params = load_tuned_params()
    clean_params = {}
    for k, v in params.items():
        if hasattr(v, 'item'):  # numpy scalar
            clean_params[k] = v.item()
        else:
            clean_params[k] = v
    all_params[model_name] = clean_params
    with open(TUNED_PARAMS_FILE, 'w') as f:
        json.dump(all_params, f, indent=2)
    print(f"Saved tuned params for '{model_name}' to {TUNED_PARAMS_FILE}")


def compute_recency_weights(dates, alpha):
    """
    Exponential decay sample weights that up-weight recent samples.

    alpha=0  → uniform weights (no decay)
    alpha=1  → a sample from 1 year ago gets weight exp(-1) ≈ 0.37x vs most recent
    alpha=2  → a sample from 1 year ago gets weight exp(-2) ≈ 0.14x vs most recent

    Weights are normalized to mean=1 so effective dataset size is unchanged.
    """
    if alpha == 0:
        return None
    dates = pd.to_datetime(dates)
    days_old = (dates.max() - dates).dt.days.values.astype(float)
    weights = np.exp(-alpha * days_old / 365.0)
    weights /= weights.mean()
    return weights


# =============================================================================
# MODEL BACKENDS
# =============================================================================

def get_lightgbm_backend():
    import lightgbm as lgb

    def get_default_params(config):
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': config['num_leaves'],
            'learning_rate': config['learning_rate'],
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
            'n_jobs': -1
        }

    def get_tuned_params(tuned):
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'seed': 42,
            'n_jobs': -1,
            'num_leaves': tuned.get('num_leaves', 63),
            'max_depth': tuned.get('max_depth', -1),
            'learning_rate': tuned.get('learning_rate', 0.05),
            'min_child_samples': tuned.get('min_child_samples', 20),
            'feature_fraction': tuned.get('colsample_bytree', 0.8),
            'bagging_fraction': tuned.get('subsample', 0.8),
            'bagging_freq': 5,
            'lambda_l1': tuned.get('reg_alpha', 0),
            'lambda_l2': tuned.get('reg_lambda', 0),
        }

    def get_tune_distributions():
        return {
            'num_leaves': randint(31, 96),
            'max_depth': randint(4, 10),
            'learning_rate': uniform(0.03, 0.12),
            'n_estimators': randint(200, 500),
            'min_child_samples': randint(10, 50),
            'subsample': uniform(0.6, 0.35),
            'colsample_bytree': uniform(0.6, 0.35),
            'reg_alpha': uniform(0, 2),
            'reg_lambda': uniform(0, 2),
        }

    def get_tuning_model():
        return lgb.LGBMRegressor(
            objective='regression',
            boosting_type='gbdt',
            verbose=-1,
            n_jobs=-1,
            random_state=42
        )

    def train(X_tr, y_tr, X_val, y_val, params, num_boost_round, sample_weight=None):
        train_data = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight)
        val_data = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        return model

    def train_final(X_train, y_train, params, num_boost_round, sample_weight=None):
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        return lgb.train(params, train_data, num_boost_round=num_boost_round)

    def predict(model, X):
        return model.predict(X)

    def get_feature_importance(model, feature_cols):
        return pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False)

    return {
        'name': 'lightgbm',
        'get_default_params': get_default_params,
        'get_tuned_params': get_tuned_params,
        'get_tune_distributions': get_tune_distributions,
        'get_tuning_model': get_tuning_model,
        'train': train,
        'train_final': train_final,
        'predict': predict,
        'get_feature_importance': get_feature_importance,
    }


def get_xgboost_backend():
    import xgboost as xgb

    def get_default_params(config):
        return {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': config['learning_rate'],
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
            'n_jobs': -1,
        }

    def get_tuned_params(tuned):
        return {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': 42,
            'n_jobs': -1,
            'max_depth': tuned.get('max_depth', 6),
            'learning_rate': tuned.get('learning_rate', 0.05),
            'min_child_weight': tuned.get('min_child_weight', 1),
            'subsample': tuned.get('subsample', 0.8),
            'colsample_bytree': tuned.get('colsample_bytree', 0.8),
            'reg_alpha': tuned.get('reg_alpha', 0),
            'reg_lambda': tuned.get('reg_lambda', 1),
        }

    def get_tune_distributions():
        return {
            'max_depth': randint(4, 10),
            'learning_rate': uniform(0.03, 0.12),
            'n_estimators': randint(200, 500),
            'min_child_weight': randint(1, 10),
            'subsample': uniform(0.6, 0.35),
            'colsample_bytree': uniform(0.6, 0.35),
            'reg_alpha': uniform(0, 2),
            'reg_lambda': uniform(0, 2),
        }

    def get_tuning_model():
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42
        )

    def train(X_tr, y_tr, X_val, y_val, params, num_boost_round, sample_weight=None):
        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sample_weight)
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=False
        )
        return model

    def train_final(X_train, y_train, params, num_boost_round, sample_weight=None):
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
        return xgb.train(params, dtrain, num_boost_round=num_boost_round)

    def predict(model, X):
        dtest = xgb.DMatrix(X)
        return model.predict(dtest)

    def get_feature_importance(model, feature_cols):
        scores = model.get_score(importance_type='gain')
        importance = [scores.get(f, 0) for f in feature_cols]
        return pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

    return {
        'name': 'xgboost',
        'get_default_params': get_default_params,
        'get_tuned_params': get_tuned_params,
        'get_tune_distributions': get_tune_distributions,
        'get_tuning_model': get_tuning_model,
        'train': train,
        'train_final': train_final,
        'predict': predict,
        'get_feature_importance': get_feature_importance,
    }


def get_catboost_backend():
    from catboost import CatBoostRegressor

    def get_default_params(config):
        return {
            'loss_function': 'RMSE',
            'depth': 6,
            'learning_rate': config['learning_rate'],
            'random_seed': 42,
            'verbose': False,
        }

    def get_tuned_params(tuned):
        return {
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': False,
            'depth': tuned.get('depth', 6),
            'learning_rate': tuned.get('learning_rate', 0.05),
            'l2_leaf_reg': tuned.get('l2_leaf_reg', 3),
            'bagging_temperature': tuned.get('bagging_temperature', 1),
        }

    def get_tune_distributions():
        return {
            'depth': randint(4, 10),
            'learning_rate': uniform(0.03, 0.12),
            'iterations': randint(200, 500),
            'l2_leaf_reg': uniform(1, 9),
            'bagging_temperature': uniform(0, 1),
        }

    def get_tuning_model():
        return CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=False)

    def train(X_tr, y_tr, X_val, y_val, params, num_boost_round, sample_weight=None):
        model = CatBoostRegressor(**params, iterations=num_boost_round)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100,
                  verbose=False, sample_weight=sample_weight)
        return model

    def train_final(X_train, y_train, params, num_boost_round, sample_weight=None):
        model = CatBoostRegressor(**params, iterations=num_boost_round)
        model.fit(X_train, y_train, verbose=False, sample_weight=sample_weight)
        return model

    def predict(model, X):
        return model.predict(X)

    def get_feature_importance(model, feature_cols):
        return pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

    return {
        'name': 'catboost',
        'get_default_params': get_default_params,
        'get_tuned_params': get_tuned_params,
        'get_tune_distributions': get_tune_distributions,
        'get_tuning_model': get_tuning_model,
        'train': train,
        'train_final': train_final,
        'predict': predict,
        'get_feature_importance': get_feature_importance,
    }


def get_ridge_backend():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    def get_default_params(_config):
        return {'alpha': 1.0}

    def get_tuned_params(tuned):
        return {'alpha': tuned.get('ridge__alpha', 1.0)}

    def get_tune_distributions():
        from scipy.stats import loguniform
        return {'ridge__alpha': loguniform(1e-3, 1e3)}

    def get_tuning_model():
        return Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])

    def train(X_tr, y_tr, _X_val, _y_val, params, _num_boost_round, sample_weight=None):
        model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(**params))])
        model.fit(X_tr, y_tr, ridge__sample_weight=sample_weight)
        return model

    def train_final(X_train, y_train, params, _num_boost_round, sample_weight=None):
        model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(**params))])
        model.fit(X_train, y_train, ridge__sample_weight=sample_weight)
        return model

    def predict(model, X):
        return model.predict(X)

    def get_feature_importance(model, feature_cols):
        coeffs = model.named_steps['ridge'].coef_
        return pd.DataFrame({
            'feature': feature_cols,
            'importance': np.abs(coeffs)
        }).sort_values('importance', ascending=False)

    return {
        'name': 'ridge',
        'get_default_params': get_default_params,
        'get_tuned_params': get_tuned_params,
        'get_tune_distributions': get_tune_distributions,
        'get_tuning_model': get_tuning_model,
        'train': train,
        'train_final': train_final,
        'predict': predict,
        'get_feature_importance': get_feature_importance,
    }


def get_elasticnet_backend():
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    def get_default_params(_config):
        return {'alpha': 1.0, 'l1_ratio': 0.5, 'max_iter': 5000}

    def get_tuned_params(tuned):
        return {
            'alpha': tuned.get('elasticnet__alpha', 1.0),
            'l1_ratio': tuned.get('elasticnet__l1_ratio', 0.5),
            'max_iter': 5000,
        }

    def get_tune_distributions():
        from scipy.stats import loguniform
        return {
            'elasticnet__alpha': loguniform(1e-3, 1e3),
            'elasticnet__l1_ratio': uniform(0.01, 0.98),
        }

    def get_tuning_model():
        return Pipeline([('scaler', StandardScaler()), ('elasticnet', ElasticNet(max_iter=5000))])

    def train(X_tr, y_tr, _X_val, _y_val, params, _num_boost_round, sample_weight=None):
        model = Pipeline([('scaler', StandardScaler()), ('elasticnet', ElasticNet(**params))])
        model.fit(X_tr, y_tr, elasticnet__sample_weight=sample_weight)
        return model

    def train_final(X_train, y_train, params, _num_boost_round, sample_weight=None):
        model = Pipeline([('scaler', StandardScaler()), ('elasticnet', ElasticNet(**params))])
        model.fit(X_train, y_train, elasticnet__sample_weight=sample_weight)
        return model

    def predict(model, X):
        return model.predict(X)

    def get_feature_importance(model, feature_cols):
        coeffs = model.named_steps['elasticnet'].coef_
        return pd.DataFrame({
            'feature': feature_cols,
            'importance': np.abs(coeffs)
        }).sort_values('importance', ascending=False)

    return {
        'name': 'elasticnet',
        'get_default_params': get_default_params,
        'get_tuned_params': get_tuned_params,
        'get_tune_distributions': get_tune_distributions,
        'get_tuning_model': get_tuning_model,
        'train': train,
        'train_final': train_final,
        'predict': predict,
        'get_feature_importance': get_feature_importance,
    }


def get_random_forest_backend():
    from sklearn.ensemble import RandomForestRegressor

    def get_default_params(_config):
        return {
            'n_estimators': 500,
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': 42,
        }

    def get_tuned_params(tuned):
        return {
            'n_estimators': tuned.get('n_estimators', 500),
            'max_depth': tuned.get('max_depth', None),
            'min_samples_split': tuned.get('min_samples_split', 5),
            'min_samples_leaf': tuned.get('min_samples_leaf', 2),
            'max_features': tuned.get('max_features', 'sqrt'),
            'n_jobs': -1,
            'random_state': 42,
        }

    def get_tune_distributions():
        return {
            'n_estimators': randint(200, 800),
            'max_depth': [None, 10, 15, 20, 25, 30],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', 0.5, 0.7],
        }

    def get_tuning_model():
        return RandomForestRegressor(n_jobs=-1, random_state=42)

    def train(X_tr, y_tr, _X_val, _y_val, params, _num_boost_round, sample_weight=None):
        model = RandomForestRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
        return model

    def train_final(X_train, y_train, params, _num_boost_round, sample_weight=None):
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model

    def predict(model, X):
        return model.predict(X)

    def get_feature_importance(model, feature_cols):
        return pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

    return {
        'name': 'randomforest',
        'get_default_params': get_default_params,
        'get_tuned_params': get_tuned_params,
        'get_tune_distributions': get_tune_distributions,
        'get_tuning_model': get_tuning_model,
        'train': train,
        'train_final': train_final,
        'predict': predict,
        'get_feature_importance': get_feature_importance,
    }


def get_stacking_backend(stack_models=None, stacking_model_params=None, use_gpu=False):
    """
    Stacking ensemble with configurable base models. Ridge as meta-learner.
    Uses out-of-fold predictions to train the meta-model (no leakage).

    Run each base model with --tune first to populate tuned_params.json:
        python src/model.py --model lightgbm --tune
        python src/model.py --model stacking --stack-models lightgbm elasticnet
    """
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    if stack_models is None:
        stack_models = ['lightgbm', 'xgboost', 'catboost']

    _stack_models = stack_models
    _stacking_model_params = stacking_model_params or {}
    _use_gpu = use_gpu

    def _make_estimator(name, saved_params):
        if name == 'lightgbm':
            params = {
                'n_estimators': saved_params.get('n_estimators', 500),
                'num_leaves': saved_params.get('num_leaves', 63),
                'max_depth': saved_params.get('max_depth', -1),
                'learning_rate': saved_params.get('learning_rate', 0.05),
                'min_child_samples': saved_params.get('min_child_samples', 20),
                'subsample': saved_params.get('subsample', 0.8),
                'colsample_bytree': saved_params.get('colsample_bytree', 0.8),
                'reg_alpha': saved_params.get('reg_alpha', 0),
                'reg_lambda': saved_params.get('reg_lambda', 0),
                'verbose': -1, 'n_jobs': -1, 'random_state': 42,
            }
            if _use_gpu:
                params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
            return ('lgb', lgb.LGBMRegressor(**params))

        elif name == 'xgboost':
            params = {
                'n_estimators': saved_params.get('n_estimators', 500),
                'max_depth': saved_params.get('max_depth', 6),
                'learning_rate': saved_params.get('learning_rate', 0.05),
                'min_child_weight': saved_params.get('min_child_weight', 1),
                'subsample': saved_params.get('subsample', 0.8),
                'colsample_bytree': saved_params.get('colsample_bytree', 0.8),
                'reg_alpha': saved_params.get('reg_alpha', 0),
                'reg_lambda': saved_params.get('reg_lambda', 1),
                'n_jobs': -1, 'random_state': 42,
            }
            if _use_gpu:
                params.update({'device': 'cuda', 'tree_method': 'hist'})
            return ('xgb', xgb.XGBRegressor(**params))

        elif name == 'catboost':
            params = {
                'iterations': saved_params.get('iterations', 500),
                'depth': saved_params.get('depth', 6),
                'learning_rate': saved_params.get('learning_rate', 0.05),
                'l2_leaf_reg': saved_params.get('l2_leaf_reg', 3),
                'bagging_temperature': saved_params.get('bagging_temperature', 1),
                'verbose': False, 'random_seed': 42,
            }
            if _use_gpu:
                params.update({'task_type': 'GPU', 'devices': '0'})
            return ('catboost', CatBoostRegressor(**params))

        elif name == 'ridge':
            return ('ridge', Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=saved_params.get('alpha', 1.0)))
            ]))

        elif name == 'elasticnet':
            return ('elasticnet', Pipeline([
                ('scaler', StandardScaler()),
                ('elasticnet', ElasticNet(
                    alpha=saved_params.get('alpha', 1.0),
                    l1_ratio=saved_params.get('l1_ratio', 0.5),
                    max_iter=5000,
                ))
            ]))

        elif name == 'randomforest':
            return ('rf', RandomForestRegressor(
                n_estimators=saved_params.get('n_estimators', 500),
                max_depth=saved_params.get('max_depth', None),
                min_samples_split=saved_params.get('min_samples_split', 5),
                min_samples_leaf=saved_params.get('min_samples_leaf', 2),
                max_features=saved_params.get('max_features', 'sqrt'),
                n_jobs=-1, random_state=42,
            ))

        else:
            raise ValueError(f"Unknown model for stacking: {name}")

    def _make_base_estimators():
        saved = load_tuned_params()
        estimators = []
        for name in _stack_models:
            if _stacking_model_params.get(name):
                model_params = {**saved.get(name, {}), **_stacking_model_params[name]}
                print(f"  {name}: using trial params")
            elif saved.get(name):
                model_params = saved[name]
                print(f"  {name}: using tuned params")
            else:
                model_params = {}
                print(f"  {name}: using defaults (run --model {name} --tune to tune)")
            estimators.append(_make_estimator(name, model_params))
        return estimators

    def _fit_base_model(short_name, estimator, X, y, sample_weight=None):
        if sample_weight is None:
            estimator.fit(X, y)
        elif hasattr(estimator, 'named_steps'):
            last_step = list(estimator.named_steps.keys())[-1]
            try:
                estimator.fit(X, y, **{f'{last_step}__sample_weight': sample_weight})
            except TypeError:
                estimator.fit(X, y)
        else:
            try:
                estimator.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                estimator.fit(X, y)

    class _ManualStackingModel:
        def __init__(self, base_models, meta_learner):
            self.base_models = base_models
            self.meta_learner = meta_learner
            self.named_estimators_ = {name: est for name, est in base_models}

        def predict(self, X):
            meta_X = np.column_stack([est.predict(X) for _, est in self.base_models])
            return self.meta_learner.predict(meta_X)

    def _train_stacking(X_tr, y_tr, params, sample_weight=None, n_meta_folds=5):
        from sklearn.base import clone

        X_arr = X_tr if isinstance(X_tr, np.ndarray) else X_tr.values
        y_arr = y_tr if isinstance(y_tr, np.ndarray) else np.array(y_tr)
        alpha = params.get('final_alpha', 1.0)
        estimators = _make_base_estimators()

        meta_X = np.zeros((len(y_arr), len(estimators)))
        kf = KFold(n_splits=n_meta_folds, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(X_arr):
            sw_fold = sample_weight[tr_idx] if sample_weight is not None else None
            for est_idx, (short_name, proto) in enumerate(estimators):
                fold_est = clone(proto)
                _fit_base_model(short_name, fold_est, X_arr[tr_idx], y_arr[tr_idx], sw_fold)
                meta_X[val_idx, est_idx] = fold_est.predict(X_arr[val_idx])

        meta_learner = Ridge(alpha=alpha)
        if sample_weight is not None:
            meta_learner.fit(meta_X, y_arr, sample_weight=sample_weight)
        else:
            meta_learner.fit(meta_X, y_arr)

        final_estimators = []
        for short_name, proto in estimators:
            final_est = clone(proto)
            _fit_base_model(short_name, final_est, X_arr, y_arr, sample_weight)
            final_estimators.append((short_name, final_est))

        return _ManualStackingModel(final_estimators, meta_learner)

    def get_default_params(_config):
        return {'final_alpha': 1.0}

    def get_tuned_params(tuned):
        return {'final_alpha': tuned.get('final_estimator__alpha', 1.0)}

    def get_tune_distributions():
        from scipy.stats import loguniform
        return {'final_estimator__alpha': loguniform(1e-3, 1e3)}

    def get_tuning_model():
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import StackingRegressor
        stacking_n_jobs = 1 if _use_gpu else -1
        return StackingRegressor(
            estimators=_make_base_estimators(),
            final_estimator=Ridge(alpha=1.0),
            cv=5, n_jobs=stacking_n_jobs, passthrough=False,
        )

    def train(X_tr, y_tr, _X_val, _y_val, params, _num_boost_round, sample_weight=None):
        return _train_stacking(X_tr, y_tr, params, sample_weight)

    def train_final(X_train, y_train, params, _num_boost_round, sample_weight=None):
        return _train_stacking(X_train, y_train, params, sample_weight)

    def predict(model, X):
        return model.predict(X)

    def get_feature_importance(model, feature_cols):
        importances = {}
        normalized = []
        name_map = {'lgb': 'lightgbm', 'xgb': 'xgboost'}
        for short_name, estimator in model.named_estimators_.items():
            full_name = name_map.get(short_name, short_name)
            imp = None
            if hasattr(estimator, 'feature_importances_'):
                imp = estimator.feature_importances_
            elif hasattr(estimator, 'named_steps'):
                for step_name in ['ridge', 'elasticnet']:
                    if step_name in estimator.named_steps:
                        imp = np.abs(estimator.named_steps[step_name].coef_)
                        break
            elif hasattr(estimator, 'coef_'):
                imp = np.abs(estimator.coef_)
            if imp is not None:
                importances[f'importance_{short_name}'] = imp
                normalized.append(imp / (imp.max() + 1e-10))

        combined = np.mean(normalized, axis=0) if normalized else np.zeros(len(feature_cols))
        result = {'feature': feature_cols, 'importance': combined}
        result.update(importances)
        return pd.DataFrame(result).sort_values('importance', ascending=False)

    return {
        'name': 'stacking',
        'get_default_params': get_default_params,
        'get_tuned_params': get_tuned_params,
        'get_tune_distributions': get_tune_distributions,
        'get_tuning_model': get_tuning_model,
        'train': train,
        'train_final': train_final,
        'predict': predict,
        'get_feature_importance': get_feature_importance,
    }


def _get_gpu_params(model_name):
    """Return GPU-specific params to overlay on top of any model config."""
    if model_name == 'lightgbm':
        return {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
    elif model_name == 'xgboost':
        return {'device': 'cuda', 'tree_method': 'hist'}
    elif model_name == 'catboost':
        return {'task_type': 'GPU', 'devices': '0'}
    return {}


def get_backend(name, config=None):
    backends = {
        'lightgbm': get_lightgbm_backend,
        'xgboost': get_xgboost_backend,
        'catboost': get_catboost_backend,
        'ridge': get_ridge_backend,
        'elasticnet': get_elasticnet_backend,
        'randomforest': get_random_forest_backend,
        'stacking': get_stacking_backend,
    }
    if name not in backends:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(backends.keys())}")
    if name == 'stacking' and config is not None:
        return backends[name](
            stack_models=config.get('stack_models', ['lightgbm', 'xgboost', 'catboost']),
            stacking_model_params=config.get('stacking_model_params', {}),
            use_gpu=config.get('use_gpu', False),
        )
    return backends[name]()


# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================

DEFAULT_CONFIG = {
    # Model selection
    'model': 'lightgbm',
    'stack_models': ['lightgbm', 'xgboost', 'catboost'],
    'stacking_model_params': {},  # Per-model param overrides for stacking (used by Optuna sweep)

    # Date column name in your DataFrame (used for time-based CV and recency weights).
    # Set to None if your data has no time axis.
    'date_col': 'date',

    # Recency weighting: exponential decay factor applied to sample weights.
    # 0 = uniform, 1 = sample from 1 year ago weighted at exp(-1)≈0.37x vs most recent.
    'recency_weight': 0.0,

    # Outlier clipping: remove training samples above this percentile of the target.
    # e.g. 0.99 removes the top 1% of targets. Applied per fold and on final training.
    'clip_target_upper': None,

    # Restrict model training to samples on or after this date (ISO format, e.g. '2022-07-01').
    # Useful for dropping older data whose distribution has shifted.
    'train_from': None,

    # Bias correction: shift predictions by the mean OOF residual (in target space).
    # Corrects systematic under/over-prediction. OOF bias is always printed.
    'use_bias_correction': False,

    # Cross-validation
    'time_based_cv': True,   # True = TimeSeriesSplit, False = random KFold
    'cv_splits': 5,

    # GPU acceleration
    'use_gpu': False,

    # Hyperparameter tuning
    'tune': False,
    'tune_iterations': 30,

    # Model params (used when not tuning)
    'num_leaves': 63,      # LightGBM
    'learning_rate': 0.05,
    'num_boost_round': 2000,

    # -------------------------------------------------------------------------
    # ADD YOUR COMPETITION-SPECIFIC FEATURE FLAGS BELOW
    # -------------------------------------------------------------------------
    # Example:
    # 'use_target_encoding': False,
    # 'use_lag_features': False,
}


def get_config(**overrides):
    """Get config with optional overrides."""
    config = DEFAULT_CONFIG.copy()
    config.update(overrides)
    return config


# =============================================================================
# TODO: IMPLEMENT THESE FOR YOUR COMPETITION
# =============================================================================

def load_and_prepare_data(config):
    """
    Load your competition data and apply any row-level pre-processing.

    Rules:
    - Load train.csv and test.csv from the data/ directory
    - Apply inflation, scaling, or other transforms that don't require fold awareness
    - For fold-aware operations (target encoding, rolling features), use prepare_features()
    - Return (train_df, test_df) as DataFrames

    The returned DataFrames should contain:
    - All raw features
    - A date column matching config['date_col'] if using time-based CV
    - The target column (training set only)
    """
    # =========================================================================
    # EXAMPLE IMPLEMENTATION: California Housing Dataset
    # Replace everything below with your competition-specific loading.
    # =========================================================================
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()

    # Synthetic date column to demonstrate time-based CV
    import datetime
    n = len(df)
    start = datetime.date(2015, 1, 1)
    end = datetime.date(2023, 12, 31)
    delta_days = (end - start).days
    np.random.seed(42)
    days_offsets = np.sort(np.random.randint(0, delta_days, size=n))
    df['date'] = [str(start + datetime.timedelta(days=int(d))) for d in days_offsets]
    df['row_id'] = range(n)

    # 80/20 temporal split for train/test
    split_idx = int(n * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"  Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")
    return train_df, test_df


def prepare_features(df, ref_df=None):
    """
    Prepare features for one fold of cross-validation.

    Called per fold:
        X_tr, y_tr, _, _, cols = prepare_features(train_fold, val_fold)
        X_val, y_val, _, _, _  = prepare_features(val_fold, train_fold)

    Called for final model:
        X_train, y_train, X_test, row_ids, _ = prepare_features(train_df, test_df)

    Args:
        df:     Primary DataFrame (features and target are extracted from here)
        ref_df: Reference DataFrame used for column alignment (one-hot encoding, etc.)
                When df=train_fold, ref_df=val_fold. When df=train_df, ref_df=test_df.

    Returns:
        X         : Feature matrix (DataFrame) from df
        y         : Target array (numpy) from df. If df has no target column, returns zeros.
        X_ref     : Feature matrix from ref_df (for column alignment), or None
        row_ids   : Row IDs from ref_df (used when ref_df is the test set)
        feature_cols: List of feature column names
    """
    # =========================================================================
    # EXAMPLE IMPLEMENTATION: California Housing Dataset
    # Replace everything below with your competition-specific feature engineering.
    # =========================================================================
    TARGET_COL = 'MedHouseVal'
    ID_COL = 'row_id'
    DROP_COLS = [TARGET_COL, ID_COL, 'date']

    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    X = df[feature_cols].copy().fillna(0)
    y = df[TARGET_COL].values if TARGET_COL in df.columns else np.zeros(len(df))

    X_ref = None
    row_ids = None
    if ref_df is not None:
        X_ref = ref_df[[c for c in feature_cols if c in ref_df.columns]].copy().fillna(0)
        row_ids = ref_df[ID_COL] if ID_COL in ref_df.columns else ref_df.index

    return X, y, X_ref, row_ids, feature_cols


# =============================================================================
# GENERIC PIPELINE (no need to modify this)
# =============================================================================

def tune_hyperparameters(X_train, y_train, backend, date_series=None, n_iter=30,
                         n_splits=3, time_based_cv=True):
    """Tune hyperparameters using RandomizedSearchCV."""
    if time_based_cv and date_series is not None:
        date_series = pd.to_datetime(date_series)
        sort_idx = date_series.argsort()
        X_tune = X_train.iloc[sort_idx].reset_index(drop=True)
        y_tune = y_train[sort_idx]
        cv = TimeSeriesSplit(n_splits=n_splits)
        print(f"\nTuning {backend['name']} on {len(X_tune):,} samples with time-based CV")
        print(f"Date range: {date_series.min().date()} to {date_series.max().date()}")
    else:
        X_tune = X_train
        y_tune = y_train
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"\nTuning {backend['name']} on {len(X_tune):,} samples with random CV")

    print(f"Running {n_iter} iterations × {n_splits} folds = {n_iter * n_splits} fits...")

    search = RandomizedSearchCV(
        backend['get_tuning_model'](),
        param_distributions=backend['get_tune_distributions'](),
        n_iter=n_iter,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        verbose=2,
        n_jobs=1,
    )
    search.fit(X_tune, y_tune)

    print(f"\nBest CV RMSE: {-search.best_score_:.4f}")
    print("Best parameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return search.best_params_


def train_and_predict_cv(train_df, test_df, config, backend, tuned_params=None):
    """
    Train model with time-based (or random) cross-validation, compute OOF predictions,
    train a final model on all data, and return test predictions.
    """
    if tuned_params:
        params = backend['get_tuned_params'](tuned_params)
        num_boost_round = tuned_params.get('n_estimators', tuned_params.get('iterations', 1000))
    else:
        params = backend['get_default_params'](config)
        num_boost_round = config['num_boost_round']

    if config.get('use_gpu'):
        params.update(_get_gpu_params(backend['name']))

    date_col = config.get('date_col')
    train_df = train_df.copy()

    # Sort for time-based CV
    if date_col and date_col in train_df.columns and config['time_based_cv']:
        train_df['_date_parsed'] = pd.to_datetime(train_df[date_col])
        train_df = train_df.sort_values('_date_parsed').reset_index(drop=True)

    n_splits = config['cv_splits']
    if config['time_based_cv']:
        cv = TimeSeriesSplit(n_splits=n_splits)
        print("Time-based CV folds:")
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        print("Random KFold CV:")

    cv_scores = []
    cv_mae_scores = []
    models = []
    feature_cols = None
    oof_actuals = []
    oof_preds = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(train_df)):
        train_fold = train_df.iloc[train_idx].copy()
        val_fold = train_df.iloc[val_idx].copy()

        if date_col and '_date_parsed' in train_df.columns:
            tr_range = f"{train_fold['_date_parsed'].min().date()} to {train_fold['_date_parsed'].max().date()}"
            val_range = f"{val_fold['_date_parsed'].min().date()} to {val_fold['_date_parsed'].max().date()}"
            print(f"  Fold {fold+1}: Train {tr_range} ({len(train_idx):,}) -> Val {val_range} ({len(val_idx):,})")
        else:
            print(f"  Fold {fold+1}: Train {len(train_idx):,} -> Val {len(val_idx):,}")

        X_tr, y_tr, _, _, cols = prepare_features(train_fold, val_fold)
        X_val, y_val, _, _, _ = prepare_features(val_fold, train_fold)

        # Outlier clipping on training labels
        clip_pct = config.get('clip_target_upper')
        clip_mask = np.ones(len(y_tr), dtype=bool)
        if clip_pct is not None:
            threshold = np.percentile(y_tr, clip_pct * 100)
            clip_mask = y_tr <= threshold
            n_removed = (~clip_mask).sum()
            print(f"  Outlier clip (>{clip_pct*100:.0f}th pct): removed {n_removed} training samples")
            X_tr = X_tr[clip_mask]
            y_tr = y_tr[clip_mask]

        # Align feature columns across folds
        if feature_cols is None:
            feature_cols = cols
        X_tr = X_tr.reindex(columns=feature_cols, fill_value=0)
        X_val = X_val.reindex(columns=feature_cols, fill_value=0)

        # Recency weighting
        alpha = config.get('recency_weight', 0.0)
        sample_weight = None
        if alpha > 0 and date_col and date_col in train_fold.columns:
            sample_weight = compute_recency_weights(train_fold[date_col], alpha)
            if clip_pct is not None:
                sample_weight = sample_weight[clip_mask]

        model = backend['train'](X_tr, y_tr, X_val, y_val, params, num_boost_round,
                                 sample_weight=sample_weight)
        models.append(model)

        y_pred = backend['predict'](model, X_val)
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        mae = np.mean(np.abs(y_val - y_pred))
        cv_scores.append(rmse)
        cv_mae_scores.append(mae)
        oof_actuals.extend(y_val.tolist())
        oof_preds.extend(y_pred.tolist())
        print(f"  Fold {fold+1}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    print(f"\nCV RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print(f"CV MAE:  {np.mean(cv_mae_scores):.4f} (+/- {np.std(cv_mae_scores):.4f})")

    oof_bias = float(np.mean(np.array(oof_actuals) - np.array(oof_preds)))
    print(f"OOF bias: {oof_bias:+.4f} ({'under' if oof_bias > 0 else 'over'}-predicting)")
    bias_correction = oof_bias if config.get('use_bias_correction', False) else 0.0
    if config.get('use_bias_correction', False):
        print(f"Bias correction ENABLED: applying {bias_correction:+.4f} to predictions")

    # Final model on all training data
    print("\nTraining final model on all data...")
    X_train, y_train, X_test, row_ids, _ = prepare_features(train_df, test_df)
    X_train = X_train.reindex(columns=feature_cols, fill_value=0)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0)

    clip_pct = config.get('clip_target_upper')
    if clip_pct is not None:
        threshold = np.percentile(y_train, clip_pct * 100)
        mask = y_train <= threshold
        print(f"Final model outlier clip: removed {(~mask).sum()} training samples")
        X_train = X_train[mask]
        y_train = y_train[mask]
        train_df_final = train_df.iloc[mask]
    else:
        train_df_final = train_df

    print("\nTop 20 Feature Importance:")
    importance = backend['get_feature_importance'](models[-1], feature_cols)
    print(importance.head(20).to_string(index=False))

    alpha = config.get('recency_weight', 0.0)
    final_weight = None
    if alpha > 0 and date_col and date_col in train_df_final.columns:
        final_weight = compute_recency_weights(train_df_final[date_col], alpha)

    final_model = backend['train_final'](X_train, y_train, params, num_boost_round,
                                         sample_weight=final_weight)

    predictions = backend['predict'](final_model, X_test) + bias_correction
    predictions = np.maximum(predictions, 0)

    oof_df = pd.DataFrame({'actual': oof_actuals, 'pred': oof_preds})
    return predictions, row_ids, cv_scores, cv_mae_scores, importance, oof_bias, oof_df


def main(config):
    with mlflow.start_run(nested=bool(mlflow.active_run())):
        _skip_log = {'model_params', 'stacking_model_params'}
        mlflow.log_params({k: v for k, v in config.items() if k not in _skip_log})

        print("=" * 60)
        print("EXPERIMENT CONFIG:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print("=" * 60)

        backend = get_backend(config['model'], config)
        print(f"\nUsing model: {backend['name']}")

        print("\nLoading data...")
        train_data, test_data = load_and_prepare_data(config)
        print(f"  Train: {train_data.shape}, Test: {test_data.shape}")

        # Restrict training window (price lookup still uses all data in original project;
        # here we simply filter the training set)
        if config.get('train_from'):
            date_col = config.get('date_col')
            if date_col and date_col in train_data.columns:
                cutoff = pd.to_datetime(config['train_from'])
                n_before = len(train_data)
                train_data = train_data[pd.to_datetime(train_data[date_col]) >= cutoff].reset_index(drop=True)
                print(f"  train_from={config['train_from']}: {len(train_data):,} of {n_before:,} rows kept")

        print(f"CV strategy: {'time-based' if config['time_based_cv'] else 'random'}")

        tuned_params = config.get('model_params', None)
        if config['tune']:
            print("\nPreparing data for hyperparameter tuning...")
            X_train, y_train, _, _, _ = prepare_features(train_data, test_data)
            date_col = config.get('date_col')
            date_series = train_data[date_col] if date_col and date_col in train_data.columns else None
            tuned_params = tune_hyperparameters(
                X_train, y_train, backend, date_series=date_series,
                n_iter=config['tune_iterations'], time_based_cv=config['time_based_cv']
            )
            mlflow.log_params({f"tuned_{k}": v for k, v in tuned_params.items()})
            save_tuned_params(config['model'], tuned_params)

        print("\nTraining with CV...")
        predictions, row_ids, cv_scores, cv_mae_scores, feature_importance, oof_bias, oof_df = \
            train_and_predict_cv(train_data, test_data, config, backend, tuned_params=tuned_params)

        # Log metrics
        mlflow.log_metric("cv_rmse_mean", np.mean(cv_scores))
        mlflow.log_metric("cv_rmse_std", np.std(cv_scores))
        for i, score in enumerate(cv_scores):
            mlflow.log_metric(f"cv_rmse_fold_{i+1}", score)

        mlflow.log_metric("cv_mae_mean", np.mean(cv_mae_scores))
        mlflow.log_metric("cv_mae_std", np.std(cv_mae_scores))
        for i, score in enumerate(cv_mae_scores):
            mlflow.log_metric(f"cv_mae_fold_{i+1}", score)

        if len(cv_scores) >= 2:
            mlflow.log_metric("cv_fold_trend", cv_scores[-1] - cv_scores[0])

        mlflow.log_metric("oof_bias", oof_bias)

        test_pred_arr = np.array(predictions)
        mlflow.log_metric("test_pred_mean", float(np.mean(test_pred_arr)))
        mlflow.log_metric("test_pred_std", float(np.std(test_pred_arr)))
        mlflow.log_metric("test_pred_iqr", float(np.percentile(test_pred_arr, 75) - np.percentile(test_pred_arr, 25)))

        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        oof_df.to_csv("oof_predictions.csv", index=False)
        mlflow.log_artifact("oof_predictions.csv")

        print("\nCreating submission...")
        run_id = mlflow.active_run().info.run_id
        run_id_short = run_id[:8]

        os.makedirs("submissions", exist_ok=True)
        submission_path = f"submissions/submission_{run_id_short}.csv"

        from utils import create_submission
        create_submission(row_ids, predictions, filename=submission_path)
        mlflow.log_artifact(submission_path)

        print(f"\nMLflow run ID: {run_id}")
        print(f"To view run: mlflow runs get -r {run_id}")

        return np.mean(cv_scores)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generic Kaggle ML Pipeline")
    parser.add_argument('--model', choices=['lightgbm', 'xgboost', 'catboost', 'ridge',
                                            'elasticnet', 'randomforest', 'stacking'])
    parser.add_argument('--stack-models', nargs='+',
                        choices=['lightgbm', 'xgboost', 'catboost', 'ridge', 'elasticnet', 'randomforest'])
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--tune-iterations', type=int)
    parser.add_argument('--random-cv', action='store_true', help='Use random KFold instead of time-based CV')
    parser.add_argument('--cv-folds', type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--recency-weight', type=float)
    parser.add_argument('--bias-correction', action='store_true')
    parser.add_argument('--clip-target', type=float, metavar='PCT',
                        help='Remove training samples above this percentile (e.g. 0.99)')
    parser.add_argument('--train-from', type=str, metavar='DATE',
                        help='Only train on samples on or after this ISO date (e.g. 2022-07-01)')
    args = parser.parse_args()

    overrides = {}
    if args.model:
        overrides['model'] = args.model
    if args.stack_models:
        overrides['stack_models'] = args.stack_models
    if args.tune:
        overrides['tune'] = True
    if args.tune_iterations:
        overrides['tune_iterations'] = args.tune_iterations
    if args.random_cv:
        overrides['time_based_cv'] = False
    if args.cv_folds:
        overrides['cv_splits'] = args.cv_folds
    if args.gpu:
        overrides['use_gpu'] = True
    if args.recency_weight is not None:
        overrides['recency_weight'] = args.recency_weight
    if args.bias_correction:
        overrides['use_bias_correction'] = True
    if args.clip_target is not None:
        overrides['clip_target_upper'] = args.clip_target
    if args.train_from:
        overrides['train_from'] = args.train_from

    config = get_config(**overrides)
    main(config)
