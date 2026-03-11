"""
Generic utilities for Kaggle ML pipelines.
"""

import pandas as pd
import numpy as np


def extract_date_features(df, date_col='date'):
    """Extract year and month features from a date column."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day_of_week'] = df[date_col].dt.dayofweek
    return df


def create_submission(row_ids, predictions, id_col='row_id', pred_col='prediction',
                      filename='submission.csv'):
    """Create a Kaggle submission CSV."""
    submission = pd.DataFrame({id_col: row_ids, pred_col: predictions})
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    print(f"  Shape: {submission.shape}")
    print(f"  Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    return submission


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def rmsle(y_true, y_pred):
    """Root Mean Squared Log Error."""
    return np.sqrt(np.mean((np.log1p(np.maximum(y_pred, 0)) - np.log1p(y_true)) ** 2))


def target_encode(train, test, col, target, smoothing=10):
    """
    Smoothed target encoding. Must be computed within each fold to avoid leakage.

    Args:
        train: Training DataFrame
        test:  Validation/test DataFrame (gets encoded using train statistics)
        col:   Column to encode
        target: Target column name
        smoothing: Regularization strength (higher = more shrinkage toward global mean)

    Returns:
        (train_encoded, test_encoded) as Series
    """
    global_mean = train[target].mean()
    agg = train.groupby(col)[target].agg(['mean', 'count'])
    smooth = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)

    train_encoded = train[col].map(smooth).fillna(global_mean)
    test_encoded = test[col].map(smooth).fillna(global_mean)
    return train_encoded, test_encoded


def encode_categoricals_as_int(train, test, cols):
    """
    Encode categorical columns as integers for tree-based models.
    Uses combined train+test vocabulary so all categories are known.

    Returns:
        (train, test, new_col_names)
    """
    train = train.copy()
    test = test.copy()
    new_cols = []

    for col in cols:
        combined = pd.concat([train[col], test[col]], axis=0).astype('category')
        categories = combined.cat.categories

        new_col = col + '_enc'
        train[new_col] = train[col].astype('category').cat.set_categories(categories).cat.codes
        test[new_col] = test[col].astype('category').cat.set_categories(categories).cat.codes
        new_cols.append(new_col)

    return train, test, new_cols


def add_fold_aware_group_stats(train_fold, val_fold, group_col, value_col, prefix=None):
    """
    Add group mean/std/count features computed from the training fold only (no leakage).
    Val fold gets train fold statistics mapped in.

    Args:
        train_fold: Training fold DataFrame
        val_fold:   Validation fold DataFrame
        group_col:  Column to group by (e.g. 'category', 'location')
        value_col:  Column to aggregate (e.g. 'log_target')
        prefix:     Prefix for new column names (default: group_col)

    Returns:
        (train_fold_with_features, val_fold_with_features)
    """
    prefix = prefix or group_col
    train_fold = train_fold.copy()
    val_fold = val_fold.copy()

    stats = train_fold.groupby(group_col)[value_col].agg(['mean', 'std', 'count'])
    stats.columns = [f'{prefix}_mean', f'{prefix}_std', f'{prefix}_count']

    train_fold = train_fold.merge(stats, on=group_col, how='left')
    val_fold = val_fold.merge(stats, on=group_col, how='left')

    # Fill NaN for unseen groups
    for col in stats.columns:
        global_val = train_fold[col].median()
        train_fold[col] = train_fold[col].fillna(global_val)
        val_fold[col] = val_fold[col].fillna(global_val)

    return train_fold, val_fold
