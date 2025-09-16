import gc
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    log_loss,
)


from utils import split_by_ticker_and_time


def _annualize_profits(y: pd.Series, profit_horizon: str) -> pd.Series:
    """
    Annualize profit values to put them on the same scale.

    Args:
        y: pd.Series of profits (can be in any of the supported horizons)
        profit_horizon: str, one of ["profit_1m", "profit_3m", "profit_6m", "profit_1y", "profit_2y", "profit_5y"]

    Returns:
        pd.Series of annualized profits
    """
    # Map profit horizon to number of years
    horizon_to_years = {
        "profit_1m": 1 / 12,
        "profit_3m": 3 / 12,
        "profit_6m": 6 / 12,
        "profit_1y": 1,
        "profit_2y": 2,
        "profit_5y": 5,
    }
    if profit_horizon not in horizon_to_years:
        raise ValueError(f"Unknown profit_horizon: {profit_horizon}")

    years = horizon_to_years[profit_horizon]
    annualized = (1 + y).pow(1 / years) - 1
    return annualized


def _encode_profit_column(series: pd.Series, binary: bool = True) -> pd.Series:
    if binary:
        return (series > 0).astype(int)
    else:
        bins = [-float("inf"), -0.08, -0.03, 0, 0.03, 0.08, float("inf")]
        labels = [-3, -2, -1, 1, 2, 3]  # Negative and positive classes
        categorized = pd.cut(series, bins=bins, labels=labels, right=False)
        # Handle NaN values by converting to a specific category or filling with 0
        # Convert to float first to handle NaN, then to int
        result = categorized.astype(float)
        result = result.fillna(0)  # Fill NaN with neutral class 0
        return result.astype(int)


def regressor_nn(
    X: pd.DataFrame,
    y: pd.Series,
    validation_size: float = 0.2,
    hidden_layer_sizes: tuple[int, ...] = (64, 64),
) -> dict:
    """
    Fit a neural network regression model using MLPRegressor from scikit-learn,
    with train/validation split and performance diagnostics.

    Args:
        X: Features (DataFrame)
        y: Target (Series)

    Returns:
        Dictionary with model, predictions, and performance metrics
    """

    # Split data
    X_train, y_train, X_val, y_val, X_val_t, y_val_t = (
        split_by_ticker_and_time(X, y, validation_size=validation_size)
    )

    # Add Gaussian noise to X_train and y_train to reduce overfitting
    noise_std_X = 0.1 * X_train.std(axis=0, ddof=0)
    noise_std_y = 0.1 * y_train.std(ddof=0)
    X_train = X_train + np.random.normal(0, noise_std_X, X_train.shape)
    y_train = y_train + np.random.normal(0, noise_std_y, y_train.shape)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_val_t_scaled = scaler.transform(X_val_t)

    # Fit neural network
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=0.001,
        learning_rate="adaptive",
        max_iter=1000,
        early_stopping=True,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    y_val_pred_t = model.predict(X_val_t_scaled)

    # Residuals
    residuals_train = y_train - y_train_pred
    residuals_val = y_val - y_val_pred
    residuals_val_t = y_val_t - y_val_pred_t

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_t_r2 = r2_score(y_val_t, y_val_pred_t)

    summary = {
        # Model & Structure
        "model": model,
        "scaler": scaler,
        "X_cols": list(X.columns),
        "hidden_layer_sizes": hidden_layer_sizes,
        "n_features": X.shape[1],
        "n_iterations": model.n_iter_,
        "loss": model.loss_,
        # Train Set
        "X_train": X_train.copy(),
        "y_train_true": y_train.copy(),
        "y_train_pred": pd.Series(y_train_pred, index=y_train.index),
        "residuals_train": pd.Series(residuals_train, index=y_train.index),
        "train_r2_score": r2_score(y_train, y_train_pred),
        "train_adjusted_r2": (
            1
            - (1 - train_r2)
            * ((len(y_train) - 1) / (len(y_train) - X.shape[1] - 1))
        ),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        # Validation Set
        "X_val": X_val.copy(),
        "y_val_true": y_val.copy(),
        "y_val_pred": pd.Series(y_val_pred, index=y_val.index),
        "residuals_val": pd.Series(residuals_val, index=y_val.index),
        "val_r2_score": r2_score(y_val, y_val_pred),
        "val_adjusted_r2": (
            1
            - (1 - val_r2) * ((len(y_val) - 1) / (len(y_val) - X.shape[1] - 1))
        ),
        "val_rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
        "val_mae": mean_absolute_error(y_val, y_val_pred),
        # Validation Set by time
        "X_val_t": X_val_t.copy(),
        "y_val_t_true": y_val_t.copy(),
        "y_val_t_pred": pd.Series(y_val_pred_t, index=y_val_t.index),
        "residuals_val_t": pd.Series(residuals_val_t, index=y_val_t.index),
        "val_t_r2_score": r2_score(y_val_t, y_val_pred_t),
        "val_t_adjusted_r2": (
            1
            - (1 - val_t_r2)
            * ((len(y_val_t) - 1) / (len(y_val_t) - X.shape[1] - 1))
        ),
        "val_t_rmse": np.sqrt(mean_squared_error(y_val_t, y_val_pred_t)),
        "val_t_mae": mean_absolute_error(y_val_t, y_val_pred_t),
        # Feature Info
        "n_region_dummies": len(
            [col for col in X.columns if "region_" in col]
        ),
        "n_sector_dummies": len(
            [col for col in X.columns if "sector_" in col]
        ),
    }

    # Clean up large objects to free memory
    del X_train, y_train, X_val, y_val, X_val_t, y_val_t
    del X_train_scaled, X_val_scaled, X_val_t_scaled
    del y_train_pred, y_val_pred, y_val_pred_t
    del residuals_train, residuals_val, residuals_val_t
    gc.collect()

    return summary


def classifier_nn(
    X: pd.DataFrame,
    y: pd.Series,
    profit_horizon: str,
    validation_size: float = 0.2,
    hidden_layer_sizes: tuple[int, int] = (64, 64),
    binary: bool = True,
) -> dict:
    """
    Fit a neural network classification model using MLPClassifier from scikit-learn,
    with train/validation split and performance diagnostics.

    Args:
        X: Features (DataFrame)
        y: Target (Series) - should contain class labels
        validation_size: Fraction of data to reserve for validation
        hidden_layer_sizes: Tuple defining neurons per hidden layer
    Returns:
        Dictionary with model, predictions, and performance metrics
    """
    y = _annualize_profits(y, profit_horizon)

    # Split data
    X_train, y_train, X_val, y_val, X_val_t, y_val_t = (
        split_by_ticker_and_time(X, y, validation_size=validation_size)
    )

    # Encode profit column
    y_train = _encode_profit_column(y_train, binary=binary)
    y_val = _encode_profit_column(y_val, binary=binary)
    y_val_t = _encode_profit_column(y_val_t, binary=binary)

    # Add Gaussian noise to X_train to reduce overfitting
    noise_std_X = 0.1 * X_train.std(axis=0, ddof=0)
    X_train = X_train + np.random.normal(0, noise_std_X, X_train.shape)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_val_t_scaled = scaler.transform(X_val_t)

    # Fit neural network classifier
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=0.001,
        learning_rate="adaptive",
        max_iter=1000,
        early_stopping=True,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    y_val_pred_t = model.predict(X_val_t_scaled)

    # Prediction probabilities (if available)
    y_train_proba = model.predict_proba(X_train_scaled)
    y_val_proba = model.predict_proba(X_val_scaled)
    y_val_proba_t = model.predict_proba(X_val_t_scaled)

    # Compute F1 score (binary or multiclass)
    average_type = "binary" if len(np.unique(y)) == 2 else "weighted"
    train_predictability = f1_score(
        y_train, y_train_pred, average=average_type
    )
    val_predictability = f1_score(y_val, y_val_pred, average=average_type)
    val_t_predictability = f1_score(
        y_val_t, y_val_pred_t, average=average_type
    )

    # Compute accuracy and log loss
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_t_accuracy = accuracy_score(y_val_t, y_val_pred_t)

    # Compute log loss (handle single-class edge case)
    def _safe_log_loss(y_true, y_proba):
        try:
            return log_loss(y_true, y_proba)
        except ValueError:
            return float("nan")

    train_strict_errors = _safe_log_loss(y_train, y_train_proba)
    val_strict_errors = _safe_log_loss(y_val, y_val_proba)
    val_t_strict_errors = _safe_log_loss(y_val_t, y_val_proba_t)

    summary = {
        # Model & Structure
        "model": model,
        "scaler": scaler,
        "X_cols": list(X_train.columns),
        "hidden_layer_sizes": hidden_layer_sizes,
        "n_features": X_train.shape[1],
        "n_iterations": model.n_iter_,
        "loss": model.loss_,
        # Train Set
        "X_train": X_train.copy(),
        "y_train_true": y_train.copy(),
        "y_train_pred": pd.Series(y_train_pred, index=y_train.index),
        "y_train_proba": pd.DataFrame(y_train_proba, index=y_train.index),
        "train_predictability": train_predictability,
        "train_accuracy": train_accuracy,
        "train_strict_errors": train_strict_errors,
        # Validation Set
        "X_val": X_val.copy(),
        "y_val_true": y_val.copy(),
        "y_val_pred": pd.Series(y_val_pred, index=y_val.index),
        "y_val_proba": pd.DataFrame(y_val_proba, index=y_val.index),
        "val_predictability": val_predictability,
        "val_accuracy": val_accuracy,
        "val_strict_errors": val_strict_errors,
        # Validation Set by time
        "X_val_t": X_val_t.copy(),
        "y_val_t_true": y_val_t.copy(),
        "y_val_t_pred": pd.Series(y_val_pred_t, index=y_val_t.index),
        "y_val_proba_t": pd.DataFrame(y_val_proba_t, index=y_val_t.index),
        "val_t_predictability": val_t_predictability,
        "val_t_accuracy": val_t_accuracy,
        "val_t_strict_errors": val_t_strict_errors,
        # Feature Info
        "n_region_dummies": len(
            [col for col in X.columns if "region_" in col]
        ),
        "n_sector_dummies": len(
            [col for col in X.columns if "sector_" in col]
        ),
    }

    # Clean up large objects to free memory
    del X_train, y_train, X_val, y_val, X_val_t, y_val_t
    del X_train_scaled, X_val_scaled, X_val_t_scaled
    del y_train_pred, y_val_pred, y_val_pred_t
    del y_train_proba, y_val_proba, y_val_proba_t
    gc.collect()

    return summary
