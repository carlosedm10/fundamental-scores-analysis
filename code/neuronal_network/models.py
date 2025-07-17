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
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from utils import split_by_ticker_and_time


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
            - (1 - r2_score(y_train, y_train_pred))
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
            - (1 - r2_score(y_val, y_val_pred))
            * ((len(y_val) - 1) / (len(y_val) - X.shape[1] - 1))
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
            - (1 - r2_score(y_val_t, y_val_pred_t))
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

    return summary


def classifier_nn(
    X: pd.DataFrame,
    y: pd.Series,
    validation_size: float = 0.2,
    hidden_layer_sizes: tuple[int, int] = (64, 64),
    alpha: float = 0.001,
    learning_rate: Literal["constant", "invscaling", "adaptive"] = "adaptive",
    max_iter: int = 1000,
    early_stopping: bool = True,
    random_state: int = 42,
) -> dict:
    """
    Fit a neural network classification model using MLPClassifier from scikit-learn,
    with train/validation split and performance diagnostics.

    Args:
        X: Features (DataFrame)
        y: Target (Series) - should contain class labels
        validation_size: Fraction of data to reserve for validation
        hidden_layer_sizes: Tuple defining neurons per hidden layer
        alpha: L2 penalty (regularization)
        learning_rate: Learning rate schedule
        max_iter: Maximum number of iterations
        early_stopping: Whether to use validation-based early stopping
        random_state: Seed for reproducibility

    Returns:
        Dictionary with model, predictions, and performance metrics
    """

    # Split data
    X_train, y_train, X_val, y_val, X_val_t, y_val_t = (
        split_by_ticker_and_time(X, y, validation_size=validation_size)
    )

    # Add Gaussian noise to X_train to reduce overfitting
    noise_std_X = 0.01 * X_train.std(axis=0, ddof=0)
    X_train = X_train + np.random.normal(0, noise_std_X, X_train.shape)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_val_t_scaled = scaler.transform(X_val_t)

    # Fit neural network classifier
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate=learning_rate,
        max_iter=max_iter,
        early_stopping=early_stopping,
        random_state=random_state,
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

    # Residuals
    residuals_train = y_train - y_train_pred
    residuals_val = y_val - y_val_pred
    residuals_val_t = y_val_t - y_val_pred_t

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
        "residuals_train": pd.Series(residuals_train, index=y_train.index),
        "train_r2_score": r2_score(y_train, y_train_pred),
        "train_adjusted_r2": (
            1
            - (1 - r2_score(y_train, y_train_pred))
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
            - (1 - r2_score(y_val, y_val_pred))
            * ((len(y_val) - 1) / (len(y_val) - X.shape[1] - 1))
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
            - (1 - r2_score(y_val_t, y_val_pred_t))
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

    return summary


def classifier_nn_2(
    X: pd.DataFrame,
    y: pd.Series,
    validation_size: float = 0.2,
    hidden_layer_sizes: tuple[int, int] = (64, 64),
    alpha: float = 0.001,
    learning_rate: Literal["constant", "invscaling", "adaptive"] = "adaptive",
    max_iter: int = 1000,
    early_stopping: bool = True,
    random_state: int = 42,
) -> dict:
    """
    Fit a neural network classification model using MLPClassifier from scikit-learn,
    with train/validation split and performance diagnostics.

    Args:
        X: Features (DataFrame)
        y: Target (Series) - should contain class labels
        validation_size: Fraction of data to reserve for validation
        hidden_layer_sizes: Tuple defining neurons per hidden layer
        alpha: L2 penalty (regularization)
        learning_rate: Learning rate schedule
        max_iter: Maximum number of iterations
        early_stopping: Whether to use validation-based early stopping
        random_state: Seed for reproducibility

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

    # Fit neural network classifier
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate=learning_rate,
        max_iter=max_iter,
        early_stopping=early_stopping,
        random_state=random_state,
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

    # Get class labels
    classes = model.classes_

    # Performance metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_accuracy_t = accuracy_score(y_val_t, y_val_pred_t)

    # Handle multiclass vs binary classification metrics
    average_method = "weighted" if len(classes) > 2 else "binary"

    train_precision = precision_score(
        y_train, y_train_pred, average=average_method, zero_division=0
    )
    train_recall = recall_score(
        y_train, y_train_pred, average=average_method, zero_division=0
    )
    train_f1 = f1_score(
        y_train, y_train_pred, average=average_method, zero_division=0
    )

    val_precision = precision_score(
        y_val, y_val_pred, average=average_method, zero_division=0
    )
    val_precision_t = precision_score(
        y_val_t, y_val_pred_t, average=average_method, zero_division=0
    )
    val_recall = recall_score(
        y_val, y_val_pred, average=average_method, zero_division=0
    )
    val_f1 = f1_score(
        y_val, y_val_pred, average=average_method, zero_division=0
    )

    # Confusion matrices
    train_cm = confusion_matrix(y_train, y_train_pred)
    val_cm = confusion_matrix(y_val, y_val_pred)
    val_cm_t = confusion_matrix(y_val_t, y_val_pred_t)

    summary = {
        # Model & Structure
        "model": model,
        "scaler": scaler,
        "X_cols": list(X.columns),
        "classes": classes,
        "n_classes": len(classes),
        "hidden_layer_sizes": hidden_layer_sizes,
        "n_features": X.shape[1],
        "n_iterations": model.n_iter_,
        "loss": model.loss_,
        # Train Set
        "X_train": X_train.copy(),
        "y_train_true": y_train.copy(),
        "y_train_pred": pd.Series(y_train_pred, index=y_train.index),
        "y_train_proba": y_train_proba,
        "train_accuracy": train_accuracy,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "train_confusion_matrix": train_cm,
        # Validation Set
        "X_val": X_val.copy(),
        "y_val_true": y_val.copy(),
        "y_val_pred": pd.Series(y_val_pred, index=y_val.index),
        "y_val_proba": y_val_proba,
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "val_confusion_matrix": val_cm,
        # Validation Set by time
        "X_val_t": X_val_t.copy(),
        "y_val_t_true": y_val_t.copy(),
        "y_val_t_pred": pd.Series(y_val_pred_t, index=y_val_t.index),
        "y_val_t_proba": y_val_proba_t,
        "val_t_accuracy": val_accuracy_t,
        "val_t_precision": val_precision_t,
        # Feature Info
        "n_region_dummies": len(
            [col for col in X.columns if "region_" in col]
        ),
        "n_sector_dummies": len(
            [col for col in X.columns if "sector_" in col]
        ),
        # Classification Reports
        "train_classification_report": classification_report(
            y_train,
            y_train_pred,
            target_names=[str(c) for c in classes],
            output_dict=True,
        ),
        "val_classification_report": classification_report(
            y_val,
            y_val_pred,
            target_names=[str(c) for c in classes],
            output_dict=True,
        ),
    }

    return summary
