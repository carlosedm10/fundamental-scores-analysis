import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


def simple_backward_regression(
    X: pd.DataFrame, y: pd.Series, threshold: float = 0.20
) -> dict:
    """
    Perform simple backward elimination regression with a threshold of 20%
    for the p-values.
    """

    # Add intercept
    X = sm.add_constant(X)

    removed_features = []

    # Backward elimination loop
    while True:
        model = sm.OLS(y, X).fit()
        pvalues = model.pvalues.drop("const", errors="ignore")

        if (pvalues <= threshold).all() or len(pvalues) == 0:
            break

        # Check for NaN p-values (numerical issues)
        if pvalues.isna().all():
            print(
                "Warning: All p-values are NaN. Stopping backward elimination."
            )
            break

        # Drop the feature with the highest p-value
        worst_feature = pvalues.idxmax()

        # Additional check in case idxmax returns NaN
        if pd.isna(worst_feature):
            print(
                "Warning: Cannot identify worst feature (NaN). Stopping backward elimination."
            )
            break

        X = X.drop(columns=worst_feature)
        removed_features.append(worst_feature)

    # Final model after elimination
    final_features = X.columns.tolist()
    final_model = sm.OLS(y, X).fit()
    summary = {
        # Model Information:
        "model": final_model,
        "summary": final_model.summary(),
        "X_cols": final_features,
        "removed_features": removed_features,
        "y_true": y.copy(),
        "y_pred": final_model.fittedvalues.copy(),
        "residuals": final_model.resid.copy(),
        # Performance metrics:
        "r2_score": final_model.rsquared,
        "adjusted_r2": final_model.rsquared_adj,
        "rmse": np.sqrt(mean_squared_error(y, final_model.fittedvalues)),
        "mae": np.mean(np.abs(y - final_model.fittedvalues)),
        # Consistency and overfitting metrics:
        "coefficients": final_model.params.to_dict(),
        "pvalues": final_model.pvalues.to_dict(),
        "stderr": final_model.bse.to_dict(),
        "signs": final_model.params.apply(np.sign).to_dict(),
        "n_features": len(final_features),
        "n_region_dummies": len(
            [col for col in final_features if "region_" in col]
        ),
        "n_sector_dummies": len(
            [col for col in final_features if "sector_" in col]
        ),
        "tvalues": final_model.tvalues.to_dict(),
        # "nobs": int(final_model.nobs),
    }
    return summary


def simple_backward_regression_2(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.20,
) -> dict:
    """
    Perform backward elimination regression on training set and evaluate on validation set.

    Args:
        X: Features
        y: Target
        threshold: p-value threshold for feature removal
        validation_size: Fraction of data to use for validation
        random_state: Seed for reproducibility

    Returns:
        A dictionary containing model, diagnostics, predictions, and both train/val performance.
    """

    # Split data into train and validation sets
    n_train = int(len(X) * 0.8)  # For a 20/80 split
    X_train, X_val = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_val = y.iloc[:n_train], y.iloc[n_train:]
    # Add intercept
    X_train_const = sm.add_constant(X_train)
    removed_features = []

    # Backward elimination on training set
    while True:
        model = sm.OLS(y_train, X_train_const).fit()
        pvalues = model.pvalues.drop("const", errors="ignore")
        if (pvalues <= threshold).all() or len(pvalues) == 0:
            break

        # Check for NaN p-values (numerical issues)
        if pvalues.isna().all():
            print(
                "Warning: All p-values are NaN. Stopping backward elimination."
            )
            break

        # Drop the feature with the highest p-value
        worst_feature = pvalues.idxmax()

        # Additional check in case idxmax returns NaN
        if pd.isna(worst_feature):
            print(
                "Warning: Cannot identify worst feature (NaN). Stopping backward elimination."
            )
            break

        X_train_const = X_train_const.drop(columns=worst_feature)
        removed_features.append(worst_feature)

    # Final model
    final_features = X_train_const.columns.tolist()
    final_model = sm.OLS(y_train, X_train_const).fit()

    # Prepare validation set using the same features
    X_val_const = sm.add_constant(X_val, has_constant="add")[final_features]

    # Predict
    y_train_pred = final_model.predict(X_train_const)
    y_val_pred = final_model.predict(X_val_const)

    # Residuals
    residuals_train = y_train - y_train_pred
    residuals_val = y_val - y_val_pred

    summary = {
        # Model Info
        "model": final_model,
        "summary": final_model.summary(),
        "X_cols": final_features,
        "removed_features": removed_features,
        # Data
        "X_train": X_train_const,
        "X_val": X_val_const,
        "y_train_true": y_train.copy(),
        "y_train_pred": y_train_pred,
        "residuals_train": residuals_train,
        "y_val_true": y_val.copy(),
        "y_val_pred": y_val_pred,
        "residuals_val": residuals_val,
        # Performance
        "train_r2_score": r2_score(y_train, y_train_pred),
        "train_adjusted_r2": final_model.rsquared_adj,
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "val_r2_score": r2_score(y_val, y_val_pred),
        "val_adjusted_r2": (
            1
            - (1 - r2_score(y_val, y_val_pred))
            * ((len(y_val) - 1) / (len(y_val) - len(final_features) - 1))
        ),
        "val_rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
        "val_mae": mean_absolute_error(y_val, y_val_pred),
        # Additional diagnostics
        "coefficients": final_model.params.to_dict(),
        "pvalues": final_model.pvalues.to_dict(),
        "stderr": final_model.bse.to_dict(),
        "tvalues": final_model.tvalues.to_dict(),
        "signs": final_model.params.apply(np.sign).to_dict(),
        "n_features": len(final_features),
        "n_region_dummies": len(
            [col for col in final_features if "region_" in col]
        ),
        "n_sector_dummies": len(
            [col for col in final_features if "sector_" in col]
        ),
    }

    return summary
