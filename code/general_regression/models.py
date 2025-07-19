import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import KFold
from pygam import LinearGAM, s, f, te
from functools import reduce


def get_spline_terms(
    feature_list: list[str], interaction_pairs: list[tuple[str, str]] = []
) -> list[str]:
    """
    Create GAM terms based on variable types:
    - s(i): for continuous variables (scores, volume, market_cap, rolling averages/std)
    - f(i): for categorical variables (sector_, region_ dummies)
    - te(i, j): for tensor interactions between variables

    Args:
        feature_list: List of feature names
        interaction_pairs: List of tuples indicating which features to interact
                          e.g., [('feature1', 'feature2'), ('feature3', 'feature4')]

    Returns:
        GAM terms combining different spline types
    """
    terms = []

    # Main effects
    for i, feature in enumerate(feature_list):
        if feature.startswith(("sector_", "region_")):
            # Categorical variables (dummies) - use factor spline
            terms.append(f(i))
        else:
            # Continuous variables (scores, rolling stats, etc.) - use smooth spline
            terms.append(s(i))

    # Interaction effects (tensor splines)
    if interaction_pairs:
        feature_to_idx = {feature: i for i, feature in enumerate(feature_list)}
        for feat1, feat2 in interaction_pairs:
            if feat1 in feature_to_idx and feat2 in feature_to_idx:
                i, j = feature_to_idx[feat1], feature_to_idx[feat2]
                terms.append(te(i, j))

    return reduce(lambda a, b: a + b, terms)


def simple_gam(
    X: pd.DataFrame,
    y: pd.Series,
    use_gridsearch: bool = True,
    interaction_pairs: list[tuple[str, str]] = [],
):
    """
    Fit a simple GAM model without any feature selection.

    Args:
        X: DataFrame with features
        y: Series with target variable
        use_gridsearch: Whether to use gridsearch for hyperparameter tuning
        interaction_pairs: List of tuples for tensor interactions

    Returns:
        Dictionary with model, predictions, and performance metrics
    """
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_columns]

    if len(X_numeric.columns) == 0:
        raise ValueError("No numeric features found in X.")

    terms = get_spline_terms(X_numeric.columns.tolist(), interaction_pairs)
    gam = LinearGAM(terms)

    if use_gridsearch:
        gam.gridsearch(X_numeric.values, y.values, progress=False)
    else:
        gam.fit(X_numeric.values, y.values)
    # Make predictions
    y_pred = gam.predict(X_numeric.values)
    residuals = y.values - y_pred

    # Calculate performance metrics
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (
        len(y) - len(X_numeric.columns) - 1
    )
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    return {
        "model": gam,
        "X_cols": X_numeric.columns.tolist(),
        "y_true": y.copy(),
        "y_pred": y_pred.copy(),
        "residuals": residuals.copy(),
        "r2_score": r2,
        "adjusted_r2": adj_r2,
        "rmse": rmse,
        "mae": mae,
        "n_features": len(X_numeric.columns),
    }
