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
    print("We got the spline terms:", len(terms))
    print(terms)
    gam = LinearGAM(terms)

    if use_gridsearch:
        gam.gridsearch(X_numeric.values, y.values, progress=False)
    else:
        gam.fit(X_numeric.values, y.values)
    print("We fitted the model")
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


def fast_backward_gam(
    X, y, threshold=0.01, max_iter=5, interaction_pairs=None
):
    features = X.columns.tolist()
    removed_features = []

    def fit_gam(feature_list, use_gridsearch=True):
        if len(feature_list) == 0:
            raise ValueError("No features left to fit the model.")

        terms = get_spline_terms(feature_list, interaction_pairs)
        gam = LinearGAM(terms)

        if use_gridsearch:
            gam.gridsearch(X[feature_list].values, y.values, progress=False)
        else:
            gam.fit(X[feature_list].values, y.values)
        return gam

    def quick_score(feature_list):
        # Fast deviance estimate: 3-fold CV + no gridsearch
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        X_vals = X[feature_list].values
        y_vals = y.values
        devs = []

        for train_idx, test_idx in kf.split(X_vals):
            X_train, X_test = X_vals[train_idx], X_vals[test_idx]
            y_train, y_test = y_vals[train_idx], y_vals[test_idx]

            terms = get_spline_terms(feature_list, interaction_pairs)
            model = LinearGAM(terms).fit(X_train, y_train)

            y_pred = model.predict(X_test)
            devs.append(mean_squared_error(y_test, y_pred))

        return np.mean(devs)

    current_features = features[:]
    best_score = quick_score(current_features)
    iteration = 0

    while len(current_features) > 1 and iteration < max_iter:
        scores = {}
        for feature in current_features:
            trial_features = [f for f in current_features if f != feature]
            try:
                score = quick_score(trial_features)
                scores[feature] = score
            except Exception as e:
                print(f"Skipping {feature} due to error: {e}")
                continue

        worst_feature, worst_score = min(scores.items(), key=lambda x: x[1])
        if best_score - worst_score > threshold:
            print(
                f"Removing {worst_feature}: improved deviance {best_score:.4f} → {worst_score:.4f}"
            )
            current_features.remove(worst_feature)
            removed_features.append(worst_feature)
            best_score = worst_score
            iteration += 1
        else:
            break

    # Final GAM with gridsearch for accuracy
    final_model = fit_gam(current_features, use_gridsearch=True)
    X_final = X[current_features]
    y_pred = final_model.predict(X_final)
    residuals = y.values - y_pred
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(current_features) - 1)

    return {
        "model": final_model,
        "summary": final_model.summary(),
        "X_cols": current_features,
        "y_true": y.copy(),
        "y_pred": y_pred.copy(),
        "residuals": residuals.copy(),
        "r2_score": r2,
        "adjusted_r2": adj_r2,
        "n_features": len(current_features),
        "removed_features": removed_features,
    }
