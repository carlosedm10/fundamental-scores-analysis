"""
Here we will create different functions to help with the data preparation.
"""

import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from pygam import LinearGAM, s
from functools import reduce


################################### DATA  ###################################
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the data from the file path.
    """
    return pd.read_csv(file_path, encoding="utf-8")


def missing_values(
    df_original: pd.DataFrame, df_cleaned: pd.DataFrame
) -> None:
    """
    Analyze and print information about missing values and retention rate.

    Args:
        df_original (pd.DataFrame): First DataFrame to analyze
        df_cleaned (pd.DataFrame): Second DataFrame to analyze
    """
    # Calculate number of rows with any missing value
    print(
        f"Retention rate: {(df_cleaned.shape[0]/df_original.shape[0]*100):.2f}%"
    )

    missing_values = df_original.isnull().sum()
    if missing_values.any():
        missing_values = missing_values[
            missing_values > 0
        ]  # Only show columns with missing values
        missing_percentages = (missing_values / len(df_original) * 100).round(
            2
        )
        print("\nMissing values and percentages by column:")
        for col in missing_values.index:
            print(
                f"{col}: {missing_values[col]} values ({missing_percentages[col]}%)"
            )


def separate_df_by_scores(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Create separate dataframes for each score type.
    """

    score_types = ["quality", "growth", "value", "dividend"]
    dataframes = {}

    for score in score_types:
        # Create separate dataframes for each score
        df_temp = df.copy()
        df_temp.drop(
            columns=[
                excluded_score
                for excluded_score in score_types
                if excluded_score != score
            ],
            inplace=True,
        )
        df_temp.dropna(inplace=True)

        dataframes[score] = df_temp

    return dataframes


############################ OUTLIERS ############################
def gaussian_yj_transform(x: pd.Series) -> pd.Series:
    """
    Transform the target variable to a Gaussian distribution using Yeo-Johnson transformation.
    """
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    return pd.Series(
        transformer.fit_transform(x.values.reshape(-1, 1)).ravel(),
        index=x.index,
    )


def iqr_outliers_removal(
    df: pd.DataFrame,
    columns: list[str],
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """
    Detect outliers using the IQR method, and removes them.
    """
    df_copy = df.copy()
    for column in columns:
        q1 = df_copy[column].quantile(tolerance)
        q3 = df_copy[column].quantile(1 - tolerance)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_copy = df_copy[df_copy[column].between(lower_bound, upper_bound)]

    return df_copy


def if_outliers(
    df: pd.DataFrame, columns: list[str], tolerance: float = 0.05
) -> np.ndarray:
    """
    Detect outliers using the Isolation Forest method.
    Note that the columns have to be scaled before using this function.
    """
    Y = df[columns]
    iso_forest = IsolationForest(
        n_estimators=200, contamination=tolerance, random_state=42
    )
    iso_forest.fit(Y)

    outliers_iso = iso_forest.predict(Y)
    return outliers_iso


def svm_outliers(
    df: pd.DataFrame, columns: list[str], tolerance: float = 0.05
) -> np.ndarray:
    """
    Detect outliers using the SVM method.
    Note that the columns have to be scaled before using this function.
    """
    Y = df[columns]
    # Initialize and fit model
    svm = OneClassSVM(nu=0.05)
    svm.fit(Y)

    outliers_ocsvm = svm.predict(Y)
    return outliers_ocsvm


def lof_outliers(
    df: pd.DataFrame, columns: list[str], tolerance: float = 0.05
) -> np.ndarray:
    """
    Detect outliers using the Local Outlier Factor method.
    Note that the columns have to be scaled before using this function.
    """
    Y = df[columns]
    lof = LocalOutlierFactor(n_neighbors=20, contamination=tolerance)

    outliers_lof = lof.fit_predict(Y)
    return outliers_lof


def multicretieria_outliers(
    df: pd.DataFrame, columns: list[str], tolerance: float = 0.05
) -> np.ndarray:
    """
    Detect outliers using the Multi-criteria method.
    """
    df_copy = df.copy()

    # Transform the data to a Gaussian distribution
    for column in columns:
        df_copy[column] = gaussian_yj_transform(df_copy[column])

    # Detect outliers using the Isolation Forest method
    outliers_iso = if_outliers(df_copy, columns, tolerance)

    # Detect outliers using the SVM method
    outliers_svm = svm_outliers(df_copy, columns, tolerance)

    # Detect outliers using the Local Outlier Factor method
    outliers_lof = lof_outliers(df_copy, columns, tolerance)

    # Combine the outliers using the intersection
    true_outliers = outliers_iso & outliers_svm & outliers_lof
    return true_outliers


############################ PLOTS ############################


def plot_numeric_distributions(
    df: pd.DataFrame, figsize: tuple = (15, 4), file_path: str | None = None
) -> None:
    """
    Plot histograms for all numeric columns in the DataFrame.
    """
    numeric_columns = df.select_dtypes(
        include=["float64", "int64", "datetime64"]
    ).columns

    plt.figure(figsize=(figsize[0], len(numeric_columns) * figsize[1]))

    for idx, column in enumerate(numeric_columns, 1):
        plt.subplot(len(numeric_columns), 1, idx)
        plt.hist(df[column], bins=100, edgecolor="black")

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if file_path:
        plt.savefig(file_path)
        plt.show()
    else:
        plt.show()


def dummy_bar_plot(
    df: pd.DataFrame,
    columns: list[str],
    figsize: tuple = (15, 4),
    file_path: str | None = None,
) -> None:
    """
    Plot bar plots for specified columns in the DataFrame.
    """
    plt.figure(figsize=(figsize[0], len(columns) * figsize[1]))

    for idx, column in enumerate(columns, 1):
        plt.subplot(len(columns), 1, idx)
        df[column].value_counts().plot(kind="bar")
        plt.title(f"Distribution of {column}")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.xlabel("")  # Remove x-axis label

    plt.tight_layout()

    if file_path:
        plt.savefig(file_path)
        plt.show()
    else:
        plt.show()


def correlation_matrix(
    df: pd.DataFrame, columns: list[str], file_path: str | None = None
) -> None:
    """
    Plot a correlation matrix for the DataFrame showing only the lower triangle.
    """
    df_columns = df[columns]
    corr_matrix = df_columns.corr()

    # Create a mask for the lower triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot using seaborn for better visualization
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        square=True,
    )
    plt.title("Correlation Matrix of Numeric Variables")
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path)
        plt.show()
    else:
        plt.show()


def pairplot(
    df: pd.DataFrame,
    file_path: str | None = None,
    kde: bool = False,
    hue: str = "region",
    hue_order: list[str] | None = None,
    palette: dict[str, str] | None = None,
) -> None:

    if len(df) > 10000:
        df = df.sample(10000, random_state=42)

    g = sns.pairplot(
        df,
        hue=hue,
        diag_kind="kde",
        hue_order=hue_order,
        palette=palette,
        corner=True,
    )
    if kde:
        g.map_lower(sns.kdeplot, levels=4, color=".2")

    if file_path:
        plt.savefig(file_path)
        plt.close()
    else:
        plt.show()


def model_summary(
    summary_df: pd.DataFrame, order: list[str], export_path: str | None = None
):
    """
    Plot a heatmap and a barplot of the model summary.
    """
    # Create pivot table and reorder columns
    heatmap_data = summary_df.pivot(
        index="score_type", columns="profit", values="r2_score"
    ).reindex(columns=order)

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Plot heatmap on first subplot
    sns.heatmap(heatmap_data, annot=True, cmap="viridis", fmt=".2f", ax=ax1)
    ax1.set_title("Models R²")
    ax1.set_xlabel("Profit Horizon")
    ax1.set_ylabel("Score Type")

    # Plot barplot on second subplot
    sns.barplot(
        data=summary_df,
        x="profit",
        y="n_features",
        hue="score_type",
        order=order,
        ax=ax2,
    )
    ax2.set_title("Number of Variables in Final Models")
    ax2.set_ylabel("Feature Count")
    ax2.set_xlabel("Profit Horizon")

    plt.tight_layout()

    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        plt.savefig(export_path)
        plt.close()
    else:
        plt.show()


def plot_residual_diagnostics(
    residuals: pd.Series,
    fitted_values: pd.Series,
    title: str,
    export_path: str | None = None,
):
    """
    Plot the residual diagnostics of a model:
    - Residuals vs. Time
    - Residuals vs. Fitted Values
    - Autocorrelation
    - Partial Autocorrelation
    - Q-Q Plot
    - Histogram of Residuals
    """

    # Ensure inputs are pandas Series
    residuals = (
        pd.Series(residuals)
        if isinstance(residuals, np.ndarray)
        else residuals
    )
    fitted_values = (
        pd.Series(fitted_values)
        if isinstance(fitted_values, np.ndarray)
        else fitted_values
    )

    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Residual Diagnostics: {title}", fontsize=16)

    # Residuals vs. Time
    axs[0, 0].plot(residuals.values)
    axs[0, 0].set_title("Residuals vs. Time")
    axs[0, 0].set_xlabel("Index")
    axs[0, 0].set_ylabel("Residuals")

    # Residuals vs. Fitted Values
    axs[0, 1].scatter(fitted_values, residuals, alpha=0.6)
    axs[0, 1].set_title("Residuals vs. Fitted Values")
    axs[0, 1].set_xlabel("Fitted Values")
    axs[0, 1].set_ylabel("Residuals")

    # Autocorrelation
    plot_acf(residuals, ax=axs[1, 1], lags=12)
    axs[1, 1].set_title("Autocorrelation (ACF)")

    # Partial Autocorrelation
    plot_pacf(residuals, ax=axs[2, 0], lags=12)
    axs[2, 0].set_title("Partial Autocorrelation (PACF)")

    # Q-Q Plot
    sm.qqplot(residuals, line="s", ax=axs[1, 0])
    axs[1, 0].set_title("Q-Q Plot")

    # Histogram
    axs[2, 1].hist(residuals, bins=100, edgecolor="black")
    axs[2, 1].set_title("Histogram of Residuals")
    axs[2, 1].set_xlabel("Residuals")
    axs[2, 1].set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot
    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        plt.savefig(export_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


######################## MODELS ############################


def simple_backward_regression(
    X: pd.DataFrame, y: pd.Series, threshold: float = 0.20
) -> dict:
    """
    Perform simple backward elimination regression.
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

        # Drop the feature with the highest p-value
        worst_feature = pvalues.idxmax()
        X = X.drop(columns=worst_feature)
        removed_features.append(worst_feature)

    # Final model after elimination
    final_features = X.columns.tolist()
    final_model = sm.OLS(y, X).fit()
    summary = {
        "model": final_model,
        "summary": final_model.summary(),
        "X_cols": final_features,
        "y_true": y.copy(),
        "y_pred": final_model.fittedvalues.copy(),
        "residuals": final_model.resid.copy(),
        "r2_score": final_model.rsquared,
        "adjusted_r2": final_model.rsquared_adj,
        "n_features": len(final_features),
        "n_region_dummies": len(
            [col for col in final_features if "region_" in col]
        ),
        "n_sector_dummies": len(
            [col for col in final_features if "sector_" in col]
        ),
        "removed_features": removed_features,
    }
    return summary


def gam_backward_regression(
    X: pd.DataFrame, y: pd.Series, threshold: float = 0.20
) -> dict:
    """
    Perform backward elimination regression using Generalized Additive Models.
    """
    removed_features = []
    current_X = X.copy()

    # Backward elimination loop
    while True:
        # Fit GAM model
        gam = LinearGAM(n_splines=25).gridsearch(current_X, y)

        # Get p-values for each feature
        pvalues = pd.Series(
            gam.statistics_["p_values"], index=current_X.columns
        )

        if (pvalues <= threshold).all() or len(pvalues) == 0:
            break

        # Drop the feature with the highest p-value
        worst_feature = pvalues.idxmax()
        current_X = current_X.drop(columns=worst_feature)
        removed_features.append(worst_feature)

    # Final model after elimination
    final_features = current_X.columns.tolist()
    final_gam = LinearGAM().fit(current_X, y)

    summary = {
        "model": final_gam,
        "summary": final_gam.summary(),
        "X_cols": final_features,
        "y_true": y.copy(),
        "y_pred": final_gam.fittedvalues.copy(),
        "residuals": final_gam.resid.copy(),
        "r2_score": final_gam.rsquared,
        "adjusted_r2": final_gam.rsquared_adj,
        "n_features": len(final_features),
        "n_region_dummies": len(
            [col for col in final_features if "region_" in col]
        ),
        "n_sector_dummies": len(
            [col for col in final_features if "sector_" in col]
        ),
        "removed_features": removed_features,
    }
    return summary


def simple_backward_gam(X, y, threshold=0.01):
    features = X.columns.tolist()
    removed_features = []

    def fit_gam(feature_list):
        if len(feature_list) == 0:
            raise ValueError("No features left to fit the model.")
        elif len(feature_list) == 1:
            terms = s(0)
        else:
            terms = reduce(
                lambda a, b: a + b, [s(i) for i in range(len(feature_list))]
            )

        gam = LinearGAM(terms).gridsearch(
            X[feature_list].values, y.values, progress=False
        )
        return gam

    def cv_deviance(feature_list):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        X_vals = X[feature_list].values
        y_vals = y.values
        devs = []

        for train_idx, test_idx in kf.split(X_vals):
            X_train, X_test = X_vals[train_idx], X_vals[test_idx]
            y_train, y_test = y_vals[train_idx], y_vals[test_idx]

            term_list = [s(i) for i in range(len(feature_list))]
            terms = term_list[0]
            for t in term_list[1:]:
                terms += t
            model = LinearGAM(terms).gridsearch(
                X_train, y_train, progress=False
            )
            y_pred = model.predict(X_test)
            devs.append(mean_squared_error(y_test, y_pred))

        return np.mean(devs)

    current_features = features[:]
    best_score = cv_deviance(current_features)

    while len(current_features) > 1:
        scores = {}
        for feature in current_features:
            trial_features = [
                feat for feat in current_features if feat != feature
            ]
            try:
                score = cv_deviance(trial_features)
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
        else:
            break

    # Fit final model
    final_features = current_features
    final_model = fit_gam(final_features)
    X_final = X[final_features]
    y_pred = final_model.predict(X_final)
    residuals = y.values - y_pred
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(final_features) - 1)

    summary = {
        "model": final_model,
        "summary": final_model.summary(),
        "X_cols": final_features,
        "y_true": y.copy(),
        "y_pred": y_pred.copy(),
        "residuals": residuals.copy(),
        "r2_score": r2,
        "adjusted_r2": adj_r2,
        "n_features": len(final_features),
        "n_region_dummies": len(
            [col for col in final_features if "region_" in col]
        ),
        "n_sector_dummies": len(
            [col for col in final_features if "sector_" in col]
        ),
        "removed_features": removed_features,
    }

    return summary


def fast_backward_gam(X, y, threshold=0.01, max_iter=5):
    features = X.columns.tolist()
    removed_features = []

    def fit_gam(feature_list, use_gridsearch=True):
        if len(feature_list) == 0:
            raise ValueError("No features left to fit the model.")
        elif len(feature_list) == 1:
            terms = s(0)
        else:
            terms = reduce(
                lambda a, b: a + b, [s(i) for i in range(len(feature_list))]
            )
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
            model = LinearGAM(
                reduce(
                    lambda a, b: a + b,
                    [s(i) for i in range(len(feature_list))],
                )
            ).fit(X_train, y_train)
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
