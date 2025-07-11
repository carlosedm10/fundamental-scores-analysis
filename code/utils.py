"""
Here we will create different functions to help with the data preparation.
"""

import os
import time
from typing import Literal

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import shap

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
    df: pd.DataFrame, columns: list[str], tolerance: float | None = 0.05
) -> np.ndarray:
    """
    Detect outliers using the Isolation Forest method.
    Note that the columns have to be scaled before using this function.
    """
    Y = df[columns]
    if tolerance is None:
        contamination = "auto"
    else:
        contamination = tolerance

    iso_forest = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=42
    )
    iso_forest.fit(Y)

    outliers_iso = iso_forest.predict(Y)
    return outliers_iso


def svm_outliers(
    df: pd.DataFrame, columns: list[str], tolerance: float | None = 0.05
) -> np.ndarray:
    """
    Detect outliers using the SVM method.
    Note that the columns have to be scaled before using this function.
    """
    if tolerance is None:
        nu = 0.5
    else:
        nu = tolerance

    Y = df[columns]
    # Initialize and fit model
    svm = OneClassSVM(nu=nu)
    svm.fit(Y)

    outliers_ocsvm = svm.predict(Y)
    return outliers_ocsvm


def lof_outliers(
    df: pd.DataFrame, columns: list[str], tolerance: float | None = 0.05
) -> np.ndarray:
    """
    Detect outliers using the Local Outlier Factor method.
    Note that the columns have to be scaled before using this function.
    """
    if tolerance is None:
        contamination = "auto"
    else:
        contamination = tolerance

    Y = df[columns]
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)

    outliers_lof = lof.fit_predict(Y)
    return outliers_lof


def multicretieria_outliers(
    df: pd.DataFrame,
    columns: list[str],
    tolerance: float | None = 0.05,
    training: bool = True,
) -> np.ndarray:
    """
    Detect outliers using the Multi-criteria method.
    """
    df_copy = df.copy()
    start_time = time.perf_counter()
    # Transform the data to a Gaussian distribution
    for column in columns:
        df_copy[column] = gaussian_yj_transform(df_copy[column])
    end_time = time.perf_counter()
    print(
        f"Time taken for gaussian_yj_transform: {end_time - start_time} seconds"
    )

    start_time = time.perf_counter()
    # Detect outliers using the Isolation Forest method
    outliers_iso = if_outliers(df_copy, columns, tolerance)
    end_time = time.perf_counter()
    print(f"Time taken for if_outliers: {end_time - start_time} seconds")

    start_time = time.perf_counter()
    # Detect outliers using the SVM method
    outliers_svm = svm_outliers(
        df_copy, columns, tolerance * 1.5 if tolerance else None
    )
    end_time = time.perf_counter()
    print(f"Time taken for svm_outliers: {end_time - start_time} seconds")

    if training is False:
        start_time = time.perf_counter()
        # Detect outliers using the Local Outlier Factor method
        outliers_lof = lof_outliers(
            df_copy,
            columns,
            (
                (tolerance * 3 if tolerance * 3 < 0.5 else 0.5)
                if tolerance is not None
                else None
            ),
        )
        end_time = time.perf_counter()
        print(f"Time taken for lof_outliers: {end_time - start_time} seconds")
        # Combine the outliers using the intersection
        true_outliers = outliers_iso & outliers_svm & outliers_lof
    else:
        true_outliers = outliers_iso & outliers_svm
    return true_outliers


def outliers_comparison(df: pd.DataFrame):
    outlier_counts_by_region = df.groupby("region")["outlier"].sum()
    # Count only the number of normal (non-outlier) data points per region
    total_counts_by_region = (
        df.groupby("region")["outlier"].apply(lambda x: (~x).sum())
        + outlier_counts_by_region
    )
    outlier_comparison = pd.DataFrame(
        {
            "total_datapoints": total_counts_by_region,
            "num_outliers": outlier_counts_by_region,
            "outlier_percentage": (
                (
                    outlier_counts_by_region / total_counts_by_region * 100
                ).round(2)
            ),
        }
    )
    print("Outlier comparison per region:")
    print(outlier_comparison)


############################ PLOTS ############################


def histogram(
    df: pd.DataFrame,
    figsize: tuple = (15, 4),
    center: int | None = None,
    export_path: str | None = None,
    numeric_columns: list[str] | None = None,
) -> None:
    """
    Plot histograms for all numeric columns in the DataFrame.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(
            include=["float64", "int64", "datetime64"]
        ).columns.tolist()

    plt.figure(figsize=(figsize[0], len(numeric_columns) * figsize[1]))

    for idx, column in enumerate(numeric_columns, 1):
        data = df[column]
        plt.subplot(len(numeric_columns), 1, idx)
        plt.hist(data, bins=100, edgecolor="black")
        # Centering logic
        if center is not None:
            max_dev = max(abs(data.min() - center), abs(data.max() - center))
            plt.xlim(center - max_dev, center + max_dev)

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        plt.savefig(export_path)
        plt.show()
    else:
        plt.show()


def boxplot(
    df: pd.DataFrame,
    figsize: tuple = (15, 4),
    columns: list[str] | None = None,
    export_path: str | None = None,
) -> None:
    """
    Plot boxplots for specified columns in the DataFrame.
    """
    if columns is None:
        columns = df.select_dtypes(
            include=["float64", "int64", "datetime64"]
        ).columns.tolist()

    plt.figure(figsize=(figsize[0], len(columns) * figsize[1]))

    for idx, column in enumerate(columns, 1):
        plt.subplot(len(columns), 1, idx)
        sns.boxplot(data=df, y=column)
        plt.title(f"Box Plot - {column}")
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        plt.savefig(export_path)
        plt.show()
    else:
        plt.show()


def dummy_bar_plot(
    df: pd.DataFrame,
    columns: list[str],
    figsize: tuple = (15, 4),
    export_path: str | None = None,
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

    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        plt.savefig(export_path)
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
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
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
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
        plt.show()
    else:
        plt.show()


def model_performance_summary(
    summary_df: pd.DataFrame,
    order: list[str],
    export_path: str | None = None,
):
    """
    Extended performance summary for regression models.
    Generates:
    - Heatmaps for R², R² adjusted, RMSE, MAE
    - Barplot of number of final variables per score and profit
    """

    # Set up figure with 2 rows: 4 heatmaps + 1 barplot
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2)
    ax_r2 = fig.add_subplot(gs[0, 0])
    ax_r2_adj = fig.add_subplot(gs[0, 1])
    ax_mae = fig.add_subplot(gs[1, 0])
    ax_rmse = fig.add_subplot(gs[1, 1])
    ax_bar = fig.add_subplot(gs[2, :])

    # R² heatmap
    heatmap_r2 = summary_df.pivot(
        index="score_type", columns="profit", values="r2_score"
    ).reindex(columns=order)
    sns.heatmap(heatmap_r2, annot=True, cmap="viridis", fmt=".2f", ax=ax_r2)
    ax_r2.set_title("R²")
    ax_r2.set_xlabel("Profit Horizon")
    ax_r2.set_ylabel("Score Type")

    # R² adjusted heatmap
    heatmap_r2_adj = summary_df.pivot(
        index="score_type", columns="profit", values="adjusted_r2"
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_r2_adj, annot=True, cmap="crest_r", fmt=".2f", ax=ax_r2_adj
    )
    ax_r2_adj.set_title("Adjusted R²")
    ax_r2_adj.set_xlabel("Profit Horizon")
    ax_r2_adj.set_ylabel("Score Type")

    # MAE heatmap
    heatmap_mae = summary_df.pivot(
        index="score_type", columns="profit", values="mae"
    ).reindex(columns=order)
    sns.heatmap(heatmap_mae, annot=True, cmap="magma_r", fmt=".2f", ax=ax_mae)
    ax_mae.set_title("MAE (Mean Absolute Error)")
    ax_mae.set_xlabel("Profit Horizon")
    ax_mae.set_ylabel("Score Type")

    # RMSE heatmap
    heatmap_rmse = summary_df.pivot(
        index="score_type", columns="profit", values="rmse"
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_rmse, annot=True, cmap="rocket_r", fmt=".2f", ax=ax_rmse
    )
    ax_rmse.set_title("RMSE (Root Mean Squared Error)")
    ax_rmse.set_xlabel("Profit Horizon")
    ax_rmse.set_ylabel("Score Type")

    # Barplot of number of final variables
    sns.barplot(
        data=summary_df,
        x="profit",
        y="n_features",
        hue="score_type",
        order=order,
        ax=ax_bar,
    )
    ax_bar.set_title("Number of Variables in Final Models")
    ax_bar.set_ylabel("Feature Count")
    ax_bar.set_xlabel("Profit Horizon")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.suptitle("Model Performance Summary", fontsize=18)

    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        plt.savefig(export_path, dpi=300, bbox_inches="tight")
        plt.show()
    else:
        plt.show()


def nn_shap_plot(
    summary: dict,
    title: str,
    sample_size: int = 100,
    plot_type: Literal["bar", "dot"] = "bar",
    max_features: int = 10,
    export_path: str | None = None,
):
    """
    Generate SHAP summary plot for a model stored in the regression summary.

    Args:
        summary: Dictionary output from simple_neural_network_regression
        sample_size: Number of validation samples to use for SHAP
        plot_type: Type of SHAP plot ("bar" or "dot")
        max_features: Maximum number of features to display (default: 10)
        title: Custom title for the plot. If None, auto-generates based on model info
        export_path: Path to save the plot

    Returns:
        Tuple of (explainer, shap_values, X_val_sampled)
    """
    model = summary["model"]
    scaler = summary["scaler"]
    X_cols = summary["X_cols"]

    # Prepare validation data
    X_val = summary["X_val"]
    X_val_scaled = scaler.transform(X_val)

    # Subsample to reduce SHAP computation cost
    if sample_size < X_val_scaled.shape[0]:
        idx = np.random.choice(
            X_val_scaled.shape[0], sample_size, replace=False
        )
        X_val_sampled = X_val_scaled[idx]
        X_val_sampled_raw = X_val.iloc[idx]
    else:
        X_val_sampled = X_val_scaled
        X_val_sampled_raw = X_val

    # Define model prediction function for SHAP
    def predict_fn(x):
        return model.predict(x)

    # Use KernelExplainer for MLPRegressor
    explainer = shap.KernelExplainer(predict_fn, X_val_sampled)
    shap_values = explainer.shap_values(X_val_sampled)

    # Plot
    shap.summary_plot(
        shap_values,
        X_val_sampled_raw,
        feature_names=X_cols,
        plot_type=plot_type,
        max_display=max_features,
        show=False,
    )

    # Add title to the plot
    plt.title(title, fontsize=14, pad=20)

    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        plt.savefig(export_path)
        plt.show()
    else:
        plt.show()

    return explainer, shap_values, X_val_sampled_raw


def models_scores_summary(summary_df, export_path=None):
    """
    For each score_type, create a transposed table showing coefficient, sign, and t-value
    of the main score variable (e.g., 'quality') across profit horizons.

    Returns a dict of transposed DataFrames keyed by score_type.
    """
    results = {}
    score_types = summary_df["score_type"].unique()

    for score in score_types:
        subset = summary_df[summary_df["score_type"] == score]
        profits = sorted(subset["profit"].unique())

        rows = []
        for profit in profits:
            model_row = subset[subset["profit"] == profit].iloc[0]
            coefs = model_row["coefficients"]
            tvals = model_row["tvalues"]

            coef = coefs.get(score, np.nan)  # main variable only
            tval = tvals.get(score, np.nan)
            sign = "+" if coef > 0 else "–" if coef < 0 else "0"

            rows.append(
                {
                    "profit_horizon": profit,
                    "coefficient": coef,
                    "sign": sign,
                    "t_value": tval,
                }
            )

        df = pd.DataFrame(rows).set_index("profit_horizon")
        df_t = df.transpose()  # transpose here

        results[score] = df_t

        if export_path:
            os.makedirs(export_path, exist_ok=True)
            df_t.to_csv(
                f"{export_path}/{score}_main_score_summary_transposed.csv"
            )

    return results


def model_performance_summary_2(
    summary_df: pd.DataFrame,
    order: list[str],
    export_path: str | None = None,
):
    """
    Extended performance summary for regression models.
    Displays:
    - Train & Val R², Adj R², MAE, RMSE heatmaps
    - Barplot of number of variables
    """
    fig = plt.figure(figsize=(24, 24))
    gs = fig.add_gridspec(5, 2)

    # Axes for train
    ax_r2_train = fig.add_subplot(gs[0, 0])
    ax_r2_adj_train = fig.add_subplot(gs[1, 0])
    ax_mae_train = fig.add_subplot(gs[2, 0])
    ax_rmse_train = fig.add_subplot(gs[3, 0])

    # Axes for val
    ax_r2_val = fig.add_subplot(gs[0, 1])
    ax_r2_adj_val = fig.add_subplot(gs[1, 1])
    ax_mae_val = fig.add_subplot(gs[2, 1])
    ax_rmse_val = fig.add_subplot(gs[3, 1])

    # Barplot of features
    ax_bar = fig.add_subplot(gs[4, :])

    # Train R²
    heatmap_train_r2 = summary_df.pivot(
        index="score_type", columns="profit", values="train_r2_score"
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_train_r2, annot=True, cmap="Blues", fmt=".2f", ax=ax_r2_train
    )
    ax_r2_train.set_title("Train R²")
    ax_r2_train.set_xlabel("Profit Horizon")
    ax_r2_train.set_ylabel("Score Type")

    # Val R²
    heatmap_val_r2 = summary_df.pivot(
        index="score_type", columns="profit", values="val_r2_score"
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_val_r2, annot=True, cmap="Blues", fmt=".2f", ax=ax_r2_val
    )
    ax_r2_val.set_title("Validation R²")
    ax_r2_val.set_xlabel("Profit Horizon")
    ax_r2_val.set_ylabel("Score Type")

    # Train Adjusted R²
    heatmap_train_r2_adj = summary_df.pivot(
        index="score_type", columns="profit", values="train_adjusted_r2"
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_train_r2_adj,
        annot=True,
        cmap="crest",
        fmt=".2f",
        ax=ax_r2_adj_train,
    )
    ax_r2_adj_train.set_title("Train Adjusted R²")
    ax_r2_adj_train.set_xlabel("Profit Horizon")
    ax_r2_adj_train.set_ylabel("Score Type")

    # Val Adjusted R²
    heatmap_val_r2_adj = summary_df.pivot(
        index="score_type", columns="profit", values="val_adjusted_r2"
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_val_r2_adj,
        annot=True,
        cmap="crest",
        fmt=".2f",
        ax=ax_r2_adj_val,
    )
    ax_r2_adj_val.set_title("Validation Adjusted R²")
    ax_r2_adj_val.set_xlabel("Profit Horizon")
    ax_r2_adj_val.set_ylabel("Score Type")

    # Train MAE
    heatmap_train_mae = summary_df.pivot(
        index="score_type", columns="profit", values="train_mae"
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_train_mae,
        annot=True,
        cmap="magma_r",
        fmt=".2f",
        ax=ax_mae_train,
    )
    ax_mae_train.set_title("Train MAE")
    ax_mae_train.set_xlabel("Profit Horizon")
    ax_mae_train.set_ylabel("Score Type")

    # Val MAE
    heatmap_val_mae = summary_df.pivot(
        index="score_type", columns="profit", values="val_mae"
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_val_mae, annot=True, cmap="magma_r", fmt=".2f", ax=ax_mae_val
    )
    ax_mae_val.set_title("Validation MAE")
    ax_mae_val.set_xlabel("Profit Horizon")
    ax_mae_val.set_ylabel("Score Type")

    # Train RMSE
    heatmap_train_rmse = summary_df.pivot(
        index="score_type", columns="profit", values="train_rmse"
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_train_rmse,
        annot=True,
        cmap="rocket_r",
        fmt=".2f",
        ax=ax_rmse_train,
    )
    ax_rmse_train.set_title("Train RMSE")
    ax_rmse_train.set_xlabel("Profit Horizon")
    ax_rmse_train.set_ylabel("Score Type")

    # Val RMSE
    heatmap_val_rmse = summary_df.pivot(
        index="score_type", columns="profit", values="val_rmse"
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_val_rmse,
        annot=True,
        cmap="rocket_r",
        fmt=".2f",
        ax=ax_rmse_val,
    )
    ax_rmse_val.set_title("Validation RMSE")
    ax_rmse_val.set_xlabel("Profit Horizon")
    ax_rmse_val.set_ylabel("Score Type")

    # Barplot of number of variables
    sns.barplot(
        data=summary_df,
        x="profit",
        y="n_features",
        hue="score_type",
        order=order,
        ax=ax_bar,
    )
    ax_bar.set_title("Number of Variables in Final Models")
    ax_bar.set_ylabel("Feature Count")
    ax_bar.set_xlabel("Profit Horizon")

    # Layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.suptitle("Train vs Validation Model Performance", fontsize=22)

    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        plt.savefig(export_path, dpi=300, bbox_inches="tight")
        plt.show()
    else:
        plt.show()


def model_consistency_and_overfitting(
    summary_df: pd.DataFrame,
    p_threshold: float = 0.2,
    export_folder: str | None = None,
):
    """
    Generate coefficient consistency tables for each score_type.
    Each table compares coefficient values, signs, and significance
    across different profit horizons.
    """

    score_types = summary_df["score_type"].unique()
    profit_horizons = summary_df["profit"].unique()

    for score in score_types:
        subset = summary_df[summary_df["score_type"] == score]

        # Get all features ever used in any model of this score
        all_features = set()
        for row in subset["coefficients"]:
            all_features.update(row.keys())

        all_features = sorted(
            f for f in all_features if f != "const"
        )  # exclude intercept

        # Create tables for coefficients, signs, significance
        coef_table = pd.DataFrame(index=all_features)
        sign_table = pd.DataFrame(index=all_features)
        signif_table = pd.DataFrame(index=all_features)

        for _, row in subset.iterrows():
            profit = row["profit"]
            coefs = row["coefficients"]
            pvals = row["pvalues"]

            # Fill values per feature
            for feature in all_features:
                coef = coefs.get(feature, np.nan)
                pval = pvals.get(feature, np.nan)

                coef_table.loc[feature, profit] = coef
                sign_table.loc[feature, profit] = (
                    np.sign(coef) if not np.isnan(coef) else 0
                )
                signif_table.loc[feature, profit] = (
                    pval < p_threshold if not np.isnan(pval) else False
                )

        # Export or return
        if export_folder:
            os.makedirs(export_folder, exist_ok=True)
            coef_table.to_csv(f"{export_folder}/{score}_coefficients.csv")
            sign_table.to_csv(f"{export_folder}/{score}_signs.csv")
            signif_table.to_csv(f"{export_folder}/{score}_significance.csv")

        else:
            print(f"\nScore: {score}")
            print(coef_table.round(4))
            print(sign_table.astype(int))
            print(signif_table.astype(bool))


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


def plot_residual_diagnostics_2(
    train_residuals: pd.Series,
    train_fitted: pd.Series,
    val_residuals: pd.Series,
    val_fitted: pd.Series,
    title: str = "",
    export_path: str | None = None,
):
    """
    Plot residual diagnostics for both training and validation sets:
    - Residuals vs Time
    - Residuals vs Fitted
    - Histogram
    - Q-Q Plot
    - ACF
    - PACF
    """
    # Ensure Series type
    train_residuals = pd.Series(train_residuals)
    train_fitted = pd.Series(train_fitted)
    val_residuals = pd.Series(val_residuals)
    val_fitted = pd.Series(val_fitted)

    fig, axs = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle(f"Residual Diagnostics: {title}", fontsize=18)

    # === TRAIN PLOTS (blue) ===
    axs[0, 0].plot(train_residuals.values, color="blue")
    axs[0, 0].set_title("Train: Residuals vs. Time")
    axs[0, 0].set_xlabel("Index")
    axs[0, 0].set_ylabel("Residuals")

    axs[0, 1].scatter(train_fitted, train_residuals, alpha=0.6, color="blue")
    axs[0, 1].set_title("Train: Residuals vs. Fitted")
    axs[0, 1].set_xlabel("Fitted")
    axs[0, 1].set_ylabel("Residuals")

    sm.qqplot(train_residuals, line="s", ax=axs[0, 2], color="blue")
    axs[0, 2].set_title("Train: Q-Q Plot")

    axs[0, 3].hist(train_residuals, bins=50, edgecolor="black", color="blue")
    axs[0, 3].set_title("Train: Histogram")
    axs[0, 3].set_xlabel("Residuals")

    # === VALIDATION PLOTS (orange) ===
    axs[1, 0].plot(val_residuals.values, color="orange")
    axs[1, 0].set_title("Val: Residuals vs. Time")
    axs[1, 0].set_xlabel("Index")
    axs[1, 0].set_ylabel("Residuals")

    axs[1, 1].scatter(val_fitted, val_residuals, alpha=0.6, color="orange")
    axs[1, 1].set_title("Val: Residuals vs. Fitted")
    axs[1, 1].set_xlabel("Fitted")
    axs[1, 1].set_ylabel("Residuals")

    sm.qqplot(val_residuals, line="s", ax=axs[1, 2], color="orange")
    axs[1, 2].set_title("Val: Q-Q Plot")

    axs[1, 3].hist(val_residuals, bins=50, edgecolor="black", color="orange")
    axs[1, 3].set_title("Val: Histogram")
    axs[1, 3].set_xlabel("Residuals")

    # === ACF and PACF ===
    plot_acf(train_residuals, ax=axs[2, 0], lags=12, zero=False)
    for line in axs[2, 0].get_lines():
        line.set_color("blue")
    axs[2, 0].set_title("Train: ACF")

    plot_pacf(train_residuals, ax=axs[2, 1], lags=12, zero=False)
    for line in axs[2, 1].get_lines():
        line.set_color("blue")
    axs[2, 1].set_title("Train: PACF")

    plot_acf(val_residuals, ax=axs[2, 2], lags=12, zero=False)
    for line in axs[2, 2].get_lines():
        line.set_color("orange")
    axs[2, 2].set_title("Val: ACF")

    plot_pacf(val_residuals, ax=axs[2, 3], lags=12, zero=False)
    for line in axs[2, 3].get_lines():
        line.set_color("orange")
    axs[2, 3].set_title("Val: PACF")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        plt.savefig(export_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def check_white_noise(residuals, exog, alpha=0.2):
    """
    Check if the residuals are white noise using the following tests:
    - Mean Value Test
    - Heteroscedasticity Tests
        - White Test
        - Breusch-Pagan Test
        - F and t tests
    - Normality Tests
        - Shapiro-Wilk Test
        - Jarque-Bera Test
    - Autocorrelation Test (Durbin-Watson Test)

    Args:
    residuals (pd.Series): Residuals of the model.
    alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
    dict: Dictionary with the results of the tests.
    """

    def _format_diagnostics(diagnostics):
        print("\nDiagnostic Test Results:")
        print("-" * 50)
        for key, value in diagnostics.items():
            print(f"{key.ljust(10)}: {value}")

    # Defining the model:
    squared_residuals = residuals**2
    y = exog
    t = np.arange(1, len(y) + 1)
    diagnostics = {}

    all_tests_passed = True

    # 1. Mean Value Test
    _, p_value_mean = stats.ttest_1samp(residuals, 0)
    diagnostics["Mean Test p-value"] = p_value_mean
    diagnostics["Mean Test"] = "Pass" if p_value_mean > alpha else "Fail"  # type: ignore
    if p_value_mean <= alpha:  # type: ignore
        all_tests_passed = False

    # 2. Heteroscedasticity Tests
    # White Test
    _, p_value_white, _, _ = het_white(squared_residuals, sm.add_constant(y))
    diagnostics["White Test p-value"] = p_value_white
    diagnostics["White Test"] = "Pass" if p_value_white > alpha else "Fail"
    if p_value_white <= alpha:
        all_tests_passed = False

    # Breusch-Pagan Test
    _, _, _, p_value_breusch_pagan = het_breuschpagan(
        residuals, sm.add_constant(y)
    )
    diagnostics["Breusch-Pagan Test p-value"] = p_value_breusch_pagan
    diagnostics["Breusch-Pagan Test"] = (
        "Pass" if p_value_breusch_pagan > alpha else "Fail"
    )
    if p_value_breusch_pagan <= alpha:
        all_tests_passed = False

    # F and t tests
    residual_linear_model = sm.OLS(squared_residuals, t).fit()
    f_pvalue = residual_linear_model.f_pvalue  # P-value for the F-statistic
    diagnostics["F Test p-value"] = f_pvalue
    diagnostics["F Test"] = "Pass" if f_pvalue > alpha else "Fail"
    if f_pvalue <= alpha:
        all_tests_passed = False

    # 3. Normality Tests
    # # Add test Kolmogorov-Smirnov
    # _, p_value = stats.kstest(residuals, "norm")
    # diagnostics["Kolmogorov-Smirnov Test p-value"] = p_value
    # diagnostics["Kolmogorov-Smirnov Test"] = (
    #     "Pass" if p_value > alpha else "Fail"
    # )
    # if p_value <= alpha:
    #     all_tests_passed = False

    # Shapiro-Wilk Test
    _, p_value_shapiro = stats.shapiro(residuals)
    diagnostics["Shapiro Test p-value"] = p_value_shapiro
    diagnostics["Shapiro Test"] = "Pass" if p_value_shapiro > alpha else "Fail"
    if p_value_shapiro <= alpha:
        all_tests_passed = False

    # Jarque-Bera Test
    _, p_value_jarque_bera, _, _ = jarque_bera(residuals)
    diagnostics["Jarque-Bera Test p-value"] = p_value_jarque_bera
    diagnostics["Jarque-Bera Test"] = (
        "Pass" if p_value_jarque_bera > alpha else "Fail"
    )
    if p_value_jarque_bera <= alpha:
        all_tests_passed = False

    # 4. Autocorrelation Test (Durbin-Watson Test)
    dw_stat = durbin_watson(residuals)
    diagnostics["Durbin-Watson stat"] = dw_stat
    # Interpret Durbin-Watson statistic
    if dw_stat < 1.5 or dw_stat > 2.5:
        diagnostics["Durbin-Watson"] = "Fail"
        all_tests_passed = False
    else:
        diagnostics["Durbin-Watson"] = "Pass"

    # Final Verdict
    diagnostics["Final Verdict"] = (
        "The Residues are White Noise"
        if all_tests_passed
        else "The Residues are not White Noise"
    )

    return _format_diagnostics(diagnostics)


######################## MODELS ############################


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


def simple_neural_network_regression(
    X: pd.DataFrame,
    y: pd.Series,
    validation_size: float = 0.2,
    hidden_layer_sizes: tuple[int, int] = (64, 64),
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
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_size, random_state=42
    )

    # Add Gaussian noise to X_train and y_train to reduce overfitting
    noise_std_X = 0.01 * X_train.std(axis=0, ddof=0)
    noise_std_y = 0.01 * y_train.std(ddof=0)
    X_train = X_train + np.random.normal(0, noise_std_X, X_train.shape)
    y_train = y_train + np.random.normal(0, noise_std_y, y_train.shape)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

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

    # Residuals
    residuals_train = y_train - y_train_pred
    residuals_val = y_val - y_val_pred

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
        # Feature Info
        "n_region_dummies": len(
            [col for col in X.columns if "region_" in col]
        ),
        "n_sector_dummies": len(
            [col for col in X.columns if "sector_" in col]
        ),
    }

    return summary


def simple_neural_network_classification(
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
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_size, random_state=random_state, stratify=y
    )

    # Add Gaussian noise to X_train to reduce overfitting
    noise_std_X = 0.01 * X_train.std(axis=0, ddof=0)
    X_train = X_train + np.random.normal(0, noise_std_X, X_train.shape)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

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

    # Prediction probabilities (if available)
    y_train_proba = model.predict_proba(X_train_scaled)
    y_val_proba = model.predict_proba(X_val_scaled)

    # Get class labels
    classes = model.classes_

    # Performance metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)

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
    val_recall = recall_score(
        y_val, y_val_pred, average=average_method, zero_division=0
    )
    val_f1 = f1_score(
        y_val, y_val_pred, average=average_method, zero_division=0
    )

    # Confusion matrices
    train_cm = confusion_matrix(y_train, y_train_pred)
    val_cm = confusion_matrix(y_val, y_val_pred)

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
