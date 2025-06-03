"""
Here we will create different functions to help with the data preparation.
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


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
def iqr_outliers(
    df: pd.DataFrame,
    columns: list[str],
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """
    Detect outliers using the IQR method.
    """
    for column in columns:
        q1 = df[column].quantile(tolerance)
        q3 = df[column].quantile(1 - tolerance)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[df[column].between(lower_bound, upper_bound)]

    return df


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
        plt.show()
    else:
        plt.show()


def model_summary(
    summary_df: pd.DataFrame, order: list[str], export_path: str | None = None
):
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
    heatmap_path = "past_r2_scores_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.show()

    if export_path:
        plt.savefig(export_path)
        plt.show()
    else:
        plt.show()


def plot_residual_diagnostics(
    residuals: pd.Series,
    fitted_values: pd.Series,
    title: str,
    export_path: str | None = None,
):
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
        plt.savefig(export_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


######################## MODELS ############################


def simple_backward_regression(
    X: pd.Series, y: pd.Series, threshold: float = 0.20
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
