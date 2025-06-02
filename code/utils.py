"""
Here we will create different functions to help with the data preparation.
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
    Analyze and print information about missing values in two DataFrames.

    Args:
        df_original (pd.DataFrame): First DataFrame to analyze
        df_cleaned (pd.DataFrame): Second DataFrame to analyze
    """
    for i, df in enumerate([df_original, df_cleaned], 1):
        print(f"\nAnalyzing DataFrame {i}:")

        # Calculate number of rows with any missing value
        print(
            f"Percentage of data maintained: {(df_cleaned.shape[0]/df.shape[0]*100):.2f}%"
        )

        # Display info about which columns had missing values
        print("\nMissing values by column before cleaning:")
        missing_values = df.isnull().sum()
        missing_values = missing_values[
            missing_values > 0
        ]  # Only show columns with missing values
        missing_percentages = (missing_values / len(df) * 100).round(2)

        print("\nMissing values and percentages by column:")
        for col in missing_values.index:
            print(
                f"{col}: {missing_values[col]} values ({missing_percentages[col]}%)"
            )


# Create a function to plot histograms for numeric columns
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
        plt.close()
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


############################ DESCRIPTIVE ANALYSIS ############################
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
) -> None:
    region_colors = {
        "United States and Canada": "blue",
        "Europe": "red",
        "Asia / Pacific": "yellow",
        "Latin America and Caribbean": "green",
        "Africa / Middle East": "purple",
    }
    if len(df) > 10000:
        df = df.sample(10000, random_state=42)

    g = sns.pairplot(
        df,
        hue="region",
        diag_kind="kde",
        hue_order=region_colors.keys(),
        palette=region_colors,
        corner=True,
    )
    if kde:
        g.map_lower(sns.kdeplot, levels=4, color=".2")

    if file_path:
        plt.savefig(file_path)
        plt.show()
    else:
        plt.show()
