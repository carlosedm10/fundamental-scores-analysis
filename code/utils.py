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

from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

import matplotlib.gridspec as gridspec
import shap

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import mimetypes


def send_email_notification(
    subject: str,
    body: str,
    image_path: str | list[str] | None = None,
):
    # Email configuration
    sender_email = os.getenv("SENDER_EMAIL")
    print(sender_email)
    sender_password = os.getenv("SENDER_PASSWORD")
    print(sender_password)
    receiver_email = os.getenv("RECEIVER_EMAIL")
    print(receiver_email)

    if not all([sender_email, sender_password, receiver_email]):
        print("Email configuration incomplete. Skipping email notification.")
        return

    # Create message
    msg = MIMEMultipart()
    msg["From"] = sender_email  # type: ignore
    msg["To"] = receiver_email  # type: ignore
    msg["Subject"] = subject

    # Add body
    msg.attach(MIMEText(body, "plain"))

    # Attach images if provided and exist
    if image_path:
        # Handle both single path and list of paths
        image_paths = (
            [image_path] if isinstance(image_path, str) else image_path
        )

        for path in image_paths:
            if not os.path.exists(path):
                print(f"Skipping attachment: {path} does not exist.")
                continue

            ctype, encoding = mimetypes.guess_type(path)
            # if ctype is None or not ctype.startswith("image/"):
            #     print(f"Skipping attachment: {path} is not a valid image.")
            #     continue

            maintype, subtype = ctype.split("/", 1)  # type: ignore
            with open(path, "rb") as f:
                img = MIMEImage(f.read(), _subtype=subtype)
                img.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=os.path.basename(path),
                )
                msg.attach(img)

    # Send email
    try:
        with smtplib.SMTP("smtp.zoho.eu", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)  # type: ignore
            server.send_message(msg)
            print("Email notification sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


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

        plt.title(
            f"Distribution of {column}",
            fontsize=22,  # Larger title font size
        )
        plt.xlabel(column, fontsize=18)  # Larger x-axis label
        plt.ylabel("Frequency", fontsize=18)  # Larger y-axis label
        plt.tick_params(
            axis="both", which="major", labelsize=14
        )  # Larger tick labels
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
        plt.title(f"Box Plot - {column}", fontsize=22)
        plt.ylabel(column, fontsize=18)
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
        plt.title(f"Distribution of {column}", fontsize=22)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Count", fontsize=18)
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
    plt.title("Correlation Matrix of Numeric Variables", fontsize=22)
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

    # Hacer la leyenda mucho más grande
    if g._legend is not None:
        for text in g._legend.texts:
            text.set_fontsize(24)
        g._legend.set_title(
            g._legend.get_title().get_text(), prop={"size": 26}
        )
    else:
        # Si no hay leyenda, intentar crearla
        g.add_legend(title=hue)
        if g._legend is not None:
            for text in g._legend.texts:
                text.set_fontsize(24)
            g._legend.set_title(
                g._legend.get_title().get_text(), prop={"size": 26}
            )

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
        try:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
        except Exception as e:
            print(f"Error creating directory: {e}")
        plt.savefig(export_path, dpi=300, bbox_inches="tight")
        plt.show()
    else:
        plt.show()


def model_performance_summary_2(
    summary_df: pd.DataFrame,
    order: list[str],
    export_path: str | None = None,
    classifier: bool = False,
):
    """
    Extended performance summary for regression models.
    Displays:
    - Train, Unknown Val, Known Val heatmaps per metric group
    - Metrics: Predictability (or R²), Accuracy (or MAE), Log Loss (or RMSE)
    - Barplot of number of variables
    """
    fig = plt.figure(figsize=(28, 20))
    gs = fig.add_gridspec(4, 3)

    # Row 0: Predictability (or R²)
    ax_r2_train = fig.add_subplot(gs[0, 0])
    ax_r2_val = fig.add_subplot(gs[0, 1])
    ax_r2_val_t = fig.add_subplot(gs[0, 2])

    # Row 1: Accuracy (or MAE)
    ax_acc_train = fig.add_subplot(gs[1, 0])
    ax_acc_val = fig.add_subplot(gs[1, 1])
    ax_acc_val_t = fig.add_subplot(gs[1, 2])

    # Row 2: Log Loss (or RMSE)
    ax_err_train = fig.add_subplot(gs[2, 0])
    ax_err_val = fig.add_subplot(gs[2, 1])
    ax_err_val_t = fig.add_subplot(gs[2, 2])

    # --------------------------- Predictability / R² ---------------------------
    # Train
    heatmap_train_r2 = summary_df.pivot(
        index="score_type",
        columns="profit",
        values="train_predictability" if classifier else "train_r2_score",
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_train_r2, annot=True, cmap="crest_r", fmt=".2f", ax=ax_r2_train
    )
    ax_r2_train.set_title("Train Predictability" if classifier else "Train R²")
    ax_r2_train.set_xlabel("Profit Horizon")
    ax_r2_train.set_ylabel("Score Type")

    # Unknown Val
    heatmap_val_r2 = summary_df.pivot(
        index="score_type",
        columns="profit",
        values="val_predictability" if classifier else "val_r2_score",
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_val_r2, annot=True, cmap="viridis", fmt=".2f", ax=ax_r2_val
    )
    ax_r2_val.set_title(
        "Unknown Val Predictability" if classifier else "Unknown Val R²"
    )
    ax_r2_val.set_xlabel("Profit Horizon")
    ax_r2_val.set_ylabel("Score Type")

    # Known Val (time-based)
    heatmap_val_t_r2 = summary_df.pivot(
        index="score_type",
        columns="profit",
        values="val_t_predictability" if classifier else "val_t_r2_score",
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_val_t_r2, annot=True, cmap="mako", fmt=".2f", ax=ax_r2_val_t
    )
    ax_r2_val_t.set_title(
        "Known Val Predictability" if classifier else "Known Val R²"
    )
    ax_r2_val_t.set_xlabel("Profit Horizon")
    ax_r2_val_t.set_ylabel("Score Type")

    # --------------------------- Accuracy / MAE ---------------------------

    # Train
    if classifier:
        heatmap_train_mae = summary_df.pivot(
            index="score_type",
            columns="profit",
            values="train_accuracy",
        ).reindex(columns=order)
        heatmap_train_mae = 1 - heatmap_train_mae
    else:
        heatmap_train_mae = summary_df.pivot(
            index="score_type",
            columns="profit",
            values="train_mae",
        ).reindex(columns=order)
    sns.heatmap(
        heatmap_train_mae,
        annot=True,
        cmap="magma_r",
        fmt=".2f",
        ax=ax_acc_train,
    )
    ax_acc_train.set_title("Train Error rate" if classifier else "Train MAE")
    ax_acc_train.set_xlabel("Profit Horizon")
    ax_acc_train.set_ylabel("Score Type")

    # Unknown Val
    if classifier:
        heatmap_val_mae = summary_df.pivot(
            index="score_type",
            columns="profit",
            values="val_accuracy",
        ).reindex(columns=order)
        heatmap_val_mae = 1 - heatmap_val_mae
    else:
        heatmap_val_mae = summary_df.pivot(
            index="score_type",
            columns="profit",
            values="val_mae",
        ).reindex(columns=order)
    sns.heatmap(
        heatmap_val_mae, annot=True, cmap="magma_r", fmt=".2f", ax=ax_acc_val
    )
    ax_acc_val.set_title(
        "Unknown Val Error rate" if classifier else "Unknown Val MAE"
    )
    ax_acc_val.set_xlabel("Profit Horizon")
    ax_acc_val.set_ylabel("Score Type")

    # Known Val (time-based)
    if classifier:
        heatmap_val_t_mae = summary_df.pivot(
            index="score_type",
            columns="profit",
            values="val_t_accuracy",
        ).reindex(columns=order)
        heatmap_val_t_mae = 1 - heatmap_val_t_mae
    else:
        heatmap_val_t_mae = summary_df.pivot(
            index="score_type",
            columns="profit",
            values="val_t_mae",
        ).reindex(columns=order)
    sns.heatmap(
        heatmap_val_t_mae,
        annot=True,
        cmap="magma_r",
        fmt=".2f",
        ax=ax_acc_val_t,
    )
    ax_acc_val_t.set_title(
        "Known Val Error rate" if classifier else "Known Val MAE"
    )
    ax_acc_val_t.set_xlabel("Profit Horizon")
    ax_acc_val_t.set_ylabel("Score Type")

    # --------------------------- Log Loss / RMSE ---------------------------

    # Train
    heatmap_train_rmse = summary_df.pivot(
        index="score_type",
        columns="profit",
        values="train_rmse" if not classifier else "train_strict_errors",
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_train_rmse,
        annot=True,
        cmap="rocket_r",
        fmt=".2f",
        ax=ax_err_train,
    )
    ax_err_train.set_title("Train Log Loss" if classifier else "Train RMSE")
    ax_err_train.set_xlabel("Profit Horizon")
    ax_err_train.set_ylabel("Score Type")

    # Unknown Val
    heatmap_val_rmse = summary_df.pivot(
        index="score_type",
        columns="profit",
        values="val_rmse" if not classifier else "val_strict_errors",
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_val_rmse, annot=True, cmap="rocket_r", fmt=".2f", ax=ax_err_val
    )
    ax_err_val.set_title(
        "Unknown Val Log Loss" if classifier else "Unknown Val RMSE"
    )
    ax_err_val.set_xlabel("Profit Horizon")
    ax_err_val.set_ylabel("Score Type")

    # Known Val (time-based)
    heatmap_val_t_rmse = summary_df.pivot(
        index="score_type",
        columns="profit",
        values="val_t_rmse" if not classifier else "val_t_strict_errors",
    ).reindex(columns=order)
    sns.heatmap(
        heatmap_val_t_rmse,
        annot=True,
        cmap="rocket_r",
        fmt=".2f",
        ax=ax_err_val_t,
    )
    ax_err_val_t.set_title(
        "Known Val Log Loss" if classifier else "Known Val RMSE"
    )
    ax_err_val_t.set_xlabel("Profit Horizon")
    ax_err_val_t.set_ylabel("Score Type")

    # Layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.suptitle("Train | Unknown Val | Known Val Performance", fontsize=22)

    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        plt.savefig(export_path, dpi=300, bbox_inches="tight")
        plt.show()
    else:
        plt.show()


def models_scores_summary(
    summary_df,
    profits: None | list[str] = None,
    export_path=None,
    windowed: bool = False,
):
    """
    For each score_type, create a transposed table showing coefficient, sign, and t-value
    of the main score variable (e.g., 'quality') across profit horizons.

    When windowed=True, creates separate tables for each score including averages and standard deviations.

    Returns a dict of transposed DataFrames keyed by score_type.
    """
    results = {}
    score_types = summary_df["score_type"].unique()

    for score in score_types:
        subset = summary_df[summary_df["score_type"] == score]
        if profits is None:
            profits = sorted(subset["profit"].unique())
        else:
            profits = profits

        if windowed:
            # Create windowed summary with averages and standard deviations
            rows = []
            for profit in profits:
                model_row = subset[subset["profit"] == profit].iloc[0]
                coefs = model_row["coefficients"]
                tvals = model_row["tvalues"]

                # Get main score coefficient and t-value
                coef = coefs.get(score, np.nan)
                tval = tvals.get(score, np.nan)
                sign = "+" if coef > 0 else "–" if coef < 0 else "0"

                row_data = {
                    "profit_horizon": profit,
                    "coefficient": coef,
                    "sign": sign,
                    "t_value": tval,
                }

                # Add windowed averages coefficients (3m, 6m, 12m, 24m)
                for window in [3, 6, 12, 24]:
                    avg_col = f"{score}_avg_{window}m"
                    if avg_col in coefs:
                        row_data[f"avg_{window}m_coef"] = coefs[avg_col]
                        row_data[f"avg_{window}m_tval"] = tvals.get(
                            avg_col, np.nan
                        )

                rows.append(row_data)

            # Create DataFrame and order by profits
            df = pd.DataFrame(rows)
            # Ensure profits are in the correct order
            df["profit_horizon"] = pd.Categorical(
                df["profit_horizon"], categories=profits, ordered=True
            )
            df = df.sort_values("profit_horizon").set_index("profit_horizon")
            df_t = df.transpose()

        else:
            # Original logic for non-windowed case
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
            df_t.to_csv(f"{export_path}.csv")

    return results


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
        summary: Dictionary output from regressor_nn
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
    print("X_cols", X_cols)

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
        plt.close()
    else:
        plt.show()

    return explainer, shap_values, X_val_sampled_raw


def nn_shap_table_analysis(
    model_summaries: list[dict],
    profits: list[str] = [
        "profit_1m",
        "profit_3m",
        "profit_6m",
        "profit_1y",
        "profit_2y",
        "profit_5y",
    ],
    sample_size: int = 100,
    export_path: str | None = None,
):
    """
    Create a comprehensive table analysis showing SHAP dependency plots
    for specific variables across profit horizons.

    Args:
        model_summaries: List of model summary dictionaries from classifier_nn
        profits: List of profit horizons to analyze
        sample_size: Number of samples for SHAP analysis
        export_path: Path to save the analysis plots

    Returns:
        Dictionary containing analysis results and plot paths
    """

    # Define the variables we want to analyze
    variables_to_analyze = [
        {
            "name": "Dividend 24m average",
            "feature": "dividend_avg_24m",
        },
        {
            "name": "Growth 24m average",
            "feature": "growth_avg_24m",
        },
        {
            "name": "Quality 24m average",
            "feature": "quality_avg_24m",
        },
        {"name": "Value", "feature": "value"},
        {
            "name": "Value 24m average",
            "feature": "value_avg_24m",
        },
    ]

    # Organize summaries by profit (we'll use all summaries to find our variables)
    summary_by_profit = {}
    for summary in model_summaries:
        profit = summary["profit"]
        if profit not in summary_by_profit:
            summary_by_profit[profit] = []
        summary_by_profit[profit].append(summary)

    # Create the comprehensive plot with more space to prevent overlap
    n_variables = len(variables_to_analyze)
    n_profits = len(profits)

    fig = plt.figure(figsize=(5 * n_profits, 3.5 * n_variables + 2))
    gs = gridspec.GridSpec(
        n_variables + 1,
        n_profits + 1,  # Extra column for variable labels
        height_ratios=[0.2] + [1] * n_variables,
        width_ratios=[1.2] + [1] * n_profits,  # More space for labels
        hspace=0.4,  # Increased spacing
        wspace=0.5,  # Increased horizontal spacing to prevent overlap
    )

    # Add title row with profit headers
    profit_labels = {
        "profit_1m": "1M Profit",
        "profit_3m": "3M Profit",
        "profit_6m": "6M Profit",
        "profit_1y": "1Y Profit",
        "profit_2y": "2Y Profit",
        "profit_5y": "5Y Profit",
    }

    for j, profit in enumerate(profits):
        ax = fig.add_subplot(gs[0, j + 1])  # +1 to account for label column
        ax.text(
            0.5,
            0.5,
            profit_labels.get(profit, profit),
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Create SHAP plots for each variable-profit combination
    analysis_results = {}

    for i, var_info in enumerate(variables_to_analyze):
        var_name = var_info["name"]
        feature = var_info["feature"]

        analysis_results[var_name] = {}

        # Add variable label in the left column
        ax_label = fig.add_subplot(gs[i + 1, 0])
        ax_label.text(
            0.5,
            0.5,
            var_name,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            wrap=True,
        )
        ax_label.set_xlim(0, 1)
        ax_label.set_ylim(0, 1)
        ax_label.axis("off")

        for j, profit in enumerate(profits):
            ax = fig.add_subplot(
                gs[i + 1, j + 1]
            )  # +1 to account for label column

            # Find a summary that contains our feature for this profit
            summary = None
            if profit in summary_by_profit:
                for candidate_summary in summary_by_profit[profit]:
                    if feature in candidate_summary["X_cols"]:
                        summary = candidate_summary
                        break

            if summary is not None:
                try:
                    # Generate SHAP values
                    model = summary["model"]
                    scaler = summary["scaler"]
                    X_cols = summary["X_cols"]
                    X_val = summary["X_val"]
                    X_val_scaled = scaler.transform(X_val)

                    # Subsample for SHAP
                    if sample_size < X_val_scaled.shape[0]:
                        idx = np.random.choice(
                            X_val_scaled.shape[0], sample_size, replace=False
                        )
                        X_val_sampled = X_val_scaled[idx]
                        X_val_sampled_raw = X_val.iloc[idx]
                    else:
                        X_val_sampled = X_val_scaled
                        X_val_sampled_raw = X_val

                    # Create SHAP explainer
                    def predict_fn(x):
                        return (
                            model.predict_proba(x)[:, 1]
                            if hasattr(model, "predict_proba")
                            else model.predict(x)
                        )

                    explainer = shap.KernelExplainer(
                        predict_fn,
                        X_val_sampled,
                    )
                    shap_values = explainer.shap_values(X_val_sampled)

                    # Focus on the specific feature for dependency plot
                    feature_idx = X_cols.index(feature)

                    # Create a dependency plot: variable vs its own SHAP values
                    plt.sca(ax)
                    shap.dependence_plot(
                        feature_idx,  # Feature to plot on X-axis
                        shap_values,  # SHAP values for Y-axis
                        X_val_sampled_raw,
                        feature_names=X_cols,
                        interaction_index=None,  # Let SHAP choose the best interaction feature for coloring
                        show=False,
                        ax=ax,
                    )

                    # Remove the title since we have variable names on the left
                    ax.set_title("", fontsize=8)

                    analysis_results[var_name][profit] = {
                        "shap_values": shap_values,
                        "feature_idx": feature_idx,
                    }

                except Exception as e:
                    ax.text(
                        0.5,
                        0.5,
                        f"Error:\n{str(e)[:30]}...",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="red",
                    )
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    print(f"Error processing {var_name}-{profit}: {e}")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="gray",
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")

    plt.suptitle(
        "Neural Network SHAP Dependency Analysis by Variable and Profit Horizon",
        fontsize=16,
        y=0.98,
    )

    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        plt.savefig(export_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"SHAP table analysis saved to: {export_path}")
    else:
        plt.show()

    return analysis_results


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
        plt.show()
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


def split_by_ticker_and_time(
    X: pd.DataFrame,
    y: pd.Series,
    validation_size: float = 0.2,
) -> tuple:
    """
    Splits (X, y) into:
      - train set on tickers in train_tickers minus their last n_holdout_per_ticker rows
      - ticker-based validation: all rows for held-out tickers
      - time-based validation: last n_holdout_per_ticker rows for each train ticker

    Args:
      X: DataFrame that MUST include columns 'ticker' and 'date' plus your features
      y: Series of the target, indexed to X
      validation_size: fraction of unique tickers to hold out completely

    Returns:
      (X_train, y_train,
       X_val_tickers, y_val_tickers,
       X_val_time,    y_val_time)
    """
    # 1. Recombine X & y into one df
    df = X.copy()
    df["_target_"] = y
    feature_cols = [c for c in X.columns if c not in ("ticker", "date")]

    # 2. split tickers
    all_tickers = df["ticker"].unique()
    train_tickers, val_tickers = train_test_split(
        all_tickers,
        test_size=validation_size,
        random_state=42,
    )

    # 3. ticker-based hold-out
    df_val_tickers = df[df["ticker"].isin(val_tickers)].copy()

    # 4. remaining for training/time-split
    df_train_all = df[df["ticker"].isin(train_tickers)].copy()
    df_train_all.sort_values(["ticker", "date"], inplace=True)

    # 5. tag last n_holdout_per_ticker rows per ticker
    df_train_all["rank_desc"] = df_train_all.groupby("ticker")["date"].rank(
        method="first", ascending=False
    )

    # 6. time-based hold-out vs train
    df_val_time = df_train_all[df_train_all["rank_desc"] <= 4].copy()
    df_train = df_train_all[df_train_all["rank_desc"] > 4].copy()

    # 7. drop helper column
    df_train.drop(columns=["rank_desc", "ticker"], inplace=True)
    df_val_tickers.drop(columns=["ticker"], inplace=True)
    df_val_time.drop(columns=["rank_desc", "ticker"], inplace=True)

    # 8. split X/y for each
    def split_xy(df_):
        return df_[feature_cols].astype(float), df_["_target_"].astype(float)

    X_train, y_train = split_xy(df_train)
    X_val_tickers, y_val_tickers = split_xy(df_val_tickers)
    X_val_time, y_val_time = split_xy(df_val_time)

    return (
        X_train,
        y_train,
        X_val_tickers,
        y_val_tickers,
        X_val_time,
        y_val_time,
    )
