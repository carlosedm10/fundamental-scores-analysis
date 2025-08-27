# # GAM Models


# # Data Prep


# ## Imports


import sys
import os

from pathlib import Path

# Add the parent directory (code/) to sys.path
sys.path.append(str(Path().resolve().parent))
from utils import (
    load_data,
    separate_df_by_scores,
    model_performance_summary,
    plot_residual_diagnostics,
    gaussian_yj_transform,
)
from models import simple_gam
import time
import pandas as pd

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# Loading the data
print("Loading data...")

save = True
outlier_method = "IF HARD Balanced"
file_name = f"Big Data future - {outlier_method}"

data_path = f"./code/data/cleaned/outliers/{file_name}.csv"

# Read the CSV file
df = load_data(data_path)
print("Data loaded", df.shape)

# Variables
print("Variables...")

# Explanatory variables:
scores = ["quality", "growth", "value", "dividend"]

# Encoing dummies
df["sector_code"] = df["sector"].astype("category").cat.codes
df["region_code"] = df["region"].astype("category").cat.codes
dummies = ["sector_code", "region_code"]

# Create list of tuples for sector and region encoding
sector_encoding = list(
    zip(
        df["sector"].astype("category").cat.categories,
        range(len(df["sector"].astype("category").cat.categories)),
    )
)
region_encoding = list(
    zip(
        df["region"].astype("category").cat.categories,
        range(len(df["region"].astype("category").cat.categories)),
    )
)

# Dependent variables:
profits = [
    "profit_1m",
    "profit_3m",
    "profit_6m",
    "profit_1y",
    "profit_2y",
    "profit_5y",
]
dataframes = separate_df_by_scores(df)


# Model Fit
print("\n" + "-" * 40)
print("Model Fitting")
print("-" * 40)
start_time = time.perf_counter()
gam_model_summaries = []
model_count = 0

for score in scores:
    # work on a copy so you don't fragment the original
    score_df = dataframes[score].copy()

    # Rolling avg/std for the raw score
    avg_cols = []
    std_cols = []
    rolling = {}
    for w in [3, 6, 12, 24]:
        avg_name = f"{score}_avg_{w}m"
        std_name = f"{score}_std_{w}m"

        rolling[avg_name] = score_df.groupby("ticker")[score].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        rolling[std_name] = score_df.groupby("ticker")[score].transform(
            lambda x: x.rolling(window=w, min_periods=1).std()
        )

        avg_cols.append(avg_name)
        std_cols.append(std_name)

    # concatenate the new main + rolling columns in one shot
    score_df = pd.concat(
        [
            score_df,
            pd.DataFrame(rolling, index=score_df.index),
        ],
        axis=1,
    )
    # Interacton pairs for the GAM model
    pairs = [(score, col) for col in dummies + ["volume", "market_cap"]]
    pairs += [(avg, dum) for avg in avg_cols for dum in dummies]
    pairs += [(std, dum) for std in std_cols for dum in dummies]

    score_df.dropna(inplace=True)

    for profit in profits:
        X_cols = (
            [score] + dummies + ["volume", "market_cap"] + avg_cols + std_cols
        )

        X = score_df[X_cols].astype(float)
        y = gaussian_yj_transform(score_df[profit])
        summary = simple_gam(X, y, False, interaction_pairs=pairs)
        summary["score_type"] = score
        summary["profit"] = profit

        gam_model_summaries.append(summary)
        model_count += 1


# Create and display summary DataFrame
summary_df = pd.DataFrame(gam_model_summaries)
summary_df = summary_df.sort_values(["score_type"]).reset_index(drop=True)

end_time = time.perf_counter()
total_time = end_time - start_time
avg_time = total_time / model_count
print("Model fitting complete")
print(f"\nTotal time: {total_time:.2f} seconds")
print(f"Average time per model: {avg_time:.2f} seconds")


# Analysis
print("\n" + "-" * 40)
print("Analysis")
print("-" * 40)

# Performance
print("Performance...")

if save:
    export_path = f"code/general_regression/{file_name} performance.png"
    model_performance_summary(summary_df, profits, export_path)
else:
    model_performance_summary(summary_df, profits)


# Residues
print("Residues...")

for summary in gam_model_summaries:
    export_path = f"code/general_regression/residuals/{file_name}/{summary['score_type']}_{summary['profit']}_residuals.png"

    plot_residual_diagnostics(
        summary["residuals"],
        summary["y_pred"],
        title=f"{summary['score_type']} - {summary['profit']}",
        export_path=export_path,
    )


# 2D Plots
print("2D Plots")

colors = sns.color_palette("tab10", len(profits))
profit_colors = {profit: colors[i] for i, profit in enumerate(profits)}

for score_type in scores:
    # Get all models for this score type
    score_models = [
        s for s in gam_model_summaries if s["score_type"] == score_type
    ]

    if not score_models:
        continue

    # Get the number of terms from the first model (should be same for all)
    first_model = score_models[0]
    gam = first_model["model"]
    X_cols = first_model.get("X_cols", None)
    n_terms = len(gam.terms)

    print(f"\n=== {score_type.title()} Score - Main Effects Only ===")

    # Create a subplot for each feature/term
    for term_idx in range(n_terms):
        # Skip intercept terms
        first_gam = score_models[0]["model"]
        if (
            hasattr(first_gam.terms[term_idx], "isintercept")
            and first_gam.terms[term_idx].isintercept
        ):
            continue

        # Skip interaction terms (tensor terms) - these cause empty plots after feature 5
        term = first_gam.terms[term_idx]
        term_str = str(term)

        # Check if this is an interaction term by looking for 'te(' in the string representation
        if "te(" in term_str:
            continue
        if hasattr(term, "istensor") and getattr(term, "istensor", False):

            continue

        # Skip terms beyond the main effects (typically after feature 5)
        # This is a safeguard for interaction terms that might not be caught above
        if term_idx > 20:  # Adjust this number based on your model structure
            term_type = type(term).__name__
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # Variable to store x_vals for categorical encoding
        sample_x_vals = None

        # Plot splines for all profit timeframes for this feature
        successful_plots = 0
        for summary in score_models:
            gam = summary["model"]
            profit = summary["profit"]

            try:
                XX = gam.generate_X_grid(term=term_idx)
                # Handle cases where XX doesn't have the right number of features
                n_features = gam.statistics_["m_features"]
                if XX.shape[1] != n_features:
                    try:
                        X_means = gam._X.mean(axis=0)
                    except Exception:
                        X_means = np.zeros(n_features)
                    XX_full = np.tile(X_means, (XX.shape[0], 1))
                    XX_full[:, term_idx] = XX[:, 0]
                    XX = XX_full

                pdep, confi = gam.partial_dependence(
                    term=term_idx, X=XX, width=0.95
                )

                # Handle confidence intervals
                if isinstance(confi, np.ndarray) and confi.shape[0] == 2:
                    lower, upper = confi[0], confi[1]
                elif isinstance(confi, np.ndarray) and confi.shape[1] == 2:
                    lower, upper = confi[:, 0], confi[:, 1]
                else:
                    confi = np.array(confi)
                    lower, upper = confi[0], confi[1]

                # Ensure same length and valid term index
                if term_idx >= XX.shape[1]:
                    continue

                x_vals = XX[:, term_idx]
                min_len = min(len(x_vals), len(pdep), len(lower), len(upper))
                x_vals = x_vals[:min_len]
                pdep = pdep[:min_len]
                lower = lower[:min_len]
                upper = upper[:min_len]

                # Check if we have valid data
                if len(x_vals) == 0 or len(pdep) == 0:
                    print(
                        f"Empty data for {score_type} - {profit} - term {term_idx}"
                    )
                    continue

                # Store sample x_vals for categorical encoding (from first successful iteration)
                if sample_x_vals is None:
                    sample_x_vals = x_vals.copy()

                # Plot with different colors for each profit timeframe
                color = profit_colors[profit]
                profit_label = (
                    profit.replace("profit_", "")
                    .replace("m", " month")
                    .replace("y", " year")
                )
                ax.plot(
                    x_vals, pdep, color=color, label=profit_label, linewidth=2
                )
                ax.fill_between(x_vals, lower, upper, color=color, alpha=0.2)
                successful_plots += 1

            except Exception as e:
                print(
                    f"Skipping {score_type} - {profit} - term {term_idx} due to error: {e}"
                )
                continue

        # Only show plot if we have successful data
        if successful_plots > 0:
            # Set labels and title
            feature_name = f"Feature {term_idx}"
            if X_cols is not None and term_idx < len(X_cols):
                feature_name = X_cols[term_idx]

            # Handle categorical encoding for sector and region features
            if sample_x_vals is not None:
                if "sector_code" in feature_name:
                    unique_codes = np.unique(sample_x_vals)
                    rounded_codes = np.unique(
                        np.round(unique_codes).astype(int)
                    )
                    codes_to_show = [
                        enc[1]
                        for enc in sector_encoding
                        if enc[1] in rounded_codes
                    ]
                    labels_to_show = [
                        enc[0]
                        for enc in sector_encoding
                        if enc[1] in codes_to_show
                    ]
                    if len(codes_to_show) > 0:
                        ax.set_xticks(codes_to_show)
                        ax.set_xticklabels(
                            labels_to_show, rotation=45, ha="right"
                        )
                elif "region_code" in feature_name:
                    unique_codes = np.unique(sample_x_vals)
                    rounded_codes = np.unique(
                        np.round(unique_codes).astype(int)
                    )
                    codes_to_show = [
                        enc[1]
                        for enc in region_encoding
                        if enc[1] in rounded_codes
                    ]
                    labels_to_show = [
                        enc[0]
                        for enc in region_encoding
                        if enc[1] in codes_to_show
                    ]
                    if len(codes_to_show) > 0:
                        ax.set_xticks(codes_to_show)
                        ax.set_xticklabels(
                            labels_to_show, rotation=45, ha="right"
                        )

            ax.set_title(
                f"{score_type.title()} - {feature_name}",
                fontsize=14,
            )
            ax.set_xlabel(feature_name, fontsize=12)
            ax.set_ylabel("Partial Dependence", fontsize=12)
            ax.legend(
                title="Profit Timeframe",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            if save:
                export_path = f"code/general_regression/splines/2D/{file_name}/{score_type}_{feature_name}.png"
                os.makedirs(os.path.dirname(export_path), exist_ok=True)
                plt.savefig(export_path)
                plt.close()
            else:
                plt.show()
        else:
            plt.close(fig)
            print(f"No valid data for term {term_idx}, plot skipped")


# 3D Interactions Plots
print("3D Interactions Plots")

# Enable 3D plotting
plt.rcParams["figure.max_open_warning"] = 50


# Function to check if a term is a tensor interaction
def _is_tensor_term(term):
    return hasattr(term, "istensor") and term.istensor


# Function to get feature indices from tensor term
def _get_tensor_feature_indices(term):
    if hasattr(term, "feature"):
        return term.feature
    elif hasattr(term, "features"):
        return term.features
    else:
        # Fallback: try to extract from term representation
        term_str = str(term)
        if "te(" in term_str:
            # Extract indices from string like "te(0, 1)"
            import re

            matches = re.findall(r"te\((\d+),\s*(\d+)\)", term_str)
            if matches:
                return [int(matches[0][0]), int(matches[0][1])]
    return None


for score_type in scores:
    # Get all models for this score type
    score_models = [
        s for s in gam_model_summaries if s["score_type"] == score_type
    ]

    if not score_models:
        continue

    # Get the first model to analyze terms
    first_model = score_models[0]
    gam = first_model["model"]
    X_cols = first_model.get("X_cols", None)
    n_terms = len(gam.terms)

    # print(f"\n=== {score_type.title()} Score - Tensor Interaction Terms ===")

    # Find and plot tensor interaction terms
    for term_idx in range(n_terms):
        term = gam.terms[term_idx]

        # Check if this is a tensor interaction term
        if _is_tensor_term(term):
            feature_indices = _get_tensor_feature_indices(term)

            if feature_indices is None or len(feature_indices) != 2:
                print(
                    f"Could not extract feature indices for term {term_idx}: {term}"
                )
                continue

            feat_idx1, feat_idx2 = feature_indices

            # Get feature names
            feature1_name = f"Feature {feat_idx1}"
            feature2_name = f"Feature {feat_idx2}"
            if X_cols is not None:
                if feat_idx1 < len(X_cols):
                    feature1_name = X_cols[feat_idx1]
                if feat_idx2 < len(X_cols):
                    feature2_name = X_cols[feat_idx2]

            # print(f"Plotting interaction: {feature1_name} × {feature2_name}")

            # Create subplots for different profit timeframes
            n_profits = len(profits)
            fig = plt.figure(figsize=(20, 4 * ((n_profits + 2) // 3)))

            plot_idx = 1
            for summary in score_models:
                gam = summary["model"]
                profit = summary["profit"]

                try:
                    # Generate 2D grid for the interaction
                    XX = gam.generate_X_grid(term=term_idx, n=20)  # 20x20 grid

                    # Calculate partial dependence
                    pdep, confi = gam.partial_dependence(
                        term=term_idx, X=XX, width=0.95
                    )

                    # Reshape for 3D plotting (assuming 20x20 grid)
                    grid_size = int(np.sqrt(len(pdep)))
                    if grid_size * grid_size != len(pdep):
                        # If not perfect square, use closest square
                        grid_size = int(np.sqrt(len(pdep)))
                        pdep = pdep[: grid_size * grid_size]
                        XX = XX[: grid_size * grid_size]

                    # Extract the two features for the interaction
                    x1_vals = XX[:, feat_idx1].reshape(grid_size, grid_size)
                    x2_vals = XX[:, feat_idx2].reshape(grid_size, grid_size)
                    z_vals = pdep.reshape(grid_size, grid_size)

                    # Create 3D subplot
                    ax = fig.add_subplot(2, 3, plot_idx, projection="3d")

                    # Create surface plot
                    try:
                        surf = ax.plot_surface(
                            x1_vals,
                            x2_vals,
                            z_vals,
                            cmap="viridis",
                            alpha=0.8,
                            linewidth=0,
                            antialiased=True,
                        )

                        # Add contour lines at the bottom
                        ax.contour(
                            x1_vals,
                            x2_vals,
                            z_vals,
                            zdir="z",
                            offset=z_vals.min(),
                            cmap="viridis",
                            alpha=0.5,
                        )
                    except AttributeError:
                        # Fallback to wireframe if surface fails
                        ax.plot_wireframe(
                            x1_vals, x2_vals, z_vals, cmap="viridis"
                        )

                    # Set labels
                    ax.set_xlabel(feature1_name, fontsize=10)
                    ax.set_ylabel(feature2_name, fontsize=10)
                    try:
                        ax.set_zlabel("Partial Dependence", fontsize=10)
                    except AttributeError:
                        pass  # Some matplotlib versions might not have set_zlabel

                    # Set title
                    profit_label = (
                        profit.replace("profit_", "")
                        .replace("m", " month")
                        .replace("y", " year")
                    )
                    ax.set_title(f"{profit_label}", fontsize=12)

                    # Removed all categorical encoding tick label code

                    plot_idx += 1

                except Exception as e:
                    print(
                        f"Error plotting {score_type} - {profit} - interaction {feature1_name}×{feature2_name}: {e}"
                    )
                    continue

            # Add overall title and show
            fig.suptitle(
                f"{score_type.title()} Score: {feature1_name} × {feature2_name} Interaction Effects",
                fontsize=16,
                y=0.98,
            )
            plt.tight_layout()
            if save:
                export_path = f"code/general_regression/splines/3D/{file_name}/{score_type}_{feature1_name}×{feature2_name}.png"
                os.makedirs(os.path.dirname(export_path), exist_ok=True)
                plt.savefig(export_path)
                plt.close()
            else:
                plt.show()
            print(
                f"Completed interaction plot: {feature1_name} × {feature2_name}\n"
            )

from utils import send_email_notification

send_email_notification(
    subject="✅ GAM models BIG",
    body="The GAM models are complete, please check the files.",
)
