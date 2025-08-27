# Data Prep


#  Imports


import os
import sys
from pathlib import Path
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory (code/) to sys.path
sys.path.append(str(Path().resolve().parent))
from utils import (
    load_data,
    separate_df_by_scores,
)
from models import regressor_nn

save = True

outlier_method = "IF HARD Balanced"
file_name = f"Big Data future - {outlier_method}"

data_path = f"./code/data/cleaned/outliers/{file_name}.csv"

# Read the CSV file
df = load_data(data_path)

df = pd.get_dummies(df, columns=["sector", "region"])

scores = ["quality", "growth", "value", "dividend"]

dummies = [col for col in df.columns if col.startswith(("sector_", "region_"))]

# Drop the reference dummy variables (i.e., base categories)
reference_dummies = ["sector_Information Technology", "region_Europe"]

df.drop(columns=reference_dummies, inplace=True)

# Update dummies list after dropping
dummies = [col for col in dummies if col not in reference_dummies]

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

# Architecture Search
hidden_layer_configs = [
    # 1-layer
    (16,),
    (32,),
    (64,),
    (128,),
    (256,),
    (512,),
    # 2-layers
    (64, 32),
    (128, 64),
    (256, 128),
    (512, 256),
    (128, 32),
    (256, 64),
    # 3-layers
    (128, 64, 32),
    (256, 128, 64),
    (512, 256, 128),
    (64, 32, 16),
    (256, 64, 16),
    (128, 64, 8),
    # 4 layers
    (512, 256, 128, 64),
    (256, 128, 64, 32),
    (128, 64, 32, 16),
    (64, 32, 16, 8),
    (1024, 512, 256, 128),
    # Bottleneck
    (256, 64, 16),
    (128, 32, 8),
    (512, 128, 32),
    (1024, 256, 64),
    # Uniform width
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (128, 128, 128),
    (256, 256, 256),
]

# To collect all results
all_model_summaries = []
model_count = 0
start_time = time.perf_counter()

print("🔍 Running architecture search across all models...\n")

# Outer loop = each NN config
for hl_config in tqdm(hidden_layer_configs, desc="Hidden Layer Configs"):
    reg_model_summaries = []
    for score in scores:
        score_df = dataframes[score].copy()

        # Rolling features
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

        score_df = pd.concat(
            [score_df, pd.DataFrame(rolling, index=score_df.index)], axis=1
        )
        score_df.dropna(inplace=True)

        # Inner loop = each profit target
        for profit in profits:
            X_feats = (
                [score]
                + dummies
                + ["volume", "market_cap"]
                + avg_cols
                + std_cols
            )

            X = pd.concat(
                [
                    score_df[["ticker", "date"]],
                    score_df[X_feats].astype(float),
                ],
                axis=1,
            )
            y = score_df[profit].astype(float)

            try:
                summary = regressor_nn(X, y, hidden_layer_sizes=hl_config)
                summary.update(
                    score_type=score, profit=profit, hidden_layer=hl_config
                )
                reg_model_summaries.append(summary)
                model_count += 1
            except Exception as e:
                print(
                    f"❌ Error with ({score}, {profit}), HL: {hl_config}: {e}"
                )
                continue

    # Save all summaries for this architecture
    all_model_summaries.extend(reg_model_summaries)

# Convert to DataFrame
summary_df = pd.DataFrame(all_model_summaries)

end_time = time.perf_counter()
total_time = end_time - start_time

# Display results
print(
    f"\n✅ Done {model_count} models in {total_time:.2f}s "
    f"({total_time/model_count:.4f}s each)"
)

# --- Analyze best architectures by score and time frame ---
print("\n🏆 Best architectures by score and time frame:")

# Create analysis for each score
for score in scores:
    print(f"\n📊 Score: {score.upper()}")
    score_data = summary_df[summary_df["score_type"] == score].copy()

    if len(score_data) == 0:
        print(f"  No data available for {score}")
        continue

    # Best architecture for each time frame for this score
    for profit in profits:
        profit_data = score_data[score_data["profit"] == profit].copy()
        if len(profit_data) == 0:
            continue

        # Sort by validation time-based R2 score (most relevant for future prediction)
        best_arch = profit_data.loc[profit_data["val_t_r2_score"].idxmax()]
        print(
            f"  {profit}: {best_arch['hidden_layer']} (R²={best_arch['val_t_r2_score']:.4f})"
        )

# --- Create plots: one image per score ---
print("\n📈 Creating plots for each score...")

# Color palette for time frames
profit_colors = {
    "profit_1m": "#FF6B6B",  # Red
    "profit_3m": "#4ECDC4",  # Teal
    "profit_6m": "#45B7D1",  # Blue
    "profit_1y": "#96CEB4",  # Green
    "profit_2y": "#FFEAA7",  # Yellow
    "profit_5y": "#DDA0DD",  # Plum
}

for score in scores:
    print(f"Creating plot for {score}...")
    score_data = summary_df[summary_df["score_type"] == score].copy()

    if len(score_data) == 0:
        print(f"  No data available for {score}")
        continue

    # Create figure with 2 subplots
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(20, 8))

    # Get unique architectures for this score
    unique_archs = score_data["hidden_layer"].unique()

    # --- LEFT PLOT: LOSS (RMSE) ---
    for profit in profits:
        if profit not in score_data["profit"].values:
            continue

        profit_data = score_data[score_data["profit"] == profit].copy()

        # Extract data for plotting
        architectures = []
        train_rmse = []
        val_rmse = []
        val_t_rmse = []

        for arch in unique_archs:
            arch_data = profit_data[profit_data["hidden_layer"] == arch]
            if len(arch_data) > 0:
                row = arch_data.iloc[0]  # Take first occurrence
                architectures.append(str(arch))
                train_rmse.append(row["train_rmse"])
                val_rmse.append(row["val_rmse"])
                val_t_rmse.append(row["val_t_rmse"])

        if len(architectures) > 0:
            x_pos = np.arange(len(architectures))
            width = 0.12

            # Offset for each profit timeframe
            profit_idx = profits.index(profit)
            offset = (profit_idx - len(profits) / 2) * width

            # Plot training RMSE
            ax_loss.bar(
                x_pos + offset,
                train_rmse,
                width,
                label=f"{profit} (train)",
                color=profit_colors[profit],
                alpha=0.7,
            )

            # Plot validation RMSE with different pattern
            ax_loss.bar(
                x_pos + offset,
                val_rmse,
                width,
                label=f"{profit} (val)",
                color=profit_colors[profit],
                alpha=0.5,
                hatch="//",
            )

            # Plot validation time RMSE with dots
            ax_loss.scatter(
                x_pos + offset,
                val_t_rmse,
                label=f"{profit} (val_t)",
                color=profit_colors[profit],
                s=50,
                marker="o",
            )

    ax_loss.set_xlabel("Hidden Layer Architecture")
    ax_loss.set_ylabel("RMSE (Loss)")
    ax_loss.set_title(f"RMSE by Architecture - {score.upper()}")
    ax_loss.set_xticks(range(len(unique_archs)))
    ax_loss.set_xticklabels(
        [str(arch) for arch in unique_archs], rotation=45, ha="right"
    )
    ax_loss.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_loss.grid(True, alpha=0.3)

    # --- RIGHT PLOT: ACCURACY (R²) ---
    for profit in profits:
        if profit not in score_data["profit"].values:
            continue

        profit_data = score_data[score_data["profit"] == profit].copy()

        # Extract data for plotting
        architectures = []
        train_r2 = []
        val_r2 = []
        val_t_r2 = []

        for arch in unique_archs:
            arch_data = profit_data[profit_data["hidden_layer"] == arch]
            if len(arch_data) > 0:
                row = arch_data.iloc[0]  # Take first occurrence
                architectures.append(str(arch))
                train_r2.append(row["train_r2_score"])
                val_r2.append(row["val_r2_score"])
                val_t_r2.append(row["val_t_r2_score"])

        if len(architectures) > 0:
            x_pos = np.arange(len(architectures))
            width = 0.12

            # Offset for each profit timeframe
            profit_idx = profits.index(profit)
            offset = (profit_idx - len(profits) / 2) * width

            # Plot training R²
            ax_acc.bar(
                x_pos + offset,
                train_r2,
                width,
                label=f"{profit} (train)",
                color=profit_colors[profit],
                alpha=0.7,
            )

            # Plot validation R² with different pattern
            ax_acc.bar(
                x_pos + offset,
                val_r2,
                width,
                label=f"{profit} (val)",
                color=profit_colors[profit],
                alpha=0.5,
                hatch="//",
            )

            # Plot validation time R² with dots
            ax_acc.scatter(
                x_pos + offset,
                val_t_r2,
                label=f"{profit} (val_t)",
                color=profit_colors[profit],
                s=50,
                marker="o",
            )

    ax_acc.set_xlabel("Hidden Layer Architecture")
    ax_acc.set_ylabel("R² Score (Accuracy)")
    ax_acc.set_title(f"R² Score by Architecture - {score.upper()}")
    ax_acc.set_xticks(range(len(unique_archs)))
    ax_acc.set_xticklabels(
        [str(arch) for arch in unique_archs], rotation=45, ha="right"
    )
    ax_acc.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    export_path = f"code/neuronal_network/regressor_nn/reg_architecture_analysis_{score}.png"
    if save:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        plt.savefig(export_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {export_path}")

    plt.close()

# --- Create training curve plots (loss over epochs) ---
print("\n📈 Creating training curve plots...")

for score in scores:
    print(f"Creating training curves for {score}...")
    score_data = summary_df[summary_df["score_type"] == score].copy()

    if len(score_data) == 0:
        print(f"  No data available for {score}")
        continue

    # Create figure with combined training curves
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    # Collect curves by timeframe
    for profit in profits:
        if profit not in score_data["profit"].values:
            continue

        profit_data = score_data[score_data["profit"] == profit].copy()

        # Collect all training curves for this timeframe
        train_curves = []
        val_curves = []

        for idx, row in profit_data.iterrows():
            model = row.get("model", None)
            if model is not None:
                # Training loss curve
                loss_curve = getattr(model, "loss_curve_", None)
                # Validation loss curve (validation_scores_ for MLPRegressor contains validation loss)
                val_loss_curve = getattr(model, "validation_scores_", None)

                if loss_curve is not None:
                    train_curves.append(np.array(loss_curve))
                if val_loss_curve is not None:
                    val_curves.append(np.array(val_loss_curve))

        # Plot training loss curves
        if len(train_curves) > 0:
            # Get minimum length to align all curves
            min_len = min([len(c) for c in train_curves])
            curves_trimmed = [c[:min_len] for c in train_curves]

            # Plot individual curves with transparency
            for curve in curves_trimmed:
                ax.plot(
                    curve,
                    color=profit_colors[profit],
                    alpha=0.1,
                    linewidth=0.5,
                    linestyle="-",
                )

            # Plot average training curve
            avg_curve = np.mean(curves_trimmed, axis=0)
            ax.plot(
                avg_curve,
                color=profit_colors[profit],
                label=f"{profit} (train)",
                linewidth=3,
                alpha=0.9,
                linestyle="-",
            )

        # Plot validation loss curves
        if len(val_curves) > 0:
            # Get minimum length to align all curves
            min_len = min([len(c) for c in val_curves])
            curves_trimmed = [c[:min_len] for c in val_curves]

            # Plot individual curves with transparency
            for curve in curves_trimmed:
                ax.plot(
                    curve,
                    color=profit_colors[profit],
                    alpha=0.1,
                    linewidth=0.5,
                    linestyle="--",
                )

            # Plot average validation curve
            avg_curve = np.mean(curves_trimmed, axis=0)
            ax.plot(
                avg_curve,
                color=profit_colors[profit],
                label=f"{profit} (val)",
                linewidth=3,
                alpha=0.9,
                linestyle="--",
            )

        # Plot validation_time as final point markers (since we don't have epoch-by-epoch data)
        if len(profit_data) > 0:
            # Get final validation_time RMSE for each model
            final_val_t_rmse = []
            final_epochs = []

            for idx, row in profit_data.iterrows():
                model = row.get("model", None)
                if model is not None:
                    val_t_rmse = row.get("val_t_rmse", None)
                    n_epochs = getattr(model, "n_iter_", None)
                    if val_t_rmse is not None and n_epochs is not None:
                        final_val_t_rmse.append(val_t_rmse)
                        final_epochs.append(n_epochs)

            if len(final_val_t_rmse) > 0:
                # Plot final validation_time points
                ax.scatter(
                    final_epochs,
                    final_val_t_rmse,
                    color=profit_colors[profit],
                    label=f"{profit} (val_t final)",
                    s=80,
                    marker="^",
                    alpha=0.8,
                )

    # Configure combined loss plot
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (RMSE)")
    ax.set_title(f"Training & Validation Loss Curves - {score.upper()}")
    ax.legend(
        title="Time Frame & Type", bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # Log scale for better visualization

    plt.tight_layout()

    # Save training curve plot
    curve_export_path = (
        f"code/neuronal_network/regressor_nn/training_curves_{score}.png"
    )
    if save:
        os.makedirs(os.path.dirname(curve_export_path), exist_ok=True)
        plt.savefig(curve_export_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {curve_export_path}")

    plt.show()

# --- Create summary tables for best architectures ---
print("\n📋 Creating summary tables...")

# Best architecture per score-timeframe combination
best_combinations = []
for score in scores:
    score_data = summary_df[summary_df["score_type"] == score].copy()
    for profit in profits:
        profit_data = score_data[score_data["profit"] == profit].copy()
        if len(profit_data) > 0:
            best_idx = profit_data["val_r2_score"].idxmax()
            best_row = profit_data.loc[best_idx]
            best_combinations.append(
                {
                    "score": score,
                    "timeframe": profit,
                    "best_architecture": best_row["hidden_layer"],
                    "val_r2_score": best_row["val_r2_score"],
                    "val_rmse": best_row["val_rmse"],
                    "val_mae": best_row["val_mae"],
                }
            )

best_df = pd.DataFrame(best_combinations)
print("\n🎯 Best architecture for each score-timeframe combination:")
print(best_df.to_string(index=False))

if save:
    csv_path = "code/neuronal_network/regressor_nn/best_architectures_by_score_timeframe.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    best_df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved: {csv_path}")

# Overall best architectures by score
print("\n🏅 Overall best architecture per score (averaged across timeframes):")
overall_best = (
    summary_df.groupby(["score_type", "hidden_layer"])["val_t_r2_score"]
    .mean()
    .reset_index()
    .sort_values("val_t_r2_score", ascending=False)
    .groupby("score_type")
    .first()
    .reset_index()
)
print(
    overall_best[["score_type", "hidden_layer", "val_t_r2_score"]].to_string(
        index=False
    )
)
