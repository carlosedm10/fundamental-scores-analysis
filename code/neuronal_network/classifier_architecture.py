# Data Prep


#  Imports


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
from models import classifier_nn, regressor_nn

#  Loading Data


save = True

region = None  # NOTE: If region is None, the model is run for all regions
outlier_method = "IF"
if region:
    file_name = f"Small Data future {region} - {outlier_method}"
else:
    file_name = f"Small Data future - {outlier_method}"

data_path = f"./code/data/cleaned/outliers/{file_name}.csv"

# Read the CSV file
df = load_data(data_path)


#  Prepping the variables


# Creating dummy variables
df = pd.get_dummies(df, columns=["sector", "region"])

# Explanatory variables:
scores = ["quality", "growth", "value", "dividend"]

dummies = [col for col in df.columns if col.startswith(("sector_", "region_"))]

# Drop the reference dummy variables (i.e., base categories)
if not region:
    reference_dummies = [
        "sector_Information Technology",
        "region_Europe",
    ]
else:
    reference_dummies = [
        "sector_Information Technology",
    ]
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


# Regression Neuronal Network

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
                summary = classifier_nn(X, y, hidden_layer_sizes=hl_config)
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

# Show best architectures by average performance
mean_scores_by_arch = (
    summary_df.groupby("hidden_layer")[
        [
            "val_r2_score",
            "val_rmse",
            "val_mae",
        ]
    ]
    .mean()
    .sort_values("val_r2_score", ascending=False)
)

end_time = time.perf_counter()
total_time = end_time - start_time

# Display results
print(
    f"\n✅ Done {model_count} models in {total_time:.2f}s "
    f"({total_time/model_count:.4f}s each)"
)
print("\n🏆 Average scores per architecture (sorted by time-based R²):")
print(mean_scores_by_arch)


# Plot loss and accuracy curves for all models, colored by hidden layer config

# Prepare color palette for hidden layer configs
unique_hl = summary_df["hidden_layer"].unique()
palette = sns.color_palette("tab10", n_colors=len(unique_hl))
hl_color_map = {hl: palette[i] for i, hl in enumerate(unique_hl)}

fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
fig_acc, ax_acc = plt.subplots(figsize=(10, 6))

for idx, row in summary_df.iterrows():
    # Each row is a model summary
    hl = row["hidden_layer"]
    color = hl_color_map[hl]
    label = f"HL={hl}"

    # Loss curves
    model = row.get("model", None)
    loss_curve = model.loss_curve_  # type: ignore
    val_scores = model.validation_scores_  # type: ignore

    # To avoid duplicate labels in legend, only label first occurrence
    if not any([l.get_label() == label for l in ax_loss.lines]):
        ax_loss.plot(loss_curve, color=color, alpha=0.7, label=label)
    else:
        ax_loss.plot(loss_curve, color=color, alpha=0.4)

    # Accuracy curves
    if not any([l.get_label() == label for l in ax_acc.lines]):
        ax_acc.plot(val_scores, color=color, alpha=0.7, label=label)
    else:
        ax_acc.plot(val_scores, color=color, alpha=0.4)

ax_loss.set_title("Training Loss Curves for All Models")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.legend(
    title="Hidden Layer Config", bbox_to_anchor=(1.05, 1), loc="upper left"
)
ax_loss.grid(True)

ax_acc.set_title("Validation Score Curves for All Models")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Validation Score (R²)")
ax_acc.legend(
    title="Hidden Layer Config", bbox_to_anchor=(1.05, 1), loc="upper left"
)
ax_acc.grid(True)

plt.tight_layout()
plt.savefig("classifier_nn/loss_and_accuracy_curves.png")
plt.show()


# Prepare to collect all loss and val curves by hidden layer config
loss_curves_by_hl = {hl: [] for hl in unique_hl}
val_curves_by_hl = {hl: [] for hl in unique_hl}

for idx, row in summary_df.iterrows():
    hl = row["hidden_layer"]
    model = row.get("model", None)
    loss_curve = model.loss_curve_  # type: ignore
    val_scores = model.validation_scores_  # type: ignore

    # Loss curve
    loss_curves_by_hl[hl].append(np.array(loss_curve))
    # Validation curve
    val_curves_by_hl[hl].append(np.array(val_scores))

# Plot averaged loss curves
fig, ax = plt.subplots(figsize=(10, 6))
for hl, curves in loss_curves_by_hl.items():
    if len(curves) == 0:
        continue
    # Pad curves to the same length (shortest)
    min_len = min([len(c) for c in curves])
    curves_trimmed = [c[:min_len] for c in curves]
    avg_curve = np.mean(curves_trimmed, axis=0)
    std_curve = np.std(curves_trimmed, axis=0)
    color = hl_color_map[hl]
    label = f"HL={hl} (avg)"
    ax.plot(avg_curve, color=color, label=label, linewidth=2)
    ax.fill_between(
        np.arange(min_len),
        avg_curve - std_curve,
        avg_curve + std_curve,
        color=color,
        alpha=0.15,
    )
ax.set_title("Average Training Loss Curve by Hidden Layer Config")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend(
    title="Hidden Layer Config", bbox_to_anchor=(1.05, 1), loc="upper left"
)
ax.grid(True)
plt.tight_layout()
plt.show()

# Plot averaged validation score curves (if available)
fig, ax = plt.subplots(figsize=(10, 6))
for hl, curves in val_curves_by_hl.items():
    if len(curves) == 0:
        continue
    min_len = min([len(c) for c in curves])
    curves_trimmed = [c[:min_len] for c in curves]
    avg_curve = np.mean(curves_trimmed, axis=0)
    std_curve = np.std(curves_trimmed, axis=0)
    color = hl_color_map[hl]
    label = f"HL={hl} (avg)"
    ax.plot(avg_curve, color=color, label=label, linewidth=2)
    ax.fill_between(
        np.arange(min_len),
        avg_curve - std_curve,
        avg_curve + std_curve,
        color=color,
        alpha=0.15,
    )
ax.set_title("Average Validation Score Curve by Hidden Layer Config")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Score (R²)")
ax.legend(
    title="Hidden Layer Config", bbox_to_anchor=(1.05, 1), loc="upper left"
)
ax.grid(True)
plt.tight_layout()
plt.savefig("classifier_nn/average_loss_and_validation_curves.png")
plt.show()
