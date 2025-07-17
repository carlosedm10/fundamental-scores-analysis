# # Regression Models
# #
# #### 1st Model: y = f_0 + f_1 * score + f_2_n * dummies + f_n+1_m * score * dummies


# # Imports


import sys
from pathlib import Path

# Add the parent directory (code/) to sys.path
sys.path.append(str(Path().resolve().parent))
from utils import (
    load_data,
    separate_df_by_scores,
    model_performance_summary,
    plot_residual_diagnostics,
    gaussian_yj_transform,
    simple_gam,
)
import time
import pandas as pd
from collections import Counter

# # Loading the data
# #
# ### First we start with basic cleaned data (with outliers)

region = "AF"
file_name = f"Small Data future {region} - Multi"
data_path = f"./code/data/cleaned/multiple_analysis/outliers/{file_name}.csv"

# Read the CSV file
df = load_data(data_path)
df.drop(columns=["market_cap", "volume"], inplace=True)


# ## Variables


# Creating dummy variables
df = pd.get_dummies(df, columns=["sector", "region"])

# Explanatory variables:
scores = ["quality", "growth", "value", "dividend"]

dummies = [col for col in df.columns if col.startswith(("sector_", "region_"))]
# Drop the reference dummy variables (i.e., base categories)
reference_dummies = [
    "sector_Information Technology",
    # "region_Europe",
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


# # GAM Model


gam_model_summaries = []
model_count = 0
total_time = 0
start_time = time.perf_counter()
for score in scores:
    score_df = dataframes[score]

    # Add interaction terms
    for dummy in dummies:
        score_df[f"{score}_{dummy}"] = score_df[score] * score_df[dummy]

    for profit in profits:

        X_cols = [score] + dummies + [f"{score}_{dummy}" for dummy in dummies]
        X = score_df[X_cols].astype(float)
        y = gaussian_yj_transform(score_df[profit])

        summary = fast_backward_gam(X, y)
        summary["score_type"] = score
        summary["profit"] = profit

        # Collect summary info
        gam_model_summaries.append(summary)
        model_count += 1

end_time = time.perf_counter()
total_time = end_time - start_time
avg_time = total_time / model_count
print(f"\nTotal time: {total_time:.2f} seconds")
print(f"Average time per model: {avg_time:.2f} seconds")

summary_df = pd.DataFrame(gam_model_summaries)
summary_df = summary_df.sort_values(["score_type"]).reset_index(drop=True)


# # Analysis


all_removed = [
    feat
    for summary in gam_model_summaries
    for feat in summary["removed_features"]
]

# Count frequency of each removed feature
removed_counts = Counter(all_removed)

# Convert to DataFrame for easier analysis
removed_df = pd.DataFrame(
    removed_counts.items(), columns=["feature", "times_removed"]
)
removed_df = removed_df.sort_values(
    by="times_removed", ascending=False
).reset_index(drop=True)

# Display top 10 most dropped features
print("\n===== Most Frequently Removed Features =====")
print(removed_df.head(10))


export_path = "code/linear_regression/third_model/LAT/GAM_models_summary.png"
model_summary(summary_df, profits, export_path)


# === Loop through all models and plot residual diagnostics ===
for summary in gam_model_summaries:
    export_path = f"code/linear_regression/third_model/LAT/{summary['score_type']}_{summary['profit']}_residuals.png"

    plot_residual_diagnostics(
        summary["residuals"],
        summary["y_pred"],
        title=f"{summary['score_type']} - {summary['profit']}",
        export_path=export_path,
    )
