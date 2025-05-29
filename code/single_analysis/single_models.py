# ## Data Prep

# %%
import pandas as pd
import warnings
import statsmodels.api as sm
from collections import Counter
from pygam import LinearGAM, s, f
from sklearn.metrics import r2_score

# %%
file_name = "Small Data future cleaned"
# Read the CSV file
df = pd.read_csv(f"../data/cleaned/{file_name}.csv", encoding="utf-8")

df = df.rename(
    columns={
        "stock__share_type__company__industry__industry_group__sector__name": "sector",
        "stock__share_type__company__country__region": "region",
        "stock__share_type__company__name": "company",
        "stock__ticker": "ticker",
    }
)

# Creating dummy variables for sectors and regions
df = pd.get_dummies(df, columns=["sector", "region"])

print("\nShape of the dataset:")
print(df.shape)

# %%
warnings.filterwarnings("ignore", category=FutureWarning)

for score in ["quality", "growth", "value", "dividend"]:
    # Create separate dataframes for each score
    df_temp = df.copy()
    df_temp.drop(
        columns=[s for s in ["quality", "growth", "value", "dividend"] if s != score],
        inplace=True,
    )

    globals()[f"df_{score}"] = df_temp

profit_columns = [
    "profit_1m",
    "profit_3m",
    "profit_6m",
    "profit_1y",
    "profit_2y",
    "profit_5y",
]
score_types = ["quality", "growth", "value", "dividend"]

# %% [markdown]
# # Regression Models

# %%
reg_models = {}
reg_model_summaries = []

for score_type in score_types:
    score_df = globals()[f"df_{score_type}"]

    dummy_cols = [
        col for col in score_df.columns if col.startswith(("sector_", "region_"))
    ]

    # Create rolling averages for each score type
    for window in [3, 6, 12, 24]:  # 3m, 6m, 1y, 2y in months
        score_df[f"{score_type}_avg_{window}m"] = score_df.groupby("ticker")[
            score_type
        ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

    # Drop any rows with NaN values that might have been created
    score_df.dropna(inplace=True)

    # Add interaction terms
    for dummy in dummy_cols:
        score_df[f"{score_type}_{dummy}"] = score_df[score_type] * score_df[dummy]

    # For each profit column, perform backward elimination
    for profit_col in profit_columns:
        X_cols = (
            [score_type]
            + dummy_cols
            + [f"{score_type}_{dummy}" for dummy in dummy_cols]
        )
        X = score_df[X_cols].copy()
        y = score_df[profit_col].copy()

        # NOTE: comment or uncomment when needed

        # min_profit = y.min()
        # if min_profit < 0:
        #     y = (
        #         y - min_profit + 1
        #     )  # Shift to make all values positive and add 1 for log
        # else:
        #     y = y + 1  # Add 1 for log
        # y = np.log(y)  # Apply log transformation

        data = pd.concat([X, y], axis=1)

        X = data[X_cols].astype(float)
        y = data[profit_col].astype(float)

        # Add intercept
        X = sm.add_constant(X)

        removed_features = []

        # Backward elimination loop
        while True:
            model = sm.OLS(y, X).fit()
            pvalues = model.pvalues.drop("const", errors="ignore")

            # If all p-values ≤ 0.05 or only const is left, break
            if (pvalues <= 0.05).all() or len(pvalues) == 0:
                break

            # Drop the feature with the highest p-value
            worst_feature = pvalues.idxmax()
            X = X.drop(columns=worst_feature)
            removed_features.append(worst_feature)

        # Final model after elimination
        final_features = X.columns.tolist()
        final_model = sm.OLS(y, X).fit()

        key = f"{score_type}_{profit_col}"
        reg_models[key] = {
            "model": final_model,
            "X_cols": final_features,
            "y_true": y.copy(),
            "y_pred": final_model.fittedvalues.copy(),
            "removed_features": removed_features,
            "r2_score": final_model.rsquared,
            "adjusted_r2": final_model.rsquared_adj,
        }

        # Collect summary info
        reg_model_summaries.append(
            {
                "score_type": score_type,
                "profit_col": profit_col,
                "r2_score": final_model.rsquared,
                "adjusted_r2": final_model.rsquared_adj,
                "n_features": len(final_features),
                "kept_features": final_features,
                "removed_features": removed_features,
                "residuals": final_model.resid.copy(),
                "y_true": y.copy(),
                "y_pred": final_model.fittedvalues.copy(),  # optionally store predictions
            }
        )

# Create and display summary DataFrame
summary_df = pd.DataFrame(reg_model_summaries)
summary_df = summary_df.sort_values(["score_type", "profit_col"]).reset_index(drop=True)
print("\n===== Summary of All Models =====")
print(summary_df[["score_type", "profit_col", "r2_score", "adjusted_r2", "n_features"]])

# %% [markdown]
# # GAM

# %%
gam_models = {}
gam_summaries = []

for score_type in score_types:
    score_df = globals()[f"df_{score_type}"].copy()

    dummy_cols = [c for c in score_df.columns if c.startswith(("sector_", "region_"))]

    # Interactions
    for d in dummy_cols:
        score_df[f"{score_type}_{d}"] = score_df[score_type] * score_df[d]

    # Build the full list of X‑columns
    X_cols = [score_type] + dummy_cols + [f"{score_type}_{d}" for d in dummy_cols]

    # For each profit horizon…
    for profit_col in profit_columns:
        # 1) slice X, y
        X_cols = (
            [score_type]
            + dummy_cols
            + [f"{score_type}_{dummy}" for dummy in dummy_cols]
        )
        X = score_df[X_cols].copy()
        y = score_df[profit_col].copy()

        X = data[X_cols].astype(float)
        y = data[profit_col].astype(float)

        # 4) fit the GAM
        gam = LinearGAM().fit(X, y)

        # 5) predict & pseudo‑R²
        y_pred = gam.predict(X.values)
        # use standard R² on the held‑in data
        r2 = r2_score(y, y_pred)
        # pull pseudo‑R² (deviance explained) from GAM
        p_r2 = gam.statistics_["pseudo_r2"]["explained_deviance"]

        # 6) compute adjusted R² (n = samples, p = number of terms)
        n = X.shape[0]

        # 7) store model
        key = f"{score_type}_{profit_col}"
        gam_models[key] = {
            "model": gam,
            "X_cols": X.columns.tolist(),
            "r2_score": r2,
            "pseudo_r2": p_r2,
            "y_true": y,
            "y_pred": y_pred,
        }

        # 8) collect summary
        gam_summaries.append(
            {
                "score_type": score_type,
                "profit_col": profit_col,
                "r2_score": r2,
                "pseudo_r2": p_r2,
            }
        )

# Build and show summary DataFrame
summary_df = (
    pd.DataFrame(gam_summaries)
    .sort_values(["score_type", "profit_col"])
    .reset_index(drop=True)
)

print("\n===== GAM Models Summary =====")
print(
    summary_df[
        ["score_type", "profit_col", "r2_score", "pseudo_r2", "adjusted_r2", "n_terms"]
    ]
)

# %% [markdown]
# ## Analyzing the models

# %%
# Flatten all removed features from all models
all_removed = [
    feat for summary in reg_model_summaries for feat in summary["removed_features"]
]

# Count frequency of each removed feature
removed_counts = Counter(all_removed)

# Convert to DataFrame for easier analysis
removed_df = pd.DataFrame(removed_counts.items(), columns=["feature", "times_removed"])
removed_df = removed_df.sort_values(by="times_removed", ascending=False).reset_index(
    drop=True
)

# Display top 10 most dropped features
print("\n===== Most Frequently Removed Features =====")
print(removed_df.head(20))

# %%
import seaborn as sns
import matplotlib.pyplot as plt


# Create pivot table and reorder columns
heatmap_data = summary_df.pivot(
    index="score_type", columns="profit_col", values="r2_score"
).reindex(columns=profit_columns)

plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, cmap="viridis", fmt=".2f")
plt.title("Models R²")
plt.xlabel("Profit Horizon")
plt.ylabel("Score Type")
plt.tight_layout()
heatmap_path = "r2_scores_heatmap.png"
plt.savefig(heatmap_path, dpi=300)
plt.plot()

# %%
# plt.figure(figsize=(12, 6))
# sns.barplot(data=summary_df, x="profit_col", y="n_features", hue="score_type")
# plt.title("Number of Features in Final Models")
# plt.ylabel("Feature Count")
# plt.xlabel("Profit Horizon")
# plt.tight_layout()
# plt.show()

# %% [markdown]
# # Time Series

# %% [markdown]
# ## Residues Analysis

# %%
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

import os

# Create output directory if it doesn't exist
output_dir = "residual_plots"
os.makedirs(output_dir, exist_ok=True)


# Diagnostic plotting function
def plot_residual_diagnostics(residuals, fitted_values, title=""):
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

    # Hide unused plot
    axs[2, 1].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot
    filename = os.path.join(
        output_dir, f"residual_diagnostics_{title.replace(' ', '_')}.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# === Loop through all models and plot residual diagnostics ===
for key, model_info in reg_models.items():
    residuals_as_y = model_info["model"].resid
    fitted_values = model_info["model"].fittedvalues

    plot_residual_diagnostics(residuals_as_y, fitted_values, title=key)

# %% [markdown]
# # ARIMA

# %%
import sys

sys.path.append("../")
from utils import send_email_notification
import pmdarima as pm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import os

warnings.filterwarnings("ignore", category=FutureWarning)
# Create output directory if it doesn't exist
output_dir = "arima_plots"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store model results
model_results = {}

# === Loop through all models and plot residual diagnostics ===
for key, model_info in reg_models.items():
    residuals_as_y = model_info["model"].resid
    fitted_values = model_info["model"].fittedvalues

    # Create figure for diagnostics
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"ARIMA Model Diagnostics for {key}", fontsize=16)

    try:
        # Fit ARIMA with automatic differencing (d)
        arima_model = pm.auto_arima(
            residuals_as_y,
            start_p=0,
            start_q=0,
            max_p=5,
            max_q=5,
            max_d=2,
            seasonal=True,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )

        # Get predictions
        predictions = arima_model.predict_in_sample()

        new_residuals = residuals_as_y - predictions

        # Plot 1: Actual vs Predicted
        axs[0, 0].plot(residuals_as_y, label="Actual", alpha=0.7)
        axs[0, 0].plot(predictions, label="Predicted", alpha=0.7)
        axs[0, 0].set_title("Actual vs Predicted Values")
        axs[0, 0].legend()

        # Plot 2: Residuals Distribution
        sns.histplot(new_residuals, kde=True, ax=axs[0, 1])
        axs[0, 1].set_title("Residuals Distribution")

        # Plot 3: Q-Q Plot
        stats.probplot(new_residuals, dist="norm", plot=axs[1, 0])
        axs[1, 0].set_title("Q-Q Plot")

        # Plot 4: Residuals vs Fitted
        axs[1, 1].scatter(predictions, new_residuals, alpha=0.5)
        axs[1, 1].axhline(y=0, color="r", linestyle="--")
        axs[1, 1].set_title("Residuals vs Fitted Values")
        axs[1, 1].set_xlabel("Fitted Values")
        axs[1, 1].set_ylabel("Residuals")

        plt.tight_layout()

        # Print model summary and additional statistics
        print(f"\n{'='*50}")
        print(f"Model Diagnostics for {key}")
        print(f"{'='*50}")
        print(arima_model.summary())

        # Ljung-Box test for autocorrelation
        lb_test = acorr_ljungbox(new_residuals, lags=[6, 12, 24])
        print("\nLjung-Box Test Results:")
        print("H0: No autocorrelation in residuals")
        print(f"p-values for lags 10, 20, 30: {lb_test['lb_pvalue'].round(4)}")

        # Add interpretation of results
        for lag, pval in zip([6, 12, 24], lb_test["lb_pvalue"]):
            if pval < 0.05:
                print(
                    f"Lag {lag}: Significant autocorrelation detected (p = {pval:.4f})"
                )
            else:
                print(f"Lag {lag}: No significant autocorrelation (p = {pval:.4f})")

        # Save the plot
        plot_path = f'arima_plots/arima_diagnostics_{key.replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Store model results
        model_results[key] = {
            "model": arima_model,
            "aic": arima_model.aic(),
            "bic": arima_model.bic(),
            "lb_test": lb_test,
            "plot_path": plot_path,
            "autocorrelation_passed": lb_test["lb_pvalue"].iloc[0] >= 0.05,
        }

    except Exception as e:
        print(f"Error processing model {key}: {str(e)}")
        plt.close()
        continue

# Prepare email content with all model results
subject = "ARIMA Models Analysis Complete"
body = "ARIMA Models Analysis Summary:\n\n"

# Add results for each model
for key, result in model_results.items():
    body += f"Model: {key}\n"
    body += f"- AIC: {result['aic']:.2f}\n"
    body += f"- BIC: {result['bic']:.2f}\n"
    body += f"- Autocorrelation test: {'Passed' if result['autocorrelation_passed'] else 'Failed'}\n"
    body += f"- Plot saved to: {result['plot_path']}\n\n"

# Find best model based on AIC
best_model_key = min(model_results.items(), key=lambda x: x[1]["aic"])[0]
best_model = model_results[best_model_key]

body += f"\nBest performing model: {best_model_key}\n"
body += f"- AIC: {best_model['aic']:.2f}\n"
body += f"- BIC: {best_model['bic']:.2f}\n"
body += f"- Autocorrelation test: {'Passed' if best_model['autocorrelation_passed'] else 'Failed'}\n\n"

# Send email with the best model's plot
# send_email_notification(subject, body, best_model["plot_path"])

# %% [markdown]
#

# %% [markdown]
# # Added Model

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from utils import send_email_notification

# Prepare a container for the new R2s
r2_records = []

for key, reg_info in reg_models.items():
    # Extract regression model fitted values and true y
    y_true = reg_info["y_true"]
    reg_pred = reg_info["y_pred"]

    # Get the ARIMA model for this key
    arima = model_results[key]["model"]

    # In-sample ARIMA predictions of the residuals
    arima_pred = arima.predict_in_sample()

    # Combine regression + ARIMA residual correction
    combined_pred = reg_pred.reindex_like(y_true) + arima_pred.reindex_like(reg_pred)

    # Compute R2
    r2 = r2_score(y_true, combined_pred)

    # Parse score_type and profit_col out of the key
    score_type, profit_col = key.split("_", 1)
    r2_records.append(
        {"score_type": score_type, "profit_col": profit_col, "r2_combined": r2}
    )

# Build a pivot table for the heatmap
r2_df = pd.DataFrame(r2_records)
heatmap_data = r2_df.pivot(
    index="score_type", columns="profit_col", values="r2_combined"
).reindex(columns=profit_columns)

# Plot and save the heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, cmap="viridis", fmt=".2f")
plt.title("Combined Models R²")
plt.tight_layout()

combined_heatmap_path = "r2_combined_models_heatmap.png"
plt.savefig(combined_heatmap_path, dpi=300)
plt.show()

# Prepare and send email
subject = "Combined Models R²"
body = (
    "Hi,\n\n"
    "Please find attached the heatmap of combined Regression + ARIMA residual-correction R²\n"
    "scores for each score type across all profit horizons.\n\n"
    "Best,\n"
    "Your Analysis Bot"
)
images = [heatmap_path, combined_heatmap_path]

# %% [markdown]
# ## Residue Analysis

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

output_dir = "residuals_comparison_plots"
os.makedirs(output_dir, exist_ok=True)

for key in reg_models:
    if key not in model_results:
        continue

    # --- Extract y and predictions ---
    y_true = reg_models[key]["y_true"]
    reg_pred = reg_models[key]["y_pred"]
    arima_pred = model_results[key]["model"].predict_in_sample()
    combined_pred = reg_pred.reindex_like(y_true) + arima_pred.reindex_like(reg_pred)

    # --- Residuals ---
    resid_regression = y_true - reg_pred
    resid_arima = reg_models[key]["model"].resid - arima_pred
    resid_combined = y_true - combined_pred

    residual_sets = {
        "Regression": resid_regression,
        "ARIMA": resid_arima,
        "Combined": resid_combined,
    }

    fig, axs = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(f"Residual Comparison: {key}", fontsize=18)

    for i, (label, residuals) in enumerate(residual_sets.items()):
        # 1. Distribution
        sns.histplot(residuals, kde=True, ax=axs[i, 0])
        axs[i, 0].set_title(f"{label}: Distribution")

        # 2. Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axs[i, 1])
        axs[i, 1].set_title(f"{label}: Q-Q Plot")

        # 3. ACF
        plot_acf(residuals, ax=axs[i, 2], lags=24)
        axs[i, 2].set_title(f"{label}: ACF")

        # 4. PACF
        plot_pacf(residuals, ax=axs[i, 3], lags=24)
        axs[i, 3].set_title(f"{label}: PACF")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = f"{output_dir}/triple_residuals_{key}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    images.append(plot_path)

    # print(f"Saved comparison plot for {key} → {plot_path}")

send_email_notification(subject, body, images)
