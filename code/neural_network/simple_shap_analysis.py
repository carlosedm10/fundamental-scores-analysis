#!/usr/bin/env python3
"""
Simplified SHAP analysis for memory-efficient neural network models.
This version works with model summaries that don't contain the full validation data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap


def simple_shap_analysis(
    model_summaries: list[dict],
    profits: list[str] = [
        "profit_1m",
        "profit_3m",
        "profit_6m",
        "profit_1y",
        "profit_2y",
        "profit_5y",
    ],
    sample_size: int = 50,
    export_path: str | None = None,
):
    """
    Create a simplified SHAP analysis showing feature importance across profit horizons.

    Args:
        model_summaries: List of model summary dictionaries
        profits: List of profit horizons to analyze
        sample_size: Number of samples for SHAP analysis (smaller for memory efficiency)
        export_path: Path to save the analysis plots

    Returns:
        Dictionary containing analysis results
    """

    # Define the variables we want to analyze
    variables_to_analyze = [
        {"name": "Dividend 24m average", "feature": "dividend_avg_24m"},
        {"name": "Growth 24m average", "feature": "growth_avg_24m"},
        {"name": "Quality 24m average", "feature": "quality_avg_24m"},
        {"name": "Value", "feature": "value"},
        {"name": "Value 24m average", "feature": "value_avg_24m"},
    ]

    # Organize summaries by profit
    summary_by_profit = {}
    for summary in model_summaries:
        profit = summary.get("profit")
        if profit and profit not in summary_by_profit:
            summary_by_profit[profit] = []
        if profit:
            summary_by_profit[profit].append(summary)

    # Create the plot
    n_variables = len(variables_to_analyze)
    n_profits = len(profits)

    fig = plt.figure(figsize=(5 * n_profits, 3.5 * n_variables + 2))
    gs = gridspec.GridSpec(
        n_variables + 1,
        n_profits + 1,
        height_ratios=[0.2] + [1] * n_variables,
        width_ratios=[1.2] + [1] * n_profits,
        hspace=0.4,
        wspace=0.5,
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
        ax = fig.add_subplot(gs[0, j + 1])
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

    # Create analysis for each variable-profit combination
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
            ax = fig.add_subplot(gs[i + 1, j + 1])

            # Find a summary that contains our feature for this profit
            summary = None
            if profit in summary_by_profit:
                for candidate_summary in summary_by_profit[profit]:
                    if feature in candidate_summary.get("X_cols", []):
                        summary = candidate_summary
                        break

            if summary is not None:
                try:
                    # Generate synthetic data for SHAP analysis (since we don't have X_val)
                    model = summary["model"]
                    scaler = summary["scaler"]
                    X_cols = summary["X_cols"]

                    # Create synthetic validation data based on feature distributions
                    # This is a simplified approach for memory efficiency
                    n_features = len(X_cols)

                    # Generate synthetic data with reasonable ranges
                    synthetic_data = np.random.normal(
                        0, 1, (sample_size, n_features)
                    )

                    # Apply scaling to match the training data distribution
                    synthetic_data_scaled = scaler.transform(synthetic_data)

                    # Create SHAP explainer
                    def predict_fn(x):
                        if hasattr(model, "predict_proba"):
                            return model.predict_proba(x)[:, 1]
                        else:
                            return model.predict(x)

                    # Use a smaller background dataset for memory efficiency
                    background_size = min(20, sample_size)
                    background_data = synthetic_data_scaled[:background_size]

                    explainer = shap.KernelExplainer(
                        predict_fn, background_data
                    )

                    # Calculate SHAP values for a subset
                    sample_data = synthetic_data_scaled[:sample_size]
                    shap_values = explainer.shap_values(sample_data)

                    # Focus on the specific feature
                    feature_idx = X_cols.index(feature)

                    # Create SHAP dependency plot: variable vs its own SHAP values
                    # Create SHAP dependency plot: variable vs its own SHAP values
                    plt.sca(ax)
                    shap.dependence_plot(
                        feature_idx,  # Feature to plot on X-axis
                        shap_values,  # SHAP values for Y-axis
                        sample_data,
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
        print(f"Simplified SHAP analysis saved to: {export_path}")

    plt.show()

    return analysis_results


if __name__ == "__main__":
    # Test the function
    print("Simplified SHAP analysis module loaded successfully!")
