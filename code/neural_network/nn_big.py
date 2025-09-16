"""
Memory-efficient version of neural network models for large datasets.
This script processes models in smaller batches to avoid memory issues.
"""

import sys
from pathlib import Path
import time
import gc
import psutil
import os
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
    model_performance_summary_2,
)
from models import regressor_nn, classifier_nn
from simple_shap_analysis import simple_shap_analysis


def monitor_memory():
    """Monitor current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.1f} MB")
    return memory_mb


def process_models_in_batches(
    scores,
    profits,
    dataframes,
    dummies,
    save=True,
    file_name="",
    model_type="binary_classifier",
    batch_size=2,
):
    """
    Process models in smaller batches to manage memory usage.

    Args:
        scores: List of score types
        profits: List of profit horizons
        dataframes: Dictionary of dataframes by score
        dummies: List of dummy variables
        save: Whether to save results
        file_name: Name for saving files
        model_type: Type of model to run
        batch_size: Number of models to process before cleanup
    """
    all_summaries = []
    total_models = len(scores) * len(profits)

    print(f"Processing {total_models} models in batches of {batch_size}")

    for batch_start in range(0, total_models, batch_size):
        batch_end = min(batch_start + batch_size, total_models)
        print(
            f"\nProcessing batch {batch_start//batch_size + 1}: models {batch_start+1}-{batch_end}"
        )

        batch_summaries = []
        model_count = 0

        for score in scores:
            score_df = dataframes[score].copy()

            # Create rolling features more efficiently
            avg_cols = []
            std_cols = []

            for w in [3, 6, 12, 24]:
                avg_name = f"{score}_avg_{w}m"
                std_name = f"{score}_std_{w}m"

                # Use transform to avoid creating intermediate dataframes
                score_df[avg_name] = score_df.groupby("ticker")[
                    score
                ].transform(
                    lambda x: x.rolling(window=w, min_periods=1).mean()
                )
                score_df[std_name] = score_df.groupby("ticker")[
                    score
                ].transform(lambda x: x.rolling(window=w, min_periods=1).std())

                avg_cols.append(avg_name)
                std_cols.append(std_name)

            # Drop NAs early
            score_df.dropna(inplace=True)

            for profit in profits:
                if model_count >= batch_start and model_count < batch_end:
                    print(f"  Training {model_type} for {score} - {profit}")

                    X_feats = (
                        [score]
                        + dummies
                        + ["volume", "market_cap"]
                        + avg_cols
                        + std_cols
                    )

                    # Create X and y more efficiently
                    # Only convert numerical features to float, keep ticker and date as is
                    X_numerical = score_df[X_feats].astype(float)
                    X = pd.concat(
                        [score_df[["ticker", "date"]], X_numerical], axis=1
                    )
                    y = score_df[profit].astype(float)

                    try:
                        if model_type == "binary_classifier":
                            summary = classifier_nn(
                                X, y, profit_horizon=profit, binary=True
                            )
                        elif model_type == "encoder_classifier":
                            summary = classifier_nn(
                                X, y, profit_horizon=profit, binary=False
                            )
                        elif model_type == "regressor":
                            summary = regressor_nn(X, y)
                        else:
                            raise ValueError(
                                f"Unknown model_type: {model_type}"
                            )

                        summary.update(score_type=score, profit=profit)
                        batch_summaries.append(summary)

                    except Exception as e:
                        print(
                            f"    Error training model for {score} - {profit}: {e}"
                        )
                        continue

                model_count += 1

                # Clean up intermediate variables
                if model_count % 10 == 0:
                    gc.collect()

            # Clean up score_df after processing all profits
            del score_df
            gc.collect()

        # Add batch results to all summaries
        all_summaries.extend(batch_summaries)

        # Memory cleanup after each batch
        print(f"  Batch complete. Memory cleanup...")
        gc.collect()
        monitor_memory()

        # Small delay to allow system to reclaim memory
        time.sleep(1)

    return all_summaries


def main():
    """Main execution function"""
    print("Starting memory-efficient neural network training...")

    # Configuration
    save = True
    outlier_method = "IF HARD Balanced"
    file_name = f"Big Data future - {outlier_method}"
    data_path = f"./code/data/cleaned/outliers/{file_name}.csv"

    # Monitor initial memory
    print("Initial memory usage:")
    monitor_memory()

    # Load data
    print(f"\nLoading data from {data_path}")
    df = load_data(data_path)
    print(f"Data loaded. Shape: {df.shape}")
    monitor_memory()

    # Create dummy variables
    print("\nCreating dummy variables...")
    df = pd.get_dummies(df, columns=["sector", "region"])

    scores = ["quality", "growth", "value", "dividend"]
    dummies = [
        col for col in df.columns if col.startswith(("sector_", "region_"))
    ]

    # Drop reference dummies
    reference_dummies = ["sector_Information Technology", "region_Europe"]
    df.drop(columns=reference_dummies, inplace=True)
    dummies = [col for col in dummies if col not in reference_dummies]

    print(f"Total number of dummies: {len(dummies)}")
    print(f"Dummies: {dummies}")

    profits = [
        "profit_1m",
        "profit_3m",
        "profit_6m",
        "profit_1y",
        "profit_2y",
        "profit_5y",
    ]

    # Data validation - check for non-numeric values in expected numeric columns
    print("\nValidating data types...")
    numeric_columns = ["volume", "market_cap"] + scores + profits
    for col in numeric_columns:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors="coerce").isna().sum()
            if non_numeric > 0:
                print(f"Warning: {col} has {non_numeric} non-numeric values")
                # Show some examples
                problematic_values = df[col][
                    pd.to_numeric(df[col], errors="coerce").isna()
                ].head()
                print(f"Examples: {problematic_values.tolist()}")

    # Separate dataframes by scores
    print("\nSeparating dataframes by scores...")
    dataframes = separate_df_by_scores(df)
    monitor_memory()

    # Process each model type in batches
    model_types = [
        # ("binary_classifier", "binary_classifier_nn"),
        ("encoder_classifier", "classifier_nn"),
        # ("regressor", "regressor_nn"),
    ]

    for model_type, output_dir in model_types:
        print(f"\n{'='*50}")
        print(f"Processing {model_type.upper()}")
        print(f"{'='*50}")

        start_time = time.perf_counter()

        try:
            summaries = process_models_in_batches(
                scores=scores,
                profits=profits,
                dataframes=dataframes,
                dummies=dummies,
                save=save,
                file_name=file_name,
                model_type=model_type,
                batch_size=3,  # Process 3 models at a time
            )

            if summaries:
                # Create summary dataframe
                summary_df = (
                    pd.DataFrame(summaries)
                    .sort_values("score_type")
                    .reset_index(drop=True)
                )

                # Performance summary
                if save:
                    export_path = f"code/neural_network/{output_dir}/{file_name}/performance_summary.png"
                    os.makedirs(os.path.dirname(export_path), exist_ok=True)
                    model_performance_summary_2(
                        summary_df,
                        profits,
                        export_path,
                        classifier=(model_type != "regressor"),
                    )

                # SHAP analysis (with smaller sample size for memory)
                if save:
                    export_path = f"code/neural_network/{output_dir}/{file_name}/shap_table_analysis.png"
                    simple_shap_analysis(
                        summaries, sample_size=200, export_path=export_path
                    )

                end_time = time.perf_counter()
                total_time = end_time - start_time
                print(
                    f"\nCompleted {len(summaries)} {model_type} models in {total_time:.2f}s"
                )

            else:
                print(f"No {model_type} models were successfully trained")

        except Exception as e:
            print(f"Error processing {model_type}: {e}")
            continue

        # Clean up after each model type
        gc.collect()
        monitor_memory()

    print("\nAll model types processed!")
    print("Final memory usage:")
    monitor_memory()


if __name__ == "__main__":
    main()
