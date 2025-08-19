# Outlier detection and analysis

# Imports
import sys
from pathlib import Path

# Add the parent directory (code/) to sys.path
sys.path.append(str(Path().resolve().parent))
from utils import (
    load_data,
    if_outliers,
    missing_values,
    pairplot,
)
import time

# Loading the data
file_name = "Big Data future"
data_path = f"./code/data/cleaned/multiple_analysis/{file_name}.csv"
# Read the CSV file
start_time = time.perf_counter()
df = load_data(data_path)
end_time = time.perf_counter()
print(f"Time taken for loading data: {end_time - start_time} seconds")

profit_columns = [
    "profit_1m",
    "profit_3m",
    "profit_6m",
    "profit_1y",
    "profit_2y",
    "profit_5y",
]


## Isolation Forest Outliers
df_multi = df.copy()
Y = df[profit_columns].copy()
start_time = time.perf_counter()
outliers_IF = if_outliers(df, profit_columns, 0.35)
end_time = time.perf_counter()
print(f"Time taken for outliers: {end_time - start_time} seconds")

start_time = time.perf_counter()
df_multi["outlier"] = outliers_IF
df_multi["outlier"] = df_multi["outlier"].map({1: False, -1: True})
end_time = time.perf_counter()
print(f"Time taken for mapping outliers: {end_time - start_time} seconds")

pair_path = f"./code/data/cleaned/outliers/{file_name} - IF HARD.png"
start_time = time.perf_counter()
# Plotting outliers
pairplot(df_multi, pair_path, hue="outlier")
end_time = time.perf_counter()
print(f"Time taken for the pairplot : {end_time - start_time} seconds")

# Deleting outliers
start_time = time.perf_counter()
df_multi_cleaned = df_multi[df_multi["outlier"] == False].copy()
missing_values(df, df_multi_cleaned)
end_time = time.perf_counter()
print(f"Time taken for deleting outliers: {end_time - start_time} seconds")


export_path = f"./code/data/cleaned/outliers/{file_name} - IF HARD.csv"
start_time = time.perf_counter()
df_multi_cleaned.to_csv(export_path, index=False)
end_time = time.perf_counter()
print(f"Time taken for exporting: {end_time - start_time} seconds")


pairplot(df_multi_cleaned, hue="sector")
