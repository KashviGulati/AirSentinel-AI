"""
clean_data.py
Load raw air-quality CSV, normalize timestamps, pivot pollutant rows into columns,
apply AQI calculation, create ML features, and save a cleaned ML-ready CSV.

Usage:
    python backend/ml/clean_data.py
"""

import os
import pandas as pd
import numpy as np

# Import AQI module
from backend.ml.aqi_calculator import apply_aqi

# ---------- CONFIG ----------
DEFAULT_PROJECT_PATH = os.path.join(os.getcwd(), "dataset", "air_quality_dataset.csv")
OUTPUT_PATH = os.path.join(os.getcwd(), "dataset", "cleaned_aqi_data.csv")
# ----------------------------


def find_input_path():
    if os.path.exists(DEFAULT_PROJECT_PATH):
        return DEFAULT_PROJECT_PATH
    raise FileNotFoundError(f"No input dataset found at: {DEFAULT_PROJECT_PATH}")


def load_raw(path):
    df = pd.read_csv(path, low_memory=False)
    return df


def normalize_columns(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_timestamp(df):
    timestamp_candidates = ["last_update", "timestamp", "date", "datetime", "time"]
    for col in timestamp_candidates:
        if col in df.columns:
            df.rename(columns={col: "last_update"}, inplace=True)
            df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")
            return df

    # If timestamp missing, create synthetic time series
    print("⚠️  No timestamp column found — generating artificial timestamps.")
    df["last_update"] = pd.date_range(start="2023-01-01", periods=len(df), freq="H")
    return df


def basic_cleaning(df):
    required_cols = ["pollutant_id", "pollutant_avg"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' missing. Available columns: {df.columns.tolist()}"
            )

    df["pollutant_avg"] = pd.to_numeric(df["pollutant_avg"], errors="coerce")
    df = df.dropna(subset=["last_update", "pollutant_id", "pollutant_avg"])
    return df


def pivot_pollutants(df):
    pivot_index = ["country", "state", "city", "station", "last_update",
                   "latitude", "longitude"]
    pivot_index = [c for c in pivot_index if c in df.columns]

    df_pivot = df.pivot_table(
        index=pivot_index,
        columns="pollutant_id",
        values="pollutant_avg",
        aggfunc="mean"
    ).reset_index()

    df_pivot.columns.name = None
    df_pivot.columns = [str(c) for c in df_pivot.columns]

    return df_pivot


def add_features(df_pivot):
    # pollutant columns exclude metadata
    pollutant_cols = [
        c for c in df_pivot.columns
        if c not in ["country","state","city","station","last_update",
                     "latitude","longitude"]
    ]

    df_pivot["num_pollutants_reported"] = df_pivot[pollutant_cols].notna().sum(axis=1)

    # Coordinate cleanup
    if "latitude" in df_pivot.columns:
        df_pivot["latitude"] = pd.to_numeric(df_pivot["latitude"], errors="coerce")
    if "longitude" in df_pivot.columns:
        df_pivot["longitude"] = pd.to_numeric(df_pivot["longitude"], errors="coerce")

    # Time-derived features
    df_pivot["ts_hour"] = df_pivot["last_update"].dt.floor("H")
    df_pivot["hour_of_day"] = df_pivot["last_update"].dt.hour
    df_pivot["day_of_week"] = df_pivot["last_update"].dt.dayofweek

    return df_pivot


def save_cleaned(df, outpath=OUTPUT_PATH):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False)
    print(f"✅ Saved cleaned dataset: {outpath}")


def main():
    input_path = find_input_path()
    print("Loading file:", input_path)

    raw = load_raw(input_path)
    print("Raw shape:", raw.shape)

    raw = normalize_columns(raw)
    raw = parse_timestamp(raw)
    raw = basic_cleaning(raw)

    pivot = pivot_pollutants(raw)
    pivot = add_features(pivot)

    print("ℹ️ Applying AQI calculations...")
    pivot = apply_aqi(pivot)

    print("Final cleaned shape:", pivot.shape)
    print("\nSample cleaned rows:\n", pivot.head().to_string(index=False))

    save_cleaned(pivot)


if __name__ == "__main__":
    main()
