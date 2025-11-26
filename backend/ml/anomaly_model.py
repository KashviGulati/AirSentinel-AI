"""
anomaly_model.py
Detect pollution anomalies using:
1. Isolation Forest
2. Z-score thresholding (backup)

Usage:
    python -m backend.ml.anomaly_model
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# -----------------------------------------
# CONFIG
# -----------------------------------------
DATA_PATH = os.path.join(os.getcwd(), "dataset", "cleaned_aqi_data.csv")
MODEL_PATH = os.path.join(os.getcwd(), "backend", "ml", "isolation_forest_model.pkl")
SCALER_PATH = os.path.join(os.getcwd(), "backend", "ml", "scaler.pkl")

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df


# -----------------------------------------
# FEATURE SELECTION
# -----------------------------------------
def get_feature_matrix(df):
    """Use pollutant columns + AQI_official for anomaly detection."""
    feature_cols = ["PM2.5", "PM10", "NO2", "SO2", "OZONE", "CO", "AQI_official"]

    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(df[feature_cols].mean())
    return X, feature_cols


# -----------------------------------------
# Z-SCORE FALLBACK METHOD
# -----------------------------------------
def zscore_anomaly_detection(df, col="AQI_official", threshold=3):
    mean = df[col].mean()
    std = df[col].std()

    df["z_score"] = (df[col] - mean) / std
    df["z_anomaly"] = df["z_score"].abs() > threshold
    return df


# -----------------------------------------
# ISOLATION FOREST MODEL
# -----------------------------------------
def train_isolation_forest(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=0.05,  # 5% anomalies
        n_estimators=200,
        random_state=42
    )
    model.fit(X_scaled)

    return model, scaler


def add_iforest_predictions(df, model, scaler, feature_cols):
    X = df[feature_cols].fillna(df[feature_cols].mean())
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)

    df["iforest_anomaly"] = preds
    df["iforest_anomaly"] = df["iforest_anomaly"].apply(lambda x: 1 if x == -1 else 0)

    return df


# -----------------------------------------
# SAVE MODEL
# -----------------------------------------
def save_model(model, scaler):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved: {MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")


# -----------------------------------------
# MAIN FUNCTION
# -----------------------------------------
def main():
    print("🔹 Loading cleaned dataset...")
    df = load_data()

    print("🔹 Preparing feature matrix...")
    X, feature_cols = get_feature_matrix(df)

    print("🔹 Training Isolation Forest...")
    model, scaler = train_isolation_forest(X)
    save_model(model, scaler)

    print("🔹 Applying Isolation Forest predictions...")
    df = add_iforest_predictions(df, model, scaler, feature_cols)

    print("🔹 Applying Z-score detection...")
    df = zscore_anomaly_detection(df)

    # SAVE OUTPUT WITH ANOMALY MARKERS
    output_path = os.path.join(os.getcwd(), "dataset", "cleaned_aqi_with_anomalies.csv")
    df.to_csv(output_path, index=False)

    print("\n✅ Anomaly detection complete.")
    print(f"Saved: {output_path}")

    # Show preview
    print(df[["city", "station", "AQI_official", "iforest_anomaly", "z_anomaly"]].head())


if __name__ == "__main__":
    main()
