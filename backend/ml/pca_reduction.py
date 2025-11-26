"""
pca_reduction.py

Performs PCA on pollutant features, saves PCA model and components,
and creates 2D & 3D projection CSVs for visualization.

Usage:
    python -m backend.ml.pca_reduction
Outputs:
    - dataset/cleaned_aqi_pca.csv        (original rows + PC1, PC2, PC3)
    - backend/ml/pca_model.pkl           (saved PCA object)
    - backend/ml/pca_scaler.pkl          (saved scaler)
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Paths
CLEANED_PATH = os.path.join(os.getcwd(), "dataset", "cleaned_aqi_data.csv")
OUTPUT_PCA_CSV = os.path.join(os.getcwd(), "dataset", "cleaned_aqi_pca.csv")
PCA_MODEL_PATH = os.path.join(os.getcwd(), "backend", "ml", "pca_model.pkl")
SCALER_PATH = os.path.join(os.getcwd(), "backend", "ml", "pca_scaler.pkl")

def load_cleaned():
    if not os.path.exists(CLEANED_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found: {CLEANED_PATH}")
    df = pd.read_csv(CLEANED_PATH, low_memory=False, parse_dates=["last_update"], dayfirst=False)
    return df

def select_pollutant_features(df):
    # Candidate pollutant columns (these keys must match your cleaned file)
    candidates = ["PM2.5", "PM10", "NO2", "SO2", "OZONE", "CO", "NH3"]
    features = [c for c in candidates if c in df.columns]
    if len(features) == 0:
        raise ValueError("No pollutant columns found in cleaned dataset. Columns: " + ", ".join(df.columns))
    return features

def run_pca(df, feature_cols, n_components=3):
    X = df[feature_cols].copy()
    # Impute missing values with column mean (safe for PCA)
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(Xs)

    # Attach components to dataframe
    for i in range(n_components):
        df[f"PC{i+1}"] = pcs[:, i]

    # Explained variance
    evr = pca.explained_variance_ratio_
    print("Explained variance ratio (PCs):", evr)
    print("Cumulative explained:", evr.cumsum())

    return df, pca, scaler

def save_artifacts(pca, scaler, df_out):
    os.makedirs(os.path.dirname(PCA_MODEL_PATH), exist_ok=True)
    joblib.dump(pca, PCA_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    df_out.to_csv(OUTPUT_PCA_CSV, index=False)
    print("Saved PCA model:", PCA_MODEL_PATH)
    print("Saved scaler:", SCALER_PATH)
    print("Saved PCA-augmented CSV:", OUTPUT_PCA_CSV)

def main():
    df = load_cleaned()
    feature_cols = select_pollutant_features(df)
    print("Using pollutant features:", feature_cols)

    df_pca, pca, scaler = run_pca(df, feature_cols, n_components=3)
    save_artifacts(pca, scaler, df_pca)

    # Optional: create a small CSV for 2D plotting
    plot2d = df_pca[["country","state","city","station","last_update","PC1","PC2","AQI_official","AQI_category"]].copy()
    plot2d_path = os.path.join(os.getcwd(), "dataset", "pca_plot2d.csv")
    plot2d.to_csv(plot2d_path, index=False)
    print("Saved 2D plotting CSV:", plot2d_path)

if __name__ == "__main__":
    main()
