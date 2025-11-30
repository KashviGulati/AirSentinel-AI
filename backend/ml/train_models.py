"""
Train models:
 - IsolationForest for anomaly detection
 - RandomForestClassifier for unsafe/safe classification (AQI threshold)
Saves models and scaler to backend/ml/models/
Run: python -m backend.ml.train_models
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "..", "dataset", "cleaned_aqi_data.csv")
DATA = os.path.abspath(DATA)
OUT_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUT_DIR, exist_ok=True)

# Features to use (adjust if your CSV has different column names)
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "OZONE", "CO", "NH3"]

def load_and_prepare(path):
    df = pd.read_csv(path, low_memory=False)
    # Make sure columns exist, coerce to numeric
    for c in POLLUTANTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Keep only rows with at least one pollutant
    df["num_poll"] = df[[c for c in POLLUTANTS if c in df.columns]].notna().sum(axis=1)
    df = df[df["num_poll"] >= 1].copy()
    # target: unsafe if AQI_official > 100
    if "AQI_official" not in df.columns:
        raise KeyError("AQI_official column not found in cleaned data")
    df["unsafe"] = (df["AQI_official"] > 100).astype(int)
    # time features
    if "hour_of_day" not in df.columns and "last_update" in df.columns:
        df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")
        df["hour_of_day"] = df["last_update"].dt.hour
        df["day_of_week"] = df["last_update"].dt.dayofweek
    features = [c for c in POLLUTANTS if c in df.columns] + ["hour_of_day", "day_of_week"]
    df = df.dropna(subset=features + ["unsafe"])
    X = df[features].fillna(0.0).values
    y = df["unsafe"].values
    return X, y, features

def train():
    print("Loading data...")
    X, y, features = load_and_prepare(DATA)
    print("Samples:", X.shape)

    # scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Anomaly detector on pollutant space
    print("Training IsolationForest...")
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    iso.fit(X_scaled)
    joblib.dump(iso, os.path.join(OUT_DIR, "isolation_forest.pkl"))
    print("Saved IsolationForest.")

    # Train classifier
    print("Training RandomForestClassifier...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classif. acc:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, os.path.join(OUT_DIR, "aqi_classifier.pkl"))

    # Save scaler and feature list
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
    joblib.dump(features, os.path.join(OUT_DIR, "features.pkl"))

    print("All models saved to", OUT_DIR)

if __name__ == "__main__":
    train()
