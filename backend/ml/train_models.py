"""
Train 5 ML models for AQI Analysis:
 - IsolationForest (anomaly detection)
 - RandomForestClassifier (safe/unsafe)
 - XGBoostClassifier (AQI category classification)
 - XGBoostRegressor (AQI prediction)
 - GradientBoostingRegressor (AQI prediction)

Saves all models + scaler + features to backend/ml/models/

Run:
    python -m backend.ml.train_models
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error
)

# XGBoost (ensure installed)
from xgboost import XGBClassifier, XGBRegressor


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.abspath(os.path.join(BASE, "..", "dataset", "cleaned_aqi_data.csv"))
OUT_DIR = os.path.join(BASE, "ml", "models")
os.makedirs(OUT_DIR, exist_ok=True)

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "OZONE", "CO", "NH3"]


# ------------------------------------------------------------
# LOAD + PREPARE DATA
# ------------------------------------------------------------
def load_and_prepare(path):
    df = pd.read_csv(path, low_memory=False)

    # Clean pollutant numeric columns
    for c in POLLUTANTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Require at least one pollutant
    df["num_poll"] = df[[c for c in POLLUTANTS if c in df.columns]].notna().sum(axis=1)
    df = df[df["num_poll"] >= 1].copy()

    # Target classification: unsafe (AQI > 100)
    if "AQI_official" not in df.columns:
        raise KeyError("AQI_official column missing")

    df["unsafe"] = (df["AQI_official"] > 100).astype(int)

    # Time features
    if "hour_of_day" not in df.columns:
        df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")
        df["hour_of_day"] = df["last_update"].dt.hour
        df["day_of_week"] = df["last_update"].dt.dayofweek

    # Features
    features = POLLUTANTS + ["hour_of_day", "day_of_week"]

    df = df.dropna(subset=features + ["unsafe", "AQI_official"])

    X = df[features].fillna(0.0).values
    y_class = df["unsafe"].values         # Safe vs Unsafe
    y_reg = df["AQI_official"].values     # AQI regression target

    return X, y_class, y_reg, features


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------
def train():
    print("\n📌 Loading dataset…")
    X, y_class, y_reg, features = load_and_prepare(DATA)
    print(f"Samples loaded: {X.shape[0]}")

    # --------------------------------------------------------
    # SCALER
    # --------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
    joblib.dump(features, os.path.join(OUT_DIR, "features.pkl"))
    print("✅ Saved scaler.pkl + features.pkl")

    # --------------------------------------------------------
    # 1. ISOLATION FOREST (Anomaly Detection)
    # --------------------------------------------------------
    print("\n🔍 Training Isolation Forest…")
    iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=42)
    iso.fit(X_scaled)

    joblib.dump(iso, os.path.join(OUT_DIR, "isolation_forest.pkl"))
    print("✅ Saved isolation_forest.pkl")

    # --------------------------------------------------------
    # Train/Test Split (for classification + regression)
    # --------------------------------------------------------
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X_scaled, y_class, y_reg,
        test_size=0.2, random_state=42, stratify=y_class
    )

    # --------------------------------------------------------
    # 2. RANDOM FOREST CLASSIFIER
    # --------------------------------------------------------
    print("\n🌲 Training RandomForestClassifier…")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_class_train)
    preds_rf = rf.predict(X_test)

    print("\nRF Classification Report:")
    print(classification_report(y_class_test, preds_rf))
    print("RF Accuracy:", accuracy_score(y_class_test, preds_rf))

    joblib.dump(rf, os.path.join(OUT_DIR, "rf_classifier.pkl"))
    print("✅ Saved rf_classifier.pkl")

    # --------------------------------------------------------
    # 3. XGBOOST CLASSIFIER
    # --------------------------------------------------------
    print("\n⚡ Training XGBoost Classifier…")
    xgb_clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    xgb_clf.fit(X_train, y_class_train)
    preds_xgb = xgb_clf.predict(X_test)

    print("\nXGB Classifier Report:")
    print(classification_report(y_class_test, preds_xgb))
    print("XGB Accuracy:", accuracy_score(y_class_test, preds_xgb))

    joblib.dump(xgb_clf, os.path.join(OUT_DIR, "xgb_classifier.pkl"))
    print("✅ Saved xgb_classifier.pkl")

    # --------------------------------------------------------
    # 4. XGBOOST REGRESSOR
    # --------------------------------------------------------
    print("\n📈 Training XGBoost Regressor…")
    xgb_reg = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        random_state=42
    )
    xgb_reg.fit(X_train, y_reg_train)
    reg_pred_xgb = xgb_reg.predict(X_test)

    print("\nXGB Regressor MAE:", mean_absolute_error(y_reg_test, reg_pred_xgb))
    print("XGB Regressor RMSE:", np.sqrt(mean_squared_error(y_reg_test, reg_pred_xgb)))

    joblib.dump(xgb_reg, os.path.join(OUT_DIR, "xgb_regressor.pkl"))
    print("✅ Saved xgb_regressor.pkl")

    # --------------------------------------------------------
    # 5. GRADIENT BOOSTING REGRESSOR
    # --------------------------------------------------------
    print("\n📉 Training Gradient Boosting Regressor…")
    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y_reg_train)
    reg_pred_gbr = gbr.predict(X_test)

    print("\nGBR MAE:", mean_absolute_error(y_reg_test, reg_pred_gbr))
    print("GBR RMSE:", np.sqrt(mean_squared_error(y_reg_test, reg_pred_gbr)))

    joblib.dump(gbr, os.path.join(OUT_DIR, "gbr_regressor.pkl"))
    print("✅ Saved gbr_regressor.pkl")

    print("\n🎉 ALL MODELS TRAINED + SAVED SUCCESSFULLY!\n")
    print(f"Saved in: {OUT_DIR}")


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    train()
