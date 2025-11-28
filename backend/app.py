from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------
# FIXED PATHS — ALWAYS CORRECT REGARDLESS OF OS
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

DATA_CLEAN = os.path.join(DATASET_DIR, "cleaned_aqi_data.csv")
DATA_ANOM = os.path.join(DATASET_DIR, "cleaned_aqi_with_anomalies.csv")
DATA_PCA = os.path.join(DATASET_DIR, "cleaned_aqi_pca.csv")

# ---------------------------------------------------
# LOAD DATA SAFELY
# ---------------------------------------------------

def safe_load(path, name):
    if not os.path.exists(path):
        print(f"❌ ERROR: {name} not found at {path}")
        return pd.DataFrame()
    print(f"✅ Loaded {name} from {path}")
    return pd.read_csv(path)

df_clean = safe_load(DATA_CLEAN, "cleaned_aqi_data.csv")
df_anom = safe_load(DATA_ANOM, "cleaned_aqi_with_anomalies.csv")
df_pca = safe_load(DATA_PCA, "cleaned_aqi_pca.csv")

# ---------------------------------------------------
# API ROUTES FOR DASHBOARD
# ---------------------------------------------------

@app.get("/api/recent-readings")
def get_recent_readings():
    if df_clean.empty:
        return jsonify([])
    # Sort newest readings
    sorted_df = df_clean.sort_values("last_update", ascending=False)
    return jsonify(sorted_df.to_dict(orient="records"))

@app.get("/api/anomalies")
def anomalies():
    if df_anom.empty:
        return jsonify([])
    anom = df_anom[
        (df_anom["iforest_anomaly"] == 1) |
        (df_anom["z_anomaly"] == True)
    ]
    return jsonify(anom.to_dict(orient="records"))

@app.get("/api/pca-data")
def pca_data():
    if df_pca.empty:
        return jsonify([])
    return jsonify(df_pca.to_dict(orient="records"))

@app.get("/api/city-stats")
def city_stats():
    if df_clean.empty:
        return jsonify([])

    stats = df_clean.groupby("city").agg({
        "AQI_official": "mean",
        "PM2.5": "mean",
        "PM10": "mean",
        "NO2": "mean"
    }).reset_index()

    stats.rename(columns={
        "AQI_official": "avg_aqi",
        "PM2.5": "avg_pm25",
        "PM10": "avg_pm10",
        "NO2": "avg_no2"
    }, inplace=True)

    return jsonify(stats.to_dict(orient="records"))

@app.get("/api/reports")
def get_reports():
    # Placeholder until automation scripts are added
    return jsonify([
        {"date": "2025-11-28", "cities": 10, "anomalies": 25, "severe": 3},
        {"date": "2025-11-27", "cities": 10, "anomalies": 19, "severe": 2},
    ])

# ---------------------------------------------------
# HOME ROUTE (OPTIONAL)
# ---------------------------------------------------

@app.get("/")
def home():
    return jsonify({"message": "Smart AQI API is running"})


# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
