from flask import Flask, jsonify, send_from_directory, render_template
from flask_cors import CORS
import pandas as pd
import os

# ---------------------------------------------------
# FLASK APP SETUP
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "dashboard", "templates"),
    static_folder=os.path.join(BASE_DIR, "dashboard", "static")
)

CORS(app)

# ---------------------------------------------------
# DATA PATHS
# ---------------------------------------------------

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

DATA_CLEAN = os.path.join(DATASET_DIR, "cleaned_aqi_data.csv")
DATA_ANOM = os.path.join(DATASET_DIR, "cleaned_aqi_with_anomalies.csv")
DATA_PCA = os.path.join(DATASET_DIR, "cleaned_aqi_pca.csv")

# ---------------------------------------------------
# SAFE LOADING
# ---------------------------------------------------

def safe_load(path, name):
    if not os.path.exists(path):
        print(f"❌ {name} NOT FOUND → {path}")
        return pd.DataFrame()
    print(f"✅ Loaded {name} → {path}")
    return pd.read_csv(path)

df_clean = safe_load(DATA_CLEAN, "cleaned_aqi_data.csv")
df_anom = safe_load(DATA_ANOM, "cleaned_aqi_with_anomalies.csv")
df_pca = safe_load(DATA_PCA, "cleaned_aqi_pca.csv")

# ---------------------------------------------------
# FRONTEND ROUTES
# ---------------------------------------------------

@app.route("/")
def home_page():
    """Serve the dashboard UI (index.html)."""
    return render_template("index.html")

@app.route("/city/<name>")
def city_detail_page(name):
    return render_template("city_detail.html")

# ---------------------------------------------------
# API ROUTES (Used by JS)
# ---------------------------------------------------

@app.get("/api/recent-readings")
def api_recent_readings():
    if df_clean.empty:
        return jsonify([])
    sorted_df = df_clean.sort_values("last_update", ascending=False)
    return jsonify(sorted_df.to_dict(orient="records"))

@app.get("/api/anomalies")
def api_anomalies():
    if df_anom.empty:
        return jsonify([])
    anom = df_anom[(df_anom["iforest_anomaly"] == 1) | (df_anom["z_anomaly"] == True)]
    return jsonify(anom.to_dict(orient="records"))

@app.get("/api/pca-data")
def api_pca():
    if df_pca.empty:
        return jsonify([])
    return jsonify(df_pca.to_dict(orient="records"))

@app.get("/api/city-stats")
def api_city_stats():
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

@app.get("/api/realtime")
def api_realtime():
    file = os.path.join(DATASET_DIR, "realtime_data.csv")
    if not os.path.exists(file):
        return jsonify({"error": "Real-time file missing"}), 404
    df = pd.read_csv(file)
    return jsonify(df.to_dict(orient="records"))

@app.get("/api/reports")
def api_reports():
    return jsonify([
        {"date": "2025-11-28", "cities": 10, "anomalies": 25, "severe": 3},
        {"date": "2025-11-27", "cities": 10, "anomalies": 19, "severe": 2},
    ])

# ---------------------------------------------------
# STATIC FILES (CSS / JS)
# ---------------------------------------------------

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# ---------------------------------------------------
# RUN FLASK
# ---------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
