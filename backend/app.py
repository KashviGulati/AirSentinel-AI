# backend/app.py

from flask import Flask, jsonify, send_from_directory, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime
import joblib
import subprocess

# ---------------------------------------------------
# JSON SAFE CONVERSION
# ---------------------------------------------------
def json_safe(x):
    if pd.isna(x):
        return None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    return x


# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(BASE_DIR, "dashboard", "templates")
STATIC = os.path.join(BASE_DIR, "dashboard", "static")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "ml", "models")

app = Flask(__name__, template_folder=TEMPLATES, static_folder=STATIC)
CORS(app)


# ---------------------------------------------------
# FILES
# ---------------------------------------------------
DATA_CLEAN = os.path.join(DATASET_DIR, "cleaned_aqi_data.csv")
DATA_ANOM = os.path.join(DATASET_DIR, "cleaned_aqi_with_anomalies.csv")
DATA_PCA  = os.path.join(DATASET_DIR, "cleaned_aqi_pca.csv")
REALTIME  = os.path.join(DATASET_DIR, "realtime_data.csv")


# ---------------------------------------------------
# FAST DATA LOADER
# ---------------------------------------------------
def safe_load(path, label):
    if not os.path.exists(path):
        print(f"❌ Missing dataset: {label}")
        return pd.DataFrame()
    print(f"📁 Loaded dataset: {label}")
    return pd.read_csv(path)


df_clean = safe_load(DATA_CLEAN, "cleaned_aqi_data.csv")
df_anom  = safe_load(DATA_ANOM, "cleaned_aqi_with_anomalies.csv")
df_pca   = safe_load(DATA_PCA, "cleaned_aqi_pca.csv")


# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
print("\n🔍 Loading ML models...")

SCALER = None
FEATURES = None
ISO_MODEL = None
CLASS_MODEL = None

for fname in ["scaler.pkl", "features.pkl", "isolation_forest.pkl", "aqi_classifier.pkl"]:
    file_path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(file_path):
        print(f"✅ Loaded {fname}")
        obj = joblib.load(file_path)
        if fname == "scaler.pkl": SCALER = obj
        elif fname == "features.pkl": FEATURES = obj
        elif fname == "isolation_forest.pkl": ISO_MODEL = obj
        elif fname == "aqi_classifier.pkl": CLASS_MODEL = obj
    else:
        print(f"❌ Missing model file: {fname}")


# ---------------------------------------------------
# FEATURE VECTOR BUILDER
# ---------------------------------------------------
def make_feature_vector(payload):
    if FEATURES is None:
        raise RuntimeError("FEATURES.pkl not loaded. Train models first.")

    vector = []
    for f in FEATURES:
        val = payload.get(f, payload.get(f.replace(".", ""), 0))
        try:
            vector.append(float(val))
        except:
            vector.append(0.0)

    return np.array(vector).reshape(1, -1)


# ---------------------------------------------------
# ROUTES — PAGES
# ---------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/city/<name>")
def city_page(name):
    return render_template("city_detail.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)


# ---------------------------------------------------
# ROUTES — DATA APIS
# ---------------------------------------------------
@app.get("/api/recent-readings")
def recent_readings():
    df = safe_load(DATA_CLEAN, "cleaned_aqi_data.csv")
    if df.empty:
        return jsonify([])

    if "last_update" in df.columns:
        df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")

    df = df.sort_values("last_update", ascending=False)

    return jsonify([{k: json_safe(v) for k, v in row.items()}
                    for _, row in df.iterrows()])


@app.get("/api/anomalies")
def anomalies():
    df = safe_load(DATA_ANOM, "cleaned_aqi_with_anomalies.csv")
    if df.empty:
        return jsonify([])

    df = df[(df.get("iforest_anomaly") == 1) |
            (df.get("z_anomaly") == True)]

    return jsonify([{k: json_safe(v) for k, v in row.items()}
                    for _, row in df.iterrows()])


@app.get("/api/pca-data")
def pca_data():
    df = safe_load(DATA_PCA, "cleaned_aqi_pca.csv")
    return jsonify([{k: json_safe(v) for k, v in row.items()}
                    for _, row in df.iterrows()])


@app.get("/api/city-stats")
def city_stats():
    df = safe_load(DATA_CLEAN, "cleaned_aqi_data.csv")
    if df.empty:
        return jsonify([])

    stats = df.groupby("city").agg({
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

    return jsonify([{k: json_safe(v) for k, v in row.items()}
                    for _, row in stats.iterrows()])


@app.get("/api/realtime")
def realtime():
    if not os.path.exists(REALTIME):
        return jsonify([])
    df = pd.read_csv(REALTIME)
    return jsonify([{k: json_safe(v) for k, v in row.items()}
                    for _, row in df.iterrows()])


# ---------------------------------------------------
# RUN INGEST SCRIPT
# ---------------------------------------------------
@app.post("/api/ingest")
def ingest():
    try:
        script_path = os.path.join(BASE_DIR, "backend", "cron", "hourly_ingest.py")
        output = subprocess.check_output(
            ["python", script_path], text=True
        )
        return jsonify({"status": "ok", "output": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------
# ML PREDICTION API
# ---------------------------------------------------
@app.post("/api/predict")
def predict():
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "Empty payload"}), 400

    result = {"anomaly": None, "unsafe_prob": None, "unsafe_label": None}

    # -------------------------
    # Anomaly Detection
    # -------------------------
    try:
        if ISO_MODEL and SCALER:
            X = make_feature_vector(payload)
            Xs = SCALER.transform(X)
            result["anomaly"] = bool(ISO_MODEL.predict(Xs)[0] == -1)
    except Exception as e:
        print("Anomaly error:", e)

    # -------------------------
    # Classifier Prediction
    # -------------------------
    try:
        if CLASS_MODEL and SCALER:
            X = make_feature_vector(payload)
            Xs = SCALER.transform(X)

            prob = CLASS_MODEL.predict_proba(Xs)[0][1]
            label = CLASS_MODEL.predict(Xs)[0]

            result["unsafe_prob"] = float(prob)
            result["unsafe_label"] = int(label)

            return jsonify({k: json_safe(v) for k, v in result.items()})
    except Exception as e:
        print("Classifier error:", e)

    # -------------------------
    # FALLBACK RULE (if models unavailable)
    # -------------------------
    pm25 = float(payload.get("PM2.5", 0))
    pm10 = float(payload.get("PM10", 0))

    score = 0
    if pm25 >= 60: score += 0.7
    if pm10 >= 100: score += 0.5

    result["unsafe_prob"] = min(1.0, score)
    result["unsafe_label"] = 1 if score > 0.4 else 0
    result["anomaly"] = result["anomaly"] or False

    return jsonify({k: json_safe(v) for k, v in result.items()})


# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Flask server running at http://127.0.0.1:5000/")
    app.run(debug=True)
