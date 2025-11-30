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
import json

# -----------------------------------------
# JSON-SAFE CONVERSION (fixes bool/NaN issues)
# -----------------------------------------
def json_safe(x):
    """Convert numpy + NaN values to JSON-safe Python types."""
    if pd.isna(x):
        return None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    return x

# -----------------------------------------
# PATHS
# -----------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(BASE_DIR, "dashboard", "templates")
STATIC = os.path.join(BASE_DIR, "dashboard", "static")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "ml", "models")

app = Flask(__name__, template_folder=TEMPLATES, static_folder=STATIC)
CORS(app)

# -----------------------------------------
# DATASET FILE PATHS
# -----------------------------------------
DATA_CLEAN = os.path.join(DATASET_DIR, "cleaned_aqi_data.csv")
DATA_ANOM = os.path.join(DATASET_DIR, "cleaned_aqi_with_anomalies.csv")
DATA_PCA = os.path.join(DATASET_DIR, "cleaned_aqi_pca.csv")
REALTIME = os.path.join(DATASET_DIR, "realtime_data.csv")


# -----------------------------------------
# SAFE LOADER
# -----------------------------------------
def safe_load(path, name):
    if not os.path.exists(path):
        print(f"❌ {name} NOT FOUND → {path}")
        return pd.DataFrame()
    print(f"✅ Loaded {name} → {path}")
    return pd.read_csv(path)


df_clean = safe_load(DATA_CLEAN, "cleaned_aqi_data.csv")
df_anom = safe_load(DATA_ANOM, "cleaned_aqi_with_anomalies.csv")
df_pca = safe_load(DATA_PCA, "cleaned_aqi_pca.csv")

# -----------------------------------------
# LOAD MODELS
# -----------------------------------------
SCALER = None
FEATURES = None
ISO_MODEL = None
CLASS_MODEL = None

for fname in ["scaler.pkl", "features.pkl", "isolation_forest.pkl", "aqi_classifier.pkl"]:
    p = os.path.join(MODEL_DIR, fname)
    if os.path.exists(p):
        print("Loading model:", fname)
        obj = joblib.load(p)
        if fname == "scaler.pkl": SCALER = obj
        elif fname == "features.pkl": FEATURES = obj
        elif fname == "isolation_forest.pkl": ISO_MODEL = obj
        elif fname == "aqi_classifier.pkl": CLASS_MODEL = obj
    else:
        print("Model file missing:", p)


# -----------------------------------------
# FEATURE VECTOR BUILDER
# -----------------------------------------
def make_feature_vector(payload):
    if FEATURES is None:
        raise RuntimeError("FEATURES not loaded. Train model first.")
    vec = []
    for f in FEATURES:
        v = payload.get(f, payload.get(f.replace(".", ""), 0))
        try: vec.append(float(v))
        except: vec.append(0.0)
    return np.array(vec).reshape(1, -1)


# -----------------------------------------
# PAGES
# -----------------------------------------
@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/city/<name>")
def city_detail(name):
    return render_template("city_detail.html")

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)


# -----------------------------------------
# API — RECENT READINGS
# -----------------------------------------
@app.get("/api/recent-readings")
def api_recent():
    df = safe_load(DATA_CLEAN, "cleaned_aqi_data.csv")
    if df.empty:
        return jsonify([])

    if "last_update" in df.columns:
        df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")

    df = df.sort_values("last_update", ascending=False)
    return jsonify([{k: json_safe(v) for k, v in row.items()} for _, row in df.iterrows()])


# -----------------------------------------
# API — ANOMALIES
# -----------------------------------------
@app.get("/api/anomalies")
def api_anomalies():
    df = safe_load(DATA_ANOM, "cleaned_aqi_with_anomalies.csv")
    if df.empty:
        return jsonify([])

    df = df[(df.get("iforest_anomaly") == 1) | (df.get("z_anomaly") == True)]
    return jsonify([{k: json_safe(v) for k, v in row.items()} for _, row in df.iterrows()])


# -----------------------------------------
# API — PCA
# -----------------------------------------
@app.get("/api/pca-data")
def api_pca_data():
    df = safe_load(DATA_PCA, "cleaned_aqi_pca.csv")
    if df.empty:
        return jsonify([])
    return jsonify([{k: json_safe(v) for k, v in row.items()} for _, row in df.iterrows()])


# -----------------------------------------
# API — CITY STATS
# -----------------------------------------
@app.get("/api/city-stats")
def api_city_stats():
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

    return jsonify([{k: json_safe(v) for k, v in row.items()} for _, row in stats.iterrows()])


# -----------------------------------------
# API — REALTIME DATA
# -----------------------------------------
@app.get("/api/realtime")
def api_realtime():
    if not os.path.exists(REALTIME):
        return jsonify([])
    df = pd.read_csv(REALTIME)
    return jsonify([{k: json_safe(v) for k, v in row.items()} for _, row in df.iterrows()])


# -----------------------------------------
# API — RUN INGEST SCRIPT (WAQI)
# -----------------------------------------
@app.post("/api/ingest")
def api_ingest():
    try:
        script_path = os.path.join(BASE_DIR, "backend", "cron", "hourly_ingest.py")

        print("Running ingest:", script_path)

        result = subprocess.check_output(["python", script_path], stderr=subprocess.STDOUT, text=True)
        print("Ingest output:", result)

        return jsonify({"status": "ok", "output": result})

    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.output}), 500


# -----------------------------------------
# API — PREDICT
# -----------------------------------------
@app.post("/api/predict")
def api_predict():
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "No payload"}), 400

    out = {"anomaly": None, "unsafe_prob": None, "unsafe_label": None}

    # Anomaly detection
    try:
        if ISO_MODEL and SCALER:
            X = make_feature_vector(payload)
            Xs = SCALER.transform(X)
            pred = ISO_MODEL.predict(Xs)
            out["anomaly"] = bool(pred[0] == -1)
    except:
        out["anomaly"] = None

    # Classifier
    try:
        if CLASS_MODEL and SCALER:
            X = make_feature_vector(payload)
            Xs = SCALER.transform(X)
            prob = CLASS_MODEL.predict_proba(Xs)[0]
            out["unsafe_prob"] = float(prob[1])
            out["unsafe_label"] = int(CLASS_MODEL.predict(Xs)[0])
            return jsonify({k: json_safe(v) for k, v in out.items()})
    except:
        pass

    # Fallback simple rule
    pm25 = float(payload.get("PM2.5", 0) or 0)
    pm10 = float(payload.get("PM10", 0) or 0)

    score = 0
    if pm25 >= 60: score += 0.7
    if pm10 >= 100: score += 0.5

    out["unsafe_prob"] = min(1.0, score)
    out["unsafe_label"] = 1 if score > 0.4 else 0
    out["anomaly"] = out["anomaly"] or False

    return jsonify({k: json_safe(v) for k, v in out.items()})


# -----------------------------------------
# RUN
# -----------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
