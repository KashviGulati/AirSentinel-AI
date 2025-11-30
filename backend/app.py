# backend/app.py
from flask import Flask, jsonify, send_from_directory, render_template, request
from flask_cors import CORS
import pandas as pd
import os
import requests
from datetime import datetime
import joblib
import numpy as np

# ---------------------------
# PATH / APP SETUP
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(BASE_DIR, "dashboard", "templates")
STATIC = os.path.join(BASE_DIR, "dashboard", "static")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml", "models")

app = Flask(__name__, template_folder=TEMPLATES, static_folder=STATIC)
CORS(app)

# ---------------------------
# DATA FILE PATHS
# ---------------------------
DATA_CLEAN = os.path.join(DATASET_DIR, "cleaned_aqi_data.csv")
DATA_ANOM = os.path.join(DATASET_DIR, "cleaned_aqi_with_anomalies.csv")
DATA_PCA = os.path.join(DATASET_DIR, "cleaned_aqi_pca.csv")
REALTIME = os.path.join(DATASET_DIR, "realtime_data.csv")  # written by ingest

# ---------------------------
# SAFE LOAD FUNCTION
# ---------------------------
def safe_load(path, name):
    if not os.path.exists(path):
        print(f"❌ {name} NOT FOUND -> {path}")
        return pd.DataFrame()
    print(f"✅ Loaded {name} -> {path}")
    return pd.read_csv(path)

# NOTE: we won't load the realtime CSV at import time (we'll read it on demand)
df_clean = safe_load(DATA_CLEAN, "cleaned_aqi_data.csv")
df_anom = safe_load(DATA_ANOM, "cleaned_aqi_with_anomalies.csv")
df_pca = safe_load(DATA_PCA, "cleaned_aqi_pca.csv")

# ---------------------------
# Optional: try load models (if you've trained them)
# ---------------------------
SCALER = None
FEATURES = None
ISO_MODEL = None
CLASS_MODEL = None

for fname in ["scaler.pkl", "features.pkl", "isolation_forest.pkl", "aqi_classifier.pkl"]:
    p = os.path.join(MODEL_DIR, fname)
    if os.path.exists(p):
        print("Loading model file:", fname)
        obj = joblib.load(p)
        if fname == "scaler.pkl":
            SCALER = obj
        elif fname == "features.pkl":
            FEATURES = obj
        elif fname == "isolation_forest.pkl":
            ISO_MODEL = obj
        elif fname == "aqi_classifier.pkl":
            CLASS_MODEL = obj
    else:
        print("Model file not found (this is okay for now):", p)

# ---------------------------
# HELPERS
# ---------------------------
def make_feature_vector(payload):
    """Create numeric vector aligned to FEATURES list. Fallbacks handled."""
    if FEATURES is None:
        raise RuntimeError("FEATURES not available (train and save features.pkl first)")
    vals = []
    for f in FEATURES:
        # try direct, then sanitized (remove dots), then 0
        v = payload.get(f, None)
        if v is None:
            alt = f.replace(".", "")
            v = payload.get(alt, 0)
        try:
            vals.append(float(v))
        except Exception:
            vals.append(0.0)
    return np.array(vals).reshape(1, -1)

# ---------------------------
# FRONTEND PAGES
# ---------------------------
@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/city/<name>")
def city_detail_page(name):
    return render_template("city_detail.html")

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# ---------------------------
# API: data endpoints (called by JS)
# ---------------------------
@app.get("/api/recent-readings")
def api_recent_readings():
    df = safe_load(DATA_CLEAN, "cleaned_aqi_data.csv")
    if df.empty:
        return jsonify([])
    # ensure timestamp parsed, then sort
    if "last_update" in df.columns:
        try:
            df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")
        except Exception:
            pass
    sorted_df = df.sort_values("last_update", ascending=False)
    return jsonify(sorted_df.to_dict(orient="records"))

@app.get("/api/anomalies")
def api_anomalies():
    df = safe_load(DATA_ANOM, "cleaned_aqi_with_anomalies.csv")
    if df.empty:
        return jsonify([])
    anom = df[(df.get("iforest_anomaly") == 1) | (df.get("z_anomaly") == True)]
    return jsonify(anom.to_dict(orient="records"))

@app.get("/api/pca-data")
def api_pca():
    df = safe_load(DATA_PCA, "cleaned_aqi_pca.csv")
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))

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
    return jsonify(stats.to_dict(orient="records"))

@app.get("/api/realtime")
def api_realtime():
    if not os.path.exists(REALTIME):
        return jsonify({"error": "Real-time file missing"}), 404
    df = pd.read_csv(REALTIME)
    return jsonify(df.to_dict(orient="records"))

@app.get("/api/reports")
def api_reports():
    # sample placeholders; your cron can write real reports into /reports
    return jsonify([
        {"date": datetime.utcnow().strftime("%Y-%m-%d"), "cities": 12, "anomalies": 7, "severe": 1}
    ])

# ---------------------------
# API: Ingest from OpenAQ (on-demand)
# ---------------------------
@app.post("/api/ingest")
def api_ingest():
    """
    Trigger ingestion from OpenAQ (country=IN). Saves/append to dataset/realtime_data.csv
    Returns number of rows written.
    """
    try:
        # OpenAQ endpoint (no API key needed)
        URL = "https://api.openaq.org/v2/latest"
        params = {
            "country": request.json.get("country", "IN") if request.is_json else "IN",
            "limit": request.json.get("limit", 1000) if request.is_json else 1000,
            "page": 1,
            "offset": 0
        }
        r = requests.get(URL, params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()

        rows = []
        for item in payload.get("results", []):
            city = item.get("city")
            location = item.get("location")
            coords = item.get("coordinates", {}) or {}
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            # take latest measurement timestamp if present
            ts = None
            meas_map = {}
            for m in item.get("measurements", []):
                param = (m.get("parameter") or "").lower()
                val = m.get("value")
                if param == "pm25":
                    meas_map["PM2.5"] = val
                elif param == "pm10":
                    meas_map["PM10"] = val
                elif param == "no2":
                    meas_map["NO2"] = val
                elif param == "so2":
                    meas_map["SO2"] = val
                elif param in ("o3", "ozone"):
                    meas_map["OZONE"] = val
                elif param == "co":
                    meas_map["CO"] = val
                elif param == "nh3":
                    meas_map["NH3"] = val
                # lastUpdated
                if not ts and m.get("lastUpdated"):
                    ts = m.get("lastUpdated")

            rows.append({
                "city": city,
                "station": location,
                "latitude": lat,
                "longitude": lon,
                "last_update": ts or datetime.utcnow().isoformat(),
                **meas_map
            })

        # save/append to realtime file
        if rows:
            out = REALTIME
            df_new = pd.DataFrame(rows)
            if os.path.exists(out):
                df_old = pd.read_csv(out)
                df = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df = df_new
            df.to_csv(out, index=False)
            return jsonify({"written": len(rows)})

        return jsonify({"written": 0})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# API: simple predict endpoint (works without trained model too)
# ---------------------------
@app.post("/api/predict")
def api_predict():
    """
    Accepts pollutant numbers and returns:
    - anomaly: true/false (if IsolationForest model exists)
    - unsafe_prob: probability of being 'unsafe' (classifier if exists)
    - unsafe_label: 0/1 label
    If models are missing we return a simple rule-based fallback.
    """
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "no payload"}), 400

    # Fallback rule if no classifier: mark unsafe if PM2.5 > 60 OR PM10 > 100
    out = {"anomaly": None, "unsafe_prob": None, "unsafe_label": None}

    # Anomaly by model if present
    try:
        if ISO_MODEL is not None and SCALER is not None and FEATURES is not None:
            X = make_feature_vector(payload)
            Xs = SCALER.transform(X)
            pred = ISO_MODEL.predict(Xs)  # -1 outlier
            out["anomaly"] = (pred[0] == -1)
    except Exception as e:
        out["anomaly"] = None

    # Classifier if present
    try:
        if CLASS_MODEL is not None and SCALER is not None and FEATURES is not None:
            X = make_feature_vector(payload)
            Xs = SCALER.transform(X)
            probs = CLASS_MODEL.predict_proba(Xs)
            # assume class 1 == unsafe
            if probs.shape[1] == 2:
                out["unsafe_prob"] = float(probs[0, 1])
            else:
                out["unsafe_prob"] = float(max(probs[0]))
            out["unsafe_label"] = int(CLASS_MODEL.predict(Xs)[0])
            return jsonify(out)
    except Exception:
        # fall through to simple rule
        pass

    # Fallback rule-based output:
    pm25 = None
    pm10 = None
    try:
        pm25 = float(payload.get("PM2.5", payload.get("PM25", 0) or 0))
    except Exception:
        pm25 = 0
    try:
        pm10 = float(payload.get("PM10", 0) or 0)
    except Exception:
        pm10 = 0

    # Simplistic unsafe scoring
    score = 0.0
    if pm25 >= 60: score += 0.7
    elif pm25 >= 30: score += 0.3
    if pm10 >= 100: score += 0.5
    elif pm10 >= 60: score += 0.2

    out["unsafe_prob"] = min(1.0, score)
    out["unsafe_label"] = 1 if out["unsafe_prob"] > 0.4 else 0
    if out["anomaly"] is None:
        out["anomaly"] = False

    return jsonify(out)

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
