# backend/app.py

from flask import Flask, jsonify, send_from_directory, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib
import subprocess
import torch
import pickle

from ml.train_transformer import AQITransformer, load_series

# --------------------------------------------------
# JSON SAFE CONVERSION
# ---------------------------------------------------
def json_safe(x):
    # Handle pandas missing types + NaN safely
    try:
        from pandas._libs.missing import NAType
        if isinstance(x, NAType):
            return None
    except Exception:
        pass

    if isinstance(x, (pd._libs.missing.NAType,)) or (isinstance(x, float) and pd.isna(x)):
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
RF_MODEL = None
XGB_CLF_MODEL = None
XGB_REG_MODEL = None
GBR_REG_MODEL = None
METRICS = {}

MODEL_FILES = [
    "scaler.pkl",
    "features.pkl",
    "isolation_forest.pkl",
    "rf_classifier.pkl",
    "xgb_classifier.pkl",
    "xgb_regressor.pkl",
    "gbr_regressor.pkl",
    "metrics.pkl",
]

for fname in MODEL_FILES:
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        print(f"❌ Missing model file: {fname}")
        continue

    try:
        obj = joblib.load(path)
        print(f"✅ Loaded {fname}")

        if fname == "scaler.pkl":
            SCALER = obj
        elif fname == "features.pkl":
            FEATURES = obj
        elif fname == "isolation_forest.pkl":
            ISO_MODEL = obj
        elif fname == "rf_classifier.pkl":
            RF_MODEL = obj
        elif fname == "xgb_classifier.pkl":
            XGB_CLF_MODEL = obj
        elif fname == "xgb_regressor.pkl":
            XGB_REG_MODEL = obj
        elif fname == "gbr_regressor.pkl":
            GBR_REG_MODEL = obj
        elif fname == "metrics.pkl" and isinstance(obj, dict):
            METRICS = obj

    except Exception as e:
        print(f"⚠ Error loading {fname}: {e}")


# ---------------------------------------------------
# FEATURE VECTOR BUILDER
# ---------------------------------------------------
def make_feature_vector(payload):
    """
    Build feature vector in the exact order stored in features.pkl.
    FEATURES typically = POLLUTANTS + ['hour_of_day', 'day_of_week']
    """
    if FEATURES is None:
        raise RuntimeError("FEATURES.pkl not loaded. Train models first.")

    vector = []
    for f in FEATURES:
        # handle keys like "PM2.5" vs "PM25"
        key_val = payload.get(f, payload.get(f.replace(".", ""), 0))
        try:
            vector.append(float(key_val))
        except Exception:
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

    return jsonify([
        {k: json_safe(v) for k, v in row.items()}
        for _, row in df.iterrows()
    ])


@app.get("/api/anomalies")
def anomalies():
    df = safe_load(DATA_ANOM, "cleaned_aqi_with_anomalies.csv")
    if df.empty:
        return jsonify([])

    df = df[
        (df.get("iforest_anomaly") == 1) |
        (df.get("z_anomaly") == True)
    ]

    return jsonify([
        {k: json_safe(v) for k, v in row.items()}
        for _, row in df.iterrows()
    ])


@app.get("/api/pca-data")
def pca_data():
    df = safe_load(DATA_PCA, "cleaned_aqi_pca.csv")
    return jsonify([
        {k: json_safe(v) for k, v in row.items()}
        for _, row in df.iterrows()
    ])


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

    return jsonify([
        {k: json_safe(v) for k, v in row.items()}
        for _, row in stats.iterrows()
    ])


@app.get("/api/realtime")
def realtime():
    if not os.path.exists(REALTIME):
        return jsonify([])
    df = pd.read_csv(REALTIME)
    return jsonify([
        {k: json_safe(v) for k, v in row.items()}
        for _, row in df.iterrows()
    ])


# ---------------------------------------------------
# MODEL METRICS ENDPOINT
# ---------------------------------------------------
@app.get("/api/model-metrics")
def model_metrics():
    file_path = os.path.join(MODEL_DIR, "metrics.pkl")

    if not os.path.exists(file_path):
        # Default keys so frontend doesn't break
        return jsonify({
            "rf_accuracy": None,
            "xgb_clf_accuracy": None,
            "xgb_reg_mae": None,
            "gb_reg_mae": None
        })

    raw = joblib.load(file_path)

    normalized = {
        "rf_accuracy": raw.get("rf", {}).get("accuracy"),
        "xgb_clf_accuracy": raw.get("xgb_classifier", {}).get("accuracy"),
        "xgb_reg_mae": raw.get("xgb_regressor", {}).get("mae"),
        "gb_reg_mae": raw.get("gradient_boosting", {}).get("mae")
    }

    return jsonify({k: json_safe(v) for k, v in normalized.items()})


# ---------------------------------------------------
# RUN INGEST SCRIPT
# ---------------------------------------------------
@app.post("/api/ingest")
def ingest():
    try:
        script_path = os.path.join(BASE_DIR, "backend", "cron", "hourly_ingest.py")
        output = subprocess.check_output(["python", script_path], text=True)
        return jsonify({"status": "ok", "output": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# ---------------------------------------------------
# TRANSFORMER FORECAST API (FINAL FIXED)
# ---------------------------------------------------
@app.get("/api/transformer-forecast")
def transformer_forecast():
    """
    Uses the trained AQITransformer to forecast next `horizon` AQI_official values.
    Returns a list of steps with timestamps and predicted AQI.
    """
    try:
        model_path = os.path.join(MODEL_DIR, "transformer_aqi.pt")
        cfg_path = os.path.join(MODEL_DIR, "transformer_config.pkl")

        if not os.path.exists(model_path) or not os.path.exists(cfg_path):
            # Keep response shape consistent
            return jsonify({"forecast": []}), 404

        # Load config
        cfg = joblib.load(cfg_path)
        seq_len = int(cfg.get("seq_len", 24))
        horizon = int(cfg.get("horizon", 24))

        # Load time-series the same way as during training
        series = load_series()          # np.array of AQI_official (float32, NaNs dropped)

        if len(series) < seq_len:
            return jsonify({"forecast": []}), 400

        last_seq = series[-seq_len:].astype(np.float32)

        # Rebuild model with same hyperparameters
        model = AQITransformer(
            d_model=cfg.get("d_model", 64),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 3),
            dim_feedforward=cfg.get("dim_feedforward", 128),
            dropout=0.1,
            seq_len=seq_len,
            horizon=horizon,
        )
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        inp = torch.from_numpy(last_seq).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)

        with torch.no_grad():
            preds = model(inp).cpu().numpy().reshape(-1)

        # Build JSON-safe forecast (convert NaN/inf -> None)
        base_ts = pd.Timestamp.now()
        forecast = []
        for i, val in enumerate(preds):
            val = float(val)
            if not np.isfinite(val):
                aqi_val = None
            else:
                aqi_val = val

            ts = base_ts + pd.Timedelta(hours=i + 1)
            forecast.append({
                "step": i + 1,
                "aqi": aqi_val,
                "timestamp": ts.isoformat()
            })

        return jsonify({"forecast": forecast})

    except Exception as e:
        print("Transformer forecast error:", e)
        return jsonify({"forecast": [], "error": str(e)}), 500


# ---------------------------------------------------
# ML PREDICTION API (MULTI-MODEL)
# ---------------------------------------------------
@app.post("/api/predict")
def predict():
    """
    Input payload (JSON) expects at least pollutant fields and time features:
    {
      "PM2.5": ...,
      "PM10": ...,
      "NO2": ...,
      "SO2": ...,
      "OZONE": ...,
      "CO": ...,
      "NH3": ...,
      "hour_of_day": ...,
      "day_of_week": ...
    }

    Output JSON:
    {
      "anomaly": bool or null,
      "unsafe_prob": float or null,     # main classifier probability
      "unsafe_label": int or null,      # main classifier label (0/1)
      "rf_pred": int or null,           # RF unsafe prediction
      "xgb_clf_pred": int or null,      # XGB classifier unsafe prediction
      "xgb_reg_pred": float or null,    # XGB regression AQI prediction
      "gb_reg_pred": float or null      # GBR regression AQI prediction
    }
    """
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "Empty payload"}), 400

    result = {
        "anomaly": None,
        "unsafe_prob": None,
        "unsafe_label": None,
        "rf_pred": None,
        "xgb_clf_pred": None,
        "xgb_reg_pred": None,
        "gb_reg_pred": None,
    }

    # If we don't have features/scaler, fall back to rules at the end
    Xs = None
    if FEATURES is not None and SCALER is not None:
        try:
            X = make_feature_vector(payload)
            Xs = SCALER.transform(X)
        except Exception as e:
            print("Feature/scaler error:", e)

    # -------------------------
    # Anomaly Detection (Isolation Forest)
    # -------------------------
    try:
        if ISO_MODEL is not None and Xs is not None:
            result["anomaly"] = bool(ISO_MODEL.predict(Xs)[0] == -1)
    except Exception as e:
        print("Anomaly error:", e)

    # -------------------------
    # Random Forest Classifier
    # -------------------------
    try:
        if RF_MODEL is not None and Xs is not None:
            rf_label = int(RF_MODEL.predict(Xs)[0])
            result["rf_pred"] = rf_label
    except Exception as e:
        print("RF classifier error:", e)

    # -------------------------
    # XGBoost Classifier
    # -------------------------
    try:
        if XGB_CLF_MODEL is not None and Xs is not None:
            xgb_label = int(XGB_CLF_MODEL.predict(Xs)[0])
            xgb_prob = float(XGB_CLF_MODEL.predict_proba(Xs)[0][1])
            result["xgb_clf_pred"] = xgb_label

            # Use XGB classifier as the *main* unsafe prediction if available
            result["unsafe_label"] = xgb_label
            result["unsafe_prob"] = xgb_prob
    except Exception as e:
        print("XGB classifier error:", e)

    # If we didn't set unsafe_* from XGB, fallback to RF if available
    if result["unsafe_label"] is None and RF_MODEL is not None and Xs is not None:
        try:
            rf_label = int(RF_MODEL.predict(Xs)[0])
            rf_prob = float(RF_MODEL.predict_proba(Xs)[0][1])
            result["unsafe_label"] = rf_label
            result["unsafe_prob"] = rf_prob
        except Exception as e:
            print("RF prob fallback error:", e)

    # -------------------------
    # XGBoost Regressor (AQI)
    # -------------------------
    try:
        if XGB_REG_MODEL is not None and Xs is not None:
            xgb_reg_val = float(XGB_REG_MODEL.predict(Xs)[0])
            result["xgb_reg_pred"] = xgb_reg_val
    except Exception as e:
        print("XGB regressor error:", e)

    # -------------------------
    # Gradient Boosting Regressor (AQI)
    # -------------------------
    try:
        if GBR_REG_MODEL is not None and Xs is not None:
            gb_reg_val = float(GBR_REG_MODEL.predict(Xs)[0])
            result["gb_reg_pred"] = gb_reg_val
    except Exception as e:
        print("GBR regressor error:", e)

    # -------------------------
    # FALLBACK RULE (if classifiers missing)
    # -------------------------
    if result["unsafe_prob"] is None or result["unsafe_label"] is None:
        try:
            pm25 = float(payload.get("PM2.5", 0) or 0)
            pm10 = float(payload.get("PM10", 0) or 0)

            score = 0.0
            if pm25 >= 60:
                score += 0.7
            if pm10 >= 100:
                score += 0.5

            score = min(1.0, score)
            result["unsafe_prob"] = score
            result["unsafe_label"] = 1 if score > 0.4 else 0

            # if anomaly still None, set default False
            if result["anomaly"] is None:
                result["anomaly"] = False
        except Exception as e:
            print("Fallback rule error:", e)

    return jsonify({k: json_safe(v) for k, v in result.items()})


# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Flask server running at http://127.0.0.1:5000/")
    app.run(debug=True)
