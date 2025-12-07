import joblib
import numpy as np
import os

# -----------------------------
# Load Models and Scaler
# -----------------------------

BASE_PATH = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_PATH, "models")

scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
anomaly_model = joblib.load(os.path.join(MODEL_PATH, "isolation_forest.pkl"))
classifier = joblib.load(os.path.join(MODEL_PATH, "rf_classifier.pkl"))


# -----------------------------
# Core Prediction Function
# -----------------------------
def predict_aqi(input_data: dict):
    """
    input_data → dictionary of numerical features
    Example:
        {
            "aqi": 150,
            "temperature": 28,
            "humidity": 70,
            "pm2_5": 55
        }
    """

    # Convert dict values to NumPy array
    try:
        X = np.array([list(input_data.values())], dtype=float)
    except Exception:
        return {"error": "Invalid input format. Only numerical values allowed."}

    # Scale data
    X_scaled = scaler.transform(X)

    # Step 1: Check for anomaly
    anomaly_pred = anomaly_model.predict(X_scaled)[0]   # -1 = anomaly, 1 = normal

    if anomaly_pred == -1:
        return {
            "status": "anomaly",
            "message": "Unusual sensor pattern detected. Please recheck input."
        }

    # Step 2: Classify safe / unsafe
    class_pred = classifier.predict(X_scaled)[0]

    return {
        "status": "ok",
        "category": int(class_pred),
        "meaning": "Unsafe AQI level" if class_pred == 1 else "Safe AQI level"
    }


# -----------------------------
# Optional: Quick Test Run
# -----------------------------
if __name__ == "__main__":
    sample = {
        "aqi": 120,
        "temperature": 26,
        "humidity": 45
    }

    print("Test prediction:", predict_aqi(sample))
