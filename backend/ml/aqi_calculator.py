"""
aqi_calculator.py
Two AQI calculation methods:
1) Official CPCB AQI computation using breakpoints
2) Simplified AQI using max pollutant method

Usage:
    from aqi_calculator import compute_aqi_row
"""

import numpy as np
import pandas as pd


# -----------------------------------------------------
# 1. OFFICIAL CPCB BREAKPOINTS (India National AQI)
# -----------------------------------------------------

CPCB_BREAKPOINTS = {
    "PM2.5": [
        (0, 30,      0, 50),
        (31, 60,    51, 100),
        (61, 90,   101, 200),
        (91, 120,  201, 300),
        (121, 250, 301, 400),
        (250, 800, 401, 500)
    ],
    "PM10": [
        (0, 50,     0, 50),
        (51, 100,  51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (430, 1000,401, 500)
    ],
    "NO2": [
        (0, 40,      0, 50),
        (41, 80,    51, 100),
        (81, 180,  101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (400, 1000,401, 500)
    ],
    "SO2": [
        (0, 40,      0, 50),
        (41, 80,    51, 100),
        (81, 380,  101, 200),
        (381, 800, 201, 300),
        (801, 1600,301, 400),
        (1600, 10000,401, 500)
    ],
    "OZONE": [
        (0, 50,        0, 50),
        (51, 100,     51, 100),
        (101, 168,   101, 200),
        (169, 208,   201, 300),
        (209, 748,   301, 400),
        (748, 2000,  401, 500)
    ],
    # Add if CO exists in your dataset
    "CO": [
        (0, 1,      0, 50),
        (1.1, 2,   51, 100),
        (2.1, 10,  101, 200),
        (10.1, 17, 201, 300),
        (17.1, 34, 301, 400),
        (34, 50,  401, 500)
    ],
}


# -----------------------------------------------------
# Helper Function: Compute sub-index for a pollutant
# -----------------------------------------------------
def compute_subindex(Cp, breakpoints):
    """
    Cp = measured concentration
    breakpoints = list of tuples (Clow, Chigh, Ilow, Ihigh)
    """
    for (Clow, Chigh, Ilow, Ihigh) in breakpoints:
        if Clow <= Cp <= Chigh:
            # linear interpolation
            return ((Ihigh - Ilow) / (Chigh - Clow)) * (Cp - Clow) + Ilow
    return None


# -----------------------------------------------------
# 2. SIMPLIFIED AQI METHOD (max pollutant)
# -----------------------------------------------------
def simplified_aqi(row, pollutant_cols):
    values = [row[col] for col in pollutant_cols if pd.notnull(row[col])]
    if len(values) == 0:
        return None
    return max(values)   # direct max


# -----------------------------------------------------
# 3. Full AQI calculator for a row
# -----------------------------------------------------
def compute_aqi_row(row, pollutant_cols):
    """
    Returns:
    - official_aqi
    - simplified_aqi_value
    - category
    """
    subindices = []

    # Calculate CPCB sub-index for each measured pollutant
    for pollutant in pollutant_cols:
        Cp = row.get(pollutant)
        if pollutant in CPCB_BREAKPOINTS and pd.notnull(Cp):
            si = compute_subindex(Cp, CPCB_BREAKPOINTS[pollutant])
            if si is not None:
                subindices.append(si)

    # Official AQI = maximum of valid sub-indices
    official_aqi = max(subindices) if subindices else None

    # Simplified AQI = max pollutant concentration
    simple_aqi = simplified_aqi(row, pollutant_cols)

    # Category based on official AQI
    category = classify_aqi(official_aqi)

    return official_aqi, simple_aqi, category


# -----------------------------------------------------
# AQI Category
# -----------------------------------------------------
def classify_aqi(aqi):
    if aqi is None:
        return "Unknown"
    aqi = float(aqi)

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


# -----------------------------------------------------
# 4. Apply AQI computation on entire DataFrame
# -----------------------------------------------------
def apply_aqi(df):
    pollutant_cols = [c for c in df.columns if c in CPCB_BREAKPOINTS.keys()]

    results = df.apply(lambda row: compute_aqi_row(row, pollutant_cols), axis=1)

    df["AQI_official"] = results.apply(lambda x: x[0])
    df["AQI_simple"] = results.apply(lambda x: x[1])
    df["AQI_category"] = results.apply(lambda x: x[2])

    return df
