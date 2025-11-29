import requests
import pandas as pd
import datetime
import os
from backend.ml.aqi_calculator import apply_aqi

SAVE_PATH = os.path.join(os.getcwd(), "dataset", "realtime_data.csv")

def fetch_realtime(city="Delhi"):
    url = f"https://api.openaq.org/v2/latest?city={city}&limit=100"
    r = requests.get(url)
    data = r.json()

    rows = []

    for item in data["results"]:
        city = item.get("city")
        location = item.get("location")
        lat = item["coordinates"]["latitude"]
        lon = item["coordinates"]["longitude"]
        time = item["measurements"][0]["lastUpdated"]

        pollutants = {m["parameter"]: m["value"] for m in item["measurements"]}

        row = {
            "city": city,
            "station": location,
            "latitude": lat,
            "longitude": lon,
            "last_update": time,
            "CO": pollutants.get("co"),
            "NO2": pollutants.get("no2"),
            "SO2": pollutants.get("so2"),
            "PM10": pollutants.get("pm10"),
            "PM2.5": pollutants.get("pm25"),
            "OZONE": pollutants.get("o3"),
            "NH3": pollutants.get("nh3")
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Apply AQI calculator
    df = apply_aqi(df)

    # Save for dashboard
    df.to_csv(SAVE_PATH, index=False)

    print(f"⏳ Saved real-time AQI: {SAVE_PATH}")

if __name__ == "__main__":
    fetch_realtime()
