import requests
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# -------------------------------------------------------
# Load token from .env
# -------------------------------------------------------
load_dotenv()   # loads the .env file
TOKEN = os.getenv("WAQI_TOKEN")

if not TOKEN:
    raise Exception("WAQI_TOKEN not found in .env file!")

# -------------------------------------------------------
# FIXED PATH → Save realtime_data.csv in the dataset folder
OUT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),  # backend/cron
        "..", "..",                 # go up to project root
        "dataset",                  # correct folder
        "realtime_data.csv"         # file name
    )
)

# Cities to fetch
CITIES = [
    "delhi", "mumbai", "bangalore", "hyderabad",
    "chennai", "kolkata", "pune", "ahmedabad",
    "lucknow", "jaipur"
]

# -------------------------------------------------------
# Fetch AQI data for one city from WAQI
# -------------------------------------------------------
def fetch_city(city):
    url = f"https://api.waqi.info/feed/{city}/?token={TOKEN}"

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()

        if data.get("status") != "ok":
            print(f" Failed for {city}: {data.get('data')}")
            return None

        d = data["data"]

        # pollutant extraction
        iaqi = d.get("iaqi", {})
        pm25 = iaqi.get("pm25", {}).get("v")
        pm10 = iaqi.get("pm10", {}).get("v")
        no2 = iaqi.get("no2", {}).get("v")
        so2 = iaqi.get("so2", {}).get("v")
        o3  = iaqi.get("o3", {}).get("v")
        co  = iaqi.get("co", {}).get("v")

        # timestamp
        ts = d.get("time", {}).get("s") or datetime.utcnow().isoformat()

        # coordinates
        lat, lon = d.get("city", {}).get("geo", [None, None])

        return {
            "city": city,
            "station": d.get("city", {}).get("name", city),
            "latitude": lat,
            "longitude": lon,
            "last_update": ts,
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "OZONE": o3,
            "CO": co,
            "NH3": None
        }

    except Exception as e:
        print(f"⚠ Error fetching {city}: {e}")
        return None

# -------------------------------------------------------
# Save fetched rows
# -------------------------------------------------------
def save(rows):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(rows)} rows -> {OUT}")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    print("Fetching WAQI real-time data...")
    rows = []

    for c in CITIES:
        result = fetch_city(c)
        if result:
            rows.append(result)

    if rows:
        save(rows)
    else:
        print("❌ No data fetched!")
