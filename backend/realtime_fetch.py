import requests
import pandas as pd

OPENAQ_URL = "https://api.openaq.org/v2/latest"

def get_live_data(city="Delhi"):
    params = {
        "city": city,
        "limit": 50,
        "sort": "desc"
    }

    res = requests.get(OPENAQ_URL, params=params)

    if res.status_code != 200:
        return None

    data = res.json()

    rows = []
    for item in data["results"]:
        for measurement in item["measurements"]:
            rows.append({
                "city": item.get("city"),
                "location": item.get("location"),
                "parameter": measurement["parameter"],
                "value": measurement["value"],
                "unit": measurement["unit"],
                "last_updated": measurement["lastUpdated"]
            })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = get_live_data("Delhi")
    print(df.head())
