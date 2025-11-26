import requests
from flask import Flask, jsonify

app = Flask(__name__)

@app.get("/realtime")
def get_realtime_aqi():
    url = "https://api.openaq.org/v2/latest?country=IN&city=Delhi"
    res = requests.get(url).json()

    return jsonify(res)
