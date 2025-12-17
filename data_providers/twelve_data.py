import os
import requests

TD_KEY = os.getenv("TWELVE_DATA_API_KEY")

def fetch_twelve_data(symbol, interval="1min", limit=200):
    if not TD_KEY:
        raise RuntimeError("Twelve Data API key missing")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "apikey": TD_KEY,
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()

    data = r.json()

    if "values" not in data:
        raise RuntimeError(f"Twelve Data error: {data}")

    return data["values"]
