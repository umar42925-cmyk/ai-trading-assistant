from health import broker_healthy
from data_providers.broker_data import broker_fetch
from data_providers.twelve_data import fetch_twelve_data

def get_market_data(symbol, interval):
    if broker_healthy():
        try:
            data = broker_fetch(symbol, interval)
            print(f"[DATA ROUTER] source={source}")

            return data, "broker"
        except Exception:
            pass

    data = fetch_twelve_data(symbol, interval)
    print(f"[DATA ROUTER] source={source}")

    return data, "twelve_data"
