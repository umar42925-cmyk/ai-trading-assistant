from health import broker_healthy
from data_providers.broker_data import broker_fetch
from data_providers.twelve_data import fetch_twelve_data

def normalize_for_twelve_data(symbol: str) -> str:
    """
    Convert broker-style symbols to Twelve Data format.
    """
    if ":" in symbol:
        symbol = symbol.split(":", 1)[1]

    symbol = symbol.replace("-INDEX", "")
    symbol = symbol.replace("-EQ", "")

    return symbol


def get_market_data(symbol, interval):
    if broker_healthy():
        try:
            data = broker_fetch(symbol, interval)
            print("[DATA ROUTER] source=broker")
            return data, "broker", "ok"

        except Exception as e:
            print(f"[DATA ROUTER] broker failed: {e}")

    try:
        td_symbol = normalize_for_twelve_data(symbol)
        data = fetch_twelve_data(td_symbol, interval)

        print("[DATA ROUTER] source=twelve_data")
        return data, "twelve_data", "ok"

    except Exception as e:
        print(f"[DATA ROUTER] twelve_data failed: {e}")
        return None, None, "unavailable"
