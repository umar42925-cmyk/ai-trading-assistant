# data_providers/broker_data.py

def broker_fetch(symbol, interval):
    try:
        from fyers_api import fyersModel  # ✅ moved INSIDE function
    except ImportError as e:
        raise RuntimeError("Fyers API not available") from e

    # ---- existing broker logic below ----
    # fyers = fyersModel.FyersModel(...)
    # data = fyers.history(...)
    # return data
