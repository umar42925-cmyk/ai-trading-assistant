# data_providers/broker_data.py

from fyers_api import fyersModel
from health import mark_broker_success

def broker_fetch(symbol, interval):
    """
    Single broker data entry point.
    Returns raw broker data or raises exception.
    """

    # ⚠️ TEMPORARY: call whatever Fyers function you already use
    data = get_fyers_data(symbol, interval)

    # ✅ ONLY mark success if no exception occurred
    mark_broker_success()

    return data
