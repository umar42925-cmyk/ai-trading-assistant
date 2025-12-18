# data/fyers_client.py

import os
from fyers_apiv3 import fyersModel

def get_fyers_client():
    client_id = os.getenv("FYERS_CLIENT_ID")
    access_token = os.getenv("FYERS_ACCESS_TOKEN")

    if not client_id or not access_token:
        raise RuntimeError("FYERS credentials not set")

    fyers = fyersModel.FyersModel(
        client_id=client_id,
        token=access_token,
        log_path=None
    )
    return fyers

def fyers_health_check():
    fyers = get_fyers_client()
    return fyers.get_profile()
