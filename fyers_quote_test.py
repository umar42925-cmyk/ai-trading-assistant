from fyers_apiv3 import fyersModel
import os

APP_ID = "0K4RH3LJYJ-100"  # e.g. QK4RHL3UYJ-100
TOKEN = os.getenv("FYERS_ACCESS_TOKEN")

if not TOKEN:
    raise RuntimeError("FYERS_ACCESS_TOKEN not found in environment")

fyers = fyersModel.FyersModel(
    client_id=APP_ID,
    token=TOKEN,
    log_path=""
)

# --- Test symbols ---
symbols = "NSE:NIFTY50-INDEX,NSE:RELIANCE-EQ"

response = fyers.quotes({"symbols": symbols})

print("\n=== FYERS LIVE QUOTE RESPONSE ===")
print(response)
