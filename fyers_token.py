import os
import hashlib
import requests

CLIENT_ID_FULL = os.getenv("FYERS_CLIENT_ID")   # XXXXXXXX-100
SECRET_KEY = os.getenv("FYERS_SECRET_KEY")

auth_code = input("Paste auth_code: ").strip()

CLIENT_ID_BASE = CLIENT_ID_FULL.split("-")[0]

app_id_hash = hashlib.sha256(
    f"{CLIENT_ID_BASE}:{SECRET_KEY}".encode()
).hexdigest()

payload = {
    "grant_type": "authorization_code",
    "appIdHash": app_id_hash,
    "code": auth_code
}

r = requests.post(
    "https://api-t1.fyers.in/api/v3/token",
    json=payload,
    headers={"Content-Type": "application/json"},
    timeout=10
)

print("STATUS:", r.status_code)
print("RESPONSE:", r.text)
