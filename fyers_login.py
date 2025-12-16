from fyers_apiv3 import fyersModel

# ===== EDIT THESE TWO LINES =====
APP_ID = "0K4RH3LJYJ-100"   # example: ABCD1234-100
REDIRECT_URI = "http://127.0.0.1:5000/redirect"
# =================================

session = fyersModel.SessionModel(
    client_id=APP_ID,
    redirect_uri=REDIRECT_URI,
    response_type="code",
    state="fyers_auth"
)

login_url = session.generate_authcode()

print("\n=== FYERS LOGIN URL ===")
print(login_url)
print("\nOpen this URL in your browser.")
