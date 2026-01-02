import os
from fyers_api import accessToken
from fyers_api import fyersModel
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
SECRET_KEY = os.getenv("FYERS_SECRET_KEY")
REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI")

print("CLIENT_ID:", CLIENT_ID)
print("REDIRECT_URI:", REDIRECT_URI)

# ---- Step 1: Create session ----
session = accessToken.SessionModel(
    client_id=CLIENT_ID,
    secret_key=SECRET_KEY,
    redirect_uri=REDIRECT_URI,
    response_type="code",
    grant_type="authorization_code"
)

login_url = session.generate_authcode()
print("\n🔐 Open this URL in browser:\n")
print(login_url)

# ---- Step 2: Paste auth_code ----
auth_code = input("\nPaste auth_code from redirected URL: ").strip()

session.set_token(auth_code)
token_response = session.generate_token()

print("\nTOKEN RESPONSE:")
print(token_response)

if "access_token" not in token_response:
    print("❌ v2 auth failed")
    exit(1)

# ---- Step 3: Test API ----
fyers = fyersModel.FyersModel(
    client_id=CLIENT_ID,
    token=token_response["access_token"],
    log_path=""
)

profile = fyers.get_profile()
print("\nPROFILE RESPONSE:")
print(profile)

if profile.get("s") == "ok":
    print("\n✅ FYERS v2 IS WORKING (SANITY CONFIRMED)")
else:
    print("\n❌ API returned error")
