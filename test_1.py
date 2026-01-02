import requests
import json
import os
from dotenv import load_dotenv
load_dotenv(override=True)

CLIENT_ID = os.getenv("FYERS_CLIENT_ID").strip()
SECRET_KEY = os.getenv("FYERS_SECRET_KEY").strip()
REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI", "https://127.0.0.1/").strip()

print("=" * 60)
print("MANUAL FYERS AUTH LINK GENERATION")
print("=" * 60)
print(f"CLIENT_ID: {CLIENT_ID}")
print(f"SECRET_KEY: {SECRET_KEY[:5]}...{SECRET_KEY[-5:]}")
print(f"REDIRECT_URI: {REDIRECT_URI}")
print(f"APP_ID: {CLIENT_ID}-100")
print("=" * 60)

# Payload for send_login_otp_v2
payload = {
    "fyers_id": CLIENT_ID,
    "app_id": f"{CLIENT_ID}-100",
    "redirect_uri": REDIRECT_URI,
    "appType": "100",
    "code_challenge": "",
    "state": "state",
    "scope": "",
    "nonce": "",
    "response_type": "code",
    "create_cookie": True
}

print("\n📤 Sending request to Fyers API...")
print(f"URL: https://api-t2.fyers.in/vagator/v2/send_login_otp_v2")
print(f"Payload:\n{json.dumps(payload, indent=2)}")

try:
    response = requests.post(
        "https://api-t2.fyers.in/vagator/v2/send_login_otp_v2",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    
    print(f"\n📥 Response Status: {response.status_code}")
    print(f"Response Body:\n{json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        data = response.json()
        if data.get("s") == "ok":
            request_key = data.get("request_key")
            auth_url = f"https://api-t2.fyers.in/vagator/v2/verify_otp?request_key={request_key}"
            
            print("\n" + "=" * 60)
            print("✅ SUCCESS! Auth link generated:")
            print("=" * 60)
            print(f"\n🔗 {auth_url}\n")
            print("📱 Next steps:")
            print("   1. Open this URL in your browser")
            print("   2. Login with your Fyers credentials")
            print("   3. Enter the OTP you receive")
            print("   4. Check if the app shows as 'Active' after login")
            print("=" * 60)
        else:
            print(f"\n❌ Error: {data.get('message', 'Unknown error')}")
            print(f"Error code: {data.get('code', 'N/A')}")
    else:
        print(f"\n❌ HTTP Error {response.status_code}")
        
except Exception as e:
    print(f"\n❌ Exception: {e}")
    import traceback
    traceback.print_exc()