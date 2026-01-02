import webbrowser
from urllib.parse import urlparse, parse_qs

from .fyers_auth import FyersAuth


def setup_fyers_auth():
    """
    Interactive FYERS v3 authentication
    """
    auth = FyersAuth()

    if auth.is_authenticated():
        print("✅ Already authenticated")
        return auth.get_access_token()

    auth_url = auth.get_auth_url()

    print("\n🔐 FYERS LOGIN")
    print("Open this URL in your browser:\n")
    print(auth_url)

    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    print("\nAfter login, FYERS will redirect you.")
    print("Copy the FULL redirected URL and paste it below.\n")

    redirected = input("Redirected URL: ").strip()

    parsed = urlparse(redirected)
    auth_code = parse_qs(parsed.query).get("code", [None])[0]

    if not auth_code:
        print("❌ Auth code not found")
        return None

    token = auth.exchange_code_for_token(auth_code)

    print("\n🎉 Authentication successful")
    print("Token saved successfully")
    return token["access_token"]


def get_fyers_client():
    auth = FyersAuth()
    return auth.create_session()


def check_auth_status():
    auth = FyersAuth()
    if auth.is_authenticated():
        print("✅ Authenticated")
    else:
        print("❌ Not authenticated")
