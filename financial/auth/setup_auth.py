import os
from dotenv import load_dotenv

load_dotenv(override=True)

def setup_env():
    print("\n🔧 FYERS .env SETUP\n")

    client_id = input("FYERS_CLIENT_ID: ").strip()
    secret = input("FYERS_SECRET_KEY: ").strip()
    redirect = input("REDIRECT_URI [https://127.0.0.1/]: ").strip() or "https://127.0.0.1/"

    with open(".env", "a") as f:
        f.write("\n# FYERS API v3\n")
        f.write(f"FYERS_CLIENT_ID={client_id}\n")
        f.write(f"FYERS_SECRET_KEY={secret}\n")
        f.write(f"FYERS_REDIRECT_URI={redirect}\n")

    print("\n✅ .env updated\n")


if __name__ == "__main__":
    setup_env()
    print("Now run:")
    print("python -m financial.auth.auth_helper")
