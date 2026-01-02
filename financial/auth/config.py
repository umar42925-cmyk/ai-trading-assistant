import os
from dotenv import load_dotenv

load_dotenv(override=True)

class FyersConfig:
    """
    FYERS API v3 Configuration
    """

    CLIENT_ID = os.getenv("FYERS_CLIENT_ID", "").strip()
    SECRET_KEY = os.getenv("FYERS_SECRET_KEY", "").strip()
    REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI", "https://127.0.0.1/").strip()

    # Endpoints (v3 – stable)
    AUTH_BASE = "https://api.fyers.in/api/v3"
    TOKEN_URL = f"{AUTH_BASE}/token"
    PROFILE_URL = f"{AUTH_BASE}/profile"

    # Token storage
    TOKEN_FILE = os.path.join(os.path.dirname(__file__), "fyers_token_v3.json")

    @classmethod
    def validate(cls):
        missing = []
        if not cls.CLIENT_ID:
            missing.append("FYERS_CLIENT_ID")
        if not cls.SECRET_KEY:
            missing.append("FYERS_SECRET_KEY")

        if missing:
            raise ValueError(f"Missing in .env: {', '.join(missing)}")

        return True
