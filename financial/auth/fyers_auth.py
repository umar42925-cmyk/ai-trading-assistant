import json
import time
import hashlib
import os
import requests
from typing import Optional
from urllib.parse import urlencode

from .config import FyersConfig


class FyersAuth:
    """
    FYERS API v3 OAuth handler (CORRECT FLOW)
    """

    def __init__(self):
        self.config = FyersConfig
        self.token_file = self.config.TOKEN_FILE

    # ------------------ helpers ------------------

    @staticmethod
    def _sha256(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    # ------------------ AUTH FLOW ------------------

    def get_auth_url(self) -> str:
        """
        Step 1: Generate browser login URL
        """
        self.config.validate()

        params = {
            "client_id": self.config.CLIENT_ID,
            "redirect_uri": self.config.REDIRECT_URI,
            "response_type": "code",
            "state": "fyers_state"
        }

        return f"{self.config.AUTH_BASE}/generate-authcode?{urlencode(params)}"

    def exchange_code_for_token(self, auth_code: str) -> dict:
        """
        Step 2: Exchange auth_code for access token
        """
        payload = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "appIdHash": self._sha256(
                f"{self.config.CLIENT_ID}:{self.config.SECRET_KEY}"
            )
        }

        response = requests.post(
            self.config.TOKEN_URL,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Token request failed {response.status_code}: {response.text}"
            )

        data = response.json()

        if data.get("s") != "ok":
            raise RuntimeError(f"FYERS error: {data}")

        data["timestamp"] = time.time()
        self._save_token(data)
        return data

    # ------------------ TOKEN ------------------

    def _save_token(self, token_data: dict):
        os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
        with open(self.token_file, "w") as f:
            json.dump(token_data, f, indent=2)

    def _load_token(self) -> Optional[dict]:
        if not os.path.exists(self.token_file):
            return None

        with open(self.token_file, "r") as f:
            return json.load(f)

    def get_access_token(self) -> Optional[str]:
        token = self._load_token()
        if not token:
            return None

        if time.time() - token.get("timestamp", 0) > 86400:
            return None

        return token.get("access_token")

    # ------------------ SESSION ------------------

    def create_session(self) -> Optional[requests.Session]:
        token = self.get_access_token()
        if not token:
            return None

        session = requests.Session()
        session.headers.update({
            "Authorization": f"{self.config.CLIENT_ID}:{token}",
            "Content-Type": "application/json"
        })
        return session

    def is_authenticated(self) -> bool:
        return self.get_access_token() is not None
