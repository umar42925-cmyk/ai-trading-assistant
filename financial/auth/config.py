# financial/auth/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class FyersConfig:
    """Fyers API v3 Configuration"""
    
    # API Credentials (v3)
    CLIENT_ID = os.getenv("FYERS_CLIENT_ID", "")
    SECRET_KEY = os.getenv("FYERS_SECRET_KEY", "")
    REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI", "https://127.0.0.1/")
    
    # API v3 Endpoints
    BASE_URL = "https://api-t2.fyers.in/vagator/v2"
    AUTH_URL = "https://api-t2.fyers.in/vagator/v2"
    TOKEN_URL = "https://api-t2.fyers.in/api/v3/validate-authcode"
    DATA_URL = "https://api.fyers.in/data-rest/v2"
    
    # App Configuration
    APP_ID = f"{CLIENT_ID}-100"  # Format: ClientID-100
    APP_TYPE = "100"  # 100 for web login
    
    # Token Storage
    TOKEN_FILE = "financial/auth/fyers_token_v3.json"
    
    @classmethod
    def validate_config(cls):
        """Validate if all required configs are set"""
        missing = []
        if not cls.CLIENT_ID:
            missing.append("FYERS_CLIENT_ID")
        if not cls.SECRET_KEY:
            missing.append("FYERS_SECRET_KEY")
        
        if missing:
            raise ValueError(f"Missing Fyers config in .env: {', '.join(missing)}")
        
        return True