# financial/auth/fyers_auth.py
import json
import time
import requests
import os
import hashlib
from typing import Optional, Dict, Tuple
from .config import FyersConfig

class FyersAuth:
    """Fyers Authentication Handler (API v3)"""
    
    def __init__(self):
        self.config = FyersConfig
        self.token_file = self.config.TOKEN_FILE
        self.access_token = None
        self.session = None
        
    def generate_hash(self, string: str) -> str:
        """Generate SHA256 hash for API v3"""
        return hashlib.sha256(string.encode()).hexdigest()
    
    def get_auth_url(self) -> Tuple[str, str]:
        """Generate authentication URL for API v3"""
        self.config.validate_config()
        
        # Step 1: Generate request key
        payload = {
            "fyers_id": self.config.CLIENT_ID,
            "app_id": self.config.APP_ID,
            "redirect_uri": self.config.REDIRECT_URI,
            "appType": self.config.APP_TYPE,
            "code_challenge": "",
            "state": "state",
            "scope": "",
            "nonce": "",
            "response_type": "code",
            "create_cookie": True
        }
        
        try:
            response = requests.post(
                f"{self.config.AUTH_URL}/send_login_otp_v2",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("s") == "ok":
                    request_key = data.get("request_key")
                    
                    # Generate auth URL
                    auth_url = f"https://api-t2.fyers.in/vagator/v2/verify_otp?request_key={request_key}"
                    return auth_url, request_key
                else:
                    print(f"Error getting request key: {data.get('message', 'Unknown error')}")
                    return None, None
            else:
                print(f"HTTP Error: {response.status_code}")
                return None, None
                
        except Exception as e:
            print(f"Auth URL generation error: {e}")
            return None, None
    
    def verify_otp(self, request_key: str, otp: str) -> Optional[Dict]:
        """Verify OTP for API v3"""
        try:
            payload = {
                "request_key": request_key,
                "otp": self.generate_hash(otp)
            }
            
            response = requests.post(
                f"{self.config.AUTH_URL}/verify_otp_v2",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("s") == "ok":
                    access_token = data.get("access_token")
                    
                    # Now get auth code
                    auth_code = self._get_auth_code(access_token)
                    if auth_code:
                        return self.validate_auth_code(auth_code)
                    
                else:
                    print(f"OTP verification error: {data.get('message', 'Unknown error')}")
                    return None
            else:
                print(f"HTTP Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"OTP verification error: {e}")
            return None
    
    def _get_auth_code(self, access_token: str) -> Optional[str]:
        """Get auth code using access token"""
        try:
            payload = {
                "fyers_id": self.config.CLIENT_ID,
                "app_id": self.config.APP_ID,
                "redirect_uri": self.config.REDIRECT_URI,
                "appType": self.config.APP_TYPE,
                "code_challenge": "",
                "state": "state",
                "scope": "",
                "nonce": "",
                "response_type": "code",
                "create_cookie": True
            }
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.config.AUTH_URL}/generate_authcode",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("s") == "ok":
                    return data.get("auth_code")
                else:
                    print(f"Auth code error: {data.get('message', 'Unknown error')}")
                    return None
            else:
                print(f"HTTP Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Auth code generation error: {e}")
            return None
    
    def validate_auth_code(self, auth_code: str) -> Optional[Dict]:
        """Validate auth code and get access token (API v3)"""
        try:
            payload = {
                "grant_type": "authorization_code",
                "appIdHash": self.generate_hash(f"{self.config.CLIENT_ID}-{self.config.APP_TYPE}"),
                "code": auth_code
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.config.TOKEN_URL,
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                token_data = response.json()
                
                if token_data.get("s") == "ok":
                    # Save token
                    self._save_token(token_data)
                    self.access_token = token_data["access_token"]
                    return token_data
                else:
                    print(f"Error from Fyers: {token_data.get('message', 'Unknown error')}")
                    return None
            else:
                print(f"HTTP Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Token validation error: {e}")
            return None
    
    def _save_token(self, token_data: Dict):
        """Save token data to file"""
        token_data["timestamp"] = time.time()
        token_data["client_id"] = self.config.CLIENT_ID
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
        
        with open(self.token_file, 'w') as f:
            json.dump(token_data, f, indent=2)
        print(f"✅ Token saved to: {self.token_file}")
    
    def _load_token(self) -> Optional[Dict]:
        """Load token from file"""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'r') as f:
                    token_data = json.load(f)
                
                # Check if token is for same client
                if token_data.get("client_id") == self.config.CLIENT_ID:
                    return token_data
                else:
                    print("⚠️ Token client ID mismatch")
                    return None
                    
            except Exception as e:
                print(f"Error loading token: {e}")
                return None
        return None
    
    def get_access_token(self) -> Optional[str]:
        """Get valid access token"""
        # Try to load from file first
        token_data = self._load_token()
        
        if token_data:
            # Check if token is expired (24 hours validity)
            token_time = token_data.get("timestamp", 0)
            current_time = time.time()
            
            # Token expires in 24 hours (86400 seconds)
            if (current_time - token_time) < 86400:
                self.access_token = token_data["access_token"]
                return self.access_token
        
        return None
    
    def create_session(self) -> Optional[requests.Session]:
        """Create authenticated session for API v3"""
        access_token = self.get_access_token()
        
        if not access_token:
            print("No valid access token available. Please authenticate first.")
            return None
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"{self.config.CLIENT_ID}:{access_token}",
            "Content-Type": "application/json"
        })
        
        return self.session
    
    def is_authenticated(self) -> bool:
        """Check if authentication is valid"""
        return self.get_access_token() is not None