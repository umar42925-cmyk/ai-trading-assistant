"""
Upstox Authentication for Streamlit - FIXED for upstox-python-sdk
"""

import os
import json
import webbrowser
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import streamlit as st

try:
    import upstox_client
    from upstox_client.rest import ApiException
    UPSTOX_AVAILABLE = True
except ImportError:
    UPSTOX_AVAILABLE = False
    print("⚠️ upstox-python-sdk not installed. Run: pip install upstox-python-sdk")


class UpstoxAuth:
    """Upstox authentication for API v2"""
    
    def __init__(self):
        if not UPSTOX_AVAILABLE:
            return
            
        self.api_key = os.getenv("UPSTOX_API_KEY", "")
        self.api_secret = os.getenv("UPSTOX_API_SECRET", "")
        self.redirect_uri = os.getenv("UPSTOX_REDIRECT_URI", "http://localhost:8501")
        
        self.token_file = os.path.join(os.path.dirname(__file__), "upstox_token.json")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
        
        self._access_token = None
        self._load_token()
    
    def _load_token(self) -> bool:
        """Load token from file"""
        if not os.path.exists(self.token_file):
            return False
        
        try:
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)
            
            access_token = token_data.get('access_token')
            expires_at = token_data.get('expires_at')
            
            if access_token and expires_at:
                # Check if token is still valid
                if datetime.fromisoformat(expires_at) > datetime.now():
                    self._access_token = access_token
                    print("✅ Upstox token loaded from file")
                    return True
                else:
                    print("⚠️ Upstox token expired")
            
            return False
            
        except Exception as e:
            print(f"Error loading token: {e}")
            return False
    
    def _save_token(self, token_data: Dict):
        """Save token to file"""
        try:
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
            print("✅ Upstox token saved")
        except Exception as e:
            print(f"Error saving token: {e}")
    
    def get_login_url(self) -> Optional[str]:
        """Get login URL for user authentication"""
        if not self.api_key:
            return None
        
        # Upstox v2 OAuth URL
        return f"https://api.upstox.com/v2/login/authorization/dialog?client_id={self.api_key}&redirect_uri={self.redirect_uri}&response_type=code"
    
    def authenticate(self, code: str) -> Tuple[bool, str]:
        """Authenticate using authorization code"""
        if not UPSTOX_AVAILABLE:
            return False, "Upstox SDK not installed"
        
        try:
            configuration = upstox_client.Configuration()
            api_instance = upstox_client.LoginApi(upstox_client.ApiClient(configuration))
            
            api_version = '2.0'
            
            # Exchange code for token
            response = api_instance.token(
                api_version=api_version,
                code=code,
                client_id=self.api_key,
                client_secret=self.api_secret,
                redirect_uri=self.redirect_uri,
                grant_type='authorization_code'
            )
            
            # Save token data
            token_data = {
                'access_token': response.access_token,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
            }
            
            self._save_token(token_data)
            self._access_token = response.access_token
            
            return True, "✅ Authentication successful! You can now access live Indian market data."
            
        except ApiException as e:
            error_body = e.body if hasattr(e, 'body') else str(e)
            
            if "invalid_grant" in str(error_body).lower():
                return False, "❌ Invalid or expired authorization code. Please generate a new one."
            elif "invalid_client" in str(error_body).lower():
                return False, "❌ Invalid API credentials. Check your UPSTOX_API_KEY and UPSTOX_API_SECRET in .env"
            else:
                return False, f"❌ Upstox API error: {str(error_body)[:150]}"
                
        except Exception as e:
            return False, f"❌ Authentication error: {str(e)[:150]}"
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return self._access_token is not None
    
    def get_access_token(self) -> Optional[str]:
        """Get current access token"""
        return self._access_token
    
    def logout(self):
        """Logout and clear token"""
        try:
            if os.path.exists(self.token_file):
                os.remove(self.token_file)
            self._access_token = None
            print("✅ Upstox token cleared")
        except Exception as e:
            print(f"Error during logout: {e}")
    
    def get_client(self):
        """Get authenticated Upstox client"""
        if not self._access_token:
            return None
        
        configuration = upstox_client.Configuration()
        configuration.access_token = self._access_token
        
        return upstox_client.ApiClient(configuration)


def setup_upstox_for_streamlit():
    """
    Setup Upstox for your Streamlit app - called from main.py
    Returns access token if authenticated, None otherwise
    """
    if not UPSTOX_AVAILABLE:
        print("⚠️ Upstox SDK not available")
        return None
    
    auth = UpstoxAuth()
    
    if auth.is_authenticated():
        print("✅ Upstox authenticated")
        return auth.get_access_token()
    else:
        print("ℹ️ Upstox not authenticated. Use UI to authenticate.")
        return None


def upstox_auth_flow():
    """
    Complete Upstox authentication flow for Streamlit UI
    """
    if not UPSTOX_AVAILABLE:
        st.error("""
        ## ❌ Upstox SDK Not Installed
        
        Please install with:
        ```bash
        pip install upstox-python-sdk
        ```
        
        Then restart your app.
        """)
        return False
    
    # Check environment variables
    api_key = os.getenv("UPSTOX_API_KEY", "")
    api_secret = os.getenv("UPSTOX_API_SECRET", "")
    
    if not api_key or not api_secret:
        st.error("""
        ## 🔧 Missing Upstox API Credentials
        
        Create or edit `.env` file in your project root:
        
        ```env
        # Upstox Configuration
        UPSTOX_API_KEY=your_api_key_here
        UPSTOX_API_SECRET=your_api_secret_here
        UPSTOX_REDIRECT_URI=http://localhost:8501
        ```
        
        **Get API credentials:**
        1. Go to https://api.upstox.com/
        2. Login with your Upstox account
        3. Create a new app
        4. Set Redirect URI: `http://localhost:8501`
        5. Copy API Key and Secret
        """)
        return False
    
    # Initialize auth
    auth = UpstoxAuth()
    
    # Check if already authenticated
    if auth.is_authenticated():
        st.success("✅ **Upstox Authenticated**")
        
        # Show account info if possible
        try:
            client = auth.get_client()
            user_api = upstox_client.UserApi(client)
            profile = user_api.get_profile('2.0')
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"👤 **Name:** {profile.data.user_name}")
                st.info(f"📧 **Email:** {profile.data.email}")
            with col2:
                if st.button("🚪 Logout", type="secondary"):
                    auth.logout()
                    st.rerun()
        except Exception as e:
            st.warning(f"Token valid but couldn't fetch profile: {str(e)[:50]}")
            if st.button("🚪 Logout"):
                auth.logout()
                st.rerun()
        
        return True
    
    # --- NOT AUTHENTICATED - Show auth flow ---
    st.markdown("## 🔐 Upstox Authentication")
    
    # Check for auth code in URL query params
    query_params = st.query_params
    
    if 'code' in query_params:
        auth_code = query_params['code']
        
        with st.spinner("🔄 Authenticating with Upstox..."):
            success, message = auth.authenticate(auth_code)
            
            if success:
                st.success(message)
                st.balloons()
                
                # Clear URL params
                st.query_params.clear()
                
                # Force reload
                st.rerun()
            else:
                st.error(message)
                
                # Show try again button
                if st.button("🔄 Try Again"):
                    st.query_params.clear()
                    st.rerun()
    else:
        # Show login instructions
        st.markdown("""
        ### 📱 Login to Access Live Market Data
        
        Click the button below to authenticate with your Upstox account.
        
        **What you'll get:**
        - Live Indian stock quotes (NSE/BSE)
        - NIFTY & BANKNIFTY indices
        - Real-time portfolio data
        - Order placement capabilities
        """)
        
        login_url = auth.get_login_url()
        
        if login_url:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Create clickable button using link
                st.markdown(f"""
                <a href="{login_url}" target="_self">
                    <button style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border: none;
                        color: white;
                        padding: 15px 32px;
                        text-align: center;
                        font-size: 16px;
                        font-weight: 600;
                        border-radius: 8px;
                        cursor: pointer;
                        width: 100%;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    ">
                        🔐 Login with Upstox
                    </button>
                </a>
                """, unsafe_allow_html=True)
            
            with col2:
                st.info("💡 You'll be redirected to Upstox. After login, you'll return here automatically.")
        
        # Troubleshooting
        with st.expander("🛠️ Troubleshooting", expanded=False):
            st.markdown("""
            ### Common Issues
            
            **1. "Invalid authorization code"**
            - Codes expire after 5 minutes
            - Click the login button again to get a new code
            
            **2. "Invalid client credentials"**
            - Check your `.env` file has correct `UPSTOX_API_KEY` and `UPSTOX_API_SECRET`
            - Verify they match your Upstox Developer Console
            
            **3. "Redirect URI mismatch"**
            - In Upstox Developer Console, set redirect URI to: `http://localhost:8501`
            - In `.env`, set: `UPSTOX_REDIRECT_URI=http://localhost:8501`
            - Both must match exactly!
            
            **4. Blank page after login**
            - This is normal! The page will auto-redirect
            - If stuck, check browser console (F12) for errors
            
            **5. Still not working?**
            - Restart your Streamlit app after editing `.env`
            - Clear browser cache
            - Try a different browser
            """)
    
    return False


def check_upstox_auth() -> bool:
    """
    Quick check if Upstox is authenticated
    """
    if not UPSTOX_AVAILABLE:
        return False
    
    try:
        auth = UpstoxAuth()
        return auth.is_authenticated()
    except:
        return False


def get_upstox_client():
    """
    Get authenticated Upstox client for data fetching
    """
    if not UPSTOX_AVAILABLE:
        return None
    
    try:
        auth = UpstoxAuth()
        if auth.is_authenticated():
            return auth.get_client()
    except:
        pass
    
    return None