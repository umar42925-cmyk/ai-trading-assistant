# financial/auth/auth_helper.py
import webbrowser
import getpass
from .fyers_auth import FyersAuth

def setup_fyers_auth():
    """Interactive Fyers authentication setup for API v3"""
    auth = FyersAuth()
    
    print("🔐 Setting up Fyers Authentication (API v3)...")
    print(f"Client ID: {auth.config.CLIENT_ID}")
    
    # Check if already authenticated
    if auth.is_authenticated():
        print("✅ Already authenticated with Fyers")
        return auth.get_access_token()
    
    # Step 1: Get auth URL and request key
    auth_url, request_key = auth.get_auth_url()
    
    if not auth_url or not request_key:
        print("❌ Failed to generate authentication URL")
        return None
    
    print(f"\n📋 Open this URL in your browser to authenticate:")
    print(f"🔗 {auth_url}")
    
    # Try to open browser automatically
    try:
        webbrowser.open(auth_url)
        print("🌐 Browser opened for authentication...")
    except:
        print("⚠️ Could not open browser automatically. Please copy the URL above.")
    
    print("\n📱 After logging in, you'll receive an OTP on your registered mobile/email.")
    
    # Step 2: Get OTP from user
    otp = getpass.getpass("\n📝 Enter the OTP you received: ").strip()
    
    if not otp:
        print("❌ No OTP provided")
        return None
    
    # Step 3: Verify OTP and get token
    print("🔐 Verifying OTP...")
    token_data = auth.verify_otp(request_key, otp)
    
    if token_data:
        print(f"✅ Authentication successful!")
        print(f"📊 Token valid until: {token_data.get('expires_in', 'Unknown')}")
        
        # Get access token
        access_token = auth.get_access_token()
        if access_token:
            print(f"🔑 Access token: {access_token[:30]}...")
        
        return access_token
    else:
        print("❌ Authentication failed")
        print("\n💡 Troubleshooting:")
        print("1. Make sure OTP is correct and entered within 60 seconds")
        print("2. Check your Fyers account is active")
        print("3. Verify client ID in .env file")
        return None

def get_fyers_client():
    """Get authenticated Fyers client"""
    auth = FyersAuth()
    
    if not auth.is_authenticated():
        print("⚠️ Fyers not authenticated. Please run setup_fyers_auth() first.")
        return None
    
    session = auth.create_session()
    if session:
        print("✅ Fyers client ready (API v3)")
        return session
    
    return None

# Quick test function
def quick_test():
    """Quick test of Fyers authentication"""
    print("⚡ Quick Fyers Test")
    print("=" * 50)
    
    auth = FyersAuth()
    
    if auth.is_authenticated():
        print("✅ Already authenticated")
        
        # Try to create session
        session = auth.create_session()
        if session:
            print("✅ Session created successfully")
            
            # Test a simple API call
            try:
                response = session.get("https://api.fyers.in/data-rest/v2/profile")
                if response.status_code == 200:
                    print("✅ Profile API test successful")
                else:
                    print(f"⚠️ Profile API returned: {response.status_code}")
            except Exception as e:
                print(f"⚠️ API test error: {e}")
    else:
        print("❌ Not authenticated")
        print("💡 Run setup_fyers_auth() to authenticate")

if __name__ == "__main__":
    quick_test()