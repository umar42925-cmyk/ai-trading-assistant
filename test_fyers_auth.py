# test_fyers_auth.py
import sys
import os
sys.path.insert(0, '.')

print("🔐 Testing Fyers Authentication & Data Fetch")
print("=" * 50)

try:
    # 1. Test authentication
    from financial.auth.auth_helper import setup_fyers_auth
    print("\n1. Testing Authentication...")
    
    token = setup_fyers_auth()
    if token:
        print(f"✅ Authentication successful!")
        print(f"📋 Token: {token[:30]}...")
    else:
        print("❌ Authentication failed")
        print("💡 Please check your .env file has FYERS_CLIENT_ID and FYERS_SECRET_KEY")
        
except ImportError as e:
    print(f"❌ Could not import auth_helper: {e}")
    print("💡 Make sure financial/auth/auth_helper.py exists")

print("\n" + "=" * 50)
print("2. Testing Market Pipeline with Fyers...")

try:
    # 2. Test market pipeline
    from financial.data.minimal_pipeline import MinimalMarketPipeline
    
    pipeline = MinimalMarketPipeline()
    print(f"📊 Available sources: {pipeline.sources}")
    
    # Test Indian symbols (should use Fyers if authenticated)
    test_symbols = [
        "NIFTY50-INDEX",        # Nifty 50 Index
        "RELIANCE-EQ",          # Reliance Industries
        "NSE:TCS-EQ",           # TCS
        "BANKNIFTY-INDEX",      # Bank Nifty
    ]
    
    for symbol in test_symbols:
        print(f"\n📈 Testing {symbol}...")
        
        # Try Fyers first
        if "fyers" in pipeline.sources:
            data = pipeline.fetch_fyers(symbol)
            if data:
                print(f"   ✅ Fyers: ₹{data['latest_price']:.2f} ({len(data['data'])} records)")
            else:
                print(f"   ❌ Fyers fetch failed")
        
        # Also test auto mode
        data_auto = pipeline.fetch_market_data(symbol, source="auto")
        if data_auto:
            print(f"   🔄 Auto ({data_auto['source']}): ₹{data_auto['latest_price']:.2f}")
        
except Exception as e:
    print(f"❌ Pipeline test error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("3. Testing Fyers Token Storage...")

# Check if token was saved
token_files = [
    "financial/auth/fyers_token.json",
    "fyers_token.txt",
    "financial/auth/fyers_token.txt"
]

for token_file in token_files:
    if os.path.exists(token_file):
        print(f"✅ Token file found: {token_file}")
        try:
            with open(token_file, 'r') as f:
                content = f.read()
                print(f"   📄 Size: {len(content)} bytes")
                if len(content) < 100:
                    print(f"   📝 Content: {content}")
                else:
                    print(f"   📝 Content preview: {content[:100]}...")
        except Exception as e:
            print(f"   ⚠️ Error reading: {e}")
        break
else:
    print("⚠️ No token file found")

print("\n" + "=" * 50)
print("✅ Test Complete!")