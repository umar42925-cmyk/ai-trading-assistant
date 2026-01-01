# quick_fyers_test.py
import sys
sys.path.insert(0, '.')

print("⚡ Quick Fyers Test")

# Test 1: Just check if we can create the pipeline
try:
    from financial.data.minimal_pipeline import MinimalMarketPipeline
    pipeline = MinimalMarketPipeline()
    
    print(f"Sources: {pipeline.sources}")
    
    # Quick NIFTY test
    if "fyers" in pipeline.sources:
        print("\n🔍 Checking Fyers token...")
        token = pipeline._get_fyers_token()
        if token:
            print(f"✅ Fyers token found: {token[:20]}...")
            
            # Try to fetch NIFTY
            print("\n📊 Fetching NIFTY50...")
            data = pipeline.fetch_fyers("NIFTY50-INDEX")
            if data:
                print(f"✅ Success! NIFTY: ₹{data['latest_price']:.2f}")
                print(f"   Source: {data['source']}")
                print(f"   Records: {len(data['data'])}")
            else:
                print("❌ Fetch failed - might need authentication")
                print("💡 Run: python -c \"from financial.auth.auth_helper import setup_fyers_auth; setup_fyers_auth()\"")
        else:
            print("❌ No Fyers token found")
            print("💡 You need to authenticate first")
    
    # Test Yahoo as fallback
    print("\n🔄 Testing Yahoo Finance fallback...")
    yahoo_data = pipeline.fetch_yfinance("AAPL")
    if yahoo_data:
        print(f"✅ Yahoo: AAPL ${yahoo_data['latest_price']:.2f}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()