# test_fyers_v3.py
import sys
sys.path.insert(0, '.')

print("🔐 Testing Fyers API v3")
print("=" * 50)

try:
    from financial.auth.auth_helper import setup_fyers_auth
    
    print("1. Setting up authentication...")
    token = setup_fyers_auth()
    
    if token:
        print(f"\n✅ Success! Token: {token[:30]}...")
        
        # Test with pipeline
        print("\n2. Testing with market pipeline...")
        from financial.data.minimal_pipeline import MinimalMarketPipeline
        
        pipeline = MinimalMarketPipeline()
        print(f"Sources: {pipeline.sources}")
        
        if "fyers" in pipeline.sources:
            print("\n3. Fetching NIFTY data...")
            data = pipeline.fetch_fyers("NIFTY50-INDEX")
            if data:
                print(f"✅ NIFTY50: ₹{data['latest_price']:.2f}")
                print(f"   Source: {data['source']}")
                print(f"   Records: {len(data['data'])}")
            else:
                print("❌ Failed to fetch data")
        else:
            print("⚠️ Fyers not in sources")
            
    else:
        print("❌ Authentication failed")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()