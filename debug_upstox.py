# debug_upstox.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from financial.auth.upstox_auth import get_upstox_client
import upstox_client

print("🔍 Debugging Upstox API Call")
print("="*60)

client = get_upstox_client()
if client:
    print("✅ Client created")
    
    market_api = upstox_client.MarketQuoteApi(client)
    
    # Test with RELIANCE ISIN
    symbol = 'NSE_EQ|INE002A01018'
    
    try:
        print(f"Calling API with symbol: {symbol}")
        response = market_api.get_full_market_quote(
            symbol=symbol,
            api_version='2.0'
        )
        
        print(f"✅ Response type: {type(response)}")
        print(f"Response: {response}")
        
        # Check if it's actually a string
        if isinstance(response, str):
            print("⚠️ API returned a string, not an object!")
            print(f"String content: {response[:100]}")
        else:
            print(f"✅ Got proper response object")
            if hasattr(response, 'data'):
                print(f"Response has data attribute")
                
    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}")
        print(f"Message: {str(e)[:200]}")
else:
    print("❌ No client available")