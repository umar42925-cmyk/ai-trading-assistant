# test_proper_symbols.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from financial.data.minimal_pipeline import MinimalMarketPipeline

print("🔍 Testing with Proper Symbol Formats")
print("="*60)

pipeline = MinimalMarketPipeline()

# Test with properly formatted symbols
test_cases = [
    ("RELIANCE.NS", "Indian stock with .NS"),
    ("TCS.NS", "TCS with .NS"),
    ("^NSEI", "Nifty 50 (Yahoo format)"),
    ("^GSPC", "S&P 500 (with ^)"),
    ("AAPL", "Apple"),
    ("INFY.NS", "Infosys"),
]

for symbol, desc in test_cases:
    print(f"\n📊 {symbol} ({desc}):")
    try:
        data = pipeline.fetch_market_data(symbol, source="auto", interval="1d")
        if data:
            print(f"   ✅ {data.get('source')}: {data.get('latest_price', 'N/A'):.2f}")
            print(f"   Records: {len(data.get('data', []))}")
        else:
            print(f"   ❌ Failed")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("💡 Use .NS suffix for Indian stocks on Yahoo Finance")
print("💡 Use ^ prefix for indices (^NSEI, ^GSPC)")