# test_yahoo_symbols.py
import yfinance as yf

print("🔍 Testing Yahoo Finance Symbol Formats")
print("="*60)

test_symbols = [
    ("^NSEI", "Nifty 50"),
    ("^BSESN", "Sensex"),
    ("^GSPC", "S&P 500"),
    ("RELIANCE.NS", "Reliance with .NS"),
    ("TCS.NS", "TCS with .NS"),
    ("AAPL", "Apple"),
    ("INFY.NS", "Infosys"),
]

for symbol, name in test_symbols:
    print(f"\n📊 {symbol} ({name}):")
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        history = ticker.history(period="1d")
        
        if not history.empty:
            print(f"   ✅ Works! Last price: {history['Close'].iloc[-1]:.2f}")
            print(f"   Company: {info.get('longName', 'Unknown')[:30]}...")
        else:
            print(f"   ❌ No data")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:50]}...")