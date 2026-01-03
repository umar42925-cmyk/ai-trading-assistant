def quick_test():
    """Quick test of basic functionality"""
    print("🚀 Quick Test of Professional Pipeline")
    print("="*60)
    
    pipeline = ProfessionalMarketPipeline()
    
    # Test with known working symbols
    test_symbols = [
        ("AAPL", "Global stock"),
        ("NIFTY", "Indian index"),
        ("INFY.NS", "Indian stock with suffix"),
        ("^GSPC", "S&P 500 index")
    ]
    
    for symbol, description in test_symbols:
        print(f"\n🔍 Testing {symbol} ({description}):")
        try:
            data = pipeline.fetch_market_data(
                symbol=symbol,
                interval="1d",
                source="yfinance",  # Force yfinance
                max_retries=1,
                validate=True
            )
            if data:
                print(f"   ✅ Success: {data.record_count} records")
                print(f"   Quality: {data.quality.score:.1f}/5")
                print(f"   Latest: {data.latest_price:.2f}")
            else:
                print(f"   ❌ No data returned")
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}")
    
    print("\n" + "="*60)
    print("✅ Quick test complete!")

if __name__ == "__main__":
    quick_test()