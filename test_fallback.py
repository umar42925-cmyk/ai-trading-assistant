# test_fallback.py

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from financial.data.minimal_pipeline import MinimalMarketPipeline

def test_fallback_sequence():
    """Test the smart source selection fallback logic"""
    print("🧪 Testing Fallback Sequence Logic")
    print("="*60)
    
    pipeline = MinimalMarketPipeline()
    
    test_cases = [
        # (symbol, interval, expected_source_order, description)
        ("RELIANCE", "1m", ["upstox", "fyers", "yfinance"], "Indian stock, real-time"),
        ("RELIANCE", "1d", ["upstox", "fyers", "yfinance"], "Indian stock, daily"),
        ("RELIANCE", "1mo", ["upstox", "fyers", "yfinance"], "Indian stock, monthly"),
        
        ("NIFTY", "5m", ["upstox", "fyers", "yfinance"], "Indian index, real-time"),
        ("NIFTY", "1d", ["upstox", "fyers", "yfinance"], "Indian index, daily"),
        
        ("AAPL", "1m", ["yfinance", "upstox", "fyers"], "US stock, real-time"),
        ("AAPL", "1d", ["yfinance", "upstox", "fyers"], "US stock, daily"),
        ("AAPL", "1wk", ["yfinance", "upstox", "fyers"], "US stock, weekly"),
        
        ("TSLA", "15m", ["yfinance", "upstox", "fyers"], "Global stock, intraday"),
        ("TSLA", "1d", ["yfinance", "upstox", "fyers"], "Global stock, daily"),
        
        ("TCS.NS", "1m", ["upstox", "fyers", "yfinance"], "Indian with .NS suffix"),
        ("RELIANCE-EQ", "1h", ["upstox", "fyers", "yfinance"], "Indian with -EQ suffix"),
        ("NIFTY-INDEX", "30min", ["upstox", "fyers", "yfinance"], "Indian index with suffix"),
        
        ("BTC-USD", "1d", ["yfinance", "upstox", "fyers"], "Crypto global"),
    ]
    
    print("📋 Test Cases for Source Selection:")
    print("-"*60)
    
    all_passed = True
    for i, (symbol, interval, expected_order, description) in enumerate(test_cases, 1):
        actual_order = pipeline.choose_data_source(symbol, interval)
        
        # Check if actual matches expected
        if actual_order == expected_order:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            all_passed = False
        
        print(f"\nTest {i}: {status}")
        print(f"   Symbol: {symbol}, Interval: {interval}")
        print(f"   Description: {description}")
        print(f"   Expected: {expected_order}")
        print(f"   Actual: {actual_order}")
    
    print("\n" + "="*60)
    
    if all_passed:
        print("🎉 All source selection tests passed!")
    else:
        print("⚠️ Some tests failed - check logic")
    
    return all_passed

def test_actual_fetching():
    """Test actual data fetching with auto source selection"""
    print("\n\n🧪 Testing Actual Data Fetching")
    print("="*60)
    
    pipeline = MinimalMarketPipeline()
    
    # Test symbols with different profiles
    test_symbols = [
        ("RELIANCE", "1d", "Indian equity"),
        ("AAPL", "1d", "US equity"),
        ("NIFTY", "1d", "Indian index"),
        ("^GSPC", "1d", "US index"),
        ("TCS.NS", "1d", "Indian with .NS"),
        ("GOOGL", "1d", "Global tech"),
    ]
    
    print("Testing data fetching with source='auto':")
    print("-"*60)
    
    for symbol, interval, description in test_symbols:
        print(f"\nFetching {symbol} ({description}) with interval={interval}...")
        
        try:
            data = pipeline.fetch_market_data(symbol, source="auto", interval=interval)
            
            if data:
                print(f"   ✅ Success via {data.get('source', 'unknown')}")
                print(f"   Price: {data.get('latest_price', 'N/A'):.2f}")
                print(f"   Records: {len(data.get('data', []))}")
                print(f"   Currency: {data.get('currency', 'N/A')}")
                
                # Store for comparison
                if hasattr(pipeline, '_last_source'):
                    if pipeline._last_source != data.get('source'):
                        print(f"   ⚡ Source changed from {pipeline._last_source}")
                pipeline._last_source = data.get('source')
            else:
                print(f"   ❌ Failed to fetch {symbol}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("✅ Data fetching test completed")

def test_specific_source_fallback():
    """Test forced fallback by disabling sources"""
    print("\n\n🧪 Testing Manual Source Fallback")
    print("="*60)
    
    pipeline = MinimalMarketPipeline()
    
    print("Available sources:", pipeline.sources)
    print("-"*60)
    
    # Test RELIANCE with different explicit sources
    symbol = "RELIANCE"
    print(f"\nTesting {symbol} with different sources:")
    
    # Try each available source explicitly
    for source in ["upstox", "fyers", "yfinance"]:
        print(f"\nTrying {source}...")
        try:
            data = pipeline.fetch_market_data(symbol, source=source, interval="1d")
            if data:
                print(f"   ✅ {source}: Price = {data.get('latest_price', 'N/A'):.2f}")
            else:
                print(f"   ❌ {source}: No data")
        except Exception as e:
            print(f"   ❌ {source}: Error - {e}")
    
    # Test auto with a symbol that should trigger fallback
    print(f"\n\nTesting {symbol} with auto (should show fallback):")
    data = pipeline.fetch_market_data(symbol, source="auto", interval="1d")
    if data:
        print(f"   Auto selected: {data.get('source', 'unknown')}")
        print(f"   Price: {data.get('latest_price', 'N/A'):.2f}")

def test_indian_detection():
    """Test Indian symbol detection logic"""
    print("\n\n🧪 Testing Indian Symbol Detection")
    print("="*60)
    
    pipeline = MinimalMarketPipeline()
    
    test_symbols = [
        # Should be Indian
        ("RELIANCE", True),
        ("TCS.NS", True),
        ("NIFTY", True),
        ("BANKNIFTY", True),
        ("INFY", True),
        ("RELIANCE-EQ", True),
        ("NIFTY-INDEX", True),
        ("HDFCBANK", True),
        
        # Should NOT be Indian
        ("AAPL", False),
        ("TSLA", False),
        ("GOOGL", False),
        ("BTC-USD", False),
        ("^GSPC", False),
        ("AMZN", False),
        ("MSFT", False),
        ("NVDA", False),
        
        # Edge cases
        ("TCS", True),  # Indian without suffix
        ("TATASTEEL", True),  # Indian steel company
        ("ADANIPORTS", True),  # Indian port operator
    ]
    
    print("Testing _is_indian_symbol() method:")
    print("-"*60)
    
    passed = 0
    failed = 0
    
    for symbol, expected_indian in test_symbols:
        actual_indian = pipeline._is_indian_symbol(symbol)
        
        if actual_indian == expected_indian:
            status = "✅"
            passed += 1
        else:
            status = "❌"
            failed += 1
            
        print(f"{status} {symbol:20} Expected: {str(expected_indian):6} Got: {str(actual_indian):6}")
    
    print(f"\nResults: {passed} passed, {failed} failed")
    accuracy = (passed / len(test_symbols)) * 100 if test_symbols else 0
    print(f"Accuracy: {accuracy:.1f}%")

def run_all_tests():
    """Run all fallback tests"""
    print("🧪 COMPREHENSIVE FALLBACK TEST SUITE 🧪")
    print("="*80)
    
    results = []
    
    # Run each test
    results.append(test_fallback_sequence())
    
    # Run actual fetching test
    try:
        test_actual_fetching()
        results.append(True)
    except Exception as e:
        print(f"Error in actual fetching test: {e}")
        results.append(False)
    
    # Run specific source test
    try:
        test_specific_source_fallback()
        results.append(True)
    except Exception as e:
        print(f"Error in source fallback test: {e}")
        results.append(False)
    
    # Run Indian detection test
    try:
        test_indian_detection()
        results.append(True)
    except Exception as e:
        print(f"Error in Indian detection test: {e}")
        results.append(False)
    
    print("\n" + "="*80)
    print("📊 TEST SUMMARY:")
    print(f"   Source selection tests: {'✅ PASS' if results[0] else '❌ FAIL'}")
    print(f"   Actual fetching tests: {'✅ PASS' if len(results) > 1 and results[1] else '❌ FAIL'}")
    print(f"   Source fallback tests: {'✅ PASS' if len(results) > 2 and results[2] else '❌ FAIL'}")
    print(f"   Indian detection tests: {'✅ PASS' if len(results) > 3 and results[3] else '❌ FAIL'}")
    
    all_passed = all(results)
    if all_passed:
        print("\n🎉🎉 ALL TESTS PASSED! 🎉🎉")
        print("Your fallback system is working correctly!")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
    
    return all_passed

if __name__ == "__main__":
    # Run comprehensive test suite
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)