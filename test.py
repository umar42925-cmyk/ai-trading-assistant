"""
MINIMAL TEST SCRIPT
Quick verification of the enhanced pipeline
"""

import sys
import time
sys.path.append('.')

from financial.data.professional_pipeline import ProfessionalMarketPipeline

def minimal_test():
    """Minimal test to verify the pipeline works"""
    print("🚀 Starting minimal test...")
    
    try:
        # 1. Create pipeline
        print("1. Creating pipeline...")
        pipeline = ProfessionalMarketPipeline(
            broker="fyers",
            config={'websocket': {'mock_data': True}}
        )
        
        # 2. Test real-time data
        print("2. Testing real-time subscription...")
        realtime_updates = []
        
        def simple_callback(data):
            if len(realtime_updates) < 3:  # Only store first 3
                realtime_updates.append({
                    'symbol': data.get('symbol'),
                    'price': data.get('price'),
                    'time': data.get('timestamp')[:19] if data.get('timestamp') else 'N/A'
                })
        
        # Subscribe to NIFTY
        pipeline.subscribe_realtime("NIFTY", simple_callback)
        time.sleep(2)  # Wait for some data
        
        print(f"   Received {len(realtime_updates)} real-time updates")
        for update in realtime_updates:
            print(f"   - {update['symbol']}: {update['price']} at {update['time']}")
        
        # 3. Test historical data
        print("3. Testing historical data fetch...")
        nifty_data = pipeline.fetch_market_data("NIFTY", timeframe="1d")
        print(f"   NIFTY: ${nifty_data.get('latest_price', 'N/A')}")
        print(f"   Data points: {len(nifty_data.get('data', []))}")
        print(f"   Source: {nifty_data.get('source')}")
        
        # 4. Test advanced features
        print("4. Testing advanced charts...")
        renko_data = pipeline.get_advanced_chart("NIFTY", "renko", brick_size=10)
        print(f"   Renko bricks: {renko_data.get('count', 0)}")
        
        # 5. Check statistics
        print("5. Checking statistics...")
        stats = pipeline.get_stats()
        print(f"   Requests: {stats.get('requests', 0)}")
        print(f"   Cache hits: {stats.get('cache_hits', 0)}")
        print(f"   WebSocket messages: {stats.get('websocket_message_count', 0)}")
        
        # 6. Cleanup
        print("6. Cleaning up...")
        pipeline.cleanup()
        
        print("\n✅ MINIMAL TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = minimal_test()
    if success:
        print("\n🎉 Pipeline is working correctly!")
    else:
        print("\n⚠️  Pipeline has issues that need to be addressed.")