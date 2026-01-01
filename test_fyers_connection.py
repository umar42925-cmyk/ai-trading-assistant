def test_fyers_connection():
    print("Testing Fyers API connectivity...")
    
    # CORRECT endpoints for v3
    endpoints = [
        "https://api.fyers.in/api/v3",
        "https://api-t1.fyers.in/api/v3",
        "https://api-t2.fyers.in/api/v3"
    ]
    
    for url in endpoints:
        try:
            start = time.time()
            response = requests.get(url, timeout=15)
            elapsed = time.time() - start
            print(f"✓ {url} - Status: {response.status_code} - Time: {elapsed:.2f}s")
            if response.status_code == 200:
                print(f"   Response: {response.text[:200]}")  # Show first 200 chars
        except requests.exceptions.Timeout:
            print(f"✗ {url} - TIMEOUT (15s)")
        except Exception as e:
            print(f"✗ {url} - Error: {type(e).__name__}")

test_fyers_connection()