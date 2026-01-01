# test_app_functionality.py
import sys
import os
sys.path.insert(0, os.getcwd())

from main import process_user_input

print("Testing basic app functionality...")

# Test 1: Simple greeting
print("\n1. Testing greeting:")
result = process_user_input("Hello")
print(f"   Response: {result['response'][:50]}...")
print(f"   Status: {result['status']}")
print(f"   Mode: {result['mode']}")

# Test 2: Memory test
print("\n2. Testing memory:")
result = process_user_input("My name is John")
print(f"   Response: {result['response'][:50]}...")

# Test 3: Market data test
print("\n3. Testing market data:")
result = process_user_input("What is NIFTY price?")
print(f"   Response: {result['response'][:100]}...")

print("\n✅ All tests completed!")