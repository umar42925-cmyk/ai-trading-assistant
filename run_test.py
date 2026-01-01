# run_test.py
import os
import sys

# Force Python to find modules
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'financial', 'auth'))

print("Testing Fyers Auth...")
try:
    # This will work at runtime even if VS Code shows warnings
    exec(open("financial/auth/fyers_auth.py").read())
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
