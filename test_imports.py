# debug_imports.py
import sys
import os

print("=" * 60)
print("DEBUG: Python Import System")
print("=" * 60)

print(f"\n1. Current directory: {os.getcwd()}")
print(f"\n2. Python path (sys.path):")
for i, path in enumerate(sys.path[:10]):  # Show first 10
    print(f"   [{i}] {path}")

print(f"\n3. Files in current directory:")
for file in os.listdir('.'):
    if file.endswith('.py') or file == 'financial':
        print(f"   - {file}")

print(f"\n4. Checking for 'financial' directory...")
if os.path.exists('financial'):
    print("   ✅ 'financial' directory exists")
    print("\n5. Contents of 'financial' directory:")
    for item in os.listdir('financial'):
        print(f"   - {item}")
else:
    print("   ❌ 'financial' directory NOT found")

print(f"\n6. Trying to import financial module...")
try:
    import financial
    print(f"   ✅ Success! Module location: {financial.__file__}")
except ImportError as e:
    print(f"   ❌ Failed: {e}")

print(f"\n7. Checking __init__.py in financial directory...")
if os.path.exists('financial/__init__.py'):
    print("   ✅ financial/__init__.py exists")
    with open('financial/__init__.py', 'r') as f:
        content = f.read().strip()
        print(f"   Content: '{content}' (empty if nothing)")
else:
    print("   ❌ financial/__init__.py does NOT exist")

print(f"\n8. Trying direct import...")
try:
    # Try importing the specific module directly
    import importlib.util
    
    # Check if file exists
    module_path = 'financial/data/minimal_pipeline.py'
    if os.path.exists(module_path):
        print(f"   ✅ File exists: {module_path}")
        
        # Try to load it
        spec = importlib.util.spec_from_file_location(
            "minimal_pipeline", 
            module_path
        )
        if spec:
            print("   ✅ Module spec created")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("   ✅ Module loaded!")
            if hasattr(module, 'MinimalMarketPipeline'):
                print("   ✅ MinimalMarketPipeline class found!")
            else:
                print("   ❌ MinimalMarketPipeline class NOT in module")
        else:
            print("   ❌ Could not create module spec")
    else:
        print(f"   ❌ File does NOT exist: {module_path}")
except Exception as e:
    print(f"   ❌ Error during direct import: {e}")

print("\n" + "=" * 60)