"""
Simple authentication setup for testing
"""

import os
import json
from pathlib import Path

def setup_authentication():
    """Setup basic authentication for testing"""
    
    # Create auth directory
    auth_dir = Path("financial/auth")
    auth_dir.mkdir(parents=True, exist_ok=True)
    
    # Create empty token files for testing
    token_files = {
        "fyers_token.json": {"access_token": "test_token", "app_id": "test_app"},
        "upstox_token.json": {"access_token": "test_token"}
    }
    
    for filename, content in token_files.items():
        filepath = auth_dir / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                json.dump(content, f)
            print(f"Created {filename}")
    
    # Set environment variables for testing
    os.environ.setdefault("FYERS_APP_ID", "test_app_id")
    os.environ.setdefault("FYERS_TOKEN", "test_token")
    os.environ.setdefault("UPSTOX_API_KEY", "test_api_key")
    
    print("✅ Authentication files created for testing")
    return True