import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

class TokenManager:
    """Manage authentication tokens"""
    
    def __init__(self, token_file='fyers_token.json'):
        self.token_file = Path(token_file)
        self.tokens = {}
        self.load_tokens()
    
    def load_tokens(self):
        """Load tokens from file"""
        if self.token_file.exists():
            try:
                with open(self.token_file, 'r') as f:
                    self.tokens = json.load(f)
            except json.JSONDecodeError:
                self.tokens = {}
        return self.tokens
    
    def save_tokens(self):
        """Save tokens to file"""
        with open(self.token_file, 'w') as f:
            json.dump(self.tokens, f, indent=2)
    
    def set_token(self, token_type, token_value, expires_in=86400):
        """Set a token with expiry"""
        self.tokens[token_type] = {
            'value': token_value,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(seconds=expires_in)).isoformat(),
            'expires_in': expires_in
        }
        self.save_tokens()
        return True
    
    def get_token(self, token_type):
        """Get token if valid"""
        if token_type in self.tokens:
            token_data = self.tokens[token_type]
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            
            if datetime.now() < expires_at:
                return token_data['value']
            else:
                print(f'Token {token_type} expired')
                del self.tokens[token_type]
                self.save_tokens()
        return None
    
    def clear_tokens(self):
        """Clear all tokens"""
        self.tokens = {}
        if self.token_file.exists():
            self.token_file.unlink()
        return True
    
    def list_tokens(self):
        """List all tokens with status"""
        tokens_info = []
        for token_type, token_data in self.tokens.items():
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            is_valid = datetime.now() < expires_at
            expires_in = (expires_at - datetime.now()).total_seconds()
            
            tokens_info.append({
                'type': token_type,
                'valid': is_valid,
                'expires_in': expires_in,
                'created_at': token_data['created_at']
            })
        return tokens_info
