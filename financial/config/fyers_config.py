import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FyersConfig:
    """Fyers API Configuration"""
    
    # Fyers API Credentials
    APP_ID = os.getenv('FYERS_APP_ID')
    CLIENT_ID = os.getenv('FYERS_CLIENT_ID')
    SECRET_KEY = os.getenv('FYERS_SECRET_KEY')
    REDIRECT_URI = os.getenv('FYERS_REDIRECT_URI')
    TOKEN = os.getenv('FYERS_TOKEN')
    
    # Validate configuration
    @classmethod
    def validate(cls):
        missing = []
        if not cls.APP_ID: missing.append('FYERS_APP_ID')
        if not cls.CLIENT_ID: missing.append('FYERS_CLIENT_ID')
        if not cls.SECRET_KEY: missing.append('FYERS_SECRET_KEY')
        if not cls.REDIRECT_URI: missing.append('FYERS_REDIRECT_URI')
        
        if missing:
            raise ValueError(f'Missing configuration: {", ".join(missing)}')
        
        print(' All configurations are set correctly')
        return True
