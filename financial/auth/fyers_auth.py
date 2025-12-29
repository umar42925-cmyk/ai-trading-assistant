import os
import webbrowser
from fyers_apiv3 import fyersModel
from config.fyers_config import FyersConfig

class FyersAuth:
    def __init__(self):
        FyersConfig.validate()
        self.client_id = FyersConfig.CLIENT_ID
        self.secret_key = FyersConfig.SECRET_KEY
        self.redirect_uri = FyersConfig.REDIRECT_URI
        self.token_file = 'fyers_token.txt'
    
    def get_auth_url(self):
        """Generate authentication URL"""
        return f'https://api.fyers.in/api/v3/generate-authcode?client_id={self.client_id}&redirect_uri={self.redirect_uri}&response_type=code&state=sample'
    
    def authenticate(self):
        """Complete authentication flow"""
        print(' Starting Fyers Authentication...')
        
        # Generate auth URL
        auth_url = self.get_auth_url()
        print(f'\n1. Visit this URL:')
        print(f'   {auth_url}')
        
        # Try to open browser
        try:
            webbrowser.open(auth_url)
            print('    Browser opened!')
        except:
            print('    Copy and paste the URL into your browser')
        
        # Get auth code from user
        print('\n2. After login, copy the auth_code from URL')
        print('   Example: https://localhost:8080/?auth_code=XXXXXX&state=sample')
        
        auth_code = input('\nEnter auth code: ').strip()
        
        # Generate token
        print('\n3. Generating access token...')
        try:
            fyers = fyersModel.FyersModel(
                client_id=self.client_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                response_type='code',
                grant_type='authorization_code'
            )
            
            response = fyers.generate_token(auth_code)
            
            if 'access_token' in response:
                token = response['access_token']
                self.save_token(token)
                print(f' Authentication successful!')
                print(f'   Token expires in: {response.get("expires_in", "N/A")} seconds')
                return token
            else:
                print(f' Failed: {response}')
                return None
                
        except Exception as e:
            print(f' Error: {e}')
            return None
    
    def save_token(self, token):
        """Save token to file"""
        with open(self.token_file, 'w') as f:
            f.write(token)
        os.environ['FYERS_TOKEN'] = token
        print(f' Token saved to {self.token_file}')
    
    def load_token(self):
        """Load saved token"""
        if os.path.exists(self.token_file):
            with open(self.token_file, 'r') as f:
                return f.read().strip()
        return None
