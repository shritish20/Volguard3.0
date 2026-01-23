"""
Authentication manager using CORRECT Upstox SDK patterns
As verified in your API tester
"""

import upstox_client
from upstox_client.rest import ApiException
from datetime import datetime, timedelta
import json
from typing import Optional, Dict

from config.settings import PATH_CONFIG


class AuthManager:
    """Manage Upstox authentication with CORRECT SDK usage"""
    
    def __init__(self, access_token: str = None):
        self.access_token = access_token
        self.token_expiry = None
        self._configuration = None
        self._api_client = None
        self._session = None
        
        if self.access_token:
            self._setup_configuration()
    
    def _setup_configuration(self):
        """Setup Upstox configuration as per SDK"""
        self._configuration = upstox_client.Configuration()
        self._configuration.access_token = self.access_token
        
        # Create API client
        self._api_client = upstox_client.ApiClient(self._configuration)
        
        # Set token expiry (24 hours from now)
        self.token_expiry = datetime.now() + timedelta(hours=23)
    
    def validate_token(self) -> bool:
        """Validate token by making a simple API call"""
        if not self.access_token or not self._api_client:
            return False
        
        try:
            user_api = upstox_client.UserApi(self._api_client)
            api_response = user_api.get_profile(api_version='2.0')
            return api_response.status == 'success'
        except ApiException as e:
            print(f"Token validation failed: {e}")
            return False
    
    def refresh_token(self, client_id: str, client_secret: str, 
                     refresh_token: str) -> bool:
        """Refresh access token using refresh token"""
        try:
            # Using REST endpoint for token refresh
            import requests
            url = "https://api.upstox.com/v2/login/authorization/token"
            payload = {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get("access_token")
                self._setup_configuration()
                return True
                
        except Exception as e:
            print(f"Token refresh failed: {e}")
            
        return False
    
    def get_websocket_url(self) -> Optional[str]:
        """Get WebSocket authorized URL"""
        try:
            market_api = upstox_client.WebsocketApi(self._api_client)
            api_response = market_api.get_market_data_feed_authorize(api_version='3.0')
            
            if api_response.status == 'success':
                return api_response.data.authorized_redirect_uri
                
        except ApiException as e:
            print(f"Failed to get WS URL: {e}")
            
        return None
    
    def get_portfolio_websocket_url(self) -> Optional[str]:
        """Get portfolio WebSocket URL"""
        try:
            portfolio_api = upstox_client.WebsocketApi(self._api_client)
            api_response = portfolio_api.get_portfolio_stream_feed_authorize(
                api_version='2.0'
            )
            
            if api_response.status == 'success':
                return api_response.data.authorized_redirect_uri
                
        except ApiException as e:
            print(f"Failed to get portfolio WS URL: {e}")
            
        return None
    
    def load_credentials(self) -> Dict:
        """Load credentials from file"""
        try:
            if PATH_CONFIG.CREDENTIALS_FILE.exists():
                with open(PATH_CONFIG.CREDENTIALS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Failed to load credentials: {e}")
        
        return {}
    
    def save_credentials(self, credentials: Dict):
        """Save credentials to file"""
        try:
            with open(PATH_CONFIG.CREDENTIALS_FILE, 'w') as f:
                json.dump(credentials, f, indent=2)
        except Exception as e:
            print(f"Failed to save credentials: {e}")
    
    @property
    def api_client(self):
        """Get API client instance"""
        return self._api_client
    
    @property
    def session(self):
        """Get requests session with auth headers"""
        if not self._session:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.access_token}",
                "accept": "application/json",
                "Api-Version": "2.0",
                "Content-Type": "application/json"
            })
        return self._session
    
    @property
    def is_authenticated(self) -> bool:
        """Check if authenticated"""
        return self.access_token is not None and self.validate_token()
