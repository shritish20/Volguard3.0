"""
WebSocket Manager using CORRECT Upstox SDK
As verified in your API tester
"""

import upstox_client
from upstox_client.rest import ApiException
from typing import Dict, List, Callable, Optional
from datetime import datetime
import threading
import time

from utils.logger import setup_logger


class WebSocketManager:
    """
    Manages WebSocket connections using official Upstox SDK
    CORRECT implementation matching your API tester
    """
    
    def __init__(self, auth_manager):
        self.auth = auth_manager
        self.logger = setup_logger("websocket")
        
        # Create API client
        self.api_client = auth_manager.api_client
        
        # Streamer instances
        self.market_streamer = None
        self.portfolio_streamer = None
        
        # State
        self.is_running = False
        self.subscribed_instruments = set()
        self.current_mode = "full"  # full, ltpc, option_greeks, full_d30
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'tick': [],
            'open': [],
            'close': [],
            'error': [],
            'reconnecting': []
        }
        
        # Data cache
        self.latest_data: Dict[str, dict] = {}
        
        # Thread management
        self.ws_thread = None
    
    def subscribe(self, instrument_keys: List[str], mode: str = "full"):
        """Subscribe to instruments for live data"""
        for key in instrument_keys:
            self.subscribed_instruments.add(key)
        
        self.current_mode = mode
        
        # If already connected, subscribe immediately
        if self.market_streamer and self.is_running:
            try:
                self.market_streamer.subscribe(instrument_keys, mode)
                self.logger.info(f"Subscribed to {len(instrument_keys)} instruments in {mode} mode")
            except Exception as e:
                self.logger.error(f"Subscription error: {e}")
        else:
            self.logger.info(f"Queued {len(instrument_keys)} instruments for subscription")
    
    def unsubscribe(self, instrument_keys: List[str]):
        """Unsubscribe from instruments"""
        if self.market_streamer and self.is_running:
            try:
                self.market_streamer.unsubscribe(instrument_keys)
                for key in instrument_keys:
                    self.subscribed_instruments.discard(key)
                self.logger.info(f"Unsubscribed from {len(instrument_keys)} instruments")
            except Exception as e:
                self.logger.error(f"Unsubscribe error: {e}")
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for specific event type"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            self.logger.debug(f"Registered callback for {event_type}")
    
    def start(self):
        """Start WebSocket connection in background thread"""
        if self.is_running:
            self.logger.warning("WebSocket already running")
            return
        
        self.is_running = True
        self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.ws_thread.start()
        
        self.logger.info("WebSocket manager started")
    
    def stop(self):
        """Stop WebSocket connection"""
        self.is_running = False
        
        if self.market_streamer:
            try:
                self.market_streamer.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting market streamer: {e}")
        
        if self.portfolio_streamer:
            try:
                self.portfolio_streamer.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting portfolio streamer: {e}")
        
        if self.ws_thread:
            self.ws_thread.join(timeout=5)
        
        self.logger.info("WebSocket manager stopped")
    
    def _run_websocket(self):
        """Run WebSocket in background thread"""
        try:
            # Get WebSocket URL
            ws_api = upstox_client.WebsocketApi(self.api_client)
            ws_response = ws_api.get_market_data_feed_authorize(api_version='3.0')
            
            if not ws_response or ws_response.status != 'success':
                self.logger.error("Failed to get WebSocket URL")
                return
            
            # Create streamer
            initial_instruments = list(self.subscribed_instruments) if self.subscribed_instruments else []
            
            self.market_streamer = upstox_client.MarketDataStreamerV3(
                self.api_client,
                initial_instruments,
                self.current_mode
            )
            
            # Register event handlers
            self.market_streamer.on("open", self._on_open)
            self.market_streamer.on("message", self._on_message)
            self.market_streamer.on("close", self._on_close)
            self.market_streamer.on("error", self._on_error)
            self.market_streamer.on("reconnecting", self._on_reconnecting)
            
            # Configure auto-reconnect
            self.market_streamer.auto_reconnect(True, 10, 5)
            
            # Connect
            self.market_streamer.connect()
            
            # Keep thread alive
            while self.is_running:
                time.sleep(1)
            
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            self.is_running = False
    
    # Event handlers
    def _on_open(self):
        self.logger.info("WebSocket connection opened")
        for callback in self.callbacks['open']:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Callback error (open): {e}")
    
    def _on_message(self, message):
        try:
            # Process market data
            self._process_market_data(message)
            
            # Trigger callbacks
            for callback in self.callbacks['tick']:
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Callback error (tick): {e}")
        
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")
    
    def _on_close(self):
        self.logger.info("WebSocket connection closed")
        self.is_running = False
        
        for callback in self.callbacks['close']:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Callback error (close): {e}")
    
    def _on_error(self, error):
        self.logger.error(f"WebSocket error: {error}")
        
        for callback in self.callbacks['error']:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"Callback error (error): {e}")
    
    def _on_reconnecting(self, message):
        self.logger.info(f"WebSocket reconnecting: {message}")
    
    def _process_market_data(self, data: dict):
        """Process and cache market data"""
        try:
            # The message structure depends on the mode
            # Extract instrument key and data
            if 'feeds' in data:
                feeds = data['feeds']
                for instrument_key, feed_data in feeds.items():
                    self.latest_data[instrument_key] = {
                        'ltp': feed_data.get('ltp') or feed_data.get('last_price'),
                        'volume': feed_data.get('volume'),
                        'oi': feed_data.get('oi'),
                        'bid': feed_data.get('bidPrice'),
                        'ask': feed_data.get('askPrice'),
                        'timestamp': datetime.now(),
                        'raw': feed_data
                    }
        
        except Exception as e:
            self.logger.error(f"Market data processing error: {e}")
    
    def get_latest_price(self, instrument_key: str) -> Optional[float]:
        """Get latest price for instrument"""
        data = self.latest_data.get(instrument_key)
        return data.get('ltp') if data else None
    
    def get_latest_data(self, instrument_key: str) -> Optional[dict]:
        """Get all latest data for instrument"""
        return self.latest_data.get(instrument_key)
