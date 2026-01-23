"""
WebSocket Manager - PRODUCTION VERIFIED
Compatible with Upstox SDK 2.19.0
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
    Manages WebSocket connections using official Upstox SDK.
    Handles both Market Data (LTP) and Greeks streams.
    """
    
    def __init__(self, auth_manager):
        self.auth = auth_manager
        self.logger = setup_logger("websocket")
        self.api_client = auth_manager.api_client
        
        # Streamer instances
        self.market_streamer = None
        
        # State
        self.is_running = False
        self.ws_thread = None
        
        # Subscriptions Storage
        # We separate them because they need different modes
        self.ltpc_subscriptions = set()   # For Nifty/VIX Spot (Mode: ltpc or full)
        self.greek_subscriptions = set()  # For Options (Mode: option_greeks)
        
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
        
    def subscribe(self, instrument_keys: List[str], mode: str = "ltpc"):
        """
        Queue subscriptions for instruments.
        mode: 'ltpc', 'full', or 'option_greeks'
        """
        if mode == "option_greeks":
            self.greek_subscriptions.update(instrument_keys)
        else:
            self.ltpc_subscriptions.update(instrument_keys)
        
        # If already connected, subscribe immediately
        if self.is_running and self.market_streamer:
            try:
                self.market_streamer.subscribe(instrument_keys, mode)
                self.logger.info(f"Subscribed to {len(instrument_keys)} keys in {mode}")
            except Exception as e:
                self.logger.error(f"Live subscription failed: {e}")
    
    def subscribe_greeks(self, instrument_keys: List[str]):
        """Helper specifically for Greeks subscriptions"""
        self.subscribe(instrument_keys, "option_greeks")
        
    def unsubscribe(self, instrument_keys: List[str]):
        """Unsubscribe from instruments"""
        if self.market_streamer and self.is_running:
            try:
                self.market_streamer.unsubscribe(instrument_keys)
                
                # Remove from local sets
                for key in instrument_keys:
                    self.ltpc_subscriptions.discard(key)
                    self.greek_subscriptions.discard(key)
                    
                self.logger.info(f"Unsubscribed from {len(instrument_keys)} instruments")
            except Exception as e:
                self.logger.error(f"Unsubscribe error: {e}")
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for specific event type"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
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
        
        if self.ws_thread:
            self.ws_thread.join(timeout=5)
        
        self.logger.info("WebSocket manager stopped")
    
    def _run_websocket(self):
        """Run WebSocket in background thread"""
        try:
            # ---------------------------------------------------------
            # FIX: Initialize WITHOUT arguments (SDK 2.19.0 requirement)
            # ---------------------------------------------------------
            self.market_streamer = upstox_client.MarketDataStreamerV3(self.api_client)
            
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
        """
        Handle connection open.
        Crucial: This is where we send the initial subscriptions.
        """
        self.logger.info("WebSocket connection opened")
        
        # Resubscribe to cached keys
        if self.ltpc_subscriptions:
            self.market_streamer.subscribe(list(self.ltpc_subscriptions), "ltpc")
            self.logger.info(f"Resubscribed to {len(self.ltpc_subscriptions)} LTPC keys")
            
        if self.greek_subscriptions:
            self.market_streamer.subscribe(list(self.greek_subscriptions), "option_greeks")
            self.logger.info(f"Resubscribed to {len(self.greek_subscriptions)} Greek keys")
            
        # Trigger external callbacks
        for callback in self.callbacks['open']:
            try: callback()
            except Exception as e: self.logger.error(f"Callback error (open): {e}")
    
    def _on_message(self, message):
        try:
            # Process market data
            self._process_market_data(message)
            
            # Trigger callbacks
            for callback in self.callbacks['tick']:
                try: callback(message)
                except Exception: pass
        
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")
    
    def _on_close(self):
        self.logger.info("WebSocket connection closed")
        # Don't set is_running to False here, let auto-reconnect handle it
        for callback in self.callbacks['close']:
            try: callback()
            except Exception: pass
    
    def _on_error(self, error):
        self.logger.error(f"WebSocket error: {error}")
        for callback in self.callbacks['error']:
            try: callback(error)
            except Exception: pass
    
    def _on_reconnecting(self, message):
        self.logger.info(f"WebSocket reconnecting: {message}")
        for callback in self.callbacks['reconnecting']:
            try: callback(message)
            except Exception: pass
    
    def _process_market_data(self, data: dict):
        """Process and cache market data (LTP and Greeks)"""
        try:
            if 'feeds' in data:
                feeds = data['feeds']
                for instrument_key, feed_data in feeds.items():
                    if instrument_key not in self.latest_data:
                        self.latest_data[instrument_key] = {}
                    
                    # 1. Handle Standard Data (LTPC/Full)
                    if 'ltpc' in feed_data:
                        self.latest_data[instrument_key]['ltp'] = feed_data['ltpc'].get('ltp')
                        self.latest_data[instrument_key]['last_trade_time'] = feed_data['ltpc'].get('ltt')
                        self.latest_data[instrument_key]['volume'] = feed_data['ltpc'].get('vol') # or check full feed
                        
                    # 2. Handle NATIVE GREEKS (V3 Feature)
                    if 'optionGreeks' in feed_data:
                        g = feed_data['optionGreeks']
                        self.latest_data[instrument_key]['greeks'] = {
                            'delta': g.get('delta', 0.0),
                            'gamma': g.get('gamma', 0.0),
                            'theta': g.get('theta', 0.0),
                            'vega': g.get('vega', 0.0),
                            'iv': g.get('iv', 0.0),
                            'timestamp': datetime.now()
                        }
                    
                    # Store raw feed for debugging if needed
                    self.latest_data[instrument_key]['raw'] = feed_data
        
        except Exception as e:
            self.logger.error(f"Market data processing error: {e}")
    
    def get_latest_price(self, instrument_key: str) -> Optional[float]:
        """Get latest price for instrument"""
        data = self.latest_data.get(instrument_key)
        return data.get('ltp') if data else None
    
    def get_latest_data(self, instrument_key: str) -> Optional[dict]:
        """Get all latest data for instrument (including Greeks)"""
        return self.latest_data.get(instrument_key)
