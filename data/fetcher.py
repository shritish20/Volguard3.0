"""
Upstox Data Fetcher - PRODUCTION VERIFIED
Matching your API tester 100% with V3 Greeks & Fixed History
"""

import upstox_client
from upstox_client.rest import ApiException
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
import time
import warnings
warnings.filterwarnings('ignore')

from config.settings import UPSTOX_CONFIG


class UpstoxDataFetcher:
    """Fetch all data using CORRECT Upstox SDK patterns"""
    
    def __init__(self, auth_manager):
        self.auth = auth_manager
        self.api_client = auth_manager.api_client
        
        # Initialize API instances
        self.user_api = upstox_client.UserApi(self.api_client)
        self.market_api = upstox_client.MarketQuoteApi(self.api_client)
        self.market_api_v3 = upstox_client.MarketQuoteV3Api(self.api_client) # ADDED V3 FOR GREEKS
        self.history_api = upstox_client.HistoryApi(self.api_client)
        self.options_api = upstox_client.OptionsApi(self.api_client)
        self.portfolio_api = upstox_client.PortfolioApi(self.api_client)
        self.order_api = upstox_client.OrderApi(self.api_client)
        self.charge_api = upstox_client.ChargeApi(self.api_client)
        
        # Cache
        self.cache = {}
        self.cache_expiry = {}
        
        # Link to websocket manager (set by main.py)
        self.websocket_manager = None
        
    def _make_request(self, func, *args, **kwargs):
        """Wrapper for API calls with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except ApiException as e:
                if e.status == 429:  # Rate limit
                    wait_time = (attempt + 1) * 2
                    print(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
        return None

    # -------------------- GREEKS SNAPSHOT (NEW V3) --------------------
    
    def get_greeks_snapshot(self, instrument_keys: List[str]) -> Dict[str, Dict]:
        """
        Fetch static snapshot of Greeks using V3 API.
        Works 24/7 (returns last closing data if market closed).
        """
        if not instrument_keys:
            return {}

        try:
            # Upstox API expects comma-separated string
            keys_str = ",".join(instrument_keys)
            
            # Use V3 API - This is the method we verified in the tester
            response = self._make_request(
                self.market_api_v3.get_market_quote_option_greek,
                instrument_key=keys_str
            )
            
            if response and response.status == 'success' and response.data:
                result = {}
                # response.data is a Dict where keys are messy symbols but values are Objects
                for symbol_key, data_obj in response.data.items():
                    
                    # Use getattr() because data_obj is a Class Object, not a dict
                    token = getattr(data_obj, 'instrument_token', None)
                    
                    if token:
                        result[token] = {
                            'delta': getattr(data_obj, 'delta', 0),
                            'gamma': getattr(data_obj, 'gamma', 0),
                            'theta': getattr(data_obj, 'theta', 0),
                            'vega':  getattr(data_obj, 'vega', 0),
                            'iv':    getattr(data_obj, 'iv', 0)
                        }
                return result
                
        except Exception as e:
            print(f"Snapshot fetch failed: {e}")
            
        return {}
    
    # -------------------- BASIC MARKET DATA --------------------
    
    def get_spot_price(self) -> Optional[float]:
        """Get current Nifty spot price"""
        try:
            api_response = self._make_request(
                self.market_api.get_full_market_quote,
                symbol=UPSTOX_CONFIG.NIFTY_50_KEY,
                api_version='2.0'
            )
            
            if api_response and api_response.status == 'success':
                # Key format can vary - try both formats
                key = UPSTOX_CONFIG.NIFTY_50_KEY.replace('|', ':')
                data = api_response.data.get(key) or api_response.data.get(UPSTOX_CONFIG.NIFTY_50_KEY)
                if data:
                    return data.last_price
                    
        except Exception as e:
            print(f"Failed to get spot price: {e}")
            
        return None
    
    def get_vix_price(self) -> Optional[float]:
        """Get current India VIX price"""
        try:
            api_response = self._make_request(
                self.market_api.get_full_market_quote,
                symbol=UPSTOX_CONFIG.INDIA_VIX_KEY,
                api_version='2.0'
            )
            
            if api_response and api_response.status == 'success':
                key = UPSTOX_CONFIG.INDIA_VIX_KEY.replace('|', ':')
                data = api_response.data.get(key) or api_response.data.get(UPSTOX_CONFIG.INDIA_VIX_KEY)
                if data:
                    return data.last_price
                    
        except Exception as e:
            print(f"Failed to get VIX price: {e}")
            
        return None
    
    def get_historical_data(self, instrument_key: str, days: int = 400) -> pd.DataFrame:
        """Get historical OHLC data"""
        cache_key = f"historical_{instrument_key}_{days}"
        
        # Check cache
        if cache_key in self.cache:
            if datetime.now() < self.cache_expiry.get(cache_key, datetime.now()):
                return self.cache[cache_key].copy()
        
        try:
            to_date = date.today().strftime("%Y-%m-%d")
            from_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # ------------------------------------------------------------------
            # FIX: Passed arguments POSITIONALLY. Removed 'api_version' kwarg.
            # Signature: get_historical_candle_data(instrument_key, interval, to_date, from_date)
            # ------------------------------------------------------------------
            api_response = self._make_request(
                self.history_api.get_historical_candle_data,
                instrument_key,  # Positional 1
                'day',           # Positional 2
                to_date,         # Positional 3
                from_date        # Positional 4
            )
            
            if api_response and api_response.status == 'success':
                data = api_response.data.candles
                
                if data:
                    df = pd.DataFrame(data, columns=[
                        "timestamp", "open", "high", "low", "close", "volume", "oi"
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    
                    # Cache for 5 minutes
                    self.cache[cache_key] = df
                    self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=5)
                    
                    return df.copy()
                    
        except Exception as e:
            print(f"Failed to get historical data: {e}")
            
        return pd.DataFrame()
    
    # -------------------- OPTION CHAIN DATA --------------------
    
    def get_expiry_dates(self) -> List[date]:
        """Get all available expiry dates for Nifty"""
        try:
            api_response = self._make_request(
                self.options_api.get_option_contracts,
                instrument_key=UPSTOX_CONFIG.NIFTY_50_KEY
            )
            
            if api_response and api_response.status == 'success':
                expiry_dates = []
                for contract in api_response.data:
                    if hasattr(contract, 'expiry') and contract.expiry:
                        if isinstance(contract.expiry, datetime):
                            expiry_dates.append(contract.expiry.date())
                        elif isinstance(contract.expiry, str):
                            expiry_dates.append(datetime.strptime(contract.expiry, "%Y-%m-%d").date())
                        else:
                            expiry_dates.append(contract.expiry)
                
                unique_expiries = sorted(list(set(expiry_dates)))
                return unique_expiries
                
        except Exception as e:
            print(f"Failed to get expiry dates: {e}")
            
        return []
    
    def get_option_chain(self, expiry_date: date) -> pd.DataFrame:
        """Get complete option chain for specific expiry"""
        cache_key = f"chain_{expiry_date}"
        
        # Check cache (1 minute cache for option chains)
        if cache_key in self.cache:
            if datetime.now() < self.cache_expiry.get(cache_key, datetime.now()):
                return self.cache[cache_key].copy()
        
        try:
            expiry_str = expiry_date.strftime("%Y-%m-%d")
            
            # First get option contracts for this expiry
            api_response = self._make_request(
                self.options_api.get_option_contracts,
                instrument_key=UPSTOX_CONFIG.NIFTY_50_KEY,
                expiry_date=expiry_str
            )
            
            if api_response and api_response.status == 'success':
                contracts = api_response.data
                
                # For each contract, get detailed data
                rows = []
                for contract in contracts:
                    # In high-performance mode, we avoid 100s of API calls for quotes here.
                    # We rely on Greeks snapshot or websocket for live data later.
                    row = {
                        'strike': contract.strike_price,
                        'expiry': expiry_date,
                        'instrument_key': contract.instrument_key,
                        'trading_symbol': contract.trading_symbol,
                        'lot_size': contract.lot_size,
                        'option_type': 'CE' if (' CE ' in contract.trading_symbol or contract.trading_symbol.endswith('CE')) else 'PE',
                        # Placeholders - populated by Portfolio/Strategy managers via Websocket
                        'ltp': 0,
                        'volume': 0,
                        'oi': 0,
                        'bid': 0,
                        'ask': 0,
                        'iv': 0,
                        'delta': 0,
                        'gamma': 0,
                        'theta': 0,
                        'vega': 0
                    }
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                
                # Organize into CE/PE structure
                if not df.empty:
                    # Pivot to get CE and PE side by side
                    ce_df = df[df['option_type'] == 'CE'].copy()
                    pe_df = df[df['option_type'] == 'PE'].copy()
                    
                    # Merge on strike price
                    merged_df = pd.merge(
                        ce_df.rename(columns=lambda x: f"ce_{x}" if x not in ['strike', 'expiry'] else x),
                        pe_df.rename(columns=lambda x: f"pe_{x}" if x not in ['strike', 'expiry'] else x),
                        on=['strike', 'expiry'],
                        how='outer'
                    )
                    
                    df = merged_df.sort_values('strike')
                
                # Cache for 1 minute
                self.cache[cache_key] = df
                self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=1)
                
                return df.copy()
                
        except Exception as e:
            print(f"Failed to get option chain: {e}")
            
        return pd.DataFrame()
    
    def get_put_call_chain(self, expiry_date: date) -> pd.DataFrame:
        """Get put-call chain (simplified option chain)"""
        try:
            expiry_str = expiry_date.strftime("%Y-%m-%d")
            
            api_response = self._make_request(
                self.options_api.get_put_call_option_chain,
                instrument_key=UPSTOX_CONFIG.NIFTY_50_KEY,
                expiry_date=expiry_str
            )
            
            if api_response and api_response.status == 'success':
                data = api_response.data
                
                rows = []
                for item in data:
                    # Safely handle objects/dicts using getattr logic
                    call_data = getattr(item, 'call_options', None) or {}
                    put_data = getattr(item, 'put_options', None) or {}
                    
                    # Helper to safely extract from potential objects or dicts
                    def safe_get(obj, attr, default=0):
                        if isinstance(obj, dict): return obj.get(attr, default)
                        return getattr(obj, attr, default)

                    call_market = safe_get(call_data, 'market_data', {})
                    call_greeks = safe_get(call_data, 'option_greeks', {})
                    put_market = safe_get(put_data, 'market_data', {})
                    put_greeks = safe_get(put_data, 'option_greeks', {})

                    row = {
                        'strike': getattr(item, 'strike_price', 0),
                        'underlying_spot': getattr(item, 'underlying_spot_price', 0),
                        'pcr': getattr(item, 'pcr', 0),
                        
                        'ce_instrument_key': safe_get(call_data, 'instrument_key', ''),
                        'ce_ltp': safe_get(call_market, 'ltp', 0),
                        'ce_volume': safe_get(call_market, 'volume', 0),
                        'ce_oi': safe_get(call_market, 'oi', 0),
                        'ce_iv': safe_get(call_greeks, 'iv', 0),
                        'ce_delta': safe_get(call_greeks, 'delta', 0),
                        'ce_gamma': safe_get(call_greeks, 'gamma', 0),
                        'ce_theta': safe_get(call_greeks, 'theta', 0),
                        'ce_vega': safe_get(call_greeks, 'vega', 0),
                        
                        'pe_instrument_key': safe_get(put_data, 'instrument_key', ''),
                        'pe_ltp': safe_get(put_market, 'ltp', 0),
                        'pe_volume': safe_get(put_market, 'volume', 0),
                        'pe_oi': safe_get(put_market, 'oi', 0),
                        'pe_iv': safe_get(put_greeks, 'iv', 0),
                        'pe_delta': safe_get(put_greeks, 'delta', 0),
                        'pe_gamma': safe_get(put_greeks, 'gamma', 0),
                        'pe_theta': safe_get(put_greeks, 'theta', 0),
                        'pe_vega': safe_get(put_greeks, 'vega', 0),
                    }
                    rows.append(row)
                
                return pd.DataFrame(rows)
                
        except Exception as e:
            print(f"Failed to get put-call chain: {e}")
            
        return pd.DataFrame()
    
    # -------------------- PORTFOLIO DATA --------------------
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            api_response = self._make_request(
                self.portfolio_api.get_positions,
                api_version='2.0'
            )
            
            if api_response and api_response.status == 'success':
                return [self._position_to_dict(pos) for pos in api_response.data]
                
        except Exception as e:
            print(f"Failed to get positions: {e}")
            
        return []
    
    def get_holdings(self) -> List[Dict]:
        """Get holdings"""
        try:
            api_response = self._make_request(
                self.portfolio_api.get_holdings,
                api_version='2.0'
            )
            
            if api_response and api_response.status == 'success':
                return [self._holding_to_dict(hold) for hold in api_response.data]
                
        except Exception as e:
            print(f"Failed to get holdings: {e}")
            
        return []
    
    def get_funds_and_margin(self) -> Dict:
        """Get available funds and margin"""
        try:
            api_response = self._make_request(
                self.user_api.get_user_fund_margin,
                api_version='2.0'
            )
            
            if api_response and api_response.status == 'success':
                # Convert to dict
                data = {}
                if hasattr(api_response.data, '__dict__'):
                    data = api_response.data.__dict__
                elif isinstance(api_response.data, dict):
                    data = api_response.data
                else:
                    # It might be an object that doesn't have __dict__ easily accessible
                    # We return the object itself, PortfolioManager knows how to handle it
                    return api_response.data
                
                return data
                
        except Exception as e:
            print(f"Failed to get funds: {e}")
            
        return {}
    
    def get_order_book(self) -> List[Dict]:
        """Get order book for the day"""
        try:
            api_response = self._make_request(
                self.order_api.get_order_book,
                api_version='2.0'
            )
            
            if api_response and api_response.status == 'success':
                return [self._order_to_dict(order) for order in api_response.data]
                
        except Exception as e:
            print(f"Failed to get order book: {e}")
            
        return []
    
    def get_trades(self) -> List[Dict]:
        """Get trades for the day"""
        try:
            api_response = self._make_request(
                self.order_api.get_trade_history,
                api_version='2.0'
            )
            
            if api_response and api_response.status == 'success':
                return [self._trade_to_dict(trade) for trade in api_response.data]
                
        except Exception as e:
            print(f"Failed to get trades: {e}")
            
        return []
    
    # -------------------- HELPER METHODS --------------------
    
    def _position_to_dict(self, position) -> Dict:
        """Convert position object to dict using getattr for safety"""
        return {
            'instrument_token': getattr(position, 'instrument_token', ''),
            'instrument_key': getattr(position, 'instrument_key', ''),
            'trading_symbol': getattr(position, 'trading_symbol', ''),
            'product': getattr(position, 'product', ''),
            'quantity': getattr(position, 'quantity', 0),
            'average_price': getattr(position, 'average_price', 0),
            'last_price': getattr(position, 'last_price', 0),
            'pnl': getattr(position, 'pnl', 0),
            'margin_used': getattr(position, 'margin_used', 0)
        }
    
    def _holding_to_dict(self, holding) -> Dict:
        """Convert holding object to dict using getattr for safety"""
        return {
            'instrument_token': getattr(holding, 'instrument_token', ''),
            'trading_symbol': getattr(holding, 'trading_symbol', ''),
            'quantity': getattr(holding, 'quantity', 0),
            'average_price': getattr(holding, 'average_price', 0),
            'last_price': getattr(holding, 'last_price', 0),
            'pnl': getattr(holding, 'pnl', 0)
        }
    
    def _order_to_dict(self, order) -> Dict:
        """Convert order object to dict using getattr for safety"""
        return {
            'order_id': getattr(order, 'order_id', ''),
            'trading_symbol': getattr(order, 'trading_symbol', ''),
            'instrument_token': getattr(order, 'instrument_token', ''),
            'transaction_type': getattr(order, 'transaction_type', ''),
            'order_type': getattr(order, 'order_type', ''),
            'quantity': getattr(order, 'quantity', 0),
            'filled_quantity': getattr(order, 'filled_quantity', 0),
            'price': getattr(order, 'price', 0),
            'trigger_price': getattr(order, 'trigger_price', 0),
            'status': getattr(order, 'status', ''),
            'tag': getattr(order, 'tag', '')
        }
    
    def _trade_to_dict(self, trade) -> Dict:
        """Convert trade object to dict using getattr for safety"""
        return {
            'trade_id': getattr(trade, 'trade_id', ''),
            'order_id': getattr(trade, 'order_id', ''),
            'trading_symbol': getattr(trade, 'trading_symbol', ''),
            'quantity': getattr(trade, 'quantity', 0),
            'price': getattr(trade, 'price', 0),
            'trade_time': getattr(trade, 'trade_time', '')
        }
    
    # -------------------- CALCULATIONS --------------------
    
    def calculate_margin(self, instruments: List[Dict]) -> Dict:
        """Calculate margin for instruments"""
        try:
            # Convert instruments to MarginRequest
            margin_instruments = []
            for instr in instruments:
                margin_instr = upstox_client.Instrument(
                    instrument_key=instr.get('instrument_key'),
                    quantity=instr.get('quantity'),
                    transaction_type=instr.get('transaction_type'),
                    product=instr.get('product', 'D')
                )
                margin_instruments.append(margin_instr)
            
            margin_request = upstox_client.MarginRequest(instruments=margin_instruments)
            
            api_response = self._make_request(
                self.charge_api.post_margin,
                margin_request
            )
            
            if api_response and api_response.status == 'success':
                # Convert to dict
                data = {}
                if hasattr(api_response.data, '__dict__'):
                    data = api_response.data.__dict__
                elif isinstance(api_response.data, dict):
                    data = api_response.data
                else:
                    # Handle object without __dict__ (common in generated clients)
                    data = {
                        'required_margin': getattr(api_response.data, 'required_margin', 0),
                        'final_margin': getattr(api_response.data, 'final_margin', 0)
                    }
                
                return data
                
        except Exception as e:
            print(f"Failed to calculate margin: {e}")
            
        return {}
    
    def get_brokerage(self, instrument_token: str, quantity: int, 
                     product: str, transaction_type: str, price: float) -> Dict:
        """Get brokerage charges"""
        try:
            api_response = self._make_request(
                self.charge_api.get_brokerage,
                instrument_token=instrument_token,
                quantity=quantity,
                product=product,
                transaction_type=transaction_type,
                price=price,
                api_version='2.0'
            )
            
            if api_response and api_response.status == 'success':
                # Convert to dict
                data = {}
                if hasattr(api_response.data, '__dict__'):
                    data = api_response.data.__dict__
                elif isinstance(api_response.data, dict):
                    data = api_response.data
                else:
                    # Extract total charges safely if object
                    charges = getattr(api_response.data, 'charges', None)
                    total = getattr(charges, 'total', 0) if charges else 0
                    data = {'total': total}
                
                return data
                
        except Exception as e:
            print(f"Failed to get brokerage: {e}")
            
        return {}
