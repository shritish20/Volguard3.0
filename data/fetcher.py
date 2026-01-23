"""
Upstox Data Fetcher using CORRECT SDK patterns
Matching your API tester 100%
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
        self.history_api = upstox_client.HistoryApi(self.api_client)
        self.options_api = upstox_client.OptionsApi(self.api_client)
        self.portfolio_api = upstox_client.PortfolioApi(self.api_client)
        self.order_api = upstox_client.OrderApi(self.api_client)
        self.charge_api = upstox_client.ChargeApi(self.api_client)
        
        # Cache
        self.cache = {}
        self.cache_expiry = {}
        
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
            
            api_response = self._make_request(
                self.history_api.get_historical_candle_data,
                instrument_key=instrument_key,
                interval='day',
                to_date=to_date,
                from_date=from_date
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
                    # Get market quote for this option
                    option_key = contract.instrument_key
                    quote_response = self._make_request(
                        self.market_api.get_full_market_quote,
                        symbol=option_key,
                        api_version='2.0'
                    )
                    
                    if quote_response and quote_response.status == 'success':
                        key = option_key.replace('|', ':')
                        market_data = quote_response.data.get(key)
                        
                        if market_data:
                            # Try to get Greeks if available
                            try:
                                greeks_response = self._make_request(
                                    self.market_api.get_option_greek,
                                    instrument_key=option_key,
                                    api_version='3.0'
                                )
                                greeks_data = greeks_response.data.get(key) if greeks_response and greeks_response.status == 'success' else {}
                            except:
                                greeks_data = {}
                            
                            row = {
                                'strike': contract.strike_price,
                                'expiry': expiry_date,
                                'instrument_key': option_key,
                                'trading_symbol': contract.trading_symbol,
                                'lot_size': contract.lot_size,
                                'option_type': contract.option_type,  # CE or PE
                                'ltp': market_data.last_price,
                                'volume': market_data.volume,
                                'oi': market_data.oi,
                                'bid': market_data.bid_price,
                                'ask': market_data.ask_price,
                                'iv': greeks_data.get('iv', 0),
                                'delta': greeks_data.get('delta', 0),
                                'gamma': greeks_data.get('gamma', 0),
                                'theta': greeks_data.get('theta', 0),
                                'vega': greeks_data.get('vega', 0)
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
                    call_data = item.call_options if hasattr(item, 'call_options') else {}
                    put_data = item.put_options if hasattr(item, 'put_options') else {}
                    
                    row = {
                        'strike': item.strike_price,
                        'underlying_spot': item.underlying_spot_price,
                        'pcr': item.pcr,
                        
                        'ce_instrument_key': call_data.get('instrument_key') if call_data else '',
                        'ce_ltp': call_data.get('market_data', {}).get('ltp', 0),
                        'ce_volume': call_data.get('market_data', {}).get('volume', 0),
                        'ce_oi': call_data.get('market_data', {}).get('oi', 0),
                        'ce_iv': call_data.get('option_greeks', {}).get('iv', 0),
                        'ce_delta': call_data.get('option_greeks', {}).get('delta', 0),
                        'ce_gamma': call_data.get('option_greeks', {}).get('gamma', 0),
                        'ce_theta': call_data.get('option_greeks', {}).get('theta', 0),
                        'ce_vega': call_data.get('option_greeks', {}).get('vega', 0),
                        
                        'pe_instrument_key': put_data.get('instrument_key') if put_data else '',
                        'pe_ltp': put_data.get('market_data', {}).get('ltp', 0),
                        'pe_volume': put_data.get('market_data', {}).get('volume', 0),
                        'pe_oi': put_data.get('market_data', {}).get('oi', 0),
                        'pe_iv': put_data.get('option_greeks', {}).get('iv', 0),
                        'pe_delta': put_data.get('option_greeks', {}).get('delta', 0),
                        'pe_gamma': put_data.get('option_greeks', {}).get('gamma', 0),
                        'pe_theta': put_data.get('option_greeks', {}).get('theta', 0),
                        'pe_vega': put_data.get('option_greeks', {}).get('vega', 0),
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
        """Convert position object to dict"""
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
        """Convert holding object to dict"""
        return {
            'instrument_token': getattr(holding, 'instrument_token', ''),
            'trading_symbol': getattr(holding, 'trading_symbol', ''),
            'quantity': getattr(holding, 'quantity', 0),
            'average_price': getattr(holding, 'average_price', 0),
            'last_price': getattr(holding, 'last_price', 0),
            'pnl': getattr(holding, 'pnl', 0)
        }
    
    def _order_to_dict(self, order) -> Dict:
        """Convert order object to dict"""
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
        """Convert trade object to dict"""
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
                
                return data
                
        except Exception as e:
            print(f"Failed to get brokerage: {e}")
            
        return {}
