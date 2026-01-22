"""
VOLGUARD v31.0 – DATA ENGINE
============================
The Eyes - Fetches all market data, funds, positions, holidays
Single source of truth for the entire system
"""

import pandas as pd
from datetime import datetime, date, timedelta
import upstox_client
from upstox_client.rest import ApiException
from typing import Dict, List, Optional, Tuple
from colorama import Fore, Style

class UpstoxDataEngine:
    """Complete data fetching engine for VOLGUARD v31.0"""
    
    def __init__(self, access_token: str):
        self.config = upstox_client.Configuration()
        self.config.access_token = access_token
        self.api_client = upstox_client.ApiClient(self.config)
        
        # Initialize all API endpoints
        self.history_api = upstox_client.HistoryV3Api(self.api_client)
        self.market_quote_api = upstox_client.MarketQuoteV3Api(self.api_client)
        self.options_api = upstox_client.OptionsApi(self.api_client)
        self.user_api = upstox_client.UserApi(self.api_client)
        self.portfolio_api = upstox_client.PortfolioApi(self.api_client)
        self.meta_api = upstox_client.MarketHolidaysAndTimingsApi(self.api_client)
        
        # Cache for efficiency
        self.holidays_cache = []
        self.instrument_cache = {}
        
        print(f"{Fore.GREEN}✓ UpstoxDataEngine initialized{Style.RESET_ALL}")
    
    def fetch_funds_and_margin(self) -> Dict:
        """CRITICAL: Checks available capital before any trade decision"""
        try:
            response = self.user_api.get_user_fund_margin("2.0")
            if response.status == "success":
                equity_funds = response.data.equity
                return {
                    "available_margin": float(equity_funds.available_margin),
                    "used_margin": float(equity_funds.used_margin),
                    "total_balance": float(equity_funds.total_balance),
                    "collateral": float(equity_funds.collateral)
                }
        except ApiException as e:
            print(f"{Fore.RED}❌ Funds Fetch Error: {e}{Style.RESET_ALL}")
            return {"available_margin": 0.0, "used_margin": 0.0, "total_balance": 0.0}
        except Exception as e:
            print(f"{Fore.RED}❌ Unexpected funds error: {e}{Style.RESET_ALL}")
            return {"available_margin": 0.0, "used_margin": 0.0, "total_balance": 0.0}
    
    def fetch_positions(self) -> List[Dict]:
        """CRITICAL: Fetches existing open positions for State Recovery"""
        try:
            response = self.portfolio_api.get_positions("2.0")
            if response.status == "success":
                positions = []
                for pos in response.data:
                    if pos.quantity != 0:  # Only open positions
                        positions.append({
                            "position_id": f"{pos.symbol}_{pos.product}_{pos.expiry_date}",
                            "instrument_token": pos.instrument_token,
                            "symbol": pos.symbol,
                            "quantity": pos.quantity,
                            "product": pos.product,
                            "expiry_date": pos.expiry_date,
                            "strike_price": float(pos.strike_price),
                            "option_type": pos.option_type,
                            "entry_price": float(pos.entry_price),
                            "current_price": float(pos.current_price),
                            "pnl": float(pos.pnl),
                            "total_value": float(pos.total_value),
                            "trading_symbol": pos.trading_symbol
                        })
                return positions
        except ApiException as e:
            print(f"{Fore.RED}❌ Portfolio Fetch Error: {e}{Style.RESET_ALL}")
            return []
        except Exception as e:
            print(f"{Fore.RED}❌ Unexpected portfolio error: {e}{Style.RESET_ALL}")
            return []
    
    def fetch_holidays(self) -> List[date]:
        """Fetches market holidays to correct DTE calculations"""
        if self.holidays_cache:
            return self.holidays_cache
            
        try:
            response = self.meta_api.get_holidays()
            if response.status == "success":
                holidays = []
                for holiday in response.data:
                    if holiday.holiday_type == "TRADING_HOLIDAY" and "NSE" in holiday.closed_exchanges:
                        holidays.append(datetime.strptime(holiday.date, "%Y-%m-%d").date())
                
                self.holidays_cache = holidays
                return holidays
        except ApiException as e:
            print(f"{Fore.RED}❌ Holiday Fetch Error: {e}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Unexpected holiday error: {e}{Style.RESET_ALL}")
        
        return []
    
    def fetch_history(self, instrument_key: str, days: int = 400) -> pd.DataFrame:
        """Fetches Daily Candles for Volatility/GARCH models"""
        try:
            to_date = date.today().strftime("%Y-%m-%d")
            from_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            response = self.history_api.get_historical_candle_data1(
                instrument_key, "days", "1", to_date, from_date
            )
            
            if response.status == "success" and response.data.candles:
                df = pd.DataFrame(response.data.candles, 
                                columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df.astype(float).sort_index()
                
        except ApiException as e:
            print(f"{Fore.RED}❌ History Error for {instrument_key}: {e}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Unexpected history error: {e}{Style.RESET_ALL}")
        
        return pd.DataFrame()
    
    def fetch_live_quote(self, keys: List[str]) -> Dict[str, float]:
        """Fetches real-time LTP for Spot & VIX"""
        try:
            key_str = ",".join(keys)
            response = self.market_quote_api.get_ltp(key_str)
            
            result = {}
            if response.status == "success":
                for key in keys:
                    data_key = key.replace("|", ":")
                    if data_key in response.data:
                        result[key] = float(response.data[data_key].last_price)
                    elif key in response.data:
                        result[key] = float(response.data[key].last_price)
            
            return result
            
        except ApiException as e:
            print(f"{Fore.RED}❌ Live Quote Error: {e}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Unexpected live quote error: {e}{Style.RESET_ALL}")
        
        return {}
    
    def fetch_expiry_details(self, instrument_key: str = "NSE_INDEX|Nifty 50") -> Tuple[Optional[date], Optional[date], Optional[date], int]:
        """Dynamically fetches Expiries and Lot Size"""
        try:
            response = self.options_api.get_option_contracts(instrument_key)
            
            if response.status == "success":
                data = response.data
                if not data:
                    return None, None, None, 0
                
                lot_size = int(data[0].lot_size) if hasattr(data[0], 'lot_size') else 50
                
                expiry_dates = []
                for contract in data:
                    if hasattr(contract, 'expiry') and contract.expiry:
                        expiry_dates.append(datetime.strptime(contract.expiry, "%Y-%m-%d").date())
                
                expiry_dates = sorted(list(set(expiry_dates)))
                valid_dates = [d for d in expiry_dates if d >= date.today()]
                
                if not valid_dates:
                    return None, None, None, lot_size
                
                weekly = valid_dates[0]
                next_weekly = valid_dates[1] if len(valid_dates) > 1 else valid_dates[0]
                
                current_month = weekly.month
                monthly_candidates = [d for d in valid_dates if d.month == current_month]
                monthly = monthly_candidates[-1] if monthly_candidates else valid_dates[-1]
                
                return weekly, monthly, next_weekly, lot_size
                
        except ApiException as e:
            print(f"{Fore.RED}❌ Expiry Error: {e}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Unexpected expiry error: {e}{Style.RESET_ALL}")
        
        return None, None, None, 0
    
    def fetch_option_chain(self, expiry_date: date, instrument_key: str = "NSE_INDEX|Nifty 50") -> pd.DataFrame:
        """Fetches Chain & Maps Strikes to Instrument Keys"""
        try:
            expiry_str = expiry_date.strftime("%Y-%m-%d")
            response = self.options_api.get_put_call_option_chain(instrument_key, expiry_str)
            
            if response.status == "success":
                data = response.data
                chain_list = []
                
                for option_data in data:
                    call_key = None
                    call_data = {'iv': 0, 'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'oi': 0, 'ltp': 0}
                    
                    if hasattr(option_data, 'call_options') and option_data.call_options:
                        call = option_data.call_options
                        call_key = call.instrument_key
                        if hasattr(call, 'option_greeks'):
                            call_data.update({
                                'iv': float(call.option_greeks.iv),
                                'delta': float(call.option_greeks.delta),
                                'gamma': float(call.option_greeks.gamma),
                                'theta': float(call.option_greeks.theta),
                                'vega': float(call.option_greeks.vega)
                            })
                        if hasattr(call, 'market_data'):
                            call_data.update({
                                'oi': float(call.market_data.oi),
                                'ltp': float(call.market_data.ltp)
                            })
                    
                    put_key = None
                    put_data = {'iv': 0, 'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'oi': 0, 'ltp': 0}
                    
                    if hasattr(option_data, 'put_options') and option_data.put_options:
                        put = option_data.put_options
                        put_key = put.instrument_key
                        if hasattr(put, 'option_greeks'):
                            put_data.update({
                                'iv': float(put.option_greeks.iv),
                                'delta': float(put.option_greeks.delta),
                                'gamma': float(put.option_greeks.gamma),
                                'theta': float(put.option_greeks.theta),
                                'vega': float(put.option_greeks.vega)
                            })
                        if hasattr(put, 'market_data'):
                            put_data.update({
                                'oi': float(put.market_data.oi),
                                'ltp': float(put.market_data.ltp)
                            })
                    
                    chain_list.append({
                        'strike': float(option_data.strike_price),
                        'ce_iv': call_data['iv'], 'ce_delta': call_data['delta'], 'ce_gamma': call_data['gamma'],
                        'ce_oi': call_data['oi'], 'ce_ltp': call_data['ltp'], 'ce_key': call_key,
                        'pe_iv': put_data['iv'], 'pe_delta': put_data['delta'], 'pe_gamma': put_data['gamma'],
                        'pe_oi': put_data['oi'], 'pe_ltp': put_data['ltp'], 'pe_key': put_key
                    })
                
                return pd.DataFrame(chain_list)
                
        except ApiException as e:
            print(f"{Fore.RED}❌ Chain Error: {e}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Unexpected chain error: {e}{Style.RESET_ALL}")
        
        return pd.DataFrame()
