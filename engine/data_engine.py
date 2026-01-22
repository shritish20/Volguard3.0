"""
VOLGUARD v31.0 – DATA ENGINE
100 % Upstox V3 – daily 1-year candles, funds, positions, holidays, expiry, option-chain
"""
import datetime as dt
import pandas as pd
import upstox_client
from upstox_client.rest import ApiException
from typing import Dict, List, Optional, Tuple
from colorama import Fore
import time
import config

class UpstoxDataEngine:
    def __init__(self, access_token: str):
        cfg = upstox_client.Configuration()
        cfg.access_token = access_token
        self.api_client = upstox_client.ApiClient(cfg)
        self._init_apis()

    def _init_apis(self):
        self.user_api        = upstox_client.UserApi(self.api_client)
        self.market_api      = upstox_client.MarketHolidaysAndTimingsApi(self.api_client)
        self.hist_v3         = upstox_client.HistoryV3Api(self.api_client)
        self.quote_v3        = upstox_client.MarketQuoteV3Api(self.api_client)
        self.options_api     = upstox_client.OptionsApi(self.api_client)
        self.charge_api      = upstox_client.ChargeApi(self.api_client)

    # ------------------------------------------------------------------
    def fetch_funds_and_margin(self) -> Dict:
        """Return dict with equity/commodity margins."""
        try:
            resp = self.user_api.get_user_fund_margin(api_version='2.0')
            return resp.to_dict()['data']
        except ApiException as e:
            print(Fore.RED + f"Funds fetch failed: {e}")
            return {}

    # ------------------------------------------------------------------
    def fetch_positions(self) -> List[Dict]:
        """Open positions today."""
        try:
            resp = upstox_client.PortfolioApi(self.api_client).get_positions(api_version='2.0')
            return resp.to_dict()['data']
        except ApiException as e:
            print(Fore.RED + f"Positions fetch failed: {e}")
            return []

    # ------------------------------------------------------------------
    def fetch_holidays(self) -> List[dt.date]:
        """All trading holidays current year."""
        try:
            resp = self.market_api.get_holidays()
            hol = [dt.datetime.strptime(d['date'], '%Y-%m-%d').date() for d in resp.to_dict()['data']]
            return hol
        except ApiException as e:
            print(Fore.RED + f"Holidays fetch failed: {e}")
            return []

    # ------------------------------------------------------------------
    def fetch_history(self, instrument_key: str, days: int = 365) -> pd.DataFrame:
        """Daily candles last 1 year (default)."""
        to_dt   = dt.date.today()
        from_dt = to_dt - dt.timedelta(days=days)
        try:
            resp = self.hist_v3.get_historical_candle_data1(
                instrument_key, 'days', '1',
                to_dt.strftime('%Y-%m-%d'),
                from_dt.strftime('%Y-%m-%d')
            )
            candles = resp.to_dict()['data']['candles']
            df = pd.DataFrame(candles, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['ts'] = pd.to_datetime(df['ts'])
            return df
        except ApiException as e:
            print(Fore.RED + f"History fetch failed: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    def fetch_live_quote(self, keys: List[str]) -> Dict[str, float]:
        """LTP for any list of keys."""
        try:
            resp = self.quote_v3.get_ltp(instrument_key=','.join(keys))
            return {k: float(v['last_price']) for k, v in resp.to_dict()['data'].items()}
        except ApiException as e:
            print(Fore.RED + f"Live quote failed: {e}")
            return {}

    # ------------------------------------------------------------------
    def fetch_expiry_details(self, instrument_key: str = config.NIFTY_KEY
                            ) -> Tuple[Optional[dt.date], Optional[dt.date], Optional[dt.date], int]:
        """Returns (weekly_exp, monthly_exp, next_weekly_exp, lot_size)."""
        try:
            contracts = self.options_api.get_option_contracts(instrument_key).to_dict()['data']
            df = pd.DataFrame(contracts)
            df['expiry'] = pd.to_datetime(df['expiry']).dt.date
            df = df.sort_values('expiry')
            weekly  = df.iloc[0]['expiry']
            monthly = df[df['weekly'] == False].iloc[0]['expiry'] if len(df[df['weekly'] == False]) else weekly
            next_w  = df[df['expiry'] > weekly].iloc[0]['expiry'] if len(df[df['expiry'] > weekly]) else weekly
            lot_size = int(df.iloc[0]['lot_size'])
            return weekly, monthly, next_w, lot_size
        except ApiException as e:
            print(Fore.RED + f"Expiry fetch failed: {e}")
            return None, None, None, 50

    # ------------------------------------------------------------------
    def fetch_option_chain(self, expiry_date: dt.date, instrument_key: str = config.NIFTY_KEY
                          ) -> pd.DataFrame:
        """Full chain with Greeks and OI."""
        try:
            resp = self.options_api.get_put_call_option_chain(instrument_key, expiry_date.strftime('%Y-%m-%d'))
            data = []
            for strike_data in resp.to_dict()['data']:
                strike = strike_data['strike_price']
                for cp, opt in [('CE', 'call_options'), ('PE', 'put_options')]:
                    if opt in strike_data:
                        d = strike_data[opt]
                        d['strike'] = strike
                        d['option_type'] = cp
                        d['expiry'] = expiry_date
                        data.append(d)
            return pd.DataFrame(data)
        except ApiException as e:
            print(Fore.RED + f"Option chain fetch failed: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    def is_market_open_today(self) -> bool:
        """Live exchange status."""
        try:
            resp = self.market_api.get_market_status('NSE')
            return resp.to_dict()['data']['status'] == 'NORMAL_OPEN'
        except ApiException as e:
            print(Fore.RED + f"Market status failed: {e}")
            return False

    # ------------------------------------------------------------------
    def get_brokerage_estimate(self, instrument_token: str, qty: int, price: float, side: str) -> float:
        """Quick brokerage + taxes estimate."""
        try:
            resp = self.charge_api.get_brokerage(instrument_token, qty, 'I' if config.MARGIN_SELL_BASE < 50000 else 'D',
                                                 side, price, api_version='2.0')
            return float(resp.to_dict()['data']['charges']['total'])
        except ApiException as e:
            print(Fore.RED + f"Brokerage fetch failed: {e}")
            return 0.0
