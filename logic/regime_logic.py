"""
VOLGUARD v31.0 – REGIME LOGIC
============================
The Brain - Analytics Engine, Regime Engine, Display Engine
Complete regime detection and analysis logic
"""

import subprocess
import sys
import warnings
from urllib.parse import quote
import io
import pandas as pd
import numpy as np
import requests
from colorama import Fore, Style, init
from tabulate import tabulate
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from arch import arch_model
import pytz
from scipy.stats import norm

init(autoreset=True)
warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
class Config:
    # API
    UPSTOX_BASE_V2 = "https://api.upstox.com/v2"
    UPSTOX_BASE_V3 = "https://api.upstox.com/v3"
    NIFTY_KEY = "NSE_INDEX|Nifty 50"
    VIX_KEY  = "NSE_INDEX|India VIX"

    # CAPITAL
    BASE_CAPITAL    = 10_00_000
    MARGIN_SELL_BASE = 1_25_000
    MARGIN_BUY_BASE  = 30_000

    # RISK
    GAMMA_DANGER_DTE      = 1
    GEX_STICKY_RATIO      = 0.03
    HIGH_VOL_IVP          = 75.0
    LOW_VOL_IVP           = 25.0
    VOV_CRASH_ZSCORE      = 2.5
    VOV_WARNING_ZSCORE    = 2.0

    # NEW v31.0
    SKEW_CRASH_FEAR         = 5.0
    SKEW_MELT_UP            = -2.0
    VIX_MOMENTUM_BREAKOUT   = 5.0
    FII_STRONG_CONVICTION   = 100_000
    FII_MODERATE_CONVICTION = 50_000
    REGIME_STABILITY_THRESHOLD = 0.60

    # WEIGHTS
    WEIGHT_VOL   = 0.40
    WEIGHT_STRUCT= 0.30
    WEIGHT_EDGE  = 0.20
    WEIGHT_RISK  = 0.10

# ----------------------------------------------------------
# DATA MODELS
# ----------------------------------------------------------
@dataclass
class TimeMetrics:
    current_date: date
    weekly_exp: date
    monthly_exp: date
    next_weekly_exp: date
    dte_weekly: int
    dte_monthly: int
    is_gamma_week: bool
    is_gamma_month: bool
    days_to_next_weekly: int

@dataclass
class VolMetrics:
    spot: float
    vix: float
    rv7: float; rv28: float; rv90: float
    garch7: float; garch28: float
    park7: float; park28: float
    vov: float; vov_zscore: float
    ivp_30d: float; ivp_90d: float; ivp_1yr: float
    ma20: float; atr14: float; trend_strength: float
    vol_regime: str
    is_fallback: bool
    vix_change_5d: float
    vix_momentum: str

@dataclass
class StructMetrics:
    net_gex: float; gex_ratio: float; total_oi_value: float
    gex_regime: str; pcr: float; max_pain: float
    skew_25d: float; oi_regime: str; lot_size: int
    pcr_atm: float
    skew_regime: str
    gex_weighted: float

@dataclass
class EdgeMetrics:
    iv_weekly: float; vrp_rv_weekly: float; vrp_garch_weekly: float; vrp_park_weekly: float
    iv_monthly: float; vrp_rv_monthly: float; vrp_garch_monthly: float; vrp_park_monthly: float
    term_spread: float; term_regime: str; primary_edge: str
    weighted_vrp_weekly: float
    weighted_vrp_monthly: float

@dataclass
class ParticipantData:
    fut_long: float; fut_short: float; fut_net: float
    call_long: float; call_short: float; call_net: float
    put_long: float; put_short: float; put_net: float
    stock_net: float

@dataclass
class ExternalMetrics:
    fii: Optional[ParticipantData]; dii: Optional[ParticipantData]
    pro: Optional[ParticipantData]; client: Optional[ParticipantData]
    fii_net_change: float
    flow_regime: str
    flow_score: float
    option_bias: float
    data_date: str
    is_fallback_data: bool
    event_risk: str
    fast_vol: bool
    fii_conviction: str
    flow_regime_full: str

@dataclass
class RegimeScore:
    vol_score: float
    struct_score: float
    edge_score: float
    risk_score: float
    composite: float
    confidence: str
    score_stability: float
    transition_probability: Dict[str, float] = field(default_factory=dict)

@dataclass
class TradingMandate:
    expiry_type: str
    expiry_date: date
    dte: int
    regime_name: str
    strategy_type: str
    allocation_pct: float
    max_lots: int
    risk_per_lot: float
    score: RegimeScore
    rationale: List[str]
    warnings: List[str]
    suggested_structure: str
    regime_persistence: float

# ----------------------------------------------------------
# NSE PARTICIPANT DATA ENGINE
# ----------------------------------------------------------
class ParticipantDataFetcher:
    @staticmethod
    def get_candidate_dates(days_back: int = 5):
        tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(tz)
        dates = []
        candidate = now
        if candidate.hour < 18:
            candidate -= timedelta(days=1)
        while len(dates) < days_back:
            if candidate.weekday() < 5:
                dates.append(candidate)
            candidate -= timedelta(days=1)
        return dates

    @staticmethod
    def fetch_oi_csv(date_obj):
        date_str = date_obj.strftime('%d%m%Y')
        url = f"https://archives.nseindia.com/content/nsccl/fao_participant_oi_{date_str}.csv"
        try:
            headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*", "Connection": "keep-alive"}
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                content = r.content.decode('utf-8')
                if "Future Index Long" in content:
                    lines = content.splitlines()
                    for idx, line in enumerate(lines[:20]):
                        if "Future Index Long" in line:
                            df = pd.read_csv(io.StringIO(content), skiprows=idx)
                            df.columns = df.columns.str.strip()
                            return df
        except Exception:
            pass
        return None

    @staticmethod
    def process_participant_data(df) -> Dict[str, ParticipantData]:
        data = {}
        for p in ["FII", "DII", "Client", "Pro"]:
            try:
                row = df[df['Client Type'].astype(str).str.contains(p, case=False, na=False)].iloc[0]
                def get_val(col): return float(str(row[col]).replace(',', ''))
                data[p] = ParticipantData(
                    fut_long=get_val('Future Index Long'),
                    fut_short=get_val('Future Index Short'),
                    fut_net=get_val('Future Index Long') - get_val('Future Index Short'),
                    call_long=get_val('Option Index Call Long'),
                    call_short=get_val('Option Index Call Short'),
                    call_net=get_val('Option Index Call Long') - get_val('Option Index Call Short'),
                    put_long=get_val('Option Index Put Long'),
                    put_short=get_val('Option Index Put Short'),
                    put_net=get_val('Option Index Put Long') - get_val('Option Index Put Short'),
                    stock_net=get_val('Future Stock Long') - get_val('Future Stock Short'))
            except Exception:
                data[p] = None
        return data

    @classmethod
    def fetch_smart_participant_data(cls):
        print("  📊 Connecting to NSE Archives...")
        dates = cls.get_candidate_dates()
        primary_data = None
        primary_date = None
        secondary_data = None
        for d in dates:
            print(f"     > Trying {d.strftime('%d-%b')}...", end=" ")
            df = cls.fetch_oi_csv(d)
            if df is not None:
                primary_data = cls.process_participant_data(df)
                primary_date = d
                print(f"{Fore.GREEN}FOUND{Style.RESET_ALL}")
                prev = d - timedelta(days=1)
                while prev.weekday() >= 5:
                    prev -= timedelta(days=1)
                df_prev = cls.fetch_oi_csv(prev)
                if df_prev is not None:
                    secondary_data = cls.process_participant_data(df_prev)
                break
            else:
                print(f"{Fore.RED}MISSING{Style.RESET_ALL}")
        if primary_data is None:
            return None, None, 0.0, "NO DATA", False
        fii_net_change = 0.0
        if primary_data.get('FII') and secondary_data and secondary_data.get('FII'):
            fii_net_change = primary_data['FII'].fut_net - secondary_data['FII'].fut_net
        is_fallback = primary_date.date() != dates[0].date()
        date_str = primary_date.strftime('%d-%b-%Y')
        return primary_data, secondary_data, fii_net_change, date_str, is_fallback

# ----------------------------------------------------------
# ANALYTICS ENGINE
# ----------------------------------------------------------
class AnalyticsEngine:
    def __init__(self):
        self.regime_history = []

    def get_time_metrics(self, weekly, monthly, next_weekly) -> TimeMetrics:
        today = date.today()
        dte_w = (weekly - today).days
        dte_m = (monthly - today).days
        dte_nw = (next_weekly - today).days
        return TimeMetrics(today, weekly, monthly, next_weekly, dte_w, dte_m,
                          dte_w <= Config.GAMMA_DANGER_DTE, dte_m <= Config.GAMMA_DANGER_DTE, dte_nw)

    def get_vol_metrics(self, nifty_hist, vix_hist, spot_live, vix_live) -> VolMetrics:
        is_fallback = False
        spot = spot_live if spot_live > 0 else (nifty_hist.iloc[-1]['close'] if not nifty_hist.empty else 0)
        vix = vix_live if vix_live > 0 else (vix_hist.iloc[-1]['close'] if not vix_hist.empty else 0)
        if spot_live <= 0 or vix_live <= 0: is_fallback = True

        returns = np.log(nifty_hist['close'] / nifty_hist['close'].shift(1)).dropna()
        rv7 = returns.rolling(7).std().iloc[-1] * np.sqrt(252) * 100
        rv28 = returns.rolling(28).std().iloc[-1] * np.sqrt(252) * 100
        rv90 = returns.rolling(90).std().iloc[-1] * np.sqrt(252) * 100

        def fit_garch(horizon):
            try:
                if len(returns) < 100: return 0
                model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal')
                res = model.fit(disp='off', show_warning=False)
                forecast = res.forecast(horizon=horizon, reindex=False)
                return np.sqrt(forecast.variance.values[-1, -1]) * np.sqrt(252)
            except: return 0

        garch7 = fit_garch(7) or rv7
        garch28 = fit_garch(28) or rv28

        const = 1.0 / (4.0 * np.log(2.0))
        park7 = np.sqrt((np.log(nifty_hist['high'] / nifty_hist['low']) ** 2).tail(7).mean() * const) * np.sqrt(252) * 100
        park28 = np.sqrt((np.log(nifty_hist['high'] / nifty_hist['low']) ** 2).tail(28).mean() * const) * np.sqrt(252) * 100

        vix_returns = np.log(vix_hist['close'] / vix_hist['close'].shift(1)).dropna()
        vov = vix_returns.rolling(30).std().iloc[-1] * np.sqrt(252) * 100
        vov_rolling = vix_returns.rolling(30).std() * np.sqrt(252) * 100
        vov_mean = vov_rolling.rolling(60).mean().iloc[-1]
        vov_std  = vov_rolling.rolling(60).std().iloc[-1]
        vov_zscore = (vov - vov_mean) / vov_std if vov_std > 0 else 0

        def calc_ivp(window):
            if len(vix_hist) < window: return 0.0
            history = vix_hist['close'].tail(window)
            return (history < vix).mean() * 100

        ivp_30d, ivp_90d, ivp_1yr = calc_ivp(30), calc_ivp(90), calc_ivp(252)

        ma20 = nifty_hist['close'].rolling(20).mean().iloc[-1]
        high_low = nifty_hist['high'] - nifty_hist['low']
        high_close = (nifty_hist['high'] - nifty_hist['close'].shift(1)).abs()
        low_close  = (nifty_hist['low']  - nifty_hist['close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr14 = true_range.rolling(14).mean().iloc[-1]
        trend_strength = abs(spot - ma20) / atr14 if atr14 > 0 else 0

        # v31 – VIX momentum
        vix_5d_ago = vix_hist['close'].iloc[-6] if len(vix_hist) >= 6 else vix
        vix_change_5d = ((vix / vix_5d_ago) - 1) * 100 if vix_5d_ago > 0 else 0
        if vix_change_5d > Config.VIX_MOMENTUM_BREAKOUT:
            vix_momentum = "RISING"
        elif vix_change_5d < -Config.VIX_MOMENTUM_BREAKOUT:
            vix_momentum = "FALLING"
        else:
            vix_momentum = "STABLE"

        # vol regime
        if vov_zscore > Config.VOV_CRASH_ZSCORE:
            vol_regime = "EXPLODING"
        elif ivp_1yr > Config.HIGH_VOL_IVP and vix_momentum == "FALLING":
            vol_regime = "MEAN_REVERTING"
        elif ivp_1yr > Config.HIGH_VOL_IVP and vix_momentum == "RISING":
            vol_regime = "BREAKOUT_RICH"
        elif ivp_1yr > Config.HIGH_VOL_IVP:
            vol_regime = "RICH"
        elif ivp_1yr < Config.LOW_VOL_IVP:
            vol_regime = "CHEAP"
        else:
            vol_regime = "FAIR"

        return VolMetrics(spot, vix, rv7, rv28, rv90, garch7, garch28, park7, park28,
                         vov, vov_zscore, ivp_30d, ivp_90d, ivp_1yr, ma20, atr14,
                         trend_strength, vol_regime, is_fallback, vix_change_5d, vix_momentum)

    def get_struct_metrics(self, chain, spot, lot_size) -> StructMetrics:
        if chain.empty or spot == 0:
            return StructMetrics(0, 0, 0, "NEUTRAL", 0, 0, 0, "NEUTRAL", lot_size,
                               0, "BALANCED", 0)

        # v31 – moneyness weighting
        chain['moneyness'] = chain['strike'] / spot
        chain['proximity_weight'] = np.exp(-((chain['strike'] - spot) / spot) ** 2 / 0.02)

        gex_weighted = ((chain['ce_gamma'] * chain['ce_oi'] * chain['proximity_weight']).sum() -
                       (chain['pe_gamma'] * chain['pe_oi'] * chain['proximity_weight']).sum()) * spot * lot_size

        # legacy GEX (10 % band)
        subset = chain[(chain['strike'] > spot * 0.90) & (chain['strike'] < spot * 1.10)]
        net_gex = ((subset['ce_gamma'] * subset['ce_oi']).sum() -
                   (subset['pe_gamma'] * subset['pe_oi']).sum()) * spot * lot_size

        total_oi_value = (chain['ce_oi'].sum() + chain['pe_oi'].sum()) * spot * lot_size
        gex_ratio = abs(gex_weighted) / total_oi_value if total_oi_value > 0 else 0
        gex_regime = "STICKY" if gex_ratio > Config.GEX_STICKY_RATIO else \
                     "SLIPPERY" if gex_ratio < Config.GEX_STICKY_RATIO * 0.5 else "NEUTRAL"

        # v31 – ATM PCR (±5 %)
        atm_chain = chain[(chain['strike'] >= spot * 0.95) & (chain['strike'] <= spot * 1.05)]
        pcr_atm = atm_chain['pe_oi'].sum() / atm_chain['ce_oi'].sum() if atm_chain['ce_oi'].sum() > 0 else 1.0

        # legacy PCR
        pcr = chain['pe_oi'].sum() / chain['ceI need to continue with the remaining files. Let me provide the complete trading_logic.py and the rest of the package.
