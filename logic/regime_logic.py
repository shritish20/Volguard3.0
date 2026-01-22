"""
VOLGUARD v31.0 – REGIME LOGIC
============================
The Brain - Analytics Engine, Regime Engine, Display Engine
Complete regime detection and analysis logic
"""

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
import io
from urllib.parse import quote

init(autoreset=True)

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
        pcr = chain['pe_oi'].sum() / chain['ce_oi'].sum() if chain['ce_oi'].sum() > 0 else 1.0

        # max pain
        strikes = chain['strike'].values
        losses = []
        for strike in strikes:
            call_loss = np.sum(np.maximum(0, strike - strikes) * chain['ce_oi'].values)
            put_loss  = np.sum(np.maximum(0, strikes - strike) * chain['pe_oi'].values)
            losses.append(call_loss + put_loss)
        max_pain = strikes[np.argmin(losses)] if losses else 0

        # 25Δ skew
        try:
            ce_25d_idx = (chain['ce_delta'].abs() - 0.25).abs().argsort()[:1]
            pe_25d_idx = (chain['pe_delta'].abs() - 0.25).abs().argsort()[:1]
            skew_25d = chain.iloc[pe_25d_idx]['pe_iv'].values[0] - chain.iloc[ce_25d_idx]['ce_iv'].values[0]
        except:
            skew_25d = 0

        # v31 – skew regime
        if skew_25d > Config.SKEW_CRASH_FEAR:
            skew_regime = "CRASH_FEAR"
        elif skew_25d < Config.SKEW_MELT_UP:
            skew_regime = "MELT_UP"
        else:
            skew_regime = "BALANCED"

        oi_regime = "BULLISH" if pcr_atm > 1.2 else "BEARISH" if pcr_atm < 0.8 else "NEUTRAL"

        return StructMetrics(net_gex, gex_ratio, total_oi_value, gex_regime, pcr, max_pain,
                           skew_25d, oi_regime, lot_size, pcr_atm, skew_regime, gex_weighted)

    def get_edge_metrics(self, weekly_chain, monthly_chain, spot, vol: VolMetrics) -> EdgeMetrics:
        def get_atm_iv(chain):
            if chain.empty or spot == 0: return 0
            atm_idx = (chain['strike'] - spot).abs().argsort()[:1]
            row = chain.iloc[atm_idx].iloc[0]
            return (row['ce_iv'] + row['pe_iv']) / 2

        iv_weekly = get_atm_iv(weekly_chain)
        iv_monthly = get_atm_iv(monthly_chain)

        vrp_rv_weekly = iv_weekly - vol.rv7
        vrp_garch_weekly = iv_weekly - vol.garch7
        vrp_park_weekly = iv_weekly - vol.park7

        vrp_rv_monthly = iv_monthly - vol.rv28
        vrp_garch_monthly = iv_monthly - vol.garch28
        vrp_park_monthly = iv_monthly - vol.park28

        # weighted VRP 70/15/15
        weighted_vrp_weekly = (vrp_garch_weekly * 0.70) + (vrp_park_weekly * 0.15) + (vrp_rv_weekly * 0.15)
        weighted_vrp_monthly = (vrp_garch_monthly * 0.70) + (vrp_park_monthly * 0.15) + (vrp_rv_monthly * 0.15)

        term_spread = iv_monthly - iv_weekly
        term_regime = "BACKWARDATION" if term_spread < -1.0 else "CONTANGO" if term_spread > 1.0 else "FLAT"

        primary_edge = "LONG_VOL" if vol.ivp_1yr < Config.LOW_VOL_IVP else \
                      "SHORT_GAMMA" if weighted_vrp_weekly > 4.0 and vol.ivp_1yr > 50 else \
                      "SHORT_VEGA" if weighted_vrp_monthly > 3.0 and vol.ivp_1yr > 50 else \
                      "CALENDAR_SPREAD" if term_regime == "BACKWARDATION" and term_spread < -2.0 else \
                      "MEAN_REVERSION" if vol.ivp_1yr > Config.HIGH_VOL_IVP else "NONE"

        return EdgeMetrics(iv_weekly, vrp_rv_weekly, vrp_garch_weekly, vrp_park_weekly,
                          iv_monthly, vrp_rv_monthly, vrp_garch_monthly, vrp_park_monthly,
                          term_spread, term_regime, primary_edge, weighted_vrp_weekly, weighted_vrp_monthly)

    def get_external_metrics(self, nifty_hist, participant_data, secondary_data, fii_net_change, data_date, is_fallback) -> ExternalMetrics:
        fast_vol = False
        if not nifty_hist.empty:
            last_bar = nifty_hist.iloc[-1]
            daily_range_pct = ((last_bar['high'] - last_bar['low']) / last_bar['open']) * 100
            fast_vol = daily_range_pct > 1.8

        flow_regime = "NEUTRAL"
        flow_score = 0.0
        option_bias = 0.0
        fii_conviction = "LOW"
        flow_regime_full = "NEUTRAL_LOW"

        if participant_data and participant_data.get('FII'):
            fii = participant_data['FII']
            fut_net = fii.fut_net
            option_bias = fii.call_net - fii.put_net
            flow_score = fut_net + (option_bias * 0.4)

            # conviction
            abs_score = abs(flow_score)
            if abs_score > Config.FII_STRONG_CONVICTION:
                fii_conviction = "VERY_HIGH"
            elif abs_score > Config.FII_MODERATE_CONVICTION:
                fii_conviction = "HIGH"
            elif abs_score > Config.FII_MODERATE_CONVICTION * 0.5:
                fii_conviction = "MODERATE"
            else:
                fii_conviction = "LOW"

            # matrix
            fut_bullish = fut_net > 0
            opt_bullish = option_bias > 0
            if fut_bullish and opt_bullish:
                flow_regime = "AGGRESSIVE_BULL"
            elif not fut_bullish and not opt_bullish:
                flow_regime = "AGGRESSIVE_BEAR"
            elif fut_bullish and not opt_bullish:
                flow_regime = "GUARDED_BULL"
            elif not fut_bullish and opt_bullish:
                flow_regime = "CONTRARIAN_TRAP"

            flow_regime_full = f"{flow_regime}_{fii_conviction}"

        return ExternalMetrics(
            fii=participant_data.get('FII') if participant_data else None,
            dii=participant_data.get('DII') if participant_data else None,
            pro=participant_data.get('Pro') if participant_data else None,
            client=participant_data.get('Client') if participant_data else None,
            fii_net_change=fii_net_change,
            flow_regime=flow_regime,
            flow_score=flow_score,
            option_bias=option_bias,
            data_date=data_date,
            is_fallback_data=is_fallback,
            event_risk="LOW",
            fast_vol=fast_vol,
            fii_conviction=fii_conviction,
            flow_regime_full=flow_regime_full)

# ----------------------------------------------------------
# WORLD-CLASS REGIME ENGINE
# ----------------------------------------------------------
class RegimeEngine:
    def __init__(self):
        self.regime_history = []
        self.transition_matrix = {}

    def calculate_scores(self, vol: VolMetrics, struct: StructMetrics, edge: EdgeMetrics,
                        external: ExternalMetrics, time: TimeMetrics, expiry_type: str) -> RegimeScore:
        weighted_vrp = edge.weighted_vrp_weekly if expiry_type == "WEEKLY" else edge.weighted_vrp_monthly

        # 1. EDGE SCORE
        edge_score = 5.0
        if weighted_vrp > 4.0: edge_score += 3.0
        elif weighted_vrp > 2.0: edge_score += 2.0
        elif weighted_vrp > 1.0: edge_score += 1.0
        elif weighted_vrp < 0: edge_score -= 3.0
        if edge.term_regime == "BACKWARDATION" and edge.term_spread < -2.0: edge_score += 1.0
        elif edge.term_regime == "CONTANGO": edge_score += 0.5
        edge_score = max(0, min(10, edge_score))

        # 2. VOL SCORE
        vol_score = 5.0
        if vol.vov_zscore > Config.VOV_CRASH_ZSCORE:
            vol_score = 0.0
        elif vol.vov_zscore > Config.VOV_WARNING_ZSCORE:
            vol_score -= 3.0
        elif vol.vov_zscore < 1.5:
            vol_score += 1.5

        if vol.ivp_1yr > Config.HIGH_VOL_IVP:
            if vol.vix_momentum == "FALLING":
                vol_score += 1.5
            elif vol.vix_momentum == "RISING":
                vol_score -= 1.0
            else:
                vol_score += 0.5
        elif vol.ivp_1yr < Config.LOW_VOL_IVP:
            vol_score -= 2.5
        else:
            vol_score += 1.0
        vol_score = max(0, min(10, vol_score))

        # 3. STRUCT SCORE
        struct_score = 5.0
        if struct.gex_regime == "STICKY":
            struct_score += 2.5 if expiry_type == "WEEKLY" and time.dte_weekly <= 1 else 1.0
        elif struct.gex_regime == "SLIPPERY":
            struct_score -= 1.0
        if 0.9 < struct.pcr_atm < 1.1:
            struct_score += 1.5
        elif struct.pcr_atm > 1.3 or struct.pcr_atm < 0.7:
            struct_score -= 0.5
        if struct.skew_regime == "CRASH_FEAR":
            struct_score -= 1.0
        elif struct.skew_regime == "MELT_UP":
            struct_score -= 0.5
        else:
            struct_score += 0.5
        struct_score = max(0, min(10, struct_score))

        # 4. RISK SCORE
        risk_score = 10.0
        if external.event_risk == "HIGH": risk_score -= 3.0
        if external.fast_vol: risk_score -= 2.0

        if external.flow_regime == "AGGRESSIVE_BEAR":
            penalty = 4.0 if external.fii_conviction == "VERY_HIGH" else 3.0
            risk_score -= penalty
        elif external.flow_regime == "AGGRESSIVE_BULL":
            bonus = 1.5 if external.fii_conviction == "VERY_HIGH" else 1.0
            risk_score += bonus
        elif external.flow_regime == "CONTRARIAN_TRAP":
            risk_score -= 2.0
        elif external.flow_regime == "GUARDED_BULL":
            risk_score -= 0.5

        if expiry_type == "WEEKLY" and time.is_gamma_week: risk_score -= 2.0
        elif expiry_type == "MONTHLY" and time.is_gamma_month: risk_score -= 2.5
        risk_score = max(0, min(10, risk_score))

        # COMPOSITE
        composite = (vol_score * Config.WEIGHT_VOL +
                    struct_score * Config.WEIGHT_STRUCT +
                    edge_score * Config.WEIGHT_EDGE +
                    risk_score * Config.WEIGHT_RISK)

        # SENSITIVITY – alternate weightings
        alt_weights = [(0.30, 0.30, 0.20, 0.20), (0.50, 0.20, 0.20, 0.10), (0.30, 0.40, 0.20, 0.10)]
        alt_scores = []
        for wv, ws, we, wr in alt_weights:
            alt_scores.append(vol_score * wv + struct_score * ws + edge_score * we + risk_score * wr)
        score_stability = 1.0 - (np.std(alt_scores) / np.mean(alt_scores)) if np.mean(alt_scores) > 0 else 0.5

        confidence = "VERY_HIGH" if composite >= 8.0 and score_stability > 0.85 else \
                    "HIGH" if composite >= 6.5 and score_stability > 0.75 else \
                    "MODERATE" if composite >= 4.0 else "LOW"

        return RegimeScore(vol_score, struct_score, edge_score, risk_score, composite, confidence, score_stability, {})

    def calculate_regime_persistence(self, current_regime: str, lookback: int = 5) -> float:
        if len(self.regime_history) < lookback: return 0.5
        recent = self.regime_history[-lookback:]
        return sum(1 for r in recent if r == current_regime) / lookback

    def update_transition_matrix(self, from_regime: str, to_regime: str):
        if from_regime not in self.transition_matrix:
            self.transition_matrix[from_regime] = {}
        if to_regime not in self.transition_matrix[from_regime]:
            self.transition_matrix[from_regime][to_regime] = 0
        self.transition_matrix[from_regime][to_regime] += 1

    def get_transition_probabilities(self, current_regime: str) -> Dict[str, float]:
        if current_regime not in self.transition_matrix: return {}
        transitions = self.transition_matrix[current_regime]
        total = sum(transitions.values())
        if total == 0: return {}
        return {regime: count / total for regime, count in transitions.items()}

    def generate_mandate(self, score: RegimeScore, vol: VolMetrics, struct: StructMetrics,
                        edge: EdgeMetrics, external: ExternalMetrics, time: TimeMetrics,
                        expiry_type: str, expiry_date: date, dte: int) -> TradingMandate:
        rationale = []
        warnings = []

        weighted_vrp = edge.weighted_vrp_weekly if expiry_type == "WEEKLY" else edge.weighted_vrp_monthly

        # regime classification
        if score.composite >= 7.5:
            regime_name = "AGGRESSIVE_SHORT"
            allocation = 60.0
            strategy = "AGGRESSIVE_SHORT"
            suggested = "STRANGLE"
            rationale.append(f"High Confidence ({score.confidence}): Weighted VRP {weighted_vrp:.2f}% is strong")
            rationale.append(f"Score Stability: {score.score_stability:.2%} (robust across weight variations)")
        elif score.composite >= 6.0:
            regime_name = "MODERATE_SHORT"
            allocation = 40.0
            strategy = "MODERATE_SHORT"
            suggested = "IRON_CONDOR" if dte > 1 else "IRON_FLY"
            rationale.append(f"Moderate Confidence: VRP {weighted_vrp:.2f}% is positive")
        elif score.composite >= 4.0:
            regime_name = "DEFENSIVE"
            allocation = 20.0
            strategy = "DEFENSIVE"
            suggested = "CREDIT_SPREAD"
            rationale.append("Defensive Posture: Focus on defined risk only")
        else:
            regime_name = "CASH"
            allocation = 0.0
            strategy = "CASH"
            suggested = "NONE"
            rationale.append("Regime Unfavorable: Cash is a position")

        # warnings
        if vol.vov_zscore > Config.VOV_WARNING_ZSCORE:
            warnings.append(f"⚠️ HIGH VOL-OF-VOL ({vol.vov_zscore:.2f}σ): Market unstable")
        if vol.vix_momentum == "RISING" and vol.ivp_1yr > 50:
            warnings.append(f"⚠️ VIX MOMENTUM: Rising {vol.vix_change_5d:+.1f}% in 5d – vol breakout risk")
        if struct.skew_regime == "CRASH_FEAR":
            warnings.append(f"⚠️ SKEW: Crash fear (25Δ skew {struct.skew_25d:+.2f}%) – puts expensive")
        if external.flow_regime == "AGGRESSIVE_BEAR":
            warnings.append(f"⚠️ FII AGGRESSIVE BEAR ({external.fii_conviction} conviction): Short Fut & Calls")
            if allocation > 0:
                allocation = min(allocation, 20.0)
                rationale.append(f"Allocation capped to 20% due to {external.fii_conviction} conviction bearish flow")
        elif external.flow_regime == "CONTRARIAN_TRAP":
            warnings.append(f"⚠️ FII TRAP SIGNAL: Short Fut but Buying Calls ({external.fii_conviction} conviction)")
            allocation = min(allocation, 30.0)

        if dte <= Config.GAMMA_DANGER_DTE and expiry_type == "WEEKLY":
            warnings.append(f"⚠️ GAMMA RISK: {dte} DTE – theta acceleration but pin risk high")
            allocation *= 0.5

        # regime persistence
        regime_persistence = self.calculate_regime_persistence(regime_name)
        if regime_persistence < Config.REGIME_STABILITY_THRESHOLD:
            warnings.append(f"⚠️ REGIME UNSTABLE: Only {regime_persistence:.0%} persistence over 5 days")
            allocation *= 0.7
        else:
            rationale.append(f"Regime Persistence: {regime_persistence:.0%} stability over 5 days")

        # history & transitions
        self.regime_history.append(regime_name)
        if len(self.regime_history) > 1:
            self.update_transition_matrix(self.regime_history[-2], regime_name)
        score.transition_probability = self.get_transition_probabilities(regime_name)

        # sizing
        deployable = Config.BASE_CAPITAL * (allocation / 100.0)
        risk_per_lot = Config.MARGIN_SELL_BASE if strategy != "DEFENSIVE" else Config.MARGIN_SELL_BASE * 0.6
        max_lots = int(deployable / risk_per_lot) if risk_per_lot > 0 else 0

        return TradingMandate(expiry_type, expiry_date, dte, regime_name, strategy,
                             allocation, max_lots, risk_per_lot, score, rationale,
                             warnings, suggested, regime_persistence)

# ----------------------------------------------------------
# DISPLAY ENGINE
# ----------------------------------------------------------
class DisplayEngine:
    def render_header(self):
        print(f"\n{Fore.CYAN}{'='*90}")
        print(f"{'VOLGUARD v31.0 – WORLD-CLASS REGIME DETECTOR':^90}")
        print(f"{'='*90}{Style.RESET_ALL}\n")

    def render_time_context(self, time: TimeMetrics):
        print(f"{Fore.YELLOW}⏰ TIME CONTEXT{Style.RESET_ALL}")
        time_data = [
            ["Current Date", time.current_date.strftime("%Y-%m-%d")],
            ["Weekly Expiry", f"{time.weekly_exp} ({time.dte_weekly} DTE)"],
            ["Monthly Expiry", f"{time.monthly_exp} ({time.dte_monthly} DTE)"],
            ["Next Weekly", f"{time.next_weekly_exp} ({time.days_to_next_weekly} DTE)"],
            ["Gamma Status", f"{Fore.RED}DANGER ZONE{Style.RESET_ALL}" if time.is_gamma_week else f"{Fore.GREEN}THETA ZONE{Style.RESET_ALL}"]
        ]
        print(tabulate(time_data, tablefmt="fancy_grid"))

    def render_vol_metrics(self, vol: VolMetrics):
        print(f"\n{Fore.YELLOW}📊 VOLATILITY ANALYSIS{Style.RESET_ALL}")
        def ivp_tag(v):
            return f"{Fore.GREEN}CHEAP{Style.RESET_ALL}" if v < Config.LOW_VOL_IVP else \
                   f"{Fore.RED}RICH{Style.RESET_ALL}" if v > Config.HIGH_VOL_IVP else "FAIR"
        momentum_color = Fore.RED if vol.vix_momentum == "RISING" else Fore.GREEN if vol.vix_momentum == "FALLING" else Fore.YELLOW
        vol_data = [
            ["Spot", f"{vol.spot:.2f}", f"MA20: {vol.ma20:.2f}"],
            ["VIX", f"{vol.vix:.2f}", vol.vol_regime],
            ["VIX 5D Change", f"{momentum_color}{vol.vix_change_5d:+.1f}%{Style.RESET_ALL}", f"{vol.vix_momentum}"],
            ["IVP 30D/90D/1Y", f"{vol.ivp_30d:.1f}% / {vol.ivp_90d:.1f}% / {vol.ivp_1yr:.1f}%", ivp_tag(vol.ivp_1yr)],
            ["RV 7D/28D/90D", f"{vol.rv7:.1f} / {vol.rv28:.1f} / {vol.rv90:.1f}%", "Realized Vol"],
            ["GARCH 7D/28D", f"{vol.garch7:.1f} / {vol.garch28:.1f}%", "Forecasted Vol"],
            ["Parkinson 7D/28D", f"{vol.park7:.1f} / {vol.park28:.1f}%", "Range-Based Vol"],
            ["Vol-of-Vol", f"{vol.vov:.1f}%", f"Z-Score: {vol.vov_zscore:+.2f}σ"],
            ["Trend Strength", f"{vol.trend_strength:.2f}", "ATR-Normalized"]
        ]
        if vol.is_fallback:
            vol_data.append([f"{Fore.YELLOW}⚠️ Note{Style.RESET_ALL}", "Using Historical Close", "Live API returned 0"])
        print(tabulate(vol_data, headers="firstrow", tablefmt="fancy_grid"))

    def render_participant_data(self, external: ExternalMetrics):
        print(f"\n{Fore.YELLOW}🏦 FII/DII/PRO/CLIENT POSITIONS ({external.data_date}){Style.RESET_ALL}")
        if not external.fii and not external.dii:
            print(f"{Fore.RED}  ❌ Participant data unavailable{Style.RESET_ALL}")
            return
        participants_data = []
        for name, data in [("FII", external.fii), ("DII", external.dii), ("Pro", external.pro), ("Client", external.client)]:
            if data:
                participants_data.append([
                    name,
                    f"{data.fut_long:>9,.0f}",
                    f"{data.fut_short:>9,.0f}",
                    f"{data.fut_net:>10,.0f}",
                    f"{data.call_net:>10,.0f}",
                    f"{data.put_net:>10,.0f}",
                    f"{data.stock_net:>10,.0f}"
                ])
        headers = ["Type", "Fut Long", "Fut Short", "Fut Net", "Call Net", "Put Net", "Stk Net"]
        print(tabulate(participants_data, headers=headers, tablefmt="fancy_grid"))
        print(f"\n  📊 Flow Regime: {Fore.CYAN}{external.flow_regime_full}{Style.RESET_ALL}")
        print(f"  💪 FII Conviction: {Fore.CYAN}{external.fii_conviction}{Style.RESET_ALL}")
        if external.fii_net_change != 0:
            color = Fore.GREEN if external.fii_net_change > 0 else Fore.RED
            print(f"  📈 FII Net Change: {color}{external.fii_net_change:+,.0f}{Style.RESET_ALL} contracts (Day-over-Day)")
        if external.is_fallback_data:
            print(f"  {Fore.YELLOW}⚠️ Fallback data used (latest NSE file may be 1-2 days old){Style.RESET_ALL}")

    def render_struct_metrics(self, struct: StructMetrics):
        print(f"\n{Fore.YELLOW}🏛️ MARKET STRUCTURE{Style.RESET_ALL}")
        struct_data = [
            ["Metric", "Value", "Regime/Context"],
            ["Net GEX (Original)", f"₹{struct.net_gex/1e7:.2f} Cr", struct.gex_regime],
            ["Weighted GEX (Moneyness)", f"₹{struct.gex_weighted/1e7:.2f} Cr", "Proximity-adjusted"],
            ["GEX Ratio", f"{struct.gex_ratio*100:.2f}%", f"Threshold: {Config.GEX_STICKY_RATIO*100}%"],
            ["PCR (All Strikes)", f"{struct.pcr:.2f}", struct.oi_regime],
            ["PCR ATM (±5%)", f"{struct.pcr_atm:.2f}", "Directional Conviction"],
            ["Max Pain", f"{struct.max_pain:.0f}", f"Spot: {struct.max_pain:.0f}"],
            ["25Δ Skew", f"{struct.skew_25d:+.2f}%", struct.skew_regime],
            ["Lot Size", f"{struct.lot_size}", "-"]
        ]
        print(tabulate(struct_data, headers="firstrow", tablefmt="fancy_grid"))

    def render_edge_metrics(self, edge: EdgeMetrics):
        print(f"\n{Fore.YELLOW}🎯 OPTION EDGES{Style.RESET_ALL}")
        def vrp_tag(v):
            return f"{Fore.GREEN}HIGH{Style.RESET_ALL}" if v > 3 else \
                   f"{Fore.RED}LOW{Style.RESET_ALL}" if v < 0 else "OK"
        edge_data = [
            ["Metric", "Weekly", "Tag", "Monthly", "Tag"],
            ["ATM IV (CE+PE)/2", f"{edge.iv_weekly:.2f}%", "-", f"{edge.iv_monthly:.2f}%", "-"],
            ["VRP (vs RV)", f"{edge.vrp_rv_weekly:+.2f}%", vrp_tag(edge.vrp_rv_weekly),
             f"{edge.vrp_rv_monthly:+.2f}%", vrp_tag(edge.vrp_rv_monthly)],
            ["VRP (vs GARCH)", f"{edge.vrp_garch_weekly:+.2f}%", vrp_tag(edge.vrp_garch_weekly),
             f"{edge.vrp_garch_monthly:+.2f}%", vrp_tag(edge.vrp_garch_monthly)],
            ["VRP (vs Park)", f"{edge.vrp_park_weekly:+.2f}%", vrp_tag(edge.vrp_park_weekly),
             f"{edge.vrp_park_monthly:+.2f}%", vrp_tag(edge.vrp_park_monthly)],
            ["WEIGHTED VRP (70/15/15)", f"{Fore.CYAN}{edge.weighted_vrp_weekly:+.2f}%{Style.RESET_ALL}",
             vrp_tag(edge.weighted_vrp_weekly), f"{Fore.CYAN}{edge.weighted_vrp_monthly:+.2f}%{Style.RESET_ALL}",
             vrp_tag(edge.weighted_vrp_monthly)],
            ["Term Spread", f"{edge.term_spread:+.2f}%", edge.term_regime, "-", "-"]
        ]
        print(tabulate(edge_data, headers="firstrow", tablefmt="fancy_grid"))
        print(f"\n  {Fore.CYAN}Primary Edge Detected:{Style.RESET_ALL} {edge.primary_edge}")

    def render_regime_scores(self, score: RegimeScore):
        print(f"\n{Fore.YELLOW}📈 REGIME SCORES (0-10 Scale){Style.RESET_ALL}")
        score_data = [
            ["Component", "Score", "Weight", "Contribution"],
            ["Volatility", f"{score.vol_score:.2f}/10", f"{Config.WEIGHT_VOL*100:.0f}%", f"{score.vol_score * Config.WEIGHT_VOL:.2f}"],
            ["Structure", f"{score.struct_score:.2f}/10", f"{Config.WEIGHT_STRUCT*100:.0f}%", f"{score.struct_score * Config.WEIGHT_STRUCT:.2f}"],
            ["Edge", f"{score.edge_score:.2f}/10", f"{Config.WEIGHT_EDGE*100:.0f}%", f"{score.edge_score * Config.WEIGHT_EDGE:.2f}"],
            ["Risk", f"{score.risk_score:.2f}/10", f"{Config.WEIGHT_RISK*100:.0f}%", f"{score.risk_score * Config.WEIGHT_RISK:.2f}"],
            ["", "", "", ""],
            ["COMPOSITE", f"{score.composite:.2f}/10", "100%", f"Confidence: {score.confidence}"],
            ["Score Stability", f"{score.score_stability:.1%}", "-", "Robust across weight variations"]
        ]
        print(tabulate(score_data, headers="firstrow", tablefmt="fancy_grid"))
        if score.transition_probability:
            print(f"\n  {Fore.CYAN}Regime Transition Probabilities:{Style.RESET_ALL}")
            for regime, prob in sorted(score.transition_probability.items(), key=lambda x: -x[1])[:3]:
                print(f"    → {regime}: {prob:.1%}")

    def render_mandate(self, mandate: TradingMandate):
        color = Fore.GREEN if mandate.max_lots > 0 else Fore.RED
        print(f"\n{color}{'='*90}")
        print(f"{'🎯 TRADING MANDATE':^90}")
        print(f"{'='*90}{Style.RESET_ALL}\n")
        mandate_data = [
            ["Expiry Type", mandate.expiry_type],
            ["Expiry Date", f"{mandate.expiry_date} ({mandate.dte} DTE)"],
            ["Regime", f"{color}{mandate.regime_name}{Style.RESET_ALL}"],
            ["Regime Persistence", f"{mandate.regime_persistence:.0%} (5-day stability)"],
            ["Strategy Type", mandate.strategy_type],
            ["Suggested Structure", mandate.suggested_structure],
            ["", ""],
            ["Capital Allocation", f"{mandate.allocation_pct:.1f}%"],
            ["Max Lots", f"{color}{mandate.max_lots}{Style.RESET_ALL}"],
            ["Risk Per Lot", f"₹{mandate.risk_per_lot:,.0f}"],
            ["Total Deployment", f"₹{mandate.max_lots * mandate.risk_per_lot:,.0f}"]
        ]
        print(tabulate(mandate_data, tablefmt="fancy_grid"))
        print(f"\n{Fore.CYAN}📋 RATIONALE:{Style.RESET_ALL}")
        for i, reason in enumerate(mandate.rationale, 1):
            print(f"  {i}. {reason}")
        if mandate.warnings:
            print(f"\n{Fore.RED}⚠️ WARNINGS:{Style.RESET_ALL}")
            for i, warning in enumerate(mandate.warnings, 1):
                print(f"  {i}. {warning}")

    def render_summary(self, weekly_mandate: TradingMandate, monthly_mandate: TradingMandate):
        print(f"\n{Fore.CYAN}{'='*90}")
        print(f"{'📊 COMPARATIVE SUMMARY':^90}")
        print(f"{'='*90}{Style.RESET_ALL}\n")
        summary_data = [
            ["Aspect", "Weekly Analysis", "Monthly Analysis"],
            ["Expiry", f"{weekly_mandate.expiry_date} ({weekly_mandate.dte} DTE)",
             f"{monthly_mandate.expiry_date} ({monthly_mandate.dte} DTE)"],
            ["Composite Score", f"{weekly_mandate.score.composite:.2f}/10 ({weekly_mandate.score.confidence})",
             f"{monthly_mandate.score.composite:.2f}/10 ({monthly_mandate.score.confidence})"],
            ["Score Stability", f"{weekly_mandate.score.score_stability:.1%}",
             f"{monthly_mandate.score.score_stability:.1%}"],
            ["Regime", weekly_mandate.regime_name, monthly_mandate.regime_name],
            ["Regime Persistence", f"{weekly_mandate.regime_persistence:.0%}",
             f"{monthly_mandate.regime_persistence:.0%}"],
            ["Structure", weekly_mandate.suggested_structure, monthly_mandate.suggested_structure],
            ["Allocation", f"{weekly_mandate.allocation_pct:.0f}%", f"{monthly_mandate.allocation_pct:.0f}%"],
            ["Max Lots", f"{weekly_mandate.max_lots}", f"{monthly_mandate.max_lots}"],
            ["Deployment", f"₹{weekly_mandate.max_lots * weekly_mandate.risk_per_lot:,.0f}",
             f"₹{monthly_mandate.max_lots * monthly_mandate.risk_per_lot:,.0f}"]
        ]
        print(tabulate(summary_data, headers="firstrow", tablefmt="fancy_grid"))
        print(f"\n{Fore.MAGENTA}💡 RECOMMENDATION:{Style.RESET_ALL}")
        if weekly_mandate.score.composite > monthly_mandate.score.composite:
            print(f"  → Focus on WEEKLY expiry (Score: {weekly_mandate.score.composite:.2f} vs {monthly_mandate.score.composite:.2f})")
        elif monthly_mandate.score.composite > weekly_mandate.score.composite:
            print(f"  → Focus on MONTHLY expiry (Score: {monthly_mandate.score.composite:.2f} vs {weekly_mandate.score.composite:.2f})")
        else:
            print(f"  → Both expiries equally favorable (Split allocation)")
        if weekly_mandate.dte <= Config.GAMMA_DANGER_DTE:
            print(f"  → Consider next week expiry ({weekly_mandate.dte + 7} DTE) to avoid gamma risk")
