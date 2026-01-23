"""
VOLGUARD Analytics Engine - Integrated with CORRECT Upstox API
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from arch import arch_model
from scipy import stats

from config.settings import ANALYTICS_CONFIG, UPSTOX_CONFIG
from analytics.models import (
    TimeMetrics, VolMetrics, StructMetrics, EdgeMetrics,
    ExternalMetrics, RegimeScore, TradingMandate, AnalysisResult
)


class AnalyticsEngine:
    """Enhanced analytics engine for VOLGUARD"""
    
    def __init__(self):
        self.historical_cache = {}
        self.regime_history = []
        
    # ------------- TIME METRICS -------------
    def get_time_metrics(self, weekly_exp: date, monthly_exp: date, 
                        next_weekly_exp: date) -> TimeMetrics:
        """Calculate time metrics"""
        today = date.today()
        dte_w = (weekly_exp - today).days
        dte_m = (monthly_exp - today).days
        dte_nw = (next_weekly_exp - today).days
        
        return TimeMetrics(
            current_date=today,
            weekly_exp=weekly_exp,
            monthly_exp=monthly_exp,
            next_weekly_exp=next_weekly_exp,
            dte_weekly=dte_w,
            dte_monthly=dte_m,
            is_gamma_week=dte_w <= ANALYTICS_CONFIG.GAMMA_DANGER_DTE,
            is_gamma_month=dte_m <= ANALYTICS_CONFIG.GAMMA_DANGER_DTE,
            days_to_next_weekly=dte_nw
        )
    
    # ------------- VOLATILITY METRICS -------------
    def calculate_vol_metrics(self, nifty_hist: pd.DataFrame, 
                            vix_hist: pd.DataFrame, spot: float, vix: float) -> VolMetrics:
        """Calculate all volatility metrics"""
        if nifty_hist.empty or vix_hist.empty:
            return self._get_fallback_vol_metrics(spot, vix)
        
        # Calculate returns
        returns = np.log(nifty_hist['close'] / nifty_hist['close'].shift(1)).dropna()
        
        # Realized volatility calculations
        rv7 = returns.rolling(7).std().iloc[-1] * np.sqrt(252) * 100
        rv28 = returns.rolling(28).std().iloc[-1] * np.sqrt(252) * 100
        rv90 = returns.rolling(90).std().iloc[-1] * np.sqrt(252) * 100
        
        # GARCH forecasts
        garch7 = self._fit_garch_forecast(returns, horizon=7) or rv7
        garch28 = self._fit_garch_forecast(returns, horizon=28) or rv28
        
        # Parkinson volatility
        park7 = self._calculate_parkinson_vol(nifty_hist, window=7)
        park28 = self._calculate_parkinson_vol(nifty_hist, window=28)
        
        # Vol-of-Vol
        vov, vov_zscore = self._calculate_vol_of_vol(vix_hist)
        
        # IV Percentiles
        ivp_30d, ivp_90d, ivp_1yr = self._calculate_iv_percentiles(vix_hist, vix)
        
        # Trend metrics
        ma20 = nifty_hist['close'].rolling(20).mean().iloc[-1]
        atr14 = self._calculate_atr(nifty_hist)
        trend_strength = abs(spot - ma20) / atr14 if atr14 > 0 else 0
        
        # VIX momentum
        vix_change_5d, vix_momentum = self._calculate_vix_momentum(vix_hist, vix)
        
        # Vol regime classification
        vol_regime = self._classify_vol_regime(vov_zscore, ivp_1yr, vix_momentum)
        
        return VolMetrics(
            spot=spot,
            vix=vix,
            rv7=rv7, rv28=rv28, rv90=rv90,
            garch7=garch7, garch28=garch28,
            park7=park7, park28=park28,
            vov=vov, vov_zscore=vov_zscore,
            ivp_30d=ivp_30d, ivp_90d=ivp_90d, ivp_1yr=ivp_1yr,
            ma20=ma20, atr14=atr14, trend_strength=trend_strength,
            vol_regime=vol_regime,
            is_fallback=False,
            vix_change_5d=vix_change_5d,
            vix_momentum=vix_momentum
        )
    
    def _fit_garch_forecast(self, returns: pd.Series, horizon: int = 7) -> float:
        """Fit GARCH(1,1) and forecast"""
        try:
            if len(returns) < 100:
                return 0
                
            model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal')
            result = model.fit(disp='off', show_warning=False)
            forecast = result.forecast(horizon=horizon, reindex=False)
            return np.sqrt(forecast.variance.values[-1, -1]) * np.sqrt(252)
            
        except Exception as e:
            print(f"GARCH forecast failed: {e}")
            return 0
    
    def _calculate_parkinson_vol(self, df: pd.DataFrame, window: int) -> float:
        """Calculate Parkinson volatility"""
        try:
            hl_ratio = np.log(df['high'] / df['low']) ** 2
            const = 1.0 / (4.0 * np.log(2.0))
            parkinson = np.sqrt(hl_ratio.rolling(window).mean() * const) * np.sqrt(252) * 100
            return parkinson.iloc[-1]
        except:
            return 0
    
    def _calculate_vol_of_vol(self, vix_hist: pd.DataFrame) -> Tuple[float, float]:
        """Calculate volatility of volatility"""
        try:
            vix_returns = np.log(vix_hist['close'] / vix_hist['close'].shift(1)).dropna()
            vov = vix_returns.rolling(30).std().iloc[-1] * np.sqrt(252) * 100
            
            # Calculate Z-score
            vov_rolling = vix_returns.rolling(30).std() * np.sqrt(252) * 100
            vov_mean = vov_rolling.rolling(60).mean().iloc[-1]
            vov_std = vov_rolling.rolling(60).std().iloc[-1]
            vov_zscore = (vov - vov_mean) / vov_std if vov_std > 0 else 0
            
            return vov, vov_zscore
        except:
            return 0, 0
    
    def _calculate_iv_percentiles(self, vix_hist: pd.DataFrame, current_vix: float) -> Tuple[float, float, float]:
        """Calculate IV percentiles"""
        def calc_percentile(window: int) -> float:
            if len(vix_hist) < window:
                return 0
            history = vix_hist['close'].tail(window)
            return (history < current_vix).mean() * 100
        
        return calc_percentile(30), calc_percentile(90), calc_percentile(252)
    
    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range"""
        try:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return true_range.rolling(14).mean().iloc[-1]
        except:
            return 0
    
    def _calculate_vix_momentum(self, vix_hist: pd.DataFrame, current_vix: float) -> Tuple[float, str]:
        """Calculate VIX momentum"""
        try:
            if len(vix_hist) >= 6:
                vix_5d_ago = vix_hist['close'].iloc[-6]
                vix_change_5d = ((current_vix / vix_5d_ago) - 1) * 100
            else:
                vix_change_5d = 0
            
            if vix_change_5d > ANALYTICS_CONFIG.VIX_MOMENTUM_BREAKOUT:
                momentum = "RISING"
            elif vix_change_5d < -ANALYTICS_CONFIG.VIX_MOMENTUM_BREAKOUT:
                momentum = "FALLING"
            else:
                momentum = "STABLE"
                
            return vix_change_5d, momentum
        except:
            return 0, "STABLE"
    
    def _classify_vol_regime(self, vov_zscore: float, ivp_1yr: float, 
                           vix_momentum: str) -> str:
        """Classify volatility regime"""
        if vov_zscore > ANALYTICS_CONFIG.VOV_CRASH_ZSCORE:
            return "EXPLODING"
        elif ivp_1yr > ANALYTICS_CONFIG.HIGH_VOL_IVP:
            if vix_momentum == "FALLING":
                return "MEAN_REVERTING"
            elif vix_momentum == "RISING":
                return "BREAKOUT_RICH"
            else:
                return "RICH"
        elif ivp_1yr < ANALYTICS_CONFIG.LOW_VOL_IVP:
            return "CHEAP"
        else:
            return "FAIR"
    
    def _get_fallback_vol_metrics(self, spot: float, vix: float) -> VolMetrics:
        """Fallback metrics when data is unavailable"""
        return VolMetrics(
            spot=spot, vix=vix,
            rv7=0, rv28=0, rv90=0,
            garch7=0, garch28=0,
            park7=0, park28=0,
            vov=0, vov_zscore=0,
            ivp_30d=0, ivp_90d=0, ivp_1yr=0,
            ma20=spot, atr14=0, trend_strength=0,
            vol_regime="UNKNOWN",
            is_fallback=True,
            vix_change_5d=0,
            vix_momentum="STABLE"
        )
    
    # ------------- STRUCTURE METRICS -------------
    def calculate_struct_metrics(self, chain_df: pd.DataFrame, spot: float, 
                               lot_size: int) -> StructMetrics:
        """Calculate market structure metrics"""
        if chain_df.empty:
            return self._get_fallback_struct_metrics(lot_size)
        
        # Moneyness weighting
        chain_df = chain_df.copy()
        chain_df['moneyness'] = chain_df['strike'] / spot
        chain_df['proximity_weight'] = np.exp(-((chain_df['strike'] - spot) / spot) ** 2 / 0.02)
        
        # Weighted GEX calculation
        gex_weighted = self._calculate_weighted_gex(chain_df, spot, lot_size)
        
        # Traditional GEX (10% band)
        subset = chain_df[(chain_df['strike'] > spot * 0.90) & (chain_df['strike'] < spot * 1.10)]
        net_gex = self._calculate_net_gex(subset, spot, lot_size)
        
        # Total OI value
        total_oi_value = (chain_df['ce_oi'].sum() + chain_df['pe_oi'].sum()) * spot * lot_size
        gex_ratio = abs(gex_weighted) / total_oi_value if total_oi_value > 0 else 0
        
        # GEX regime
        gex_regime = self._classify_gex_regime(gex_ratio)
        
        # PCR calculations
        pcr = self._calculate_pcr(chain_df)
        pcr_atm = self._calculate_atm_pcr(chain_df, spot)
        
        # Max pain
        max_pain = self._calculate_max_pain(chain_df)
        
        # Skew calculation
        skew_25d = self._calculate_skew_25d(chain_df)
        skew_regime = self._classify_skew_regime(skew_25d)
        
        # OI regime
        oi_regime = self._classify_oi_regime(pcr_atm)
        
        return StructMetrics(
            net_gex=net_gex,
            gex_ratio=gex_ratio,
            total_oi_value=total_oi_value,
            gex_regime=gex_regime,
            pcr=pcr,
            max_pain=max_pain,
            skew_25d=skew_25d,
            oi_regime=oi_regime,
            lot_size=lot_size,
            pcr_atm=pcr_atm,
            skew_regime=skew_regime,
            gex_weighted=gex_weighted
        )
    
    def _calculate_weighted_gex(self, chain_df: pd.DataFrame, spot: float, 
                              lot_size: int) -> float:
        """Calculate moneyness-weighted GEX"""
        try:
            weighted_gex = (
                (chain_df['ce_gamma'] * chain_df['ce_oi'] * chain_df['proximity_weight']).sum() -
                (chain_df['pe_gamma'] * chain_df['pe_oi'] * chain_df['proximity_weight']).sum()
            ) * spot * lot_size
            return weighted_gex
        except:
            return 0
    
    def _calculate_net_gex(self, chain_df: pd.DataFrame, spot: float, 
                          lot_size: int) -> float:
        """Calculate net GEX"""
        try:
            net_gex = (
                (chain_df['ce_gamma'] * chain_df['ce_oi']).sum() -
                (chain_df['pe_gamma'] * chain_df['pe_oi']).sum()
            ) * spot * lot_size
            return net_gex
        except:
            return 0
    
    def _classify_gex_regime(self, gex_ratio: float) -> str:
        """Classify GEX regime"""
        if gex_ratio > ANALYTICS_CONFIG.GEX_STICKY_RATIO:
            return "STICKY"
        elif gex_ratio < ANALYTICS_CONFIG.GEX_STICKY_RATIO * 0.5:
            return "SLIPPERY"
        else:
            return "NEUTRAL"
    
    def _calculate_pcr(self, chain_df: pd.DataFrame) -> float:
        """Calculate Put-Call Ratio"""
        try:
            total_pe_oi = chain_df['pe_oi'].sum()
            total_ce_oi = chain_df['ce_oi'].sum()
            return total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0
        except:
            return 1.0
    
    def _calculate_atm_pcr(self, chain_df: pd.DataFrame, spot: float) -> float:
        """Calculate ATM PCR (±5%)"""
        try:
            atm_chain = chain_df[
                (chain_df['strike'] >= spot * 0.95) & 
                (chain_df['strike'] <= spot * 1.05)
            ]
            total_pe_oi = atm_chain['pe_oi'].sum()
            total_ce_oi = atm_chain['ce_oi'].sum()
            return total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0
        except:
            return 1.0
    
    def _calculate_max_pain(self, chain_df: pd.DataFrame) -> float:
        """Calculate max pain"""
        try:
            strikes = chain_df['strike'].values
            losses = []
            
            for strike in strikes:
                call_loss = np.sum(np.maximum(0, strike - strikes) * chain_df['ce_oi'].values)
                put_loss = np.sum(np.maximum(0, strikes - strike) * chain_df['pe_oi'].values)
                losses.append(call_loss + put_loss)
            
            if losses:
                return strikes[np.argmin(losses)]
        except:
            pass
            
        return 0
    
    def _calculate_skew_25d(self, chain_df: pd.DataFrame) -> float:
        """Calculate 25Δ skew"""
        try:
            # Find 25Δ call and put
            ce_25d_idx = (chain_df['ce_delta'].abs() - 0.25).abs().argsort()[:1]
            pe_25d_idx = (chain_df['pe_delta'].abs() - 0.25).abs().argsort()[:1]
            
            ce_25d_iv = chain_df.iloc[ce_25d_idx]['ce_iv'].values[0]
            pe_25d_iv = chain_df.iloc[pe_25d_idx]['pe_iv'].values[0]
            
            return pe_25d_iv - ce_25d_iv
        except:
            return 0
    
    def _classify_skew_regime(self, skew_25d: float) -> str:
        """Classify skew regime"""
        if skew_25d > ANALYTICS_CONFIG.SKEW_CRASH_FEAR:
            return "CRASH_FEAR"
        elif skew_25d < ANALYTICS_CONFIG.SKEW_MELT_UP:
            return "MELT_UP"
        else:
            return "BALANCED"
    
    def _classify_oi_regime(self, pcr_atm: float) -> str:
        """Classify OI regime"""
        if pcr_atm > 1.2:
            return "BULLISH"
        elif pcr_atm < 0.8:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _get_fallback_struct_metrics(self, lot_size: int) -> StructMetrics:
        """Fallback structure metrics"""
        return StructMetrics(
            net_gex=0, gex_ratio=0, total_oi_value=0,
            gex_regime="NEUTRAL", pcr=1.0, max_pain=0,
            skew_25d=0, oi_regime="NEUTRAL", lot_size=lot_size,
            pcr_atm=1.0, skew_regime="BALANCED", gex_weighted=0
        )
    
    # ------------- EDGE METRICS -------------
    def calculate_edge_metrics(self, weekly_chain: pd.DataFrame, 
                             monthly_chain: pd.DataFrame, spot: float, 
                             vol_metrics: VolMetrics) -> EdgeMetrics:
        """Calculate edge metrics"""
        # Get ATM IV for weekly and monthly
        iv_weekly = self._get_atm_iv(weekly_chain, spot)
        iv_monthly = self._get_atm_iv(monthly_chain, spot)
        
        # Calculate VRPs
        vrp_rv_weekly = iv_weekly - vol_metrics.rv7
        vrp_garch_weekly = iv_weekly - vol_metrics.garch7
        vrp_park_weekly = iv_weekly - vol_metrics.park7
        
        vrp_rv_monthly = iv_monthly - vol_metrics.rv28
        vrp_garch_monthly = iv_monthly - vol_metrics.garch28
        vrp_park_monthly = iv_monthly - vol_metrics.park28
        
        # Weighted VRP (70/15/15)
        weighted_vrp_weekly = (
            vrp_garch_weekly * ANALYTICS_CONFIG.VRP_WEIGHTS["garch"] +
            vrp_park_weekly * ANALYTICS_CONFIG.VRP_WEIGHTS["parkinson"] +
            vrp_rv_weekly * ANALYTICS_CONFIG.VRP_WEIGHTS["standard"]
        )
        
        weighted_vrp_monthly = (
            vrp_garch_monthly * ANALYTICS_CONFIG.VRP_WEIGHTS["garch"] +
            vrp_park_monthly * ANALYTICS_CONFIG.VRP_WEIGHTS["parkinson"] +
            vrp_rv_monthly * ANALYTICS_CONFIG.VRP_WEIGHTS["standard"]
        )
        
        # Term structure
        term_spread = iv_monthly - iv_weekly
        term_regime = self._classify_term_regime(term_spread)
        
        # Primary edge
        primary_edge = self._determine_primary_edge(
            weighted_vrp_weekly, weighted_vrp_monthly, 
            vol_metrics.ivp_1yr, term_spread, term_regime
        )
        
        return EdgeMetrics(
            iv_weekly=iv_weekly,
            vrp_rv_weekly=vrp_rv_weekly,
            vrp_garch_weekly=vrp_garch_weekly,
            vrp_park_weekly=vrp_park_weekly,
            iv_monthly=iv_monthly,
            vrp_rv_monthly=vrp_rv_monthly,
            vrp_garch_monthly=vrp_garch_monthly,
            vrp_park_monthly=vrp_park_monthly,
            term_spread=term_spread,
            term_regime=term_regime,
            primary_edge=primary_edge,
            weighted_vrp_weekly=weighted_vrp_weekly,
            weighted_vrp_monthly=weighted_vrp_monthly
        )
    
    def _get_atm_iv(self, chain_df: pd.DataFrame, spot: float) -> float:
        """Get ATM IV from option chain"""
        if chain_df.empty:
            return 0
        
        try:
            # Find ATM strike (closest to spot)
            atm_idx = (chain_df['strike'] - spot).abs().argsort()[:1]
            row = chain_df.iloc[atm_idx].iloc[0]
            
            # Average of CE and PE IV
            return (row['ce_iv'] + row['pe_iv']) / 2
        except:
            return 0
    
    def _classify_term_regime(self, term_spread: float) -> str:
        """Classify term structure regime"""
        if term_spread < -1.0:
            return "BACKWARDATION"
        elif term_spread > 1.0:
            return "CONTANGO"
        else:
            return "FLAT"
    
    def _determine_primary_edge(self, weighted_vrp_weekly: float, 
                              weighted_vrp_monthly: float, ivp_1yr: float,
                              term_spread: float, term_regime: str) -> str:
        """Determine primary trading edge"""
        if ivp_1yr < ANALYTICS_CONFIG.LOW_VOL_IVP:
            return "LONG_VOL"
        elif weighted_vrp_weekly > ANALYTICS_CONFIG.VRP_THRESHOLD_STRONG and ivp_1yr > 50:
            return "SHORT_GAMMA"
        elif weighted_vrp_monthly > 3.0 and ivp_1yr > 50:
            return "SHORT_VEGA"
        elif term_regime == "BACKWARDATION" and term_spread < -2.0:
            return "CALENDAR_SPREAD"
        elif ivp_1yr > ANALYTICS_CONFIG.HIGH_VOL_IVP:
            return "MEAN_REVERSION"
        else:
            return "NONE"
