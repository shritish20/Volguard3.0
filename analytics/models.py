"""
Data models for VOLGUARD analytics
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd


@dataclass
class TimeMetrics:
    """Time-related metrics"""
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
    """Volatility metrics"""
    spot: float
    vix: float
    rv7: float
    rv28: float
    rv90: float
    garch7: float
    garch28: float
    park7: float
    park28: float
    vov: float
    vov_zscore: float
    ivp_30d: float
    ivp_90d: float
    ivp_1yr: float
    ma20: float
    atr14: float
    trend_strength: float
    vol_regime: str
    is_fallback: bool
    vix_change_5d: float
    vix_momentum: str  # RISING, FALLING, STABLE


@dataclass
class StructMetrics:
    """Market structure metrics"""
    net_gex: float
    gex_ratio: float
    total_oi_value: float
    gex_regime: str  # STICKY, SLIPPERY, NEUTRAL
    pcr: float
    max_pain: float
    skew_25d: float
    oi_regime: str  # BULLISH, BEARISH, NEUTRAL
    lot_size: int
    pcr_atm: float
    skew_regime: str  # CRASH_FEAR, MELT_UP, BALANCED
    gex_weighted: float


@dataclass
class EdgeMetrics:
    """Option edge metrics"""
    iv_weekly: float
    vrp_rv_weekly: float
    vrp_garch_weekly: float
    vrp_park_weekly: float
    iv_monthly: float
    vrp_rv_monthly: float
    vrp_garch_monthly: float
    vrp_park_monthly: float
    term_spread: float
    term_regime: str  # BACKWARDATION, CONTANGO, FLAT
    primary_edge: str
    weighted_vrp_weekly: float
    weighted_vrp_monthly: float


@dataclass
class ParticipantData:
    """Participant data (FII/DII/Pro/Client)"""
    fut_long: float
    fut_short: float
    fut_net: float
    call_long: float
    call_short: float
    call_net: float
    put_long: float
    put_short: float
    put_net: float
    stock_net: float


@dataclass
class ExternalMetrics:
    """External market metrics"""
    fii: Optional[ParticipantData]
    dii: Optional[ParticipantData]
    pro: Optional[ParticipantData]
    client: Optional[ParticipantData]
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
    """Regime scoring results"""
    vol_score: float
    struct_score: float
    edge_score: float
    risk_score: float
    composite: float
    confidence: str  # VERY_HIGH, HIGH, MODERATE, LOW
    score_stability: float
    transition_probability: Dict[str, float] = field(default_factory=dict)


@dataclass
class TradingMandate:
    """Trading mandate output"""
    expiry_type: str  # WEEKLY, MONTHLY
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


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    timestamp: datetime
    time_metrics: TimeMetrics
    vol_metrics: VolMetrics
    struct_weekly: StructMetrics
    struct_monthly: StructMetrics
    edge_metrics: EdgeMetrics
    external_metrics: ExternalMetrics
    weekly_mandate: TradingMandate
    monthly_mandate: TradingMandate
    weekly_score: RegimeScore
    monthly_score: RegimeScore
