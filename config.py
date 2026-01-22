"""
VOLGUARD v31.0 – CENTRAL CONFIGURATION
Production constants – 100 % Upstox API aligned
"""
import os
from datetime import date, timedelta

# ------------------------------------------------------------------
# 1. Upstox API – absolute endpoints
# ------------------------------------------------------------------
UPSTOX_BASE_V2      = "https://api.upstox.com/v2"
UPSTOX_BASE_V3      = "https://api.upstox.com/v3"
UPSTOX_HFT_V3       = "https://api-hft.upstox.com/v3"   # order placement
WS_MARKET_V3        = "wss://api.upstox.com/v3/feed/market-data-feed"
WS_PORTFOLIO_V2     = "wss://api.upstox.com/v2/feed/portfolio-stream-feed"

# instrument keys
NIFTY_KEY           = "NSE_INDEX|Nifty 50"
VIX_KEY             = "NSE_INDEX|India VIX"

# ------------------------------------------------------------------
# 2. Capital & margin (₹)
# ------------------------------------------------------------------
BASE_CAPITAL        = 10_00_000
MARGIN_SELL_BASE    = 1_25_000   # per short lot
MARGIN_BUY_BASE     = 30_000     # per long lot

# ------------------------------------------------------------------
# 3. Risk thresholds
# ------------------------------------------------------------------
GAMMA_DANGER_DTE            = 1
GEX_STICKY_RATIO            = 0.03
HIGH_VOL_IVP                = 75.0
LOW_VOL_IVP                 = 25.0
VOV_CRASH_ZSCORE            = 2.5
VOV_WARNING_ZSCORE          = 1.5
SKEW_CRASH_FEAR             = 5.0
SKEW_MELT_UP                = -2.0
VIX_MOMENTUM_BREAKOUT       = 5.0
FII_STRONG_CONVICTION       = 100_000
FII_MODERATE_CONVICTION     = 50_000
REGIME_STABILITY_THRESHOLD  = 0.6

# ------------------------------------------------------------------
# 4. Regime scoring weights
# ------------------------------------------------------------------
WEIGHT_VOL    = 0.40
WEIGHT_STRUCT = 0.30
WEIGHT_EDGE   = 0.20
WEIGHT_RISK   = 0.10

# ------------------------------------------------------------------
# 5. Trading rules
# ------------------------------------------------------------------
PROFIT_TARGET_PCT      = 50
STOP_LOSS_UNDEFINED_PCT = 200
THETA_TARGET_PCT        = 50
DELTA_BREACH_ABS        = 0.15
MAX Lots_PER_STRATEGY   = 20
ORDER_TAG               = "VOLGUARD"

# ------------------------------------------------------------------
# 6. Market timing helpers
# ------------------------------------------------------------------
def is_market_open_today() -> bool:
    """Live check – uses Upstox /market/status"""
    # injected dynamically via data_engine – here for typing
    return True

# ------------------------------------------------------------------
# 7. Brokerage & charges (conservative %)
# ------------------------------------------------------------------
BROKERAGE_PERC      = 0.03   # intraday F&O
STT_PERC            = 0.05
GST_PERC            = 18
EXCHANGE_TXN_PERC   = 0.003
SEBI_TURNOVER_PERC  = 0.0001
STAMP_DUTY_PERC     = 0.002

# ------------------------------------------------------------------
# 8. Logging & DB
# ------------------------------------------------------------------
DB_PATH             = "volguard_journal.db"
LOG_LEVEL           = "INFO"
