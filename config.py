"""
VOLGUARD v31.0 – CONFIGURATION
===============================
Central configuration for the entire system
"""

# API Configuration
UPSTOX_BASE_V2 = "https://api.upstox.com/v2"
UPSTOX_BASE_V3 = "https://api.upstox.com/v3"
NIFTY_KEY = "NSE_INDEX|Nifty 50"
VIX_KEY = "NSE_INDEX|India VIX"

# Capital Configuration
BASE_CAPITAL = 10_00_000  # 10 Lakhs
MARGIN_SELL_BASE = 1_25_000  # ₹1.25L per short lot
MARGIN_BUY_BASE = 30_000  # ₹30K per long lot

# Risk Configuration
GAMMA_DANGER_DTE = 1
GEX_STICKY_RATIO = 0.03
HIGH_VOL_IVP = 75.0
LOW_VOL_IVP = 25.0
VOV_CRASH_ZSCORE = 2.5
VOV_WARNING_ZSCORE = 2.0

# New v31.0 Configuration
SKEW_CRASH_FEAR = 5.0
SKEW_MELT_UP = -2.0
VIX_MOMENTUM_BREAKOUT = 5.0
FII_STRONG_CONVICTION = 100_000
FII_MODERATE_CONVICTION = 50_000
REGIME_STABILITY_THRESHOLD = 0.60

# Scoring Weights
WEIGHT_VOL = 0.40
WEIGHT_STRUCT = 0.30
WEIGHT_EDGE = 0.20
WEIGHT_RISK = 0.10
