"""
Configuration for VOLGUARD Options Cockpit
Using CORRECT Upstox API patterns as verified
"""

from pathlib import Path
from datetime import timedelta
from dataclasses import dataclass
from enum import Enum


class TradingMode(Enum):
    SHADOW = "SHADOW"
    LIVE = "LIVE"


@dataclass
class UpstoxConfig:
    """Upstox API Configuration - VERIFIED PATTERNS"""
    # API URLs (Verified in your test)
    BASE_URL_V2 = "https://api.upstox.com/v2"
    BASE_URL_V3 = "https://api.upstox.com/v3"
    HFT_URL = "https://api-hft.upstox.com/v3"
    
    # Instrument Keys (Verified)
    NIFTY_50_KEY = "NSE_INDEX|Nifty 50"
    INDIA_VIX_KEY = "NSE_INDEX|India VIX"
    SEGMENT_NSE_FO = "NSE_FO"
    
    # API Limits (Adjusted based on your test)
    MAX_INSTRUMENTS_WS = 2000
    MAX_OPTION_CHAIN_REQUESTS = 50
    RATE_LIMIT_DELAY = 0.1
    
    # Default parameters
    DEFAULT_HISTORICAL_DAYS = 400
    DEFAULT_INTERVAL = "day"
    DEFAULT_LOT_SIZE = 75  # Nifty lot size
    DEFAULT_STRIKE_INTERVAL = 50


@dataclass
class AnalyticsConfig:
    """VOLGUARD Analytics Configuration"""
    # Capital & Margin
    BASE_CAPITAL = 10_00_000  # 10 Lakhs
    MARGIN_SELL_BASE = 1_25_000
    MARGIN_BUY_BASE = 30_000
    
    # Risk Parameters
    GAMMA_DANGER_DTE = 1
    GEX_STICKY_RATIO = 0.03
    HIGH_VOL_IVP = 75.0
    LOW_VOL_IVP = 25.0
    VOV_CRASH_ZSCORE = 2.5
    VOV_WARNING_ZSCORE = 2.0
    
    # VRP Weights (70/15/15 - YOUR ORIGINAL)
    VRP_WEIGHTS = {
        "garch": 0.70,
        "parkinson": 0.15,
        "standard": 0.15
    }
    
    # VRP Thresholds
    VRP_THRESHOLD_STRONG = 4.0
    VRP_THRESHOLD_MODERATE = 2.0
    
    # Regime Parameters
    REGIME_STABILITY_THRESHOLD = 0.60
    SKEW_CRASH_FEAR = 5.0
    SKEW_MELT_UP = -2.0
    VIX_MOMENTUM_BREAKOUT = 5.0
    
    # FII Conviction
    FII_STRONG_CONVICTION = 100_000
    FII_MODERATE_CONVICTION = 50_000
    
    # Composite Score Weights
    WEIGHT_VOL = 0.40
    WEIGHT_STRUCT = 0.30
    WEIGHT_EDGE = 0.20
    WEIGHT_RISK = 0.10


@dataclass
class PathConfig:
    """Directory Configuration"""
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Subdirectories
    DATA_DIR = PROJECT_ROOT / "data_storage"
    LOGS_DIR = PROJECT_ROOT / "logs"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    STATE_DIR = PROJECT_ROOT / "state"
    CACHE_DIR = PROJECT_ROOT / "cache"
    
    # Files
    CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
    ENV_FILE = PROJECT_ROOT / ".env"
    STATE_FILE = STATE_DIR / "cockpit_state.json"
    
    def create_directories(self):
        """Create all required directories"""
        for directory in [self.DATA_DIR, self.LOGS_DIR, self.REPORTS_DIR, 
                         self.STATE_DIR, self.CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class DashboardConfig:
    """Dashboard Configuration"""
    REFRESH_INTERVAL = 5  # seconds
    MAX_POSITIONS_DISPLAY = 20
    ENABLE_SOUNDS = False
    
    # Chart Settings
    CHART_HEIGHT = 400
    CHART_WIDTH = None  # Use container width
    
    # Colors
    COLORS = {
        "positive": "#2ecc71",
        "negative": "#e74c3c",
        "warning": "#f39c12",
        "neutral": "#3498db",
        "background": "#1a1a2e",
        "text": "#ecf0f1"
    }


# Global instances
UPSTOX_CONFIG = UpstoxConfig()
ANALYTICS_CONFIG = AnalyticsConfig()
PATH_CONFIG = PathConfig()
DASHBOARD_CONFIG = DashboardConfig()
