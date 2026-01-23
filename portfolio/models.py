"""
Portfolio data models
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class PositionType(Enum):
    """Position type enumeration"""
    OPTION = "OPTION"
    FUTURE = "FUTURE"
    EQUITY = "EQUITY"


@dataclass
class Position:
    """Position data structure"""
    instrument_token: str
    instrument_key: str
    trading_symbol: str
    instrument_type: str  # CE, PE, FUT, EQ
    quantity: int
    average_price: float
    current_price: float
    lot_size: int
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    iv: float = 0.0
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    margin_used: float = 0.0
    expiry_date: Optional[date] = None
    strike_price: Optional[float] = None
    
    @property
    def position_type(self) -> PositionType:
        """Get position type"""
        if self.instrument_type in ["CE", "PE"]:
            return PositionType.OPTION
        elif self.instrument_type == "FUT":
            return PositionType.FUTURE
        else:
            return PositionType.EQUITY
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value"""
        return abs(self.quantity) * self.current_price * self.lot_size
    
    @property
    def investment_value(self) -> float:
        """Calculate investment value"""
        return abs(self.quantity) * self.average_price * self.lot_size
    
    def update_greeks(self, greeks: Dict[str, float]):
        """Update Greek values"""
        self.delta = greeks.get('delta', 0)
        self.gamma = greeks.get('gamma', 0)
        self.theta = greeks.get('theta', 0)
        self.vega = greeks.get('vega', 0)
        self.iv = greeks.get('iv', 0)
    
    def calculate_pnl(self, current_price: Optional[float] = None):
        """Calculate P&L"""
        if current_price is not None:
            self.current_price = current_price
        
        price_diff = self.current_price - self.average_price
        
        # Determine multiplier based on position type
        if self.instrument_type in ["PE", "FUT"]:
            multiplier = -1  # Short positions
        else:
            multiplier = 1   # Long positions
        
        self.pnl = price_diff * abs(self.quantity) * self.lot_size * multiplier
        
        if self.investment_value > 0:
            self.pnl_percentage = (self.pnl / self.investment_value) * 100
        else:
            self.pnl_percentage = 0


@dataclass
class PortfolioSummary:
    """Portfolio summary"""
    total_positions: int
    total_investment: float
    total_current_value: float
    total_pnl: float
    total_pnl_percentage: float
    total_margin_used: float
    available_margin: float
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    max_risk: float
    var_95: float  # Value at Risk 95%
    concentration_ratio: float  # Top 3 positions / total
