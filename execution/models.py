"""
Execution data models
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from enum import Enum


class ExecutionMode(Enum):
    """Execution mode"""
    LIVE = "LIVE"
    SHADOW = "SHADOW"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    PLACED = "PLACED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    """Order data structure"""
    order_id: str
    correlation_id: str
    instrument_token: str
    trading_symbol: str
    transaction_type: str  # BUY/SELL
    order_type: str  # MARKET/LIMIT/SL/SL-M
    quantity: int
    price: float
    trigger_price: float
    product: str  # I/D/MTF
    validity: str  # DAY/IOC
    disclosed_quantity: int
    tag: str
    status: OrderStatus
    filled_quantity: int
    average_price: float
    placed_at: datetime
    updated_at: datetime
    mode: ExecutionMode
    is_amo: bool = False
    slice: bool = False
    
    @property
    def pending_quantity(self) -> int:
        """Get pending quantity"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete"""
        return self.status in [OrderStatus.COMPLETED, OrderStatus.REJECTED, OrderStatus.CANCELLED]
    
    @property
    def is_active(self) -> bool:
        """Check if order is active"""
        return self.status in [OrderStatus.PLACED, OrderStatus.PARTIALLY_FILLED]


@dataclass
class StrategyOrder:
    """Multi-leg strategy order"""
    strategy_id: str
    strategy_type: str  # IRON_FLY, IRON_CONDOR, etc.
    legs: List[Order]
    net_premium: float
    max_loss: float
    target_profit: float
    status: str
    placed_at: datetime
    expiry_date: datetime
