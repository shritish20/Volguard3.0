"""
Risk Management Module - CRITICAL FOR LIVE TRADING
Enforces position limits, stop losses, and circuit breakers.
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from config.settings import ANALYTICS_CONFIG
from utils.logger import setup_logger

class RiskStatus(Enum):
    """Risk status levels"""
    NORMAL = "NORMAL"           # All systems go
    WARNING = "WARNING"         # Approaching limits
    CRITICAL = "CRITICAL"       # Breached soft limits
    HALT = "HALT"               # Trading must stop immediately

@dataclass
class RiskLimits:
    """Risk limit configuration"""
    # Capital limits
    max_total_capital_deployed: float = 6_00_000  # Max 60% of 10L
    max_single_position_size: float = 2_00_000    # Max 20% per position
    
    # Loss limits (CRITICAL)
    max_daily_loss: float = 50_000          # Stop trading if -50K in a day
    max_weekly_loss: float = 1_00_000       # Stop trading if -1L in a week
    max_drawdown_pct: float = 15.0          # Max 15% drawdown from peak
    
    # Position limits
    max_total_positions: int = 10           # Max 10 concurrent positions
    max_positions_per_expiry: int = 3       # Max 3 per expiry
    
    # Greek limits
    max_portfolio_delta: float = 1000       # Absolute delta limit
    max_portfolio_gamma: float = 100        # Absolute gamma limit
    max_portfolio_vega: float = 5000        # Absolute vega limit
    
    # Margin limits
    max_margin_utilization_pct: float = 75.0  # Max 75% margin usage
    min_buffer_margin: float = 1_00_000       # Always keep 1L buffer
    
    # Trading velocity limits
    max_orders_per_minute: int = 10
    max_orders_per_hour: int = 100
    
    # Regime-based limits
    halt_on_vix_spike: float = 30.0         # Stop if VIX > 30
    halt_on_vov_zscore: float = 3.0         # Stop if VoV > 3σ

@dataclass
class RiskMetrics:
    """Current risk metrics snapshot"""
    timestamp: datetime
    total_capital_deployed: float
    total_pnl_today: float
    total_pnl_week: float
    current_drawdown_pct: float
    peak_equity: float
    
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_vega: float
    portfolio_theta: float
    
    margin_utilization_pct: float
    total_positions: int
    
    orders_last_minute: int
    orders_last_hour: int
    
    current_vix: float
    vov_zscore: float
    
    risk_status: RiskStatus
    violations: List[str]
    warnings: List[str]

class RiskManager:
    """Comprehensive Risk Management System"""
    
    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self.logger = setup_logger("risk_manager")
        
        # State tracking
        self.peak_equity = ANALYTICS_CONFIG.BASE_CAPITAL
        self.daily_pnl_history = []
        self.order_timestamps = []
        self.halt_status = False
        self.halt_reason = None
        
        # Trade journal for analytics (Simple in-memory for now)
        self.pnl_by_day = {}
        
        self.logger.info("🛡️ Risk Manager initialized")
        self._log_limits()
    
    def _log_limits(self):
        """Log all risk limits on startup"""
        self.logger.info("="*80)
        self.logger.info("RISK LIMITS CONFIGURATION")
        self.logger.info("="*80)
        self.logger.info(f"Max Daily Loss: ₹{self.limits.max_daily_loss:,.0f}")
        self.logger.info(f"Max Drawdown: {self.limits.max_drawdown_pct}%")
        self.logger.info(f"Max Delta: {self.limits.max_portfolio_delta}")
        self.logger.info("="*80)
    
    def check_pre_trade_risk(self, proposed_order: Dict, 
                             current_portfolio: Dict,
                             current_market: Dict) -> Tuple[bool, str]:
        """
        Pre-trade risk check - MUST pass before any order placement
        """
        
        # 1. HALT STATUS CHECK (CRITICAL)
        if self.halt_status:
            return False, f"❌ TRADING HALTED: {self.halt_reason}"
        
        violations = []
        
        # 2. POSITION COUNT CHECK
        if current_portfolio.get('total_positions', 0) >= self.limits.max_total_positions:
            violations.append(f"Max positions reached ({self.limits.max_total_positions})")
        
        # 3. POSITION SIZE CHECK
        # Estimate size: Qty * Price * LotSize
        proposed_size = abs(proposed_order.get('quantity', 0)) * proposed_order.get('price', 0) * proposed_order.get('lot_size', 75)
        if proposed_size > self.limits.max_single_position_size:
            violations.append(f"Position size ₹{proposed_size:,.0f} exceeds limit ₹{self.limits.max_single_position_size:,.0f}")
        
        # 4. CAPITAL DEPLOYMENT CHECK
        new_capital_deployed = current_portfolio.get('total_investment', 0) + proposed_size
        if new_capital_deployed > self.limits.max_total_capital_deployed:
            violations.append(f"Total capital ₹{new_capital_deployed:,.0f} exceeds limit")
        
        # 5. GREEK CHECKS (Delta/Gamma)
        # Simple additive check for now
        proposed_delta = proposed_order.get('delta', 0) * proposed_order.get('quantity', 0) * proposed_order.get('lot_size', 75)
        new_portfolio_delta = current_portfolio.get('net_delta', 0) + proposed_delta
        
        if abs(new_portfolio_delta) > self.limits.max_portfolio_delta:
            violations.append(f"Portfolio delta {new_portfolio_delta:.0f} exceeds limit {self.limits.max_portfolio_delta}")

        # 6. ORDER VELOCITY CHECK
        now = datetime.now()
        # Clean old timestamps
        self.order_timestamps = [ts for ts in self.order_timestamps if now - ts < timedelta(hours=1)]
        
        orders_last_minute = sum(1 for ts in self.order_timestamps if now - ts < timedelta(minutes=1))
        
        if orders_last_minute >= self.limits.max_orders_per_minute:
            violations.append(f"Order velocity exceeded: {orders_last_minute} orders in last minute")
        
        # VERDICT
        if violations:
            reason = " | ".join(violations)
            self.logger.warning(f"⚠️ PRE-TRADE RISK CHECK FAILED: {reason}")
            return False, reason
        
        # Approved - record timestamp
        self.order_timestamps.append(now)
        self.logger.info(f"✅ Pre-trade risk check PASSED for {proposed_order.get('trading_symbol')}")
        return True, "Approved"
    
    def check_portfolio_risk(self, portfolio_summary: Dict, 
                            current_market: Dict) -> RiskMetrics:
        """
        Comprehensive portfolio risk assessment.
        Called every minute or on portfolio update.
        """
        violations = []
        warnings = []
        
        # Calculate current drawdown
        # Assuming base capital + PnL = Current Equity
        current_equity = ANALYTICS_CONFIG.BASE_CAPITAL + portfolio_summary.get('total_pnl', 0)
        self.peak_equity = max(self.peak_equity, current_equity)
        
        current_drawdown_pct = 0.0
        if self.peak_equity > 0:
            current_drawdown_pct = ((self.peak_equity - current_equity) / self.peak_equity * 100)
        
        # 1. DRAWDOWN CHECK (CRITICAL)
        if current_drawdown_pct > self.limits.max_drawdown_pct:
            violations.append(f"DRAWDOWN BREACH: {current_drawdown_pct:.1f}%")
            self._trigger_halt(f"Max drawdown exceeded: {current_drawdown_pct:.1f}%")
        
        # 2. DAILY LOSS CHECK (CRITICAL)
        pnl_today = portfolio_summary.get('total_pnl', 0) # Simplified: Assumes PnL resets daily or is tracked elsewhere
        
        if pnl_today < -self.limits.max_daily_loss:
            violations.append(f"DAILY LOSS BREACH: ₹{pnl_today:,.0f}")
            self._trigger_halt(f"Max daily loss exceeded: ₹{pnl_today:,.0f}")
        
        # 3. MARGIN CHECK
        margin_used = portfolio_summary.get('total_margin_used', 0)
        margin_avail = portfolio_summary.get('available_margin', 0)
        total_margin = margin_used + margin_avail
        
        margin_utilization = 0.0
        if total_margin > 0:
            margin_utilization = (margin_used / total_margin) * 100
            
        if margin_utilization > self.limits.max_margin_utilization_pct:
            warnings.append(f"High margin utilization: {margin_utilization:.1f}%")

        # 4. MARKET CHECKS
        vix = current_market.get('vix', 0)
        if vix > self.limits.halt_on_vix_spike:
            violations.append(f"VIX Spike: {vix:.2f}")
            self._trigger_halt(f"VIX Spike to {vix:.2f}")

        # Determine Status
        if self.halt_status:
            risk_status = RiskStatus.HALT
        elif violations:
            risk_status = RiskStatus.CRITICAL
        elif warnings:
            risk_status = RiskStatus.WARNING
        else:
            risk_status = RiskStatus.NORMAL
            
        return RiskMetrics(
            timestamp=datetime.now(),
            total_capital_deployed=portfolio_summary.get('total_investment', 0),
            total_pnl_today=pnl_today,
            total_pnl_week=0.0, # Implement weekly tracking logic
            current_drawdown_pct=current_drawdown_pct,
            peak_equity=self.peak_equity,
            portfolio_delta=portfolio_summary.get('net_delta', 0),
            portfolio_gamma=portfolio_summary.get('net_gamma', 0),
            portfolio_vega=portfolio_summary.get('net_vega', 0),
            portfolio_theta=portfolio_summary.get('net_theta', 0),
            margin_utilization_pct=margin_utilization,
            total_positions=portfolio_summary.get('total_positions', 0),
            orders_last_minute=0, # Simplified for metrics object
            orders_last_hour=0,
            current_vix=vix,
            vov_zscore=current_market.get('vov_zscore', 0),
            risk_status=risk_status,
            violations=violations,
            warnings=warnings
        )
    
    def _trigger_halt(self, reason: str):
        """Trigger trading halt"""
        if not self.halt_status:
            self.halt_status = True
            self.halt_reason = reason
            self.logger.critical(f"🛑 TRADING HALTED: {reason}")
    
    def get_risk_adjusted_allocation(self, base_allocation_pct: float,
                                    regime_score: float,
                                    current_risk_metrics: RiskMetrics) -> float:
        """Calculate risk-adjusted allocation percentage"""
        adjusted = base_allocation_pct
        
        # Reduce if approaching limits
        if current_risk_metrics.current_drawdown_pct > self.limits.max_drawdown_pct * 0.5:
            adjusted *= 0.5
            
        # Boost if regime is perfect and risk is low
        if regime_score > 8.0 and current_risk_metrics.risk_status == RiskStatus.NORMAL:
            adjusted *= 1.2
            
        return max(0, min(adjusted, base_allocation_pct))
