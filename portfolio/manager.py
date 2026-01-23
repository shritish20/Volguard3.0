"""
Portfolio Manager with CORRECT Upstox API usage
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from config.settings import ANALYTICS_CONFIG, UPSTOX_CONFIG
from portfolio.models import Position, PositionType, PortfolioSummary
from utils.logger import setup_logger


class PortfolioManager:
    """Manage portfolio with real Upstox data"""
    
    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher
        self.logger = setup_logger("portfolio")
        
        # State
        self.positions: Dict[str, Position] = {}
        self.portfolio_history = []
        self.greek_cache = {}
        self.greek_cache_time = {}
        
    def refresh_positions(self) -> bool:
        """Refresh all positions from Upstox"""
        try:
            # Get positions from Upstox
            positions_data = self.data_fetcher.get_positions()
            
            # Clear existing positions
            self.positions.clear()
            
            for pos_data in positions_data:
                try:
                    position = self._parse_position_data(pos_data)
                    if position:
                        self.positions[position.instrument_key] = position
                except Exception as e:
                    self.logger.error(f"Error parsing position: {e}")
            
            # Update Greek values for options
            self._update_position_greeks()
            
            # Calculate P&L
            self._calculate_all_pnl()
            
            # Record portfolio snapshot
            self._record_portfolio_snapshot()
            
            self.logger.info(f"Refreshed {len(self.positions)} positions")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to refresh positions: {e}")
            return False
    
    def _parse_position_data(self, pos_data: Dict) -> Optional[Position]:
        """Parse Upstox position data"""
        try:
            instrument_key = pos_data.get("instrument_key", "")
            trading_symbol = pos_data.get("trading_symbol", "")
            
            # Skip non-NIFTY positions for our cockpit
            if not any(x in trading_symbol for x in ["NIFTY", "BANKNIFTY", "FINNIFTY"]):
                return None
            
            # Determine instrument type
            if "CE" in trading_symbol:
                instrument_type = "CE"
            elif "PE" in trading_symbol:
                instrument_type = "PE"
            elif "FUT" in trading_symbol:
                instrument_type = "FUT"
            else:
                instrument_type = "EQ"
            
            # Parse quantity (positive for long, negative for short)
            quantity = int(pos_data.get("quantity", 0))
            
            # Get average price
            average_price = float(pos_data.get("average_price", 0))
            
            # Get current price (use LTP if available)
            current_price = float(pos_data.get("last_price", average_price))
            
            # Get lot size (default to 75 for NIFTY)
            lot_size = int(pos_data.get("lot_size", UPSTOX_CONFIG.DEFAULT_LOT_SIZE))
            
            # Parse expiry date if option
            expiry_date = None
            strike_price = None
            
            if instrument_type in ["CE", "PE"]:
                # Extract from trading symbol: NIFTY 22000 CE 30 JAN 25
                parts = trading_symbol.split()
                if len(parts) >= 4:
                    try:
                        strike_price = float(parts[1])
                        # Parse expiry: "30-JAN-25"
                        expiry_str = f"{parts[3]}-{parts[4]}-{parts[5]}"
                        expiry_date = datetime.strptime(expiry_str, "%d-%b-%y").date()
                    except:
                        pass
            
            position = Position(
                instrument_token=pos_data.get("instrument_token", ""),
                instrument_key=instrument_key,
                trading_symbol=trading_symbol,
                instrument_type=instrument_type,
                quantity=quantity,
                average_price=average_price,
                current_price=current_price,
                lot_size=lot_size,
                expiry_date=expiry_date,
                strike_price=strike_price,
                margin_used=float(pos_data.get("margin_used", 0))
            )
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error in position parsing: {e}")
            return None
    
    def _update_position_greeks(self):
        """Update Greek values for option positions"""
        try:
            # Get option positions
            option_positions = [
                p for p in self.positions.values() 
                if p.position_type == PositionType.OPTION
            ]
            
            if not option_positions:
                return
            
            # Get instrument keys
            instrument_keys = [p.instrument_key for p in option_positions]
            
            # Try to get Greeks from market quotes
            # Note: This requires proper API endpoint, might need adjustment
            for position in option_positions:
                # Placeholder for Greek calculation
                # In production, you'd call Upstox API for option Greeks
                position.delta = 0.5 if position.instrument_type == "CE" else -0.5
                position.gamma = 0.02
                position.theta = -0.05
                position.vega = 0.15
                position.iv = 15.0
                    
        except Exception as e:
            self.logger.error(f"Failed to update Greeks: {e}")
    
    def _calculate_all_pnl(self):
        """Calculate P&L for all positions"""
        for position in self.positions.values():
            position.calculate_pnl()
    
    def _record_portfolio_snapshot(self):
        """Record portfolio snapshot for history"""
        summary = self.get_portfolio_summary()
        snapshot = {
            'timestamp': datetime.now(),
            'summary': summary,
            'positions_count': len(self.positions)
        }
        self.portfolio_history.append(snapshot)
        
        # Keep only last 1000 snapshots
        if len(self.portfolio_history) > 1000:
            self.portfolio_history.pop(0)
    
    def get_portfolio_summary(self) -> PortfolioSummary:
        """Get comprehensive portfolio summary"""
        total_positions = len(self.positions)
        
        # Calculate totals
        total_investment = sum(p.investment_value for p in self.positions.values())
        total_current_value = sum(p.notional_value for p in self.positions.values())
        total_pnl = sum(p.pnl for p in self.positions.values())
        total_margin_used = sum(p.margin_used for p in self.positions.values())
        
        # Calculate P&L percentage
        total_pnl_percentage = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        
        # Calculate net Greeks
        net_delta = sum(p.delta * p.quantity * p.lot_size for p in self.positions.values())
        net_gamma = sum(p.gamma * p.quantity * p.lot_size for p in self.positions.values())
        net_theta = sum(p.theta * p.quantity * p.lot_size for p in self.positions.values())
        net_vega = sum(p.vega * p.quantity * p.lot_size for p in self.positions.values())
        
        # Get available margin
        funds_data = self.data_fetcher.get_funds_and_margin()
        
        # Parse available margin from funds data
        available_margin = 0
        if isinstance(funds_data, dict):
            # Try to extract from the complex structure
            equity = funds_data.get('equity', {})
            if isinstance(equity, dict):
                available_margin = equity.get('available_margin', 0)
            else:
                # Try to access as object
                available_margin = getattr(equity, 'available_margin', 0)
        
        # Calculate concentration ratio
        if total_current_value > 0:
            position_values = [p.notional_value for p in self.positions.values()]
            position_values.sort(reverse=True)
            top_3_value = sum(position_values[:3])
            concentration_ratio = top_3_value / total_current_value
        else:
            concentration_ratio = 0
        
        # Simple VaR calculation (95% confidence)
        var_95 = self._calculate_var_95()
        
        # Max risk (simplified)
        max_risk = total_investment * 0.05  # 5% of investment
        
        return PortfolioSummary(
            total_positions=total_positions,
            total_investment=total_investment,
            total_current_value=total_current_value,
            total_pnl=total_pnl,
            total_pnl_percentage=total_pnl_percentage,
            total_margin_used=total_margin_used,
            available_margin=available_margin,
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_theta=net_theta,
            net_vega=net_vega,
            max_risk=max_risk,
            var_95=var_95,
            concentration_ratio=concentration_ratio
        )
    
    def _calculate_var_95(self) -> float:
        """Calculate Value at Risk at 95% confidence"""
        try:
            if len(self.portfolio_history) < 20:
                return 0
            
            # Get recent P&L changes
            pnl_changes = []
            for i in range(1, min(20, len(self.portfolio_history))):
                pnl_change = (
                    self.portfolio_history[i]['summary'].total_pnl -
                    self.portfolio_history[i-1]['summary'].total_pnl
                )
                pnl_changes.append(pnl_change)
            
            if not pnl_changes:
                return 0
            
            # Calculate 5th percentile (95% VaR)
            var_95 = np.percentile(pnl_changes, 5)
            return abs(var_95)
            
        except Exception as e:
            self.logger.error(f"VaR calculation failed: {e}")
            return 0
    
    def get_position_concentration(self) -> pd.DataFrame:
        """Get position concentration analysis"""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for position in self.positions.values():
            data.append({
                'Symbol': position.trading_symbol,
                'Type': position.instrument_type,
                'Quantity': position.quantity,
                'Value (₹)': position.notional_value,
                'P&L (₹)': position.pnl,
                'P&L (%)': position.pnl_percentage,
                'Delta': position.delta * position.quantity * position.lot_size,
                'Gamma': position.gamma * position.quantity * position.lot_size,
                'Margin (₹)': position.margin_used
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values('Value (₹)', ascending=False)
            df['% of Portfolio'] = (df['Value (₹)'] / df['Value (₹)'].sum() * 100).round(2)
        
        return df
    
    def get_greek_exposure_report(self) -> Dict:
        """Get detailed Greek exposure report"""
        summary = self.get_portfolio_summary()
        
        # Calculate individual Greek contributions
        delta_contributors = []
        gamma_contributors = []
        vega_contributors = []
        
        for position in self.positions.values():
            delta_exposure = position.delta * position.quantity * position.lot_size
            gamma_exposure = position.gamma * position.quantity * position.lot_size
            vega_exposure = position.vega * position.quantity * position.lot_size
            
            if abs(delta_exposure) > 0.1:
                delta_contributors.append({
                    'symbol': position.trading_symbol,
                    'exposure': delta_exposure,
                    'percentage': (delta_exposure / summary.net_delta * 100) if summary.net_delta != 0 else 0
                })
            
            if abs(gamma_exposure) > 0.001:
                gamma_contributors.append({
                    'symbol': position.trading_symbol,
                    'exposure': gamma_exposure,
                    'percentage': (gamma_exposure / summary.net_gamma * 100) if summary.net_gamma != 0 else 0
                })
            
            if abs(vega_exposure) > 0.01:
                vega_contributors.append({
                    'symbol': position.trading_symbol,
                    'exposure': vega_exposure,
                    'percentage': (vega_exposure / summary.net_vega * 100) if summary.net_vega != 0 else 0
                })
        
        # Sort contributors
        delta_contributors.sort(key=lambda x: abs(x['exposure']), reverse=True)
        gamma_contributors.sort(key=lambda x: abs(x['exposure']), reverse=True)
        vega_contributors.sort(key=lambda x: abs(x['exposure']), reverse=True)
        
        # Margin utilization
        margin_utilization = 0
        if summary.total_margin_used > 0:
            margin_utilization = (
                summary.total_margin_used / 
                (summary.total_margin_used + summary.available_margin) * 100 
                if (summary.total_margin_used + summary.available_margin) > 0 else 0
            )
        
        return {
            'net_exposures': {
                'delta': summary.net_delta,
                'gamma': summary.net_gamma,
                'theta': summary.net_theta,
                'vega': summary.net_vega
            },
            'top_delta_contributors': delta_contributors[:5],
            'top_gamma_contributors': gamma_contributors[:5],
            'top_vega_contributors': vega_contributors[:5],
            'margin_utilization': margin_utilization
          }
