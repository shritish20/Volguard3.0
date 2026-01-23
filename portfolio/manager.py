"""
Portfolio Manager - PRODUCTION VERIFIED
Implements Tri-Layer Greek Engine (WebSocket -> Snapshot -> BSM)
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
from analytics.pricing import BlackScholesEngine  # Backup Engine


class PortfolioManager:
    """Manage portfolio with real Upstox data"""
    
    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher
        self.logger = setup_logger("portfolio")
        
        # State
        self.positions: Dict[str, Position] = {}
        self.portfolio_history = []
        
    def refresh_positions(self) -> bool:
        """Refresh all positions from Upstox and Subscribe to Greeks"""
        try:
            # 1. Get positions from Upstox API
            positions_data = self.data_fetcher.get_positions()
            
            # Clear existing to handle closed positions
            self.positions.clear()
            
            option_keys = []
            
            for pos_data in positions_data:
                try:
                    position = self._parse_position_data(pos_data)
                    if position:
                        self.positions[position.instrument_key] = position
                        
                        # Collect option keys for WebSocket subscription
                        if position.position_type == PositionType.OPTION:
                            option_keys.append(position.instrument_key)
                            
                except Exception as e:
                    self.logger.error(f"Error parsing position: {e}")
            
            # 2. CRITICAL: Subscribe to Greeks via WebSocket
            if option_keys and self.data_fetcher.websocket_manager:
                self.data_fetcher.websocket_manager.subscribe_greeks(option_keys)
                self.logger.info(f"Subscribed to Greeks for {len(option_keys)} options")
            
            # 3. Update Greeks (Tri-Layer Logic)
            self._update_position_greeks()
            
            # 4. Calculate P&L
            self._calculate_all_pnl()
            
            # 5. Record History
            self._record_portfolio_snapshot()
            
            self.logger.info(f"Refreshed {len(self.positions)} positions")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to refresh positions: {e}")
            return False
    
    def _parse_position_data(self, pos_data: Any) -> Optional[Position]:
        """Parse Upstox position data safely (handles Dict or Object)"""
        try:
            # Helper to get value from Dict or Object
            def get_val(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            trading_symbol = get_val(pos_data, "trading_symbol", "")
            instrument_key = get_val(pos_data, "instrument_key", "")
            
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
            
            # Parse quantity
            quantity = int(get_val(pos_data, "quantity", 0))
            if quantity == 0: return None # Filter out closed positions
            
            # Prices
            average_price = float(get_val(pos_data, "average_price", 0))
            last_price = float(get_val(pos_data, "last_price", 0) or average_price)
            
            # Lot size
            lot_size = int(get_val(pos_data, "lot_size", 0) or UPSTOX_CONFIG.DEFAULT_LOT_SIZE)
            
            # Parse expiry date & strike if option
            expiry_date = None
            strike_price = None
            
            if instrument_type in ["CE", "PE"]:
                parts = trading_symbol.split()
                # Example: NIFTY 22000 CE 30 JAN 25
                if len(parts) >= 4:
                    try:
                        strike_price = float(parts[1])
                        expiry_str = f"{parts[3]}-{parts[4]}-{parts[5]}"
                        expiry_date = datetime.strptime(expiry_str, "%d-%b-%y").date()
                    except:
                        pass
            
            position = Position(
                instrument_token=get_val(pos_data, "instrument_token", ""),
                instrument_key=instrument_key,
                trading_symbol=trading_symbol,
                instrument_type=instrument_type,
                quantity=quantity,
                average_price=average_price,
                current_price=last_price,
                lot_size=lot_size,
                expiry_date=expiry_date,
                strike_price=strike_price,
                margin_used=float(get_val(pos_data, "margin_used", 0))
            )
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error in position parsing: {e}")
            return None
    
    def _update_position_greeks(self):
        """
        Tri-Layer Greek Engine:
        1. WebSocket (Fastest, <10ms)
        2. API Snapshot (Verified, ~200ms)
        3. Local BSM (Safety Net, Fallback)
        """
        # Filter for active option positions
        option_positions = [
            p for p in self.positions.values() 
            if p.position_type == PositionType.OPTION
        ]
        
        if not option_positions:
            return
            
        missing_greeks_keys = []

        # --- LAYER 1: WEBSOCKET CACHE ---
        for position in option_positions:
            ws_data = None
            if self.data_fetcher.websocket_manager:
                ws_data = self.data_fetcher.websocket_manager.get_latest_data(position.instrument_key)
            
            if ws_data and 'greeks' in ws_data:
                # 🟢 Live Data Found
                g = ws_data['greeks']
                position.delta = float(g.get('delta', 0))
                position.gamma = float(g.get('gamma', 0))
                position.theta = float(g.get('theta', 0))
                position.vega  = float(g.get('vega', 0))
                position.iv    = float(g.get('iv', 0))
            else:
                # 🟡 Mark for Snapshot
                missing_greeks_keys.append(position.instrument_key)

        # --- LAYER 2: API SNAPSHOT ---
        if missing_greeks_keys:
            # Fetch snapshot for all missing keys at once
            snapshot_data = self.data_fetcher.get_greeks_snapshot(missing_greeks_keys)
            
            for position in option_positions:
                if position.instrument_key in snapshot_data:
                    # 🟢 Snapshot Data Found
                    g = snapshot_data[position.instrument_key]
                    position.delta = float(g.get('delta', 0))
                    position.gamma = float(g.get('gamma', 0))
                    position.theta = float(g.get('theta', 0))
                    position.vega  = float(g.get('vega', 0))
                    position.iv    = float(g.get('iv', 0))
                
                # --- LAYER 3: LOCAL BSM (Fallback) ---
                elif position.delta == 0 and position.instrument_key in missing_greeks_keys:
                    # 🔴 Fallback to Math
                    spot = self.data_fetcher.get_spot_price()
                    if spot and position.strike_price and position.expiry_date:
                        # Estimate IV from VIX
                        vix = self.data_fetcher.get_vix_price() or 15.0
                        
                        bsm_greeks = BlackScholesEngine.calculate_greeks(
                            S=spot,
                            K=position.strike_price,
                            expiry_date=position.expiry_date,
                            sigma=vix/100.0,
                            option_type=position.instrument_type
                        )
                        position.delta = bsm_greeks['delta']
                        position.gamma = bsm_greeks['gamma']
                        position.theta = bsm_greeks['theta']
                        position.vega  = bsm_greeks['vega']
                        position.iv    = vix # Proxy

    def _calculate_all_pnl(self):
        """Calculate P&L using Live WebSocket Price if available"""
        for position in self.positions.values():
            ws_data = None
            if self.data_fetcher.websocket_manager:
                ws_data = self.data_fetcher.websocket_manager.get_latest_data(position.instrument_key)
            
            if ws_data and ws_data.get('ltp'):
                # Use Live Tick
                position.calculate_pnl(float(ws_data['ltp']))
            else:
                # Use Last Known
                position.calculate_pnl(position.current_price)
    
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
        
        # Parse available margin safely
        available_margin = 0
        def safe_get(obj, key, default=0):
            if isinstance(obj, dict): return obj.get(key, default)
            return getattr(obj, key, default)

        if funds_data:
            equity = safe_get(funds_data, 'equity', {})
            available_margin = float(safe_get(equity, 'available_margin', 0))
        
        # Calculate concentration ratio
        concentration_ratio = 0
        if total_current_value > 0:
            position_values = [p.notional_value for p in self.positions.values()]
            position_values.sort(reverse=True)
            top_3_value = sum(position_values[:3])
            concentration_ratio = top_3_value / total_current_value
        
        # VaR & Risk
        var_95 = self._calculate_var_95()
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
            
            pnl_changes = []
            for i in range(1, min(20, len(self.portfolio_history))):
                pnl_change = (
                    self.portfolio_history[i]['summary'].total_pnl -
                    self.portfolio_history[i-1]['summary'].total_pnl
                )
                pnl_changes.append(pnl_change)
            
            if not pnl_changes:
                return 0
            
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
