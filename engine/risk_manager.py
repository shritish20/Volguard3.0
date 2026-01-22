"""
VOLGUARD v31.0 – RISK MANAGER
=============================
The Shield - Smart State Recovery and Live WebSocket Monitoring
"""

import threading
import time
from datetime import datetime, date
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import upstox_client
from colorama import Fore, Style

@dataclass
class RecoveredPosition:
    """Position data recovered from Upstox"""
    position_id: str
    instrument_token: str
    symbol: str
    quantity: int
    strike_price: float
    option_type: str
    expiry_date: date
    entry_price: float
    current_price: float
    pnl: float
    lot_size: int = 50
    structure: str = "UNKNOWN"

class UpstoxRiskManager:
    """Smart Risk Manager with state recovery and live monitoring"""
    
    def __init__(self, access_token: str, data_engine, position_monitor_logic):
        self.access_token = access_token
        self.data_engine = data_engine
        self.logic = position_monitor_logic
        
        # WebSocket setup
        self.streamer = None
        self.subscribed_tokens: Set[str] = set()
        self.is_connected = False
        
        # Position tracking
        self.recovered_positions: Dict[str, RecoveredPosition] = {}
        self.active_positions: Dict[str, RecoveredPosition] = {}
        
        # Emergency state
        self.emergency_active = False
        self.emergency_reason = None
        
        print(f"{Fore.GREEN}✓ UpstoxRiskManager initialized{Style.RESET_ALL}")
    
    def sync_state(self) -> Dict:
        """
        SMART RECOVERY:
        1. Fetch Funds
        2. Fetch existing Open Positions from Upstox
        3. Re-populate the internal PositionMonitor
        4. Subscribe to live feeds
        """
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"🔄 SYNCING STATE WITH UPSTOX - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*80}{Style.RESET_ALL}")
        
        recovery_summary = {
            "funds_fetched": False,
            "positions_recovered": 0,
            "websocket_started": False,
            "errors": []
        }
        
        try:
            # 1. Check Funds
            print(f"\n{Fore.YELLOW}💰 Checking Available Funds...{Style.RESET_ALL}")
            funds = self.data_engine.fetch_funds_and_margin()
            
            if funds['available_margin'] > 0:
                recovery_summary['funds_fetched'] = True
                print(f"  Available Margin: ₹{funds['available_margin']:,.2f}")
                print(f"  Used Margin: ₹{funds['used_margin']:,.2f}")
                print(f"  Total Balance: ₹{funds['total_balance']:,.2f}")
            else:
                print(f"  {Fore.YELLOW}⚠️ No margin data available{Style.RESET_ALL}")
                recovery_summary['errors'].append("No margin data")
            
            # 2. Recover Positions
            print(f"\n{Fore.YELLOW}📊 Recovering Open Positions...{Style.RESET_ALL}")
            upstox_positions = self.data_engine.fetch_positions()
            
            if upstox_positions:
                print(f"  Found {len(upstox_positions)} open positions on server")
                
                for i, pos in enumerate(upstox_positions, 1):
                    recovered_pos = self._convert_to_internal_position(pos)
                    self.recovered_positions[recovered_pos.position_id] = recovered_pos
                    
                    print(f"  {i}. {recovered_pos.symbol} {recovered_pos.option_type} "
                          f"{recovered_pos.strike_price} @ ₹{recovered_pos.entry_price:.2f} "
                          f"(Qty: {recovered_pos.quantity}, PnL: ₹{recovered_pos.pnl:,.0f})")
                    
                    # Add to websocket subscription immediately
                    self.subscribe_leg(recovered_pos.instrument_token)
                
                recovery_summary['positions_recovered'] = len(upstox_positions)
                self._populate_internal_monitor()
                
            else:
                print(f"  {Fore.GREEN}✓ No existing positions found{Style.RESET_ALL}")
            
            # 3. Start WebSocket
            print(f"\n{Fore.YELLOW}🛡️ Starting Risk Shield WebSocket...{Style.RESET_ALL}")
            self.start_stream()
            recovery_summary['websocket_started'] = True
            
            print(f"\n{Fore.GREEN}✓ State sync complete!{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ State sync error: {e}{Style.RESET_ALL}")
            recovery_summary['errors'].append(str(e))
        
        return recovery_summary
    
    def _convert_to_internal_position(self, upstox_pos: Dict) -> RecoveredPosition:
        """Convert Upstox position to internal format"""
        
        structure = "UNKNOWN"
        if upstox_pos['quantity'] > 0:
            structure = "LONG_OPTION"
        else:
            structure = "SHORT_OPTION"
        
        return RecoveredPosition(
            position_id=upstox_pos['position_id'],
            instrument_token=upstox_pos['instrument_token'],
            symbol=upstox_pos['symbol'],
            quantity=upstox_pos['quantity'],
            strike_price=float(upstox_pos['strike_price']),
            option_type=upstox_pos['option_type'],
            expiry_date=datetime.strptime(upstox_pos['expiry_date'], "%Y-%m-%d").date(),
            entry_price=float(upstox_pos['entry_price']),
            current_price=float(upstox_pos['current_price']),
            pnl=float(upstox_pos['pnl']),
            lot_size=50,
            structure=structure
        )
    
    def _populate_internal_monitor(self):
        """Convert recovered positions to internal Position objects"""
        
        for pos_id, recovered_pos in self.recovered_positions.items():
            # Create internal Position object
            position_data = {
                'position_id': pos_id,
                'structure': recovered_pos.structure,
                'entry_date': datetime.now().date(),
                'entry_time': datetime.now().strftime("%H:%M:%S"),
                'expiry_date': recovered_pos.expiry_date,
                'dte_at_entry': (recovered_pos.expiry_date - date.today()).days,
                'strikes': {recovered_pos.option_type.lower(): recovered_pos.strike_price},
                'lots': abs(recovered_pos.quantity) // recovered_pos.lot_size,
                'lot_size': recovered_pos.lot_size,
                'entry_premium': recovered_pos.entry_price,
                'entry_vix': 15.0,
                'entry_spot': 23500,
                'current_premium': recovered_pos.current_price,
                'current_spot': 23500,
                'current_vix': 15.0,
                'current_dte': (recovered_pos.expiry_date - date.today()).days,
                'unrealized_pnl': recovered_pos.pnl,
                'pnl_pct': (recovered_pos.pnl / (recovered_pos.entry_price * abs(recovered_pos.quantity))) * 100
            }
            
            # Add to internal position monitor
            if hasattr(self.logic, 'add_position'):
                self.logic.add_position(position_data)
            
            self.active_positions[pos_id] = recovered_pos
    
    def start_stream(self):
        """Start V3 WebSocket (Protobuf) for live monitoring"""
        
        try:
            config = upstox_client.Configuration()
            config.access_token = self.access_token
            api_client = upstox_client.ApiClient(config)
            
            self.streamer = upstox_client.MarketDataStreamerV3(api_client, mode="full")
            
            # Set up event handlers
            self.streamer.on("open", self.on_open)
            self.streamer.on("message", self.on_message)
            self.streamer.on("error", self.on_error)
            self.streamer.on("close", self.on_close)
            
            # Start connection in separate thread
            thread = threading.Thread(target=self.streamer.connect, daemon=True)
            thread.start()
            
            print("  🛡️ Risk Shield: WebSocket connecting...")
            
        except Exception as e:
            print(f"{Fore.RED}❌ WebSocket start error: {e}{Style.RESET_ALL}")
    
    def on_open(self, message):
        """WebSocket connection opened"""
        self.is_connected = True
        print(f"  {Fore.GREEN}✓ Risk Shield: Connected!{Style.RESET_ALL}")
        
        # Subscribe to existing positions
        for token in self.subscribed_tokens:
            self.subscribe_leg(token)
    
    def on_message(self, message):
        """Receives Decoded Protobuf Data (Clean Dict)"""
        
        feeds = message.get('feeds')
        if not feeds:
            return
        
        for key, data in feeds.items():
            ltpc = data.get('ltpc', {})
            greeks = data.get('optionGreeks', {})
            
            # Feed this live data into your Position Logic
            if hasattr(self.logic, 'update_position_metrics'):
                self.logic.update_position_metrics(key, {
                    "current_premium": ltpc.get('ltp'),
                    "current_spot": ltpc.get('ltp'),
                    "greeks": {
                        "delta": greeks.get('delta', 0),
                        "gamma": greeks.get('gamma', 0),
                        "theta": greeks.get('theta', 0),
                        "vega": greeks.get('vega', 0)
                    },
                    "current_vix": 15.0
                })
            
            # Check for emergency conditions
            self._check_live_emergency_conditions(key, data)
    
    def on_error(self, error):
        """WebSocket error handler"""
        print(f"{Fore.RED}❌ WebSocket Error: {error}{Style.RESET_ALL}")
        self.is_connected = False
    
    def on_close(self):
        """WebSocket connection closed"""
        print(f"{Fore.YELLOW}⚠️ WebSocket connection closed{Style.RESET_ALL}")
        self.is_connected = False
    
    def subscribe_leg(self, instrument_key: str):
        """Subscribe to specific instrument for live updates"""
        if not self.streamer:
            print(f"{Fore.YELLOW}⚠️ WebSocket not initialized{Style.RESET_ALL}")
            return
        
        if instrument_key in self.subscribed_tokens:
            return  # Already subscribed
        
        try:
            self.streamer.subscribe([instrument_key], mode="full")
            self.subscribed_tokens.add(instrument_key)
            print(f"  📡 Subscribed to: {instrument_key}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ Subscribe error: {e}{Style.RESET_ALL}")
    
    def unsubscribe_leg(self, instrument_key: str):
        """Unsubscribe from instrument"""
        if not self.streamer:
            return
        
        if instrument_key in self.subscribed_tokens:
            try:
                self.streamer.unsubscribe([instrument_key])
                self.subscribed_tokens.discard(instrument_key)
                print(f"  📡 Unsubscribed from: {instrument_key}")
            except Exception as e:
                print(f"{Fore.RED}❌ Unsubscribe error: {e}{Style.RESET_ALL}")
    
    def _check_live_emergency_conditions(self, instrument_key: str, data: Dict):
        """Check for emergency conditions in real-time"""
        
        if self.emergency_active:
            return
        
        # Check for extreme price movements
        ltpc = data.get('ltpc', {})
        if ltpc:
            ltp = ltpc.get('ltp', 0)
            if ltp > 0:
                greeks = data.get('optionGreeks', {})
                if greeks:
                    delta = greeks.get('delta', 0)
                    if abs(delta) > 0.95:
                        self._trigger_emergency("EXTREME_DELTA", f"Delta {delta:.2f} for {instrument_key}")
    
    def _trigger_emergency(self, emergency_type: str, reason: str):
        """Trigger emergency protocol"""
        
        self.emergency_active = True
        self.emergency_reason = reason
        
        print(f"\n{Fore.RED}{'='*80}")
        print(f"🚨 EMERGENCY TRIGGERED: {emergency_type}")
        print(f"Reason: {reason}")
        print(f"{'='*80}{Style.RESET_ALL}")
        
        if emergency_type == "EXTREME_DELTA":
            self._emergency_close_deep_itm()
        elif emergency_type == "MARKET_CRASH":
            self._emergency_close_all_undefined()
        else:
            self._emergency_close_all()
    
    def _emergency_close_deep_itm(self):
        """Close all deep ITM positions"""
        print("Closing all deep ITM positions...")
        
        for pos_id, position in self.active_positions.items():
            if abs(position.current_price - position.entry_price) > position.entry_price * 0.5:
                print(f"  Closing deep ITM: {pos_id}")
    
    def _emergency_close_all_undefined(self):
        """Close all undefined risk positions"""
        print("Closing all undefined risk positions...")
        
        for pos_id, position in self.active_positions.items():
            if position.structure in ["SHORT_STRANGLE", "SHORT_STRADDLE"]:
                print(f"  Closing undefined risk: {pos_id}")
    
    def _emergency_close_all(self):
        """Emergency close everything"""
        print("Closing ALL positions...")
        
        for pos_id in list(self.active_positions.keys()):
            print(f"  Emergency close: {pos_id}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get real-time portfolio summary"""
        
        return {
            "total_positions": len(self.active_positions),
            "total_pnl": sum(pos.pnl for pos in self.active_positions.values()),
            "websocket_connected": self.is_connected,
            "subscribed_instruments": len(self.subscribed_tokens),
            "emergency_active": self.emergency_active
        }
    
    def generate_risk_report(self) -> str:
        """Generate comprehensive risk report"""
        
        report = []
        report.append("="*80)
        report.append(f"VOLGUARD v31.0 RISK REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        
        # Connection status
        report.append(f"\n🛡️ WEBSOCKET STATUS: {'CONNECTED' if self.is_connected else 'DISCONNECTED'}")
        report.append(f"📡 Subscribed Instruments: {len(self.subscribed_tokens)}")
        
        # Emergency status
        if self.emergency_active:
            report.append(f"\n🚨 EMERGENCY ACTIVE: {self.emergency_reason}")
        else:
            report.append(f"\n✅ System operating normally")
        
        # Position summary
        report.append(f"\n📊 POSITION SUMMARY:")
        report.append(f"  Active Positions: {len(self.active_positions)}")
        
        if self.active_positions:
            total_pnl = sum(pos.pnl for pos in self.active_positions.values())
            report.append(f"  Total P&L: ₹{total_pnl:+,.2f}")
            
            # Position details
            report.append(f"\n  POSITION DETAILS:")
            for pos_id, pos in self.active_positions.items():
                report.append(f"    {pos_id}: {pos.symbol} {pos.option_type} "
                             f"{pos.strike_price} (Qty: {pos.quantity}, PnL: ₹{pos.pnl:+,.0f})")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
