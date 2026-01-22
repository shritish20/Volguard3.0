"""
VOLGUARD v31.0 – MAIN COMMANDER
===============================
Integrates Logic (Regime/Trading) with Infrastructure (Upstox Engines)
"""
import time
import sys
from datetime import datetime, date, timedelta
from colorama import Fore, Style, init

# 1. Import The Brain (Your Logic)
from logic.regime_logic import Config, AnalyticsEngine, RegimeEngine, DisplayEngine, TimeMetrics, VolMetrics, StructMetrics, EdgeMetrics, ExternalMetrics, RegimeScore, TradingMandate, ParticipantDataFetcher
from logic.trading_logic import StrategySelector, StrikeSelector, PositionSizer, PositionMonitor, Position, OrderExecutor

# 2. Import The Infrastructure (Upstox Integration)
from engines.data_engine import UpstoxDataEngine
from engines.execution_engine import UpstoxExecutionEngine
from engines.risk_manager import UpstoxRiskManager

init(autoreset=True)

class VolGuardSystem:
    def __init__(self, token):
        print(f"\n{Fore.GREEN}╔{'═'*60}╗")
        print(f"║{'SYSTEM STARTUP: VOLGUARD v31.0':^60}║")
        print(f"║{'UPSTOX V3 INTEGRATION - PRODUCTION READY':^60}║")
        print(f"╚{'═'*60}╝{Style.RESET_ALL}")

        # A. Initialize Infrastructure
        print(f"{Fore.CYAN}1️⃣  Initializing Upstox Engines...{Style.RESET_ALL}")
        self.data_engine = UpstoxDataEngine(token)
        self.executor = UpstoxExecutionEngine(token)
        
        # B. Initialize Logic
        print(f"{Fore.CYAN}2️⃣  Initializing Logic Cores...{Style.RESET_ALL}")
        self.analytics = AnalyticsEngine()
        self.regime = RegimeEngine()
        self.display = DisplayEngine()
        self.position_monitor = PositionMonitor()
        
        # C. Initialize Risk Shield
        print(f"{Fore.CYAN}3️⃣  Initializing Risk Shield...{Style.RESET_ALL}")
        self.risk_manager = UpstoxRiskManager(token, self.data_engine, self.position_monitor)

    def run_daily_cycle(self):
        """Runs the main analysis and execution cycle"""
        
        # --- STEP 1: HEALTH & RECOVERY ---
        print(f"\n{Fore.YELLOW}🛡️ SYSTEM HEALTH CHECK{Style.RESET_ALL}")
        
        # Fetch Funds
        funds = self.data_engine.fetch_funds_and_margin()
        print(f"   💰 Available Capital: ₹{funds['available_margin']:,.2f}")
        
        # Fetch Holidays
        holidays = self.data_engine.fetch_holidays()
        print(f"   📅 Market Holidays Loaded: {len(holidays)}")

        # Sync Portfolio (Crash Recovery)
        recovery = self.risk_manager.sync_state()

        # --- STEP 2: DATA ACQUISITION ---
        print(f"\n{Fore.YELLOW}📥 DATA ACQUISITION{Style.RESET_ALL}")
        
        # Fetch History
        nifty_hist = self.data_engine.fetch_history(Config.NIFTY_KEY)
        vix_hist = self.data_engine.fetch_history(Config.VIX_KEY)
        live_data = self.data_engine.fetch_live_quote([Config.NIFTY_KEY, Config.VIX_KEY])
        
        # Fetch Expiries (Dynamic)
        weekly, monthly, next_weekly, lot_size = self.data_engine.fetch_expiry_details()
        if not weekly:
            print(f"{Fore.RED}❌ Critical Error: Could not fetch Expiries.{Style.RESET_ALL}")
            return

        # Fetch Option Chains (Dynamic Tokens)
        print("   ⛓️ Fetching Option Chains...")
        weekly_chain = self.data_engine.fetch_option_chain(weekly)
        monthly_chain = self.data_engine.fetch_option_chain(monthly)

        # --- STEP 3: REGIME ANALYSIS ---
        print(f"\n{Fore.YELLOW}🧠 REGIME ANALYSIS{Style.RESET_ALL}")
        
        # Fetch Participant Data
        participant_data, participant_yest, fii_net_change, data_date, is_fallback = ParticipantDataFetcher.fetch_smart_participant_data()
        
        # Calculate Metrics
        time_metrics = self.analytics.get_time_metrics(weekly, monthly, next_weekly)
        vol_metrics = self.analytics.get_vol_metrics(nifty_hist, vix_hist, live_data.get(Config.NIFTY_KEY, 0), live_data.get(Config.VIX_KEY, 0))
        struct_weekly = self.analytics.get_struct_metrics(weekly_chain, vol_metrics.spot, lot_size)
        struct_monthly = self.analytics.get_struct_metrics(monthly_chain, vol_metrics.spot, lot_size)
        edge_metrics = self.analytics.get_edge_metrics(weekly_chain, monthly_chain, vol_metrics.spot, vol_metrics)
        external_metrics = self.analytics.get_external_metrics(nifty_hist, participant_data, participant_yest, fii_net_change, data_date, is_fallback)
        
        # Calculate Regime Scores
        weekly_score = self.regime.calculate_scores(vol_metrics, struct_weekly, edge_metrics, external_metrics, time_metrics, "WEEKLY")
        monthly_score = self.regime.calculate_scores(vol_metrics, struct_monthly, edge_metrics, external_metrics, time_metrics, "MONTHLY")
        
        # Generate Mandates
        weekly_mandate = self.regime.generate_mandate(weekly_score, vol_metrics, struct_weekly, edge_metrics, external_metrics, time_metrics, "WEEKLY", weekly, time_metrics.dte_weekly)
        monthly_mandate = self.regime.generate_mandate(monthly_score, vol_metrics, struct_monthly, edge_metrics, external_metrics, time_metrics, "MONTHLY", monthly, time_metrics.dte_monthly)

        # --- STEP 4: DISPLAY RESULTS ---
        self.display.render_header()
        self.display.render_time_context(time_metrics)
        self.display.render_vol_metrics(vol_metrics)
        self.display.render_participant_data(external_metrics)
        self.display.render_struct_metrics(struct_weekly)
        self.display.render_edge_metrics(edge_metrics)
        
        print(f"\n{Fore.CYAN}{'='*90}")
        print(f"{'📅 WEEKLY EXPIRY ANALYSIS':^90}")
        print(f"{'='*90}{Style.RESET_ALL}")
        self.display.render_regime_scores(weekly_score)
        self.display.render_mandate(weekly_mandate)
        
        print(f"\n{Fore.CYAN}{'='*90}")
        print(f"{'📅 MONTHLY EXPIRY ANALYSIS':^90}")
        print(f"{'='*90}{Style.RESET_ALL}")
        self.display.render_regime_scores(monthly_score)
        self.display.render_mandate(monthly_mandate)
        
        self.display.render_summary(weekly_mandate, monthly_mandate)

        # --- STEP 5: EXECUTION (If Mandate Exists) ---
        if weekly_mandate.max_lots > 0 and funds['available_margin'] > Config.MARGIN_SELL_BASE:
            print(f"\n{Fore.GREEN}🚀 EXECUTING STRATEGY: {weekly_mandate.strategy_type}{Style.RESET_ALL}")
            
            # 1. Select Strikes
            strategy_func = StrategySelector.select_strategy(
                weekly_mandate.strategy_type, "WEEKLY", weekly_mandate.dte,
                vol_metrics, struct_weekly, edge_metrics, external_metrics
            )[1]
            
            trade_plan = strategy_func(weekly_chain, vol_metrics.spot, weekly_mandate.dte)
            
            # 2. Size Position
            sizer = PositionSizer(funds['available_margin'])
            sizing = sizer.calculate_position_size(
                trade_plan, weekly_score.composite, weekly_score.confidence
            )
            
            # 3. Execute
            result = self.executor.execute_strategy(
                strategy_name=weekly_mandate.strategy_type,
                trade_plan=trade_plan,
                chain_df=weekly_chain,
                expiry=weekly,
                lot_size=lot_size
            )
            
            if result['status'] == "SUCCESS":
                print(f"{Fore.GREEN}✅ Trade Executed! Order IDs: {result['order_ids']}{Style.RESET_ALL}")
                # Update Risk Manager immediately
                self.risk_manager.sync_state()
            else:
                print(f"{Fore.RED}❌ Execution Failed: {result.get('error', 'Unknown error')}{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        # Prompt for token
        token = input("🔑 Enter Upstox Access Token: ").strip()
        if not token:
            print(f"{Fore.RED}❌ Token required. Exiting.{Style.RESET_ALL}")
            sys.exit(1)
            
        system = VolGuardSystem(token)
        system.run_daily_cycle()
        
        # Keep process alive for WebSocket Monitoring
        print(f"\n{Fore.CYAN}🛡️ System is Live & Monitoring. Press Ctrl+C to stop.{Style.RESET_ALL}")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 System Shutdown.")
    except Exception as e:
        print(f"\n{Fore.RED}❌ System Error: {e}{Style.RESET_ALL}")
