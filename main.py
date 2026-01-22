"""
VOLGUARD v31.0 MAIN COMMANDER
Integrates Logic (Regime/Trading) with Infrastructure (Upstox Engines)
AND Persistent Database Journaling
"""

import time
import sys
from datetime import datetime, date, timedelta
from colorama import Fore, Style, init

# 1. Import The Brain (Your Logic)
from logic.regime_logic import Config, AnalyticsEngine, RegimeEngine, DisplayEngine, ParticipantDataFetcher
from logic.trading_logic import StrategySelector, PositionSizer, PositionMonitor

# 2. Import The Infrastructure (Upstox Integration + Database)
from engine.data_engine import UpstoxDataEngine
from engine.execution_engine import UpstoxExecutionEngine
from engine.risk_manager import UpstoxRiskManager
from engine.journal_engine import TradeJournal, TradeRecord

init(autoreset=True)

class VolGuardSystem:
    def __init__(self, token):
        print(f"\n{Fore.GREEN}{'='*60}")
        print(f" || {'SYSTEM STARTUP: VOLGUARD V31.0':^52} ||")
        print(f" || {'UPSTOX V3 INTEGRATION PRODUCTION READY':^52} ||")
        print(f"{'='*60}{Style.RESET_ALL}")

        # A. Initialize Infrastructure
        print(f"{Fore.CYAN} [1/4] Initializing Upstox Engines...{Style.RESET_ALL}")
        self.data_engine = UpstoxDataEngine(token)
        self.executor = UpstoxExecutionEngine(token)
        
        # B. Initialize Logic
        print(f"{Fore.CYAN} [2/4] Initializing Logic Cores...{Style.RESET_ALL}")
        self.analytics = AnalyticsEngine()
        self.regime = RegimeEngine()
        self.display = DisplayEngine()
        self.position_monitor = PositionMonitor()

        # C. Initialize Risk Shield
        print(f"{Fore.CYAN} [3/4] Initializing Risk Shield...{Style.RESET_ALL}")
        self.risk_manager = UpstoxRiskManager(token, self.data_engine, self.position_monitor)

        # D. Initialize Database Journal (NEW)
        print(f"{Fore.CYAN} [4/4] Connecting to Trade Journal...{Style.RESET_ALL}")
        self.journal = TradeJournal("volguard.db")

    def run_daily_cycle(self):
        """Runs the main analysis and execution cycle"""

        # --- STEP 1: HEALTH & RECOVERY ---
        print(f"\n{Fore.YELLOW}>>> STEP 1: SYSTEM HEALTH CHECK{Style.RESET_ALL}")
        
        # Fetch Funds
        funds = self.data_engine.fetch_funds_and_margin()
        print(f" Available Capital: ₹{funds['available_margin']:,.2f}")

        # Fetch Holidays
        holidays = self.data_engine.fetch_holidays()
        print(f" Market Holidays Loaded: {len(holidays)}")

        # Sync Portfolio (Crash Recovery)
        self.risk_manager.sync_state()

        # --- STEP 2: DATA ACQUISITION ---
        print(f"\n{Fore.YELLOW}>>> STEP 2: DATA ACQUISITION{Style.RESET_ALL}")

        # Fetch History
        nifty_hist = self.data_engine.fetch_history(Config.NIFTY_KEY)
        vix_hist = self.data_engine.fetch_history(Config.VIX_KEY)
        live_data = self.data_engine.fetch_live_quote([Config.NIFTY_KEY, Config.VIX_KEY])

        # Fetch Expiries (Dynamic)
        weekly, monthly, next_weekly, lot_size = self.data_engine.fetch_expiry_details()
        if not weekly:
            print(f"{Fore.RED} Critical Error: Could not fetch Expiries.{Style.RESET_ALL}")
            return

        # Fetch Option Chains
        print(" Fetching Option Chains...")
        weekly_chain = self.data_engine.fetch_option_chain(weekly)
        monthly_chain = self.data_engine.fetch_option_chain(monthly)

        # --- STEP 3: REGIME ANALYSIS ---
        print(f"\n{Fore.YELLOW}>>> STEP 3: REGIME ANALYSIS{Style.RESET_ALL}")

        # Fetch Participant Data
        participant_data, _, fii_net_change, data_date, is_fallback = ParticipantDataFetcher.fetch_smart_participant_data()

        # Calculate Metrics
        time_metrics = self.analytics.get_time_metrics(weekly, monthly, next_weekly)
        vol_metrics = self.analytics.get_vol_metrics(nifty_hist, vix_hist, live_data.get(Config.NIFTY_KEY, 0), live_data.get(Config.VIX_KEY, 0))
        struct_weekly = self.analytics.get_struct_metrics(weekly_chain, vol_metrics.spot, lot_size)
        struct_monthly = self.analytics.get_struct_metrics(monthly_chain, vol_metrics.spot, lot_size)
        edge_metrics = self.analytics.get_edge_metrics(weekly_chain, monthly_chain, vol_metrics.spot, vol_metrics)
        external_metrics = self.analytics.get_external_metrics(nifty_hist, participant_data, None, fii_net_change, data_date, is_fallback)

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
        print(f"{' WEEKLY EXPIRY ANALYSIS ':^90}")
        print(f"{'='*90}{Style.RESET_ALL}")
        self.display.render_regime_scores(weekly_score)
        self.display.render_mandate(weekly_mandate)

        print(f"\n{Fore.CYAN}{'='*90}")
        print(f"{' MONTHLY EXPIRY ANALYSIS ':^90}")
        print(f"{'='*90}{Style.RESET_ALL}")
        self.display.render_regime_scores(monthly_score)
        self.display.render_mandate(monthly_mandate)

        self.display.render_summary(weekly_mandate, monthly_mandate)

        # --- STEP 5: EXECUTION (With Database Logging) ---
        if weekly_mandate.max_lots > 0 and funds['available_margin'] > Config.MARGIN_SELL_BASE:
            print(f"\n{Fore.GREEN}>>> EXECUTING STRATEGY: {weekly_mandate.strategy_type}{Style.RESET_ALL}")

            # 1. Select Strikes
            strategy_func = StrategySelector.select_strategy(
                weekly_mandate.regime_name, "WEEKLY", weekly_mandate.dte,
                vol_metrics, struct_weekly, edge_metrics, external_metrics
            )[1]
            
            if strategy_func is None:
                print(f"{Fore.RED} X No valid strategy function found.{Style.RESET_ALL}")
                return

            trade_plan = strategy_func(weekly_chain, vol_metrics.spot, weekly_mandate.dte)

            # 2. Size Position
            sizer = PositionSizer(funds['available_margin'])
            sizing = sizer.calculate_position_size(
                trade_plan, weekly_score.composite, weekly_score.confidence
            )
            
            # Apply sizing to plan
            trade_plan['lots'] = sizing['lots']

            # 3. Execute
            result = self.executor.execute_strategy(
                strategy_name=weekly_mandate.strategy_type,
                trade_plan=trade_plan,
                chain_df=weekly_chain,
                expiry=weekly,
                lot_size=lot_size
            )

            if result['status'] == "SUCCESS":
                print(f"{Fore.GREEN} Trade Executed! Order IDs: {result['order_ids']} {Style.RESET_ALL}")
                
                # --- NEW: LOG TO DATABASE ---
                try:
                    trade_id = f"TRD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    record = TradeRecord(
                        trade_id=trade_id,
                        entry_date=datetime.now().strftime('%Y-%m-%d'),
                        entry_time=datetime.now().strftime('%H:%M:%S'),
                        expiry_date=str(weekly),
                        structure=weekly_mandate.strategy_type,
                        regime_name=weekly_mandate.regime_name,
                        entry_vrp=edge_metrics.weighted_vrp_weekly,
                        entry_vov=vol_metrics.vov_zscore,
                        entry_gex=struct_weekly.gex_regime,
                        regime_score=weekly_score.composite,
                        strikes=trade_plan,
                        lots=sizing['lots'],
                        entry_premium=result.get('total_premium', 0.0),
                        entry_spot=vol_metrics.spot,
                        entry_vix=vol_metrics.vix
                    )
                    
                    self.journal.log_entry(record)
                    
                except Exception as e:
                    print(f"{Fore.RED} X Database Log Error: {e}{Style.RESET_ALL}")
                
                self.risk_manager.sync_state()

            else:
                print(f"{Fore.RED} Execution Failed: {result.get('error', 'Unknown error')}{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW} No trade execution: Margin or Mandate constraints not met.{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        token = input(" Enter Upstox Access Token: ").strip()
        if not token:
            print(f"{Fore.RED} X Token required. Exiting.{Style.RESET_ALL}")
            sys.exit(1)

        system = VolGuardSystem(token)
        system.run_daily_cycle()

        print(f"\n{Fore.CYAN} System is Live & Monitoring. Press Ctrl+C to stop.{Style.RESET_ALL}")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n System Shutdown.")
    except Exception as e:
        print(f"\n{Fore.RED} X System Error: {e}{Style.RESET_ALL}")
