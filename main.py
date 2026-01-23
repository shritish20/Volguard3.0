"""
VOLGUARD OPTIONS COCKPIT - Main Entry Point
Production-grade options trading system
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import TradingMode, PATH_CONFIG
from auth.manager import AuthManager
from data.fetcher import UpstoxDataFetcher
from analytics.engine import AnalyticsEngine
from analytics.regime_engine import RegimeEngine
from data.participant_fetcher import ParticipantDataFetcher
from portfolio.manager import PortfolioManager
from execution.engine import ExecutionEngine
from data.websocket_manager import WebSocketManager
from utils.logger import setup_logger
from utils.helpers import load_json_file, save_json_file


class OptionsCockpit:
    """Main orchestrator for VOLGUARD Options Cockpit"""
    
    def __init__(self, mode: TradingMode = TradingMode.SHADOW):
        self.mode = mode
        self.logger = setup_logger(f"cockpit_{mode.value.lower()}")
        self.state_file = PATH_CONFIG.STATE_DIR / f"cockpit_state_{mode.value.lower()}.json"
        
        # Core components
        self.auth_manager = None
        self.data_fetcher = None
        self.analytics_engine = None
        self.regime_engine = None
        self.participant_fetcher = ParticipantDataFetcher()
        self.portfolio_manager = None
        self.execution_engine = None
        self.websocket_manager = None
        
        # State
        self.is_running = False
        self.last_analysis_time = None
        self.current_analysis = None
        
    def initialize(self, access_token: str = None) -> bool:
        """Initialize all system components"""
        self.logger.info("="*80)
        self.logger.info(f"INITIALIZING VOLGUARD OPTIONS COCKPIT - {self.mode.value} MODE")
        self.logger.info("="*80)
        
        try:
            # Create directories
            PATH_CONFIG.create_directories()
            
            # 1. Authentication
            self.logger.info("1/7 Initializing authentication...")
            if not access_token:
                credentials = load_json_file(PATH_CONFIG.CREDENTIALS_FILE)
                access_token = credentials.get('access_token')
            
            if not access_token:
                raise ValueError("Access token required")
            
            self.auth_manager = AuthManager(access_token)
            
            if not self.auth_manager.validate_token():
                raise ValueError("Invalid or expired access token")
            
            self.logger.info("   ✓ Authentication successful")
            
            # 2. Data Fetcher
            self.logger.info("2/7 Initializing data fetcher...")
            self.data_fetcher = UpstoxDataFetcher(self.auth_manager)
            self.logger.info("   ✓ Data fetcher ready")
            
            # 3. Analytics Engine
            self.logger.info("3/7 Initializing analytics engine...")
            self.analytics_engine = AnalyticsEngine()
            self.regime_engine = RegimeEngine()
            self.logger.info("   ✓ Analytics engine ready")
            
            # 4. Portfolio Manager
            self.logger.info("4/7 Initializing portfolio manager...")
            self.portfolio_manager = PortfolioManager(self.data_fetcher)
            self.logger.info("   ✓ Portfolio manager ready")
            
            # 5. Execution Engine
            self.logger.info("5/7 Initializing execution engine...")
            self.execution_engine = ExecutionEngine(
                self.auth_manager,
                self.data_fetcher,
                self.mode
            )
            self.logger.info(f"   ✓ Execution engine ready ({self.mode.value} mode)")
            
            # 6. WebSocket Manager
            self.logger.info("6/7 Initializing WebSocket manager...")
            self.websocket_manager = WebSocketManager(self.auth_manager)
            self.logger.info("   ✓ WebSocket manager ready")
            
            # 7. Load previous state
            self.logger.info("7/7 Loading previous state...")
            self._load_state()
            
            self.is_running = True
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"✓ VOLGUARD OPTIONS COCKPIT INITIALIZED SUCCESSFULLY")
            self.logger.info(f"{'='*80}\n")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def run_analysis(self) -> dict:
        """Run complete VOLGUARD analysis"""
        self.logger.info("\n" + "="*80)
        self.logger.info("RUNNING VOLGUARD ANALYSIS")
        self.logger.info("="*80)
        
        try:
            # 1. Fetch participant data
            self.logger.info("1/7 Fetching FII/DII data...")
            participant_data, secondary_data, fii_net_change, data_date, is_fallback = \
                self.participant_fetcher.fetch_smart_participant_data()
            
            # 2. Fetch market data
            self.logger.info("2/7 Fetching market data...")
            spot = self.data_fetcher.get_spot_price()
            vix = self.data_fetcher.get_vix_price()
            
            nifty_hist = self.data_fetcher.get_historical_data("NSE_INDEX|Nifty 50")
            vix_hist = self.data_fetcher.get_historical_data("NSE_INDEX|India VIX")
            
            if nifty_hist.empty or vix_hist.empty:
                raise ValueError("Failed to fetch historical data")
            
            # 3. Get expiries
            self.logger.info("3/7 Fetching expiry dates...")
            expiry_dates = self.data_fetcher.get_expiry_dates()
            if not expiry_dates:
                raise ValueError("No expiry dates available")
            
            weekly, monthly, next_weekly = expiry_dates[0], expiry_dates[-1], expiry_dates[1] if len(expiry_dates) > 1 else expiry_dates[0]
            
            # Adjust monthly expiry if same as weekly
            if weekly == monthly:
                # Find next monthly expiry
                monthly_candidates = [d for d in expiry_dates if d.month != weekly.month]
                if monthly_candidates:
                    monthly = monthly_candidates[0]
            
            # Get lot size
            lot_size = 75  # Default, should fetch from API
            
            # 4. Get option chains
            self.logger.info("4/7 Fetching option chains...")
            weekly_chain = self.data_fetcher.get_option_chain(weekly)
            monthly_chain = self.data_fetcher.get_option_chain(monthly)
            
            # 5. Run analytics
            self.logger.info("5/7 Running analytics...")
            time_metrics = self.analytics_engine.get_time_metrics(weekly, monthly, next_weekly)
            vol_metrics = self.analytics_engine.calculate_vol_metrics(nifty_hist, vix_hist, spot, vix)
            struct_weekly = self.analytics_engine.calculate_struct_metrics(weekly_chain, vol_metrics.spot, lot_size)
            struct_monthly = self.analytics_engine.calculate_struct_metrics(monthly_chain, vol_metrics.spot, lot_size)
            edge_metrics = self.analytics_engine.calculate_edge_metrics(weekly_chain, monthly_chain, vol_metrics.spot, vol_metrics)
            
            # 6. Calculate external metrics
            self.logger.info("6/7 Calculating external metrics...")
            external_metrics = self.analytics_engine.get_external_metrics(
                nifty_hist, participant_data, secondary_data,
                fii_net_change, data_date, is_fallback
            )
            
            # 7. Generate mandates
            self.logger.info("7/7 Generating trading mandates...")
            weekly_score = self.regime_engine.calculate_scores(
                vol_metrics, struct_weekly, edge_metrics,
                external_metrics, time_metrics, "WEEKLY"
            )
            monthly_score = self.regime_engine.calculate_scores(
                vol_metrics, struct_monthly, edge_metrics,
                external_metrics, time_metrics, "MONTHLY"
            )
            
            weekly_mandate = self.regime_engine.generate_mandate(
                weekly_score, vol_metrics, struct_weekly,
                edge_metrics, external_metrics, time_metrics,
                "WEEKLY", weekly, time_metrics.dte_weekly
            )
            
            monthly_mandate = self.regime_engine.generate_mandate(
                monthly_score, vol_metrics, struct_monthly,
                edge_metrics, external_metrics, time_metrics,
                "MONTHLY", monthly, time_metrics.dte_monthly
            )
            
            # Compile results
            self.current_analysis = {
                'timestamp': datetime.now(),
                'time_metrics': time_metrics,
                'vol_metrics': vol_metrics,
                'struct_weekly': struct_weekly,
                'struct_monthly': struct_monthly,
                'edge_metrics': edge_metrics,
                'external_metrics': external_metrics,
                'weekly_mandate': weekly_mandate,
                'monthly_mandate': monthly_mandate,
                'weekly_score': weekly_score,
                'monthly_score': monthly_score
            }
            
            self.last_analysis_time = datetime.now()
            self._save_state()
            
            self.logger.info("✓ Analysis complete")
            return self.current_analysis
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return None
    
    def refresh_portfolio(self) -> dict:
        """Refresh portfolio positions"""
        self.logger.info("Refreshing portfolio...")
        
        try:
            success = self.portfolio_manager.refresh_positions()
            
            if success:
                summary = self.portfolio_manager.get_portfolio_summary()
                self.logger.info(f"✓ Portfolio refreshed: {summary.total_positions} positions")
                return summary
            else:
                self.logger.warning("Portfolio refresh failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Portfolio refresh error: {e}")
            return None
    
    def execute_strategy(self, strategy_config: dict) -> dict:
        """Execute a trading strategy"""
        self.logger.info(f"Executing strategy in {self.mode.value} mode...")
        
        if not self.current_analysis:
            self.logger.error("No analysis available. Run analysis first.")
            return None
        
        try:
            result = self.execution_engine.place_multi_leg_strategy(
                strategy_type=strategy_config.get('strategy_type'),
                legs=strategy_config.get('legs', []),
                tag=strategy_config.get('tag', '')
            )
            
            if result.get('success'):
                self.logger.info(f"✓ Strategy executed: {result.get('successful_orders')} orders placed")
                # Refresh portfolio after execution
                self.refresh_portfolio()
            else:
                self.logger.error(f"Strategy execution failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Strategy execution error: {e}")
            return None
    
    def start_websocket(self, instruments: list = None):
        """Start WebSocket streaming"""
        self.logger.info("Starting WebSocket...")
        
        try:
            if not instruments:
                instruments = ["NSE_INDEX|Nifty 50", "NSE_INDEX|India VIX"]
            
            self.websocket_manager.subscribe(instruments)
            self.websocket_manager.start()
            
            self.logger.info(f"✓ WebSocket started for {len(instruments)} instruments")
            
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
    
    def stop_websocket(self):
        """Stop WebSocket streaming"""
        if self.websocket_manager:
            self.websocket_manager.stop()
            self.logger.info("WebSocket stopped")
    
    def generate_report(self, filename: str = None) -> dict:
        """Generate analysis report"""
        self.logger.info("Generating report...")
        
        if not self.current_analysis:
            self.logger.warning("No analysis data for report")
            return None
        
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'mode': self.mode.value,
                'analysis': self._serialize_analysis(self.current_analysis),
                'portfolio': self._serialize_portfolio()
            }
            
            if filename:
                report_file = PATH_CONFIG.REPORTS_DIR / filename
                save_json_file(report, report_file)
                self.logger.info(f"✓ Report saved to {report_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return None
    
    def _serialize_analysis(self, analysis: dict) -> dict:
        """Serialize analysis for JSON output"""
        # Simplified serialization - implement full serialization as needed
        if not analysis:
            return {}
        
        return {
            'timestamp': analysis.get('timestamp').isoformat() if analysis.get('timestamp') else None,
            'spot_price': getattr(analysis.get('vol_metrics'), 'spot', 0),
            'vix': getattr(analysis.get('vol_metrics'), 'vix', 0),
            'weekly_regime': getattr(analysis.get('weekly_mandate'), 'regime_name', ''),
            'monthly_regime': getattr(analysis.get('monthly_mandate'), 'regime_name', '')
        }
    
    def _serialize_portfolio(self) -> dict:
        """Serialize portfolio for JSON output"""
        if not self.portfolio_manager:
            return {}
        
        summary = self.portfolio_manager.get_portfolio_summary()
        if not summary:
            return {}
        
        return {
            'total_positions': summary.total_positions,
            'total_pnl': summary.total_pnl,
            'total_pnl_percentage': summary.total_pnl_percentage,
            'net_delta': summary.net_delta,
            'net_theta': summary.net_theta
        }
    
    def _save_state(self):
        """Save current state to disk"""
        try:
            state = {
                'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
                'mode': self.mode.value
            }
            save_json_file(state, self.state_file)
        except Exception as e:
            self.logger.error(f"State save error: {e}")
    
    def _load_state(self):
        """Load previous state from disk"""
        try:
            state = load_json_file(self.state_file)
            if state and 'last_analysis_time' in state and state['last_analysis_time']:
                self.last_analysis_time = datetime.fromisoformat(state['last_analysis_time'])
                self.logger.info(f"   ✓ Previous state loaded")
        except Exception as e:
            self.logger.warning(f"Could not load previous state: {e}")
    
    def shutdown(self):
        """Clean shutdown"""
        self.logger.info("Shutting down...")
        
        try:
            self.stop_websocket()
            self._save_state()
            self.generate_report(f"shutdown_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            self.is_running = False
            self.logger.info("✓ Shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='VOLGUARD Options Cockpit')
    
    parser.add_argument('--mode', choices=['live', 'shadow'], default='shadow',
                      help='Trading mode (default: shadow)')
    parser.add_argument('--token', type=str, help='Upstox access token')
    parser.add_argument('--analyze', action='store_true', help='Run analysis')
    parser.add_argument('--portfolio', action='store_true', help='Refresh portfolio')
    parser.add_argument('--report', type=str, help='Generate report file')
    parser.add_argument('--dashboard', action='store_true', help='Launch dashboard')
    
    args = parser.parse_args()
    
    # Determine mode
    mode = TradingMode.LIVE if args.mode == 'live' else TradingMode.SHADOW
    
    # Launch dashboard if requested
    if args.dashboard:
        print("Launching dashboard...")
        import subprocess
        dashboard_file = Path(__file__).parent / "dashboard" / "streamlit_app.py"
        if dashboard_file.exists():
            subprocess.run(["streamlit", "run", str(dashboard_file), "--", "--mode", args.mode])
        else:
            print("Dashboard not found")
        return
    
    # Initialize cockpit
    cockpit = OptionsCockpit(mode)
    
    if not cockpit.initialize(access_token=args.token):
        print("Initialization failed")
        return
    
    try:
        # Run analysis if requested
        if args.analyze:
            result = cockpit.run_analysis()
            if result:
                print("✓ Analysis complete")
        
        # Refresh portfolio if requested
        if args.portfolio:
            summary = cockpit.refresh_portfolio()
            if summary:
                print(f"Portfolio: {summary.total_positions} positions, P&L: ₹{summary.total_pnl:,.0f}")
        
        # Generate report if requested
        if args.report:
            cockpit.generate_report(args.report)
        
        # If no specific action, show help
        if not any([args.analyze, args.portfolio, args.report, args.dashboard]):
            print("\nUsage:")
            print("  --analyze      Run VOLGUARD analysis")
            print("  --portfolio    Refresh portfolio")
            print("  --report FILE  Generate report")
            print("  --dashboard    Launch Streamlit dashboard")
            print("  --mode MODE    Set mode (live/shadow)")
            print("  --token TOKEN  Set Upstox access token")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cockpit.shutdown()


if __name__ == "__main__":
    main()
