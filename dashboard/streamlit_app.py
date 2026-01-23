"""
Streamlit Dashboard for VOLGUARD Options Cockpit
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import TradingMode, DASHBOARD_CONFIG
from utils.helpers import format_currency, format_percentage


# Page configuration
st.set_page_config(
    page_title="VOLGUARD Options Cockpit",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3498db;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #2c3e50;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin-bottom: 1rem;
    }
    .warning-card {
        background-color: #f39c12;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .success-card {
        background-color: #2ecc71;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


class Dashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.cockpit = None
        
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
        if 'analysis_result' not in st.session_state:
            st.session_state.analysis_result = None
        if 'portfolio_summary' not in st.session_state:
            st.session_state.portfolio_summary = None
        if 'mode' not in st.session_state:
            st.session_state.mode = TradingMode.SHADOW
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.title("⚙️ Configuration")
            
            # Mode selection
            mode = st.radio(
                "Trading Mode",
                ["SHADOW", "LIVE"],
                index=0 if st.session_state.mode == TradingMode.SHADOW else 1,
                help="SHADOW mode for paper trading, LIVE mode for real execution"
            )
            st.session_state.mode = TradingMode.SHADOW if mode == "SHADOW" else TradingMode.LIVE
            
            # Access token
            st.subheader("🔐 Authentication")
            token = st.text_input("Upstox Access Token", type="password")
            
            # Initialize button
            if st.button("🚀 Initialize System", type="primary", use_container_width=True):
                if token:
                    with st.spinner("Initializing..."):
                        # Initialize cockpit
                        from main import OptionsCockpit
                        self.cockpit = OptionsCockpit(st.session_state.mode)
                        if self.cockpit.initialize(token):
                            st.session_state.initialized = True
                            st.success("System initialized!")
                        else:
                            st.error("Initialization failed")
                else:
                    st.warning("Please enter access token")
            
            st.divider()
            
            # Actions
            st.subheader("🚀 Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Run Analysis", use_container_width=True):
                    if st.session_state.initialized and self.cockpit:
                        with st.spinner("Running analysis..."):
                            result = self.cockpit.run_analysis()
                            if result:
                                st.session_state.analysis_result = result
                                st.success("Analysis complete!")
                            else:
                                st.error("Analysis failed")
                    else:
                        st.warning("System not initialized")
            
            with col2:
                if st.button("💼 Refresh Portfolio", use_container_width=True):
                    if st.session_state.initialized and self.cockpit:
                        with st.spinner("Refreshing portfolio..."):
                            summary = self.cockpit.refresh_portfolio()
                            if summary:
                                st.session_state.portfolio_summary = summary
                                st.success("Portfolio refreshed!")
                            else:
                                st.error("Portfolio refresh failed")
                    else:
                        st.warning("System not initialized")
            
            # WebSocket controls
            st.divider()
            st.subheader("📡 Live Data")
            
            if st.button("▶️ Start WebSocket", use_container_width=True):
                if st.session_state.initialized and self.cockpit:
                    self.cockpit.start_websocket()
                    st.success("WebSocket started")
                else:
                    st.warning("System not initialized")
            
            if st.button("⏹️ Stop WebSocket", use_container_width=True):
                if st.session_state.initialized and self.cockpit:
                    self.cockpit.stop_websocket()
                    st.info("WebSocket stopped")
            
            st.divider()
            
            # Status indicators
            st.subheader("📊 System Status")
            
            status_color = "🟢" if st.session_state.initialized else "🔴"
            st.write(f"{status_color} System: {'Ready' if st.session_state.initialized else 'Not Initialized'}")
            
            if st.session_state.analysis_result:
                st.write("🟡 Analysis: Available")
            else:
                st.write("⚪ Analysis: Not Run")
            
            if st.session_state.portfolio_summary:
                st.write("🟡 Portfolio: Available")
            else:
                st.write("⚪ Portfolio: Not Loaded")
    
    def render_header(self):
        """Render main header"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.image("https://img.icons8.com/color/96/000000/stock-share.png", width=80)
        
        with col2:
            st.markdown('<h1 class="main-header">VOLGUARD OPTIONS COCKPIT</h1>', unsafe_allow_html=True)
            mode_color = "orange" if st.session_state.mode == TradingMode.SHADOW else "green"
            st.markdown(f'<p style="text-align: center; color: {mode_color}; font-size: 1.2rem;">'
                       f'{st.session_state.mode.value} MODE</p>', unsafe_allow_html=True)
        
        with col3:
            st.write("")
            current_time = datetime.now().strftime("%H:%M:%S")
            st.metric("Current Time", current_time)
    
    def render_analysis_tab(self):
        """Render analysis tab"""
        st.header("📈 VOLGUARD Analysis")
        
        if not st.session_state.analysis_result:
            st.info("No analysis data. Click 'Run Analysis' to analyze market.")
            return
        
        result = st.session_state.analysis_result
        
        # Overview metrics
        st.subheader("📊 Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            spot = result.get('vol_metrics').spot if result.get('vol_metrics') else 0
            st.metric("Nifty Spot", f"{spot:,.0f}")
        
        with col2:
            vix = result.get('vol_metrics').vix if result.get('vol_metrics') else 0
            st.metric("India VIX", f"{vix:.2f}")
        
        with col3:
            vol_regime = result.get('vol_metrics').vol_regime if result.get('vol_metrics') else "UNKNOWN"
            st.metric("Vol Regime", vol_regime)
        
        with col4:
            if result.get('weekly_mandate'):
                allocation = result['weekly_mandate'].allocation_pct
                st.metric("Weekly Allocation", f"{allocation:.0f}%")
        
        st.divider()
        
        # Weekly and Monthly mandates
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_mandate_card("📅 Weekly Expiry", result.get('weekly_mandate'))
        
        with col2:
            self._render_mandate_card("📅 Monthly Expiry", result.get('monthly_mandate'))
        
        st.divider()
        
        # Volatility metrics
        st.subheader("📊 Volatility Analysis")
        
        if result.get('vol_metrics'):
            vol = result['vol_metrics']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("RV 7D", f"{vol.rv7:.1f}%")
                st.metric("RV 28D", f"{vol.rv28:.1f}%")
            
            with col2:
                st.metric("IVP 1Y", f"{vol.ivp_1yr:.1f}%")
                st.metric("Vol-of-Vol", f"{vol.vov:.1f}%")
            
            with col3:
                st.metric("VIX Change 5D", format_percentage(vol.vix_change_5d))
                st.metric("VIX Momentum", vol.vix_momentum)
        
        # Edge metrics
        st.divider()
        st.subheader("🎯 Option Edges")
        
        if result.get('edge_metrics'):
            edge = result['edge_metrics']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Weighted VRP Weekly", format_percentage(edge.weighted_vrp_weekly))
                st.metric("IV Weekly", f"{edge.iv_weekly:.1f}%")
            
            with col2:
                st.metric("Weighted VRP Monthly", format_percentage(edge.weighted_vrp_monthly))
                st.metric("IV Monthly", f"{edge.iv_monthly:.1f}%")
            
            with col3:
                st.metric("Term Spread", format_percentage(edge.term_spread))
                st.metric("Term Regime", edge.term_regime)
    
    def _render_mandate_card(self, title: str, mandate):
        """Render mandate card"""
        if not mandate:
            st.warning(f"No {title} mandate")
            return
        
        with st.container():
            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
            st.subheader(title)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Regime:** {mandate.regime_name}")
                st.write(f"**Strategy:** {mandate.strategy_type}")
                st.write(f"**Structure:** {mandate.suggested_structure}")
            
            with col2:
                st.write(f"**Allocation:** {mandate.allocation_pct:.0f}%")
                st.write(f"**Max Lots:** {mandate.max_lots}")
                st.write(f"**Score:** {mandate.score.composite:.1f}/10")
            
            # Confidence indicator
            confidence_color = {
                "VERY_HIGH": "green",
                "HIGH": "lightgreen",
                "MODERATE": "yellow",
                "LOW": "red"
            }.get(mandate.score.confidence, "gray")
            
            st.write(f"**Confidence:** :{confidence_color}[{mandate.score.confidence}]")
            
            # Warnings
            if mandate.warnings:
                with st.expander("⚠️ Warnings"):
                    for warning in mandate.warnings:
                        st.warning(warning)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_portfolio_tab(self):
        """Render portfolio tab"""
        st.header("💼 Portfolio Management")
        
        if not st.session_state.portfolio_summary:
            st.info("No portfolio data. Click 'Refresh Portfolio' to load.")
            return
        
        summary = st.session_state.portfolio_summary
        
        # Overview metrics
        st.subheader("📊 Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Positions", summary.total_positions)
            st.metric("Current Value", format_currency(summary.total_current_value))
        
        with col2:
            pnl_color = "normal" if summary.total_pnl >= 0 else "inverse"
            st.metric(
                "Total P&L",
                format_currency(summary.total_pnl),
                delta=format_percentage(summary.total_pnl_percentage),
                delta_color=pnl_color
            )
            st.metric("Investment", format_currency(summary.total_investment))
        
        with col3:
            st.metric("Net Delta", f"{summary.net_delta:,.2f}")
            st.metric("Net Gamma", f"{summary.net_gamma:,.4f}")
        
        with col4:
            st.metric("Net Theta", f"{summary.net_theta:,.2f}")
            st.metric("Net Vega", f"{summary.net_vega:,.2f}")
        
        # Greeks visualization
        st.divider()
        st.subheader("📈 Greeks Exposure")
        
        greeks_data = pd.DataFrame({
            'Greek': ['Delta', 'Gamma', 'Theta', 'Vega'],
            'Value': [
                summary.net_delta,
                summary.net_gamma * 1000,
                summary.net_theta,
                summary.net_vega
            ]
        })
        
        fig = go.Figure()
        
        for idx, row in greeks_data.iterrows():
            color = 'green' if row['Value'] >= 0 else 'red'
            fig.add_trace(go.Bar(
                x=[row['Greek']],
                y=[row['Value']],
                name=row['Greek'],
                marker_color=color,
                text=f"{row['Value']:.2f}",
                textposition='outside'
            ))
        
        fig.update_layout(
            title="Net Greeks Exposure",
            height=400,
            showlegend=False,
            yaxis_title="Exposure"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Margin & Risk
        st.divider()
        st.subheader("💰 Margin & Risk")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            margin_util = (summary.total_margin_used / 
                          (summary.total_margin_used + summary.available_margin) * 100
                          if summary.total_margin_used > 0 else 0)
            st.metric("Margin Used", format_currency(summary.total_margin_used))
            st.metric("Margin Utilization", f"{margin_util:.1f}%")
        
        with col2:
            st.metric("Available Margin", format_currency(summary.available_margin))
            st.metric("VaR (95%)", format_currency(summary.var_95))
        
        with col3:
            st.metric("Max Risk", format_currency(summary.max_risk))
            st.metric("Concentration", f"{summary.concentration_ratio:.1%}")
    
    def render_execution_tab(self):
        """Render execution tab"""
        st.header("⚡ Strategy Execution")
        
        if not st.session_state.initialized:
            st.warning("System not initialized")
            return
        
        # Strategy configuration
        st.subheader("📝 Configure Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            expiry_type = st.selectbox("Expiry Type", ["WEEKLY", "MONTHLY"])
            strategy_type = st.selectbox(
                "Strategy Type",
                ["IRON_CONDOR", "IRON_FLY", "STRANGLE", "CREDIT_SPREAD"]
            )
        
        with col2:
            if st.session_state.analysis_result:
                mandate = (st.session_state.analysis_result.get('weekly_mandate') 
                          if expiry_type == "WEEKLY" 
                          else st.session_state.analysis_result.get('monthly_mandate'))
                
                if mandate:
                    st.info(f"**Suggested:** {mandate.suggested_structure}")
                    st.info(f"**Max Lots:** {mandate.max_lots}")
                    st.info(f"**Allocation:** {mandate.allocation_pct:.0f}%")
        
        # Strategy parameters
        st.subheader("🎯 Strategy Parameters")
        
        if strategy_type == "IRON_CONDOR":
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                put_long = st.number_input("Put Long Strike", value=22500, step=50)
            with col2:
                put_short = st.number_input("Put Short Strike", value=22700, step=50)
            with col3:
                call_short = st.number_input("Call Short Strike", value=23300, step=50)
            with col4:
                call_long = st.number_input("Call Long Strike", value=23500, step=50)
            
            lots = st.slider("Number of Lots", 1, 10, 1)
            
            # Calculate metrics
            max_profit = (call_short - put_short - (call_long - call_short) - (put_short - put_long)) * 75 * lots
            max_loss = ((call_long - call_short) + (put_short - put_long)) * 75 * lots - max_profit
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Profit", format_currency(max_profit))
            with col2:
                st.metric("Max Loss", format_currency(max_loss))
            with col3:
                st.metric("Risk:Reward", f"1:{abs(max_profit/max_loss):.2f}" if max_loss > 0 else "N/A")
        
        # Execution controls
        st.divider()
        st.subheader("🚀 Execute")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Preview Order", type="secondary", use_container_width=True):
                st.info("Order preview - coming soon")
        
        with col2:
            if st.button("✅ Execute Strategy", type="primary", use_container_width=True):
                if st.session_state.mode == TradingMode.LIVE:
                    confirm = st.checkbox("⚠️ Confirm LIVE execution")
                    if confirm:
                        st.success("Executing strategy...")
                        # Call execution logic
                    else:
                        st.warning("Please confirm to execute in LIVE mode")
                else:
                    st.success("Executing in SHADOW mode...")
        
        with col3:
            if st.button("❌ Cancel", use_container_width=True):
                st.info("Cancelled")
    
    def render_monitoring_tab(self):
        """Render monitoring tab"""
        st.header("📡 Live Monitoring")
        
        # Placeholder for live data
        placeholder = st.empty()
        
        with placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Nifty Spot", "23,050.75", "+125.50 (+0.55%)")
            
            with col2:
                st.metric("India VIX", "14.25", "-0.35 (-2.40%)")
            
            with col3:
                st.metric("PCR", "1.15", "+0.05")
            
            with col4:
                st.metric("Max Pain", "23,000", "")
        
        # Greeks monitoring
        if st.session_state.portfolio_summary:
            st.divider()
            st.subheader("📈 Portfolio Greeks (Live)")
            
            # Create sample time series
            import numpy as np
            times = pd.date_range(start=datetime.now() - timedelta(minutes=30),
                                 end=datetime.now(), freq='1min')
            
            greeks_series = pd.DataFrame({
                'Time': times,
                'Delta': np.random.randn(len(times)).cumsum() + 100,
                'Theta': np.random.randn(len(times)).cumsum() - 50
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=greeks_series['Time'], y=greeks_series['Delta'],
                                   mode='lines', name='Delta', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=greeks_series['Time'], y=greeks_series['Theta'],
                                   mode='lines', name='Theta', line=dict(color='red'),
                                   yaxis='y2'))
            
            fig.update_layout(
                title='Portfolio Greeks - Last 30 Minutes',
                xaxis_title='Time',
                yaxis_title='Delta',
                yaxis2=dict(title='Theta', overlaying='y', side='right'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_settings_tab(self):
        """Render settings tab"""
        st.header("⚙️ Settings")
        
        # Risk parameters
        st.subheader("🛡️ Risk Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Base Capital (₹)", value=10_00_000, step=100000)
            st.number_input("Max Allocation (%)", value=60, min_value=0, max_value=100)
            st.number_input("Stop Loss (%)", value=20, min_value=0, max_value=100)
        
        with col2:
            st.number_input("Max Lots Per Trade", value=10, min_value=1, max_value=50)
            st.number_input("Max Positions", value=5, min_value=1, max_value=20)
            st.number_input("Profit Target (%)", value=50, min_value=0, max_value=200)
        
        # Save button
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            st.success("Settings saved!")
    
    def run(self):
        """Main dashboard runner"""
        self.render_sidebar()
        self.render_header()
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Analysis",
            "💼 Portfolio",
            "⚡ Execution",
            "📡 Live Monitor",
            "⚙️ Settings"
        ])
        
        with tab1:
            self.render_analysis_tab()
        
        with tab2:
            self.render_portfolio_tab()
        
        with tab3:
            self.render_execution_tab()
        
        with tab4:
            self.render_monitoring_tab()
        
        with tab5:
            self.render_settings_tab()


# Run the dashboard
if __name__ == "__main__":
    # Parse command line arguments
    import sys
    mode = TradingMode.SHADOW
    if len(sys.argv) > 1 and '--mode' in sys.argv:
        idx = sys.argv.index('--mode')
        if idx + 1 < len(sys.argv):
            mode_str = sys.argv[idx + 1]
            mode = TradingMode.LIVE if mode_str.lower() == 'live' else TradingMode.SHADOW
    
    st.session_state.mode = mode
    dashboard = Dashboard()
    dashboard.run()
