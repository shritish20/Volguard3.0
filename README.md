# 🚀 VOLGUARD OPTIONS COCKPIT

Professional-grade options trading system for NSE Nifty 50 using Upstox API.

## 📋 FEATURES

### ✅ **CORRECT API USAGE**
- Verified Upstox SDK patterns (2.19.0)
- All endpoints tested 100% working
- Proper error handling and retry logic

### 🧠 **VOLGUARD Analytics Engine**
- Moneyness-weighted GEX calculation
- ATM PCR (±5% strikes only)
- 25Δ skew regime classification
- VIX momentum tracking
- FII conviction scoring
- Regime persistence analysis

### 💼 **Portfolio Management**
- Real-time position tracking
- Greeks aggregation
- P&L calculation with attribution
- Margin utilization monitoring
- VaR (95%) calculation

### ⚡ **Execution Engine**
- **SHADOW MODE**: Paper trading with realistic fills
- **LIVE MODE**: Real order execution
- Multi-leg strategy support
- Order validation and risk checks

### 📡 **Real-Time Monitoring**
- WebSocket streaming for live prices
- Portfolio update notifications
- Greeks monitoring dashboard

## 🚀 QUICK START

### 1. Installation
```bash
# Clone repository
git clone https://github.com/yourusername/volguard-cockpit.git
cd volguard-cockpit

# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp credentials.json.example credentials.json
cp .env.example .env

# Edit credentials.json with your Upstox API tokens
