"""
Helper functions for VOLGUARD
"""

import json
from pathlib import Path
from datetime import datetime, date
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np

from config.settings import PATH_CONFIG


def load_json_file(file_path: Path) -> Dict:
    """Load JSON file"""
    try:
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
    
    return {}


def save_json_file(data: Dict, file_path: Path):
    """Save data to JSON file"""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Failed to save {file_path}: {e}")


def format_currency(value: float) -> str:
    """Format currency value"""
    if abs(value) >= 1_00_00_000:  # 1 crore
        return f"₹{value/1_00_00_000:,.2f} Cr"
    elif abs(value) >= 1_00_000:  # 1 lakh
        return f"₹{value/1_00_000:,.2f} L"
    else:
        return f"₹{value:,.0f}"


def format_percentage(value: float) -> str:
    """Format percentage value"""
    return f"{value:+.2f}%"


def calculate_moneyness_weight(spot: float, strike: float, bandwidth: float = 0.02) -> float:
    """Calculate moneyness weight using Gaussian decay"""
    return np.exp(-((strike - spot) / spot) ** 2 / bandwidth)


def calculate_max_profit_loss(strategy_type: str, legs: List[Dict], lot_size: int) -> Dict:
    """Calculate max profit/loss for strategy"""
    # Simplified calculation - implement based on strategy type
    if strategy_type == "IRON_CONDOR":
        # Iron Condor: Sell OTM Put, Buy further OTM Put, Sell OTM Call, Buy further OTM Call
        # Max profit = Net premium received
        # Max loss = Width of spread - premium received
        pass
    
    return {
        'max_profit': 0,
        'max_loss': 0,
        'breakevens': [],
        'risk_reward_ratio': 0
    }


def validate_instrument_key(instrument_key: str) -> bool:
    """Validate Upstox instrument key format"""
    if not instrument_key or '|' not in instrument_key:
        return False
    
    parts = instrument_key.split('|')
    if len(parts) != 2:
        return False
    
    return True


def parse_trading_symbol(symbol: str) -> Dict:
    """Parse trading symbol to extract information"""
    # Example: "NIFTY 22000 CE 30 JAN 25"
    parts = symbol.split()
    
    if len(parts) < 4:
        return {}
    
    try:
        result = {
            'underlying': parts[0],
            'strike': float(parts[1]),
            'option_type': parts[2],  # CE or PE
            'expiry_str': ' '.join(parts[3:])
        }
        
        # Parse expiry date
        expiry_parts = parts[3:]
        if len(expiry_parts) >= 3:
            expiry_str = f"{expiry_parts[0]}-{expiry_parts[1]}-{expiry_parts[2]}"
            result['expiry_date'] = datetime.strptime(expiry_str, "%d-%b-%y").date()
        
        return result
    except:
        return {}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame by removing NaN and infinite values"""
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Replace inf with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with appropriate values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['open', 'high', 'low', 'close']:
            df_clean[col] = df_clean[col].ffill().bfill()
        else:
            df_clean[col] = df_clean[col].fillna(0)
    
    return df_clean
