import os
import sys
import pandas as pd
from colorama import Fore, Style, init

# Initialize color formatting for clear output
init(autoreset=True)

print(f"\n{Fore.CYAN}{'='*60}")
print(f"{'VOLGUARD v31.0 SYSTEM AUDIT':^60}")
print(f"{'='*60}{Style.RESET_ALL}")

# --- CHECK 1: FILE STRUCTURE & IMPORTS ---
print(f"\n{Fore.YELLOW}1. CHECKING FILE STRUCTURE...{Style.RESET_ALL}")
try:
    from engine.data_engine import UpstoxDataEngine
    from engine.execution_engine import UpstoxExecutionEngine
    from engine.journal_engine import TradeJournal
    from logic.regime_logic import AnalyticsEngine
    from config import Config
    print(f"[{Fore.GREEN}OK{Style.RESET_ALL}] All internal modules found (Engine + Logic).")
except ImportError as e:
    print(f"[{Fore.RED}FAIL{Style.RESET_ALL}] Critical Module Missing: {e}")
    print(f"Make sure you are running this from the folder containing 'main.py'")
    sys.exit(1)

# --- CHECK 2: AUTHENTICATION ---
print(f"\n{Fore.YELLOW}2. CHECKING API AUTHENTICATION...{Style.RESET_ALL}")
# Try to get token from environment variable, or ask user
token = os.getenv("UPSTOX_ACCESS_TOKEN")
if not token:
    print(f"{Fore.CYAN}Tip: You can set UPSTOX_ACCESS_TOKEN in your environment to skip this.{Style.RESET_ALL}")
    token = input(f"{Fore.WHITE}Please paste your Upstox Access Token: {Style.RESET_ALL}").strip()

if not token:
    print(f"[{Fore.RED}FAIL{Style.RESET_ALL}] No token provided. Exiting.")
    sys.exit(1)

try:
    # Initialize the Data Engine (The Body)
    data_engine = UpstoxDataEngine(token)
    
    # We make a lightweight call to 'User Profile' to test validity
    # Note: If your data_engine doesn't have a direct 'get_profile' method,
    # we use the internal api_client if available, or try fetching funds.
    try:
        funds = data_engine.fetch_funds_and_margin()
        print(f"[{Fore.GREEN}OK{Style.RESET_ALL}] Token is Valid. Connection established.")
        print(f"    -> Available Funds: ₹{funds.get('available_margin', 0):,.2f}")
    except Exception as e:
        print(f"[{Fore.RED}FAIL{Style.RESET_ALL}] Token Invalid or API Error: {e}")
        sys.exit(1)
        
except Exception as e:
    print(f"[{Fore.RED}FAIL{Style.RESET_ALL}] Could not initialize Data Engine: {e}")
    sys.exit(1)

# --- CHECK 3: DATA FEED (THE EYES) ---
print(f"\n{Fore.YELLOW}3. CHECKING DATA FEEDS (NIFTY & VIX)...{Style.RESET_ALL}")
try:
    # 1. Check History (Critical for Volatility Math)
    print("    Fetching Nifty historical data...", end=" ")
    nifty_hist = data_engine.fetch_history(Config.NIFTY_KEY, days=50)
    
    if not nifty_hist.empty and len(nifty_hist) > 20:
        print(f"[{Fore.GREEN}OK{Style.RESET_ALL}] ({len(nifty_hist)} candles)")
    else:
        print(f"[{Fore.RED}FAIL{Style.RESET_ALL}] Empty data returned for Nifty History.")

    # 2. Check Option Chain (Critical for Greeks)
    print("    Fetching Expiry details...", end=" ")
    weekly, monthly, _, _ = data_engine.fetch_expiry_details()
    
    if weekly:
        print(f"[{Fore.GREEN}OK{Style.RESET_ALL}] Next Expiry: {weekly}")
        print("    Fetching Option Chain...", end=" ")
        chain = data_engine.fetch_option_chain(weekly)
        
        if not chain.empty and 'ce_gamma' in chain.columns:
            print(f"[{Fore.GREEN}OK{Style.RESET_ALL}] Chain fetched & Greeks present.")
        else:
            print(f"[{Fore.RED}FAIL{Style.RESET_ALL}] Chain missing or Greeks not calculated.")
            # If this fails, your data_engine might have column naming mismatches
    else:
        print(f"[{Fore.RED}FAIL{Style.RESET_ALL}] Could not determine expiry dates.")

except Exception as e:
    print(f"[{Fore.RED}FAIL{Style.RESET_ALL}] Data Feed Crash: {e}")

# --- CHECK 4: BRAIN INTEGRATION ---
print(f"\n{Fore.YELLOW}4. CHECKING LOGIC INTEGRATION (BRAIN)...{Style.RESET_ALL}")
try:
    analytics = AnalyticsEngine()
    
    # We construct a fake 'live' scenario using the data we just fetched
    if not nifty_hist.empty:
        last_close = nifty_hist.iloc[-1]['close']
        
        # Test the VolMetrics calculator
        vol = analytics.get_vol_metrics(
            nifty_hist=nifty_hist,
            vix_hist=pd.DataFrame(), # Empty VIX hist is handled by logic fallback?
            spot_live=last_close,
            vix_live=13.0
        )
        
        if vol and vol.vol_regime:
            print(f"[{Fore.GREEN}OK{Style.RESET_ALL}] Brain successfully processed Data.")
            print(f"    -> Calculated Vol Regime: {vol.vol_regime}")
            print(f"    -> GARCH Volatility: {vol.garch7:.2f}%")
        else:
            print(f"[{Fore.RED}FAIL{Style.RESET_ALL}] Brain returned empty results.")
    else:
        print(f"[{Fore.RED}SKIP{Style.RESET_ALL}] Skipping Brain test (No history data).")

except Exception as e:
    print(f"[{Fore.RED}FAIL{Style.RESET_ALL}] Logic Crash: {e}")
    print("    (This usually means column names in Data Engine don't match what Logic expects)")

# --- CHECK 5: DATABASE (MEMORY) ---
print(f"\n{Fore.YELLOW}5. CHECKING DATABASE WRITE ACCESS...{Style.RESET_ALL}")
try:
    # Use a specific 'audit' database so we don't mess up your real trading log
    journal = TradeJournal(db_path="volguard_audit.db")
    print(f"[{Fore.GREEN}OK{Style.RESET_ALL}] Database 'volguard_audit.db' created successfully.")
except Exception as e:
    print(f"[{Fore.RED}FAIL{Style.RESET_ALL}] Database permission error: {e}")

print(f"\n{Fore.CYAN}{'='*60}")
print(f"{'AUDIT COMPLETE':^60}")
print(f"{'='*60}{Style.RESET_ALL}")
