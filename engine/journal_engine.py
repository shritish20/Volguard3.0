"""
VOLGUARD v31.0 JOURNAL ENGINE
=============================
Handles persistent storage of all trade data using SQLite.
"""

import sqlite3
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any

@dataclass
class TradeRecord:
    """Immutable record of the trade context."""
    trade_id: str
    entry_date: str
    entry_time: str
    expiry_date: str
    structure: str          # e.g. "IRON_CONDOR"
    regime_name: str        # e.g. "AGGRESSIVE_SHORT"
    entry_vrp: float        # Vol Risk Premium at entry
    entry_vov: float        # Vol of Vol at entry
    entry_gex: str          # Gamma Exposure Regime
    regime_score: float     # Composite Score (0-10)
    strikes: Dict[str, Any] # Dictionary of selected strikes
    lots: int
    entry_premium: float
    entry_spot: float
    entry_vix: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_realized: Optional[float] = None

class TradeJournal:
    """SQLite Database Manager for VolGuard"""
    
    def __init__(self, db_path: str = "volguard.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the trades table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table with JSON support for complex fields like 'strikes'
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    entry_date TEXT,
                    entry_time TEXT,
                    expiry_date TEXT,
                    structure TEXT,
                    regime_name TEXT,
                    entry_vrp REAL,
                    entry_vov REAL,
                    entry_gex TEXT,
                    regime_score REAL,
                    lots INTEGER,
                    entry_premium REAL,
                    entry_spot REAL,
                    entry_vix REAL,
                    strikes_json TEXT,
                    exit_date TEXT,
                    exit_price REAL,
                    exit_reason TEXT,
                    pnl_realized REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
            print(" [DB] Database connection established.")
        except Exception as e:
            print(f" [DB] Init Error: {e}")

    def log_entry(self, record: TradeRecord):
        """Log a new trade entry to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialize the strikes dictionary to JSON string
            strikes_json = json.dumps(record.strikes, default=str)
            
            sql = '''
                INSERT OR REPLACE INTO trades 
                (trade_id, entry_date, entry_time, expiry_date, structure, regime_name, 
                 entry_vrp, entry_vov, entry_gex, regime_score, lots, entry_premium, 
                 entry_spot, entry_vix, strikes_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            values = (
                record.trade_id, record.entry_date, record.entry_time, record.expiry_date,
                record.structure, record.regime_name, record.entry_vrp, record.entry_vov,
                record.entry_gex, record.regime_score, record.lots, record.entry_premium,
                record.entry_spot, record.entry_vix, strikes_json
            )
            
            cursor.execute(sql, values)
            conn.commit()
            conn.close()
            print(f" [DB] Trade {record.trade_id} successfully logged.")
            
        except Exception as e:
            print(f" [DB] Logging Error: {e}")

    def log_exit(self, trade_id: str, exit_price: float, exit_reason: str, pnl: float):
        """Update an existing trade with exit details."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            sql = '''
                UPDATE trades 
                SET exit_date = ?, exit_price = ?, exit_reason = ?, pnl_realized = ?
                WHERE trade_id = ?
            '''
            
            exit_date = datetime.now().strftime("%Y-%m-%d")
            cursor.execute(sql, (exit_date, exit_price, exit_reason, pnl, trade_id))
            
            if cursor.rowcount > 0:
                print(f" [DB] Trade {trade_id} updated with exit data.")
            else:
                print(f" [DB] Warning: Trade ID {trade_id} not found for exit.")
                
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f" [DB] Exit Update Error: {e}")

    def get_trade_history(self):
        """Fetch all trades for analysis."""
        conn = sqlite3.connect(self.db_path)
        # Returns a list of rows
        cursor = conn.execute("SELECT * FROM trades ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows
