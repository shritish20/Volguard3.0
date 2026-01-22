"""
VOLGUARD v31.0 – JOURNAL ENGINE
SQLite persistence for every trade with JSON strikes column
"""
import sqlite3
import json
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class TradeRecord:
    trade_id: str
    entry_date: str
    expiry_date: str
    dte_at_entry: int
    structure: str
    strikes: dict
    lots: int
    entry_premium: float
    regime_name: str
    regime_score: float
    regime_confidence: str
    spot_at_entry: float
    vix_at_entry: float
    ivp_at_entry: float
    weighted_vrp_at_entry: float
    gex_regime: str
    skew_regime: str
    fii_flow: str
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_realized: Optional[float] = None

class TradeJournal:
    def __init__(self, db_path: str = config.DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS trades(
                    trade_id TEXT PRIMARY KEY,
                    entry_date TEXT,
                    expiry_date TEXT,
                    dte_at_entry INTEGER,
                    structure TEXT,
                    strikes_json TEXT,
                    lots INTEGER,
                    entry_premium REAL,
                    regime_name TEXT,
                    regime_score REAL,
                    regime_confidence TEXT,
                    spot_at_entry REAL,
                    vix_at_entry REAL,
                    ivp_at_entry REAL,
                    weighted_vrp_at_entry REAL,
                    gex_regime TEXT,
                    skew_regime TEXT,
                    fii_flow TEXT,
                    exit_date TEXT,
                    exit_price REAL,
                    exit_reason TEXT,
                    pnl_realized REAL
                )
            """)

    def log_entry(self, record: TradeRecord):
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO trades VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.trade_id, record.entry_date, record.expiry_date, record.dte_at_entry,
                 record.structure, json.dumps(record.strikes), record.lots, record.entry_premium,
                 record.regime_name, record.regime_score, record.regime_confidence,
                 record.spot_at_entry, record.vix_at_entry, record.ivp_at_entry,
                 record.weighted_vrp_at_entry, record.gex_regime, record.skew_regime,
                 record.fii_flow, record.exit_date, record.exit_price, record.exit_reason,
                 record.pnl_realized)
            )

    def log_exit(self, trade_id: str, exit_price: float, exit_reason: str, pnl: float):
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "UPDATE trades SET exit_date=?, exit_price=?, exit_reason=?, pnl_realized=? WHERE trade_id=?",
                (dt.date.today().isoformat(), exit_price, exit_reason, pnl, trade_id)
            )

    def get_trade_history(self):
        with sqlite3.connect(self.db_path) as con:
            return pd.read_sql("SELECT * FROM trades", con)
