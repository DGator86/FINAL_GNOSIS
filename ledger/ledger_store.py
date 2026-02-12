"""Ledger store for tracking pipeline results with SQLite persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from schemas.core_schemas import PipelineResult


class LedgerStore:
    """
    JSONL ledger store for pipeline results with thread-safe SQLite.
    
    Uses thread-local connections to handle SQLite's thread safety requirements.
    Each thread gets its own connection to avoid "SQLite objects created in a 
    thread can only be used in that same thread" errors.
    """
    
    def __init__(self, ledger_path: Path):
        """
        Initialize ledger store.
        
        Args:
            ledger_path: Path to JSONL ledger file
        """
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = self.ledger_path.with_suffix(".db")
        
        # Thread-local storage for connections
        self._local = threading.local()
        
        # Lock for file operations
        self._file_lock = threading.Lock()
        
        # Initialize schema using a fresh connection
        self._ensure_schema()
        logger.info(f"LedgerStore initialized at {self.ledger_path} (thread-safe)")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            # Create new connection for this thread
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,  # We manage thread safety ourselves
                timeout=30.0,  # Wait up to 30s for locks
            )
            # Enable WAL mode for better concurrent access
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn
    
    def append(self, result: PipelineResult) -> None:
        """
        Append a pipeline result to the ledger (thread-safe).
        
        Args:
            result: Pipeline result to store
        """
        try:
            result_dict = result.model_dump(mode="json")
            
            # Write to JSONL file with lock
            with self._file_lock:
                with open(self.ledger_path, "a") as f:
                    f.write(json.dumps(result_dict) + "\n")

            # Write to SQLite using thread-local connection
            conn = self._get_connection()
            db_row = pd.DataFrame(
                [
                    {
                        "timestamp": result.timestamp.isoformat(),
                        "symbol": result.symbol,
                        "payload": json.dumps(result_dict),
                    }
                ]
            )
            db_row.to_sql("ledger", con=conn, if_exists="append", index=False)
            conn.commit()
            
            logger.debug(
                "Appended result to ledger: %s at %s (sqlite + jsonl)",
                result.symbol,
                result.timestamp,
            )
        except Exception as e:
            logger.error(f"Error appending to ledger: {e}")

    def _ensure_schema(self) -> None:
        """Create ledger table when missing (thread-safe)."""
        create_stmt = """
        CREATE TABLE IF NOT EXISTS ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            payload TEXT
        );
        """
        # Create index for faster lookups
        index_stmt = """
        CREATE INDEX IF NOT EXISTS idx_ledger_symbol_timestamp 
        ON ledger(symbol, timestamp);
        """
        
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(create_stmt)
        cur.execute(index_stmt)
        conn.commit()

    def load_dataframe(self) -> pd.DataFrame:
        """Return ledger as DataFrame for analytics/backtests (thread-safe)."""
        try:
            conn = self._get_connection()
            return pd.read_sql("SELECT * FROM ledger ORDER BY timestamp", conn)
        except Exception as exc:
            logger.error(f"Failed to load ledger dataframe: {exc}")
            return pd.DataFrame()
    
    def close(self) -> None:
        """Close the thread-local connection if it exists."""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

    def recent_flows(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Fetch recent flows for elasticity regression."""

        df = self.load_dataframe()
        if df.empty:
            return []
        df = df.tail(limit)
        flows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            try:
                payload = json.loads(row.get("payload") or "{}")
            except json.JSONDecodeError:
                payload = {}
            hedge = payload.get("hedge_snapshot") or {}
            flows.append(
                {
                    "flow": hedge.get("pressure_net", 0.0),
                    "price": hedge.get("elasticity", 0.0),
                }
            )
        return flows
