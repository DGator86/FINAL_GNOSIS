"""Ledger store for tracking pipeline results with SQLite persistence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from schemas.core_schemas import PipelineResult


class LedgerStore:
    """JSONL ledger store for pipeline results."""
    
    def __init__(self, ledger_path: Path):
        """
        Initialize ledger store.
        
        Args:
            ledger_path: Path to JSONL ledger file
        """
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.ledger_path.with_suffix(".db"))
        self._ensure_schema()
        logger.info(f"LedgerStore initialized at {self.ledger_path}")
    
    def append(self, result: PipelineResult) -> None:
        """
        Append a pipeline result to the ledger.
        
        Args:
            result: Pipeline result to store
        """
        try:
            with open(self.ledger_path, "a") as f:
                result_dict = result.model_dump(mode="json")
                f.write(json.dumps(result_dict) + "\n")

            db_row = pd.DataFrame(
                [
                    {
                        "timestamp": result.timestamp.isoformat(),
                        "symbol": result.symbol,
                        "payload": json.dumps(result_dict),
                    }
                ]
            )
            db_row.to_sql("ledger", con=self.conn, if_exists="append", index=False)
            logger.debug(
                "Appended result to ledger: %s at %s (sqlite + jsonl)",
                result.symbol,
                result.timestamp,
            )
        except Exception as e:
            logger.error(f"Error appending to ledger: {e}")

    def _ensure_schema(self) -> None:
        """Create ledger table when missing."""

        create_stmt = """
        CREATE TABLE IF NOT EXISTS ledger (
            timestamp TEXT,
            symbol TEXT,
            payload TEXT
        );
        """
        cur = self.conn.cursor()
        cur.execute(create_stmt)
        self.conn.commit()

    def load_dataframe(self) -> pd.DataFrame:
        """Return ledger as DataFrame for analytics/backtests."""

        try:
            return pd.read_sql("SELECT * FROM ledger", self.conn)
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to load ledger dataframe: {exc}")
            return pd.DataFrame()

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
