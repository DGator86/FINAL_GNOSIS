"""Ledger store for tracking pipeline results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
        logger.info(f"LedgerStore initialized at {self.ledger_path}")
    
    def append(self, result: PipelineResult) -> None:
        """
        Append a pipeline result to the ledger.
        
        Args:
            result: Pipeline result to store
        """
        try:
            with open(self.ledger_path, "a") as f:
                # Convert to dict for JSON serialization
                result_dict = result.model_dump(mode="json")
                f.write(json.dumps(result_dict) + "\n")
            logger.debug(f"Appended result to ledger: {result.symbol} at {result.timestamp}")
        except Exception as e:
            logger.error(f"Error appending to ledger: {e}")
