"""Convenience helpers for running the full pipeline for a symbol."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Tuple

from config import load_config
from main import build_pipeline
from schemas.core_schemas import PipelineResult


def run_full_pipeline_for_symbol(
    symbol: str,
    as_of: datetime,
    mode: str = "live",
    config_path: Optional[str] = None,
) -> Tuple[PipelineResult, Dict]:
    """
    Run the full DHPE pipeline for a single symbol.

    Returns the pipeline result along with lightweight diagnostics metadata.
    """
    config = load_config(config_path)
    runner = build_pipeline(symbol, config)
    result = runner.run_once(as_of)

    diagnostics: Dict = {"mode": mode}
    return result, diagnostics
