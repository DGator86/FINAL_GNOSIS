#!/usr/bin/env python
"""
scripts/run_gnosis_backtest.py

Single entrypoint for running Composer backtests from the command line.

This script is designed for both human use and AI agents (e.g., Google Anti-Gravity).
It:
- Builds a BacktestConfig
- Runs run_composer_backtest()
- Saves a JSON summary to runs/backtests/<tag>.json
- Prints a one-line JSON summary to STDOUT
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path for local imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtesting.composer_backtest import (  # noqa: E402
    BacktestConfig,
    BacktestResult,
    run_composer_backtest,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Gnosis Composer backtest.")
    parser.add_argument(
        "--symbol",
        type=str,
        help="Single symbol to backtest (e.g., SPY).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (overrides --symbol if provided).",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        required=True,
        help="Timeframe string (e.g., 1D, 1H, 15m).",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag/identifier for this run (used in output filename).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Optional path to a JSON config with additional BacktestConfig overrides.",
    )
    return parser.parse_args()


def load_additional_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load optional BacktestConfig overrides from JSON file."""
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_backtest_config(
    symbols: List[str],
    timeframe: str,
    start: str,
    end: str,
    overrides: Dict[str, Any],
) -> BacktestConfig:
    """
    Build a BacktestConfig instance.

    This assumes BacktestConfig has at least:
    - symbols: List[str]
    - timeframe: str
    - start_date: str or datetime
    - end_date: str or datetime

    Any additional fields are passed via **overrides.
    """
    base_kwargs: Dict[str, Any] = {
        "symbols": symbols,
        "timeframe": timeframe,
        "start_date": start,
        "end_date": end,
    }
    base_kwargs.update(overrides)
    return BacktestConfig(**base_kwargs)  # type: ignore[arg-type]


def summarize_backtest_result(result: BacktestResult) -> Dict[str, Any]:
    """
    Convert BacktestResult into a JSON-serializable summary.

    Assumes BacktestResult is a dataclass or has similar attributes.
    """
    if is_dataclass(result):
        data = asdict(result)
    else:
        # Fallback: extract common metrics explicitly if needed
        data = {
            "total_return": getattr(result, "total_return", None),
            "sharpe": getattr(result, "sharpe", None),
            "max_drawdown": getattr(result, "max_drawdown", None),
            "directional_accuracy": getattr(result, "directional_accuracy", None),
        }

    data.setdefault("timestamp", datetime.utcnow().isoformat())
    return data


def ensure_output_dir() -> str:
    """Ensure runs/backtests directory exists and return its path."""
    out_dir = os.path.join("runs", "backtests")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_summary(summary: Dict[str, Any], tag: str) -> str:
    """Save JSON summary to runs/backtests/<tag>.json and return the filepath."""
    out_dir = ensure_output_dir()
    path = os.path.join(out_dir, f"{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def run_gnosis_backtest() -> None:
    """Main entrypoint for running a Gnosis backtest."""
    args = parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.symbol:
        symbols = [args.symbol.strip()]
    else:
        raise ValueError("You must provide either --symbol or --symbols.")

    overrides = load_additional_config(args.config_path)
    config = build_backtest_config(
        symbols=symbols,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        overrides=overrides,
    )

    result = run_composer_backtest(config)
    summary = summarize_backtest_result(result)
    summary["tag"] = args.tag
    summary["symbols"] = symbols
    summary["timeframe"] = args.timeframe
    summary["start"] = args.start
    summary["end"] = args.end

    path = save_summary(summary, args.tag)

    # Print a one-line JSON summary to STDOUT so an agent can parse it easily.
    output = {
        "tag": args.tag,
        "summary_path": path,
        "metrics": {
            "total_return": summary.get("total_return"),
            "sharpe": summary.get("sharpe"),
            "max_drawdown": summary.get("max_drawdown"),
            "directional_accuracy": summary.get("directional_accuracy"),
        },
    }
    print(json.dumps(output))


if __name__ == "__main__":
    run_gnosis_backtest()
