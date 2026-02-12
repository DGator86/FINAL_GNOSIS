#!/usr/bin/env python
"""
scripts/evaluate_backtest_pair.py

Compare two backtest JSON summaries (baseline vs candidate)
and decide whether the candidate should be kept or reverted.

Designed for AI agents and humans.
"""

import argparse
import json
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate two Gnosis backtest runs.")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline JSON summary (e.g., runs/backtests/baseline_spy_daily.json).",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help="Path to candidate JSON summary.",
    )
    return parser.parse_args()


def load_summary(path: str) -> Dict[str, Any]:
    """Load JSON summary from given path."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> float:
    """Convert value to float safely; return 0.0 on failure."""
    try:
        return float(value)
    except Exception:
        return 0.0


def decide_keep_candidate(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Decide whether to keep the candidate run.

    Rules (can be tuned):
    - candidate.sharpe >= baseline.sharpe * 1.02  (>= 2% improvement)
    - candidate.max_drawdown <= baseline.max_drawdown * 1.05 (no more than 5% worse)
    - candidate.total_return >= baseline.total_return * 0.95 (no catastrophic PnL drop)
    """
    b_sharpe = safe_float(baseline.get("sharpe"))
    c_sharpe = safe_float(candidate.get("sharpe"))

    b_mdd = abs(safe_float(baseline.get("max_drawdown")))
    c_mdd = abs(safe_float(candidate.get("max_drawdown")))

    b_ret = safe_float(baseline.get("total_return"))
    c_ret = safe_float(candidate.get("total_return"))

    sharpe_ok = c_sharpe >= b_sharpe * 1.02 or (b_sharpe == 0 and c_sharpe > 0)
    mdd_ok = c_mdd <= b_mdd * 1.05 if b_mdd > 0 else True
    ret_ok = c_ret >= b_ret * 0.95

    keep = bool(sharpe_ok and mdd_ok and ret_ok)

    reason_parts = []
    reason_parts.append(f"baseline_sharpe={b_sharpe:.4f}, candidate_sharpe={c_sharpe:.4f}")
    reason_parts.append(f"baseline_max_drawdown={b_mdd:.4f}, candidate_max_drawdown={c_mdd:.4f}")
    reason_parts.append(f"baseline_total_return={b_ret:.4f}, candidate_total_return={c_ret:.4f}")

    if keep:
        decision = "KEEP"
        reason_parts.append("Candidate meets improvement criteria.")
    else:
        decision = "REVERT"
        reason_parts.append("Candidate fails improvement criteria.")

    return {
        "keep": keep,
        "decision": decision,
        "baseline_metrics": {
            "sharpe": b_sharpe,
            "max_drawdown": b_mdd,
            "total_return": b_ret,
        },
        "candidate_metrics": {
            "sharpe": c_sharpe,
            "max_drawdown": c_mdd,
            "total_return": c_ret,
        },
        "reason": " | ".join(reason_parts),
    }


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    baseline = load_summary(args.baseline)
    candidate = load_summary(args.candidate)
    decision = decide_keep_candidate(baseline, candidate)
    print(json.dumps(decision))


if __name__ == "__main__":
    main()
