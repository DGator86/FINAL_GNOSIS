#!/usr/bin/env python
"""
Automate research experiments across multiple parameter tweaks.

Workflow:
- Load an experiment JSON (see config/research/example_experiment.json).
- Run a baseline backtest (once) using run_gnosis_backtest.py.
- For each tweak:
  - Apply overrides to the backtest config JSON.
  - Run a candidate backtest tagged with the tweak name.
  - Evaluate baseline vs candidate using evaluate_backtest_pair.py.
  - Revert overrides if the tweak is rejected.
  - Append the decision to docs/research_log.md.

This keeps experiments reproducible and traceable for both humans and agents.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

RESEARCH_LOG = Path("docs/research_log.md")


def load_experiment_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_overrides(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_overrides(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def run_command(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    return result.stdout.strip()


def run_backtest(
    symbols: List[str],
    timeframe: str,
    start: str,
    end: str,
    tag: str,
    config_path: Path | None,
) -> Dict[str, Any]:
    cmd = [
        "python",
        "scripts/run_gnosis_backtest.py",
        "--symbols",
        ",".join(symbols),
        "--timeframe",
        timeframe,
        "--start",
        start,
        "--end",
        end,
        "--tag",
        tag,
    ]
    if config_path:
        cmd.extend(["--config-path", str(config_path)])

    output = run_command(cmd)
    return json.loads(output)


def evaluate_pair(baseline_path: Path, candidate_path: Path) -> Dict[str, Any]:
    cmd = [
        "python",
        "scripts/evaluate_backtest_pair.py",
        "--baseline",
        str(baseline_path),
        "--candidate",
        str(candidate_path),
    ]
    output = run_command(cmd)
    return json.loads(output)


def append_log_entry(entry: Dict[str, Any]) -> None:
    RESEARCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat()
    lines = [
        f"## {timestamp} – {entry['tweak_name']}",
        f"- Baseline: `{entry['baseline_path']}` (tag `{entry['baseline_tag']}`)",
        f"- Candidate: `{entry['candidate_path']}` (tag `{entry['candidate_tag']}`)",
        f"- Decision: **{entry['decision']}** (keep={entry['keep']})",
        f"- Reason: {entry['reason']}",
        f"- Baseline metrics: {json.dumps(entry['baseline_metrics'])}",
        f"- Candidate metrics: {json.dumps(entry['candidate_metrics'])}",
        "",
    ]
    mode = "a" if RESEARCH_LOG.exists() else "w"
    with RESEARCH_LOG.open(mode, encoding="utf-8") as f:
        f.write("\n".join(lines))


def ensure_baseline(
    *,
    symbols: List[str],
    timeframe: str,
    start: str,
    end: str,
    tag: str,
    config_path: Path | None,
) -> Path:
    baseline_path = Path("runs/backtests") / f"{tag}.json"
    if baseline_path.exists():
        return baseline_path

    result = run_backtest(
        symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        tag=tag,
        config_path=config_path,
    )
    return Path(result["summary_path"])


def process_tweak(
    tweak: Dict[str, Any],
    *,
    current_overrides: Dict[str, Any],
    symbols: List[str],
    timeframe: str,
    start: str,
    end: str,
    baseline_tag: str,
    config_path: Path,
    baseline_path: Path,
) -> Dict[str, Any]:
    original_overrides = copy.deepcopy(current_overrides)
    candidate_overrides = deep_merge(current_overrides, tweak.get("overrides", {}))
    write_overrides(config_path, candidate_overrides)

    candidate_tag = f"{baseline_tag}_{tweak['name']}"
    backtest_result = run_backtest(
        symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        tag=candidate_tag,
        config_path=config_path,
    )
    candidate_path = Path(backtest_result["summary_path"])

    decision = evaluate_pair(baseline_path, candidate_path)

    keep_change = bool(decision.get("keep"))
    if not keep_change:
        write_overrides(config_path, original_overrides)
        final_overrides = original_overrides
    else:
        final_overrides = candidate_overrides

    log_entry = {
        "tweak_name": tweak.get("name", "unknown_tweak"),
        "baseline_path": str(baseline_path),
        "baseline_tag": baseline_tag,
        "candidate_path": str(candidate_path),
        "candidate_tag": candidate_tag,
        "decision": decision.get("decision", "UNKNOWN"),
        "keep": keep_change,
        "reason": decision.get("reason", ""),
        "baseline_metrics": decision.get("baseline_metrics", {}),
        "candidate_metrics": decision.get("candidate_metrics", {}),
    }
    append_log_entry(log_entry)

    return final_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of research experiments.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment JSON (e.g., config/research/example_experiment.json).",
    )
    args = parser.parse_args()

    experiment_path = Path(args.config)
    experiment = load_experiment_config(experiment_path)

    symbols = experiment.get("symbols") or []
    if isinstance(symbols, str):
        symbols = [symbols]
    if not symbols:
        raise ValueError("Experiment config must include at least one symbol.")

    timeframe = experiment["timeframe"]
    start = experiment["start"]
    end = experiment["end"]
    baseline_tag = experiment.get("baseline_tag", f"baseline_{symbols[0]}_{timeframe}")

    config_path = Path(experiment.get("backtest_config_path", "config/research/backtest_overrides.json"))
    current_overrides = read_overrides(config_path)
    write_overrides(config_path, current_overrides)

    baseline_path = ensure_baseline(
        symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        tag=baseline_tag,
        config_path=config_path,
    )

    tweaks = experiment.get("tweaks", [])
    for tweak in tweaks:
        current_overrides = process_tweak(
            tweak,
            current_overrides=current_overrides,
            symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            baseline_tag=baseline_tag,
            config_path=config_path,
            baseline_path=baseline_path,
        )

    write_overrides(config_path, current_overrides)


if __name__ == "__main__":
    main()
