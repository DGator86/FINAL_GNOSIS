# Repository Reanalysis

This document captures the current structure and orchestration flow of the Super Gnosis / DHPE trading stack as implemented in this repository.

## Entry Points and Pipeline Assembly
- `main.py` exposes Typer commands for single-run evaluation, a live trading loop, and opportunity scanning. It wires Hedge, Liquidity, Sentiment, and Elasticity engines alongside a composer, trade agent, and optional broker integration, returning a `PipelineRunner` ready for execution. ML helpers (Kats forecaster, FAISS retriever, anomaly detector, curriculum RL evaluator) are attached by default.【F:main.py†L13-L152】【F:main.py†L155-L360】

## Configuration Model
- All runtime knobs are centralized in `config/config_models.py`, which defines Pydantic models for engines, agents, scanner ranking, and trade risk controls. These schemas back `config.AppConfig` and are consumed by the CLI when building pipelines.【F:config/config_models.py†L1-L120】

## Orchestration Flow
- `engines/orchestration/pipeline_runner.py` coordinates a single pipeline pass: executing enabled engines, collecting agent suggestions, composing consensus, updating the adaptive watchlist, generating trade ideas, optionally executing them, recording ledger entries, and feeding results to adaptation/tracking hooks. Errors inside each phase are contained with targeted logging so later phases continue when possible.【F:engines/orchestration/pipeline_runner.py†L1-L185】

## Adaptive Watchlist and Universe Control
- `watchlist/adaptive_watchlist.py` maintains a ranked symbol list using hedge, elasticity, and liquidity metrics. It scores opportunities, enforces liquidity/volatility/vanna-charm thresholds, guards against duplicate positions, and surfaces a bounded active set for trade gating. Freshness windows and candidate limits keep the watchlist focused on recent data.【F:watchlist/adaptive_watchlist.py†L1-L161】

## Ledgering and Observability
- `ledger/ledger_store.py` appends each pipeline result to a JSONL ledger path, creating parent directories as needed for persistence. Results are serialized with Pydantic’s `model_dump` to retain full context for later analysis.【F:ledger/ledger_store.py†L1-L42】

Use this reanalysis as a starting point when navigating the codebase or planning enhancements; it highlights how configuration, orchestration, gating, and persistence connect across the trading workflow.
