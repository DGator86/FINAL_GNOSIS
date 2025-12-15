"""Vectorized backtest runner integrating consensus signals and hedge elasticity."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import vectorbt as vbt
from joblib import Parallel, delayed
from loguru import logger
from scipy import stats

from engines.orchestration.pipeline_runner import PipelineRunner


class BacktestRunnerV2:
    """VectorBT-powered backtester with Monte Carlo VaR and slippage modeling."""

    def __init__(self, runner_factory, config: Dict[str, Any]):
        self.runner_factory = runner_factory
        self.config = config

    def _consensus_to_signals(self, ledger_df: pd.DataFrame) -> pd.DataFrame:
        if ledger_df.empty:
            return pd.DataFrame()
        entries = []
        for _, row in ledger_df.iterrows():
            payload = row.get("payload") or {}
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {}
            consensus = payload.get("consensus") or {}
            entries.append(
                {
                    "timestamp": pd.to_datetime(row.get("timestamp")),
                    "long": consensus.get("direction") == "long",
                    "short": consensus.get("direction") == "short",
                }
            )
        return pd.DataFrame(entries).set_index("timestamp").sort_index()

    def run_single(self, symbol: str, start: datetime, end: datetime) -> Dict[str, Any]:
        runner: PipelineRunner = self.runner_factory(symbol)
        ledger_rows: List[Dict[str, Any]] = []
        for ts in pd.date_range(start, end, freq=self.config.get("freq", "1D")):
            result = runner.run_once(ts)
            ledger_rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "payload": result.model_dump(mode="json"),
                }
            )
        ledger_df = pd.DataFrame(ledger_rows)
        signals = self._consensus_to_signals(ledger_df)
        if signals.empty:
            return {"symbol": symbol, "message": "no signals"}

        price_series = vbt.YFData.download(symbol, start=start, end=end).get("Close")
        price_series = price_series.reindex(signals.index).ffill()
        pf = vbt.Portfolio.from_signals(
            price_series,
            entries=signals["long"],
            exits=signals["short"],
            freq="1D",
            fees=0.001,
            slippage=self._slippage_model(price_series),
        )

        stats = pf.stats()
        calmar = stats.get("Calmar Ratio", np.nan)
        mean_ret = pf.returns.mean()
        std_ret = pf.returns.std()
        sim_returns = stats.norm.rvs(loc=mean_ret, scale=std_ret, size=(1000, len(pf.returns)))
        var = float(np.percentile(sim_returns, 5))

        return {
            "symbol": symbol,
            "calmar": float(calmar),
            "sharpe": float(stats.get("Sharpe Ratio", np.nan)),
            "var": var,
            "pf": pf,
        }

    def run(self, symbols: Iterable[str], start: datetime, end: datetime) -> List[Dict[str, Any]]:
        n_jobs = self.config.get("n_jobs", -1)
        return Parallel(n_jobs=n_jobs)(delayed(self.run_single)(sym, start, end) for sym in symbols)

    def _slippage_model(self, prices: pd.Series) -> float:
        spread = self.config.get("spread", 0.01)
        volume = self.config.get("volume", 1e6)
        vol = prices.pct_change().rolling(20).std().mean()
        return float(spread * 0.5 + np.sqrt(volume) * float(vol))


# Test: instantiate with dummy factory
