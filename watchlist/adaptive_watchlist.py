"""Adaptive, dynamic watchlist scoring and filtering."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Set

from loguru import logger

from schemas.core_schemas import (
    ElasticitySnapshot,
    HedgeSnapshot,
    LiquiditySnapshot,
    PipelineResult,
    WatchlistEntry,
)


class AdaptiveWatchlist:
    """Ranks symbols each cycle using Hedge/Elasticity/Liquidity signals."""

    def __init__(
        self,
        universe: Sequence[str],
        min_candidates: int = 3,
        max_candidates: int = 8,
        volume_threshold: float = 10_000_000,
        vanna_charm_threshold: float = 2.0,
        freshness: timedelta = timedelta(minutes=30),
    ) -> None:
        self.universe: Set[str] = set(universe)
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.volume_threshold = volume_threshold
        self.vanna_charm_threshold = vanna_charm_threshold
        self.freshness = freshness
        self.entries: Dict[str, WatchlistEntry] = {}

    def update_from_pipeline(
        self,
        result: PipelineResult,
        active_positions: Optional[Set[str]] = None,
    ) -> WatchlistEntry:
        """Update a symbol's entry using the latest pipeline result."""

        metrics = self._extract_metrics(result)
        score = self._compute_score(metrics)
        passes_filters, reasons = self._apply_filters(metrics, result, active_positions or set())

        entry = WatchlistEntry(
            symbol=result.symbol,
            score=score,
            timestamp=result.timestamp,
            passes_filters=passes_filters,
            reasons=reasons,
            metrics=metrics,
        )
        self.entries[result.symbol] = entry
        return entry

    def get_active_watchlist(self) -> List[WatchlistEntry]:
        """Return ranked, filtered symbols bounded by candidate limits."""

        fresh_entries = []
        for entry in self.entries.values():
            now = datetime.now(tz=entry.timestamp.tzinfo) if entry.timestamp.tzinfo else datetime.now()
            if now - entry.timestamp <= self.freshness:
                fresh_entries.append(entry)

        filtered = [e for e in fresh_entries if e.passes_filters]
        ranked = sorted(filtered, key=lambda e: e.score, reverse=True)

        if len(ranked) < self.min_candidates:
            logger.debug(
                "Active watchlist below minimum candidates: "
                f"{len(ranked)}/{self.min_candidates}"
            )

        if len(ranked) > self.max_candidates:
            ranked = ranked[: self.max_candidates]

        return ranked

    def is_symbol_active(self, symbol: str) -> bool:
        """Check if a symbol is currently tradable based on the adaptive list."""

        active_symbols = {entry.symbol for entry in self.get_active_watchlist()}
        return symbol in active_symbols

    def _compute_score(self, metrics: Dict[str, float]) -> float:
        """Weighted opportunity score using the specified composite formula."""

        score = (
            0.4 * metrics.get("movement_energy_asymmetry_score", 0.0)
            + 0.3 * metrics.get("elasticity_regime_volatility_edge", 0.0)
            + 0.15 * metrics.get("gamma_exposure_imbalance", 0.0)
            + 0.1 * metrics.get("liquidity_friction_score", 0.0)
            + 0.05 * metrics.get("recent_options_flow_momentum", 0.0)
        )
        return score

    def _extract_metrics(self, result: PipelineResult) -> Dict[str, float]:
        hedge: Optional[HedgeSnapshot] = result.hedge_snapshot
        liquidity: Optional[LiquiditySnapshot] = result.liquidity_snapshot
        elasticity: Optional[ElasticitySnapshot] = result.elasticity_snapshot

        metrics: Dict[str, float] = {
            "movement_energy_asymmetry_score": (hedge.energy_asymmetry if hedge else 0.0),
            "elasticity_regime_volatility_edge": (elasticity.volatility if elasticity else 0.0),
            "gamma_exposure_imbalance": (hedge.gamma_pressure if hedge else 0.0),
            "liquidity_friction_score": 1 - ((liquidity.bid_ask_spread or 0.0) / 1.0) if liquidity else 0.0,
            # Use pressure_net as proxy for options flow momentum (HedgeSnapshot has no .data attribute)
            "recent_options_flow_momentum": (hedge.pressure_net if hedge else 0.0),
            "average_daily_volume": liquidity.volume if liquidity else 0.0,
            "elasticity_percentile": hedge.elasticity if hedge else 0.5,
            "dealer_gamma_sign": hedge.dealer_gamma_sign if hedge else 0.0,
            "vanna_pressure": hedge.vanna_pressure if hedge else 0.0,
            "charm_pressure": hedge.charm_pressure if hedge else 0.0,
        }
        return metrics

    def _apply_filters(
        self,
        metrics: Dict[str, float],
        result: PipelineResult,
        active_positions: Set[str],
    ) -> tuple[bool, List[str]]:
        reasons: List[str] = []

        if result.symbol not in self.universe:
            reasons.append("symbol not in configured universe")

        volume = metrics.get("average_daily_volume", 0.0)
        if volume < self.volume_threshold:
            reasons.append(f"avg volume {volume:,.0f} below {self.volume_threshold:,.0f}")

        elasticity_pct = metrics.get("elasticity_percentile", 0.5)
        if not (elasticity_pct >= 0.7 or elasticity_pct <= 0.3):
            reasons.append("elasticity not in extreme regime")

        dealer_gamma = metrics.get("dealer_gamma_sign", 0.0)
        if abs(dealer_gamma) > 0.25 and dealer_gamma > 0:
            reasons.append("dealer gamma stabilizing (long gamma)")

        vanna = abs(metrics.get("vanna_pressure", 0.0))
        charm = abs(metrics.get("charm_pressure", 0.0))
        if vanna < self.vanna_charm_threshold and charm < self.vanna_charm_threshold:
            reasons.append("vanna/charm pressure below 2σ proxy")

        if result.symbol in active_positions:
            reasons.append("position already open")

        passes_filters = len(reasons) == 0
        if not passes_filters:
            logger.debug(f"{result.symbol} filtered out: {', '.join(reasons)}")

        return passes_filters, reasons

    def debug_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return raw metrics for observability/debugging."""

        return {symbol: entry.metrics for symbol, entry in self.entries.items()}
