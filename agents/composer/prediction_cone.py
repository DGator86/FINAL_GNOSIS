"""Prediction cone utilities.

Provides symbol-agnostic prediction cone construction that adapts
volatility and scaling based on timeframe and the symbol's own market data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PredictionConeConfig:
    """
    Config for the field-based prediction cone.

    NOTE: This is symbol-agnostic. The only thing that cares about cadence
    is bars_per_year, which must be set per timeframe/asset class.
    """

    horizon_bars: int = 30
    confidence_1: float = 0.68  # inner band
    confidence_2: float = 0.90  # outer band

    # Vol & scaling
    base_annual_vol: float = 0.20  # fallback if no vol engine & no realized vol
    bars_per_year: int = 252  # default: 1 bar = 1 trading day; override upstream

    max_drift_per_bar: float = 0.004
    greek_vol_scale: float = 0.5
    liquidity_vol_scale: float = 0.5
    oflow_vol_scale: float = 0.5
    cone_smoothing: float = 0.3


@dataclass
class PredictionCone:
    """Container for the blended prediction cone paths."""

    steps: int
    field_path: np.ndarray
    tech_path: np.ndarray
    blended_path: np.ndarray
    metadata: Dict[str, Any]


def estimate_bars_per_year(timeframe: str, is_crypto: bool) -> int:
    """Estimate bars per year for the provided timeframe and asset class."""

    if timeframe.endswith("d"):
        return 365 if is_crypto else 252
    if timeframe.endswith("h"):
        hours = int(timeframe[:-1])
        if is_crypto:
            return int(24 / hours * 365)
        return int(6.5 / hours * 252)
    if timeframe.endswith("m"):
        mins = int(timeframe[:-1])
        if is_crypto:
            return int(24 * 60 / mins * 365)
        return int(390 / mins * 252)
    return 252


def _infer_annual_vol(
    vol_output: Optional[Any],
    close: pd.Series,
    cfg: PredictionConeConfig,
) -> float:
    """
    Choose an annualized vol proxy per symbol.

    Priority:
    1) Implied vol (from options) if available
    2) Realized vol from this symbol's close prices
    3) Config fallback
    """

    if vol_output is not None:
        if getattr(vol_output, "implied_vol_annual", None):
            return float(vol_output.implied_vol_annual)
        if getattr(vol_output, "realized_vol_annual", None):
            return float(vol_output.realized_vol_annual)

    returns = np.log(close / close.shift(1)).dropna()
    if len(returns) > 10:
        sigma_bar = float(returns.std())
        sigma_annual = sigma_bar * np.sqrt(cfg.bars_per_year)
        if 0 < sigma_annual < 5.0:
            return sigma_annual

    return cfg.base_annual_vol


def _field_drift_per_bar(
    directive: Any,
    hedge: Any,
    liquidity: Any,
    sentiment: Any,
    cfg: PredictionConeConfig,
) -> float:
    """Derive drift per bar from directive and engine state."""

    drift_inputs = [
        getattr(directive, "direction_bias", 0.0),
        getattr(hedge, "energy_asymmetry", 0.0),
        getattr(liquidity, "liquidity_score", 0.0) - 0.5,
        getattr(sentiment, "sentiment_score", 0.0),
    ]
    raw_drift = float(np.tanh(np.nansum(drift_inputs)))
    return float(np.clip(raw_drift * cfg.max_drift_per_bar, -cfg.max_drift_per_bar, cfg.max_drift_per_bar))


def _field_vol_per_bar(
    hedge: Any,
    liquidity: Any,
    sentiment: Any,
    annual_vol: float,
    cfg: PredictionConeConfig,
) -> float:
    """Scale per-bar volatility using hedge, liquidity, and sentiment fields."""

    base_vol_bar = annual_vol / np.sqrt(cfg.bars_per_year)
    greek_scale = 1 + cfg.greek_vol_scale * abs(getattr(hedge, "gamma_pressure", 0.0))
    liq_scale = 1 + cfg.liquidity_vol_scale * max(0.0, 0.5 - getattr(liquidity, "liquidity_score", 0.5))
    sent_scale = 1 + cfg.oflow_vol_scale * abs(getattr(sentiment, "sentiment_score", 0.0))
    scaled = base_vol_bar * greek_scale * liq_scale * sent_scale
    return float(np.clip(scaled, 0.0, base_vol_bar * 5))


def _build_field_cone_path(spot: float, drift_per_bar: float, vol_per_bar: float, cfg: PredictionConeConfig) -> np.ndarray:
    """Build deterministic field-driven cone path."""

    path = np.zeros(cfg.horizon_bars)
    price = spot
    for i in range(cfg.horizon_bars):
        price *= 1 + drift_per_bar + vol_per_bar
        path[i] = price
    return path


def compute_donchian_predictive_snapshot(ohlc: pd.DataFrame, config: Optional[Any] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute a lightweight Donchian-style predictive snapshot from OHLC data."""

    window = getattr(config, "lookback", None) or min(len(ohlc), 20)
    highs = ohlc["high"].rolling(window=window).max()
    lows = ohlc["low"].rolling(window=window).min()
    don_hi = float(highs.iloc[-1]) if not highs.empty else float("nan")
    don_lo = float(lows.iloc[-1]) if not lows.empty else float("nan")
    don_md = (don_hi + don_lo) / 2 if np.isfinite(don_hi) and np.isfinite(don_lo) else float("nan")

    proj_hi = don_hi + (don_hi - don_md) * 0.1 if np.isfinite(don_hi) else don_hi
    proj_lo = don_lo - (don_md - don_lo) * 0.1 if np.isfinite(don_lo) else don_lo
    proj_md = don_md

    snapshot = {"don_hi": don_hi, "don_lo": don_lo, "don_md": don_md}
    raw = {"proj_hi": proj_hi, "proj_md": proj_md, "proj_lo": proj_lo}
    return snapshot, raw


def _build_technical_cone_path_from_donchian(
    spot: float,
    don_snapshot: Dict[str, float],
    proj_hi: float,
    proj_md: float,
    proj_lo: float,
    cfg: PredictionConeConfig,
) -> np.ndarray:
    """Project a technical cone path from Donchian levels."""

    upper_step = (proj_hi - spot) / max(cfg.horizon_bars, 1)
    lower_step = (proj_lo - spot) / max(cfg.horizon_bars, 1)
    mid_step = (proj_md - spot) / max(cfg.horizon_bars, 1)

    path = np.zeros(cfg.horizon_bars)
    for i in range(cfg.horizon_bars):
        weight = (i + 1) / cfg.horizon_bars
        technical_level = spot + mid_step * (i + 1)
        band = (upper_step - lower_step) * weight * cfg.cone_smoothing
        path[i] = technical_level + band
    return path


def _blend_cones(field_path: np.ndarray, tech_path: np.ndarray, weight_tech: float, cfg: PredictionConeConfig) -> np.ndarray:
    """Blend field and technical cones."""

    weight_tech = float(np.clip(weight_tech, 0.0, 1.0))
    weight_field = 1.0 - weight_tech
    blended = weight_field * field_path + weight_tech * tech_path
    smoothed = np.convolve(blended, np.ones(3) / 3, mode="same")
    return smoothed


def build_prediction_cone(
    spot: float,
    directive: Any,
    ohlc: pd.DataFrame,
    hedge: Any,
    liquidity: Any,
    sentiment: Any,
    vol: Optional[Any],
    cone_cfg: PredictionConeConfig,
    donchian_weight: float = 0.25,
) -> PredictionCone:
    """Build a prediction cone for the provided symbol data."""

    close = ohlc["close"]

    annual_vol = _infer_annual_vol(vol_output=vol, close=close, cfg=cone_cfg)
    drift_bar = _field_drift_per_bar(directive, hedge, liquidity, sentiment, cone_cfg)
    vol_bar = _field_vol_per_bar(hedge, liquidity, sentiment, annual_vol, cone_cfg)

    field_path = _build_field_cone_path(
        spot=spot,
        drift_per_bar=drift_bar,
        vol_per_bar=vol_bar,
        cfg=cone_cfg,
    )

    don_snapshot, raw = compute_donchian_predictive_snapshot(
        ohlc=ohlc,
        config=None,
    )
    tech_path = _build_technical_cone_path_from_donchian(
        spot=spot,
        don_snapshot=don_snapshot,
        proj_hi=raw["proj_hi"],
        proj_md=raw["proj_md"],
        proj_lo=raw["proj_lo"],
        cfg=cone_cfg,
    )

    blended_path = _blend_cones(
        field_path=field_path,
        tech_path=tech_path,
        weight_tech=donchian_weight,
        cfg=cone_cfg,
    )

    meta = {
        "drift_per_bar": drift_bar,
        "vol_per_bar": vol_bar,
        "annual_vol": annual_vol,
        "donchian_weight": donchian_weight,
    }

    return PredictionCone(
        steps=cone_cfg.horizon_bars,
        field_path=field_path,
        tech_path=tech_path,
        blended_path=blended_path,
        metadata=meta,
    )
