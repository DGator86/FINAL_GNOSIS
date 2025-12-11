"""Pydantic schemas for ML training examples.

These schemas define the structure of ML training samples
derived from trade decisions.
"""

from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel


class TradeMLExample(BaseModel):
    """
    A single ML training example derived from a TradeDecision row.

    This represents one labeled datapoint for ML training:
    - Inputs: flattened engine/agent features + universe metrics
    - Outputs: realized labels (PnL, R-multiple, hit_target, etc.)

    Use this for:
    - Supervised learning (predict expected_R, win_prob, etc.)
    - Policy learning (suggest trades given engine/agent state)
    """

    # ========== Identity ==========
    trade_id: str
    timestamp: datetime
    mode: str
    symbol: str
    direction: str
    structure: str
    config_version: str

    # ========== Core Numeric Features (flattened) ==========
    price: float
    adv: float
    iv_rank: float
    realized_vol_30d: float
    options_liq_score: float

    # ========== Engine & Agent Derived Features ==========
    # This dict contains flattened JSONB fields:
    # - dealer.gex_pressure_score
    # - liq.dark_pool_activity_score
    # - sentiment.sentiment_score
    # - agent_hedge.bias_long_vol
    # etc.
    features: Dict[str, Any]

    # ========== Labels ==========
    realized_return: float  # Percentage PnL on the trade
    r_multiple: float  # PnL / risk_per_trade
    max_drawdown_pct: float  # Worst intratrade drawdown
    hit_target: int  # 1 if target hit before stop/expiry, else 0
    stopped_out: int  # 1 if stop hit, else 0
    horizon_return: float  # PnL at fixed horizon regardless of exit
