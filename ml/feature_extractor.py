"""Feature extraction from trade decisions.

This module flattens JSONB engine/agent snapshots into
ML-ready flat dictionaries.
"""

from typing import Any, Dict


def flatten_dict(
    data: Dict[str, Any],
    prefix: str = "",
    sep: str = ".",
    max_depth: int = 3,
) -> Dict[str, Any]:
    """
    Recursively flattens a nested dict into a single level dict with dotted keys.

    Example:
        {"a": {"b": 1}} -> {"a.b": 1}

    Args:
        data: Nested dictionary to flatten
        prefix: Prefix for all keys (used in recursion)
        sep: Separator for nested keys
        max_depth: Maximum recursion depth (prevents infinite loops)

    Returns:
        Flattened dictionary
    """
    flat: Dict[str, Any] = {}

    def _flatten(obj: Any, parent_key: str, depth: int) -> None:
        """Recursive flattening helper."""
        if depth > max_depth:
            # Max depth reached, store as-is
            flat[parent_key] = obj
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
                _flatten(v, new_key, depth + 1)
        else:
            # Leaf value, store it
            flat[parent_key] = obj

    _flatten(data, prefix, 0)
    return flat


def extract_features_from_trade(trade) -> Dict[str, Any]:
    """
    Extracts and flattens engine and agent JSON fields from a TradeDecision
    into a single feature dict.

    This produces keys like:
    - dealer.gex_pressure_score
    - liq.dark_pool_activity_score
    - sentiment.sentiment_score
    - agent_hedge.bias_long_vol
    - composer.reason_codes
    etc.

    Args:
        trade: TradeDecision ORM object

    Returns:
        Flattened feature dictionary suitable for ML training
    """
    features: Dict[str, Any] = {}

    # ========== Engines ==========
    features.update(
        flatten_dict(
            trade.dealer_features,
            prefix="dealer",
        )
    )
    features.update(
        flatten_dict(
            trade.liquidity_features,
            prefix="liq",
        )
    )
    features.update(
        flatten_dict(
            trade.sentiment_features,
            prefix="sentiment",
        )
    )

    # ========== Agents ==========
    features.update(
        flatten_dict(
            trade.hedge_agent_vote,
            prefix="agent_hedge",
        )
    )
    features.update(
        flatten_dict(
            trade.liquidity_agent_vote,
            prefix="agent_liq",
        )
    )
    features.update(
        flatten_dict(
            trade.sentiment_agent_vote,
            prefix="agent_sentiment",
        )
    )
    features.update(
        flatten_dict(
            trade.composer_decision,
            prefix="composer",
        )
    )

    # ========== Portfolio Context ==========
    # Optional: include portfolio context signals
    features.update(
        flatten_dict(
            trade.portfolio_context,
            prefix="portfolio",
        )
    )

    return features
