#!/usr/bin/env python3
"""
Example: How to log a trade decision from the GNOSIS pipeline.

This shows the integration pattern for logging trade decisions
with full context before submitting orders to the broker.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db import SessionLocal
from crud.trade_decision import create_trade_decision
from schemas.trade_decision import TradeDecisionCreate


def log_trade_decision_example():
    """
    Example of logging a trade decision from GNOSIS pipeline.

    In your actual pipeline, replace the hardcoded values with
    real data from your engines and agents.
    """
    # Build decision payload from GNOSIS pipeline state
    decision = TradeDecisionCreate(
        # Meta
        timestamp=datetime.utcnow(),
        mode="paper",  # or "live" or "backtest"
        symbol="SPY",
        direction="long",
        structure="call_spread",
        config_version="v1.0.0",

        # Universe / Filters
        universe_eligible=True,
        universe_reasons=["adv_ok", "min_price_ok", "options_liq_ok"],
        price=550.25,
        adv=85000000.0,
        iv_rank=0.65,
        realized_vol_30d=0.18,
        options_liq_score=0.85,

        # Engine Snapshots
        dealer_features={
            "gex_pressure_score": 0.72,
            "vanna_bias_score": 0.35,
            "charm_bias_score": -0.15,
            "spot_vs_gamma_pivot": 1.02,
            "gamma_wall_above": 555.0,
            "gamma_wall_below": 545.0,
            "dealer_gamma_sign": -1.0,
        },
        liquidity_features={
            "liquidity_stability_score": 0.88,
            "dark_pool_activity_score": 0.45,
            "recent_liquidity_shock_score": 0.12,
            "nearest_hvn": 550.0,
            "nearest_lvn": 552.5,
            "distance_to_vwap": 0.0025,
        },
        sentiment_features={
            "wyckoff_phase": "markup",
            "micro_regime": "bullish_trending",
            "macro_regime": "risk_on",
            "sentiment_score": 0.68,
            "risk_on_off_score": 0.75,
        },

        # Agent Votes
        hedge_agent_vote={
            "bias": "long_vol",
            "direction_bias": "up",
            "confidence": 0.78,
            "reasoning": "Positive GEX pressure with dealers short gamma",
        },
        liquidity_agent_vote={
            "zone": "breakout_zone",
            "confidence": 0.82,
            "reasoning": "Price approaching HVN resistance",
        },
        sentiment_agent_vote={
            "risk_posture": "aggressive",
            "trend_alignment": "with_trend",
            "confidence": 0.85,
            "reasoning": "Markup phase confirmed, risk-on regime",
        },

        # Composer Decision
        composer_decision={
            "final_direction": "long",
            "final_structure": "call_spread",
            "sizing_hint": 0.05,
            "invalidation_level": 548.0,
            "target_level": 556.0,
            "reason_codes": ["markup_phase", "positive_gex", "strong_liquidity_shelf"],
        },

        # Portfolio Context
        portfolio_context={
            "portfolio_gross_exposure_before": 0.45,
            "portfolio_gross_exposure_after": 0.50,
            "symbol_exposure_before": 0.0,
            "symbol_exposure_after": 0.05,
            "risk_per_trade_pct": 0.02,
            "max_drawdown_limit_active": False,
        },
    )

    # Log to database
    db = SessionLocal()
    try:
        trade_record = create_trade_decision(db, decision)
        trade_id = trade_record.id

        print("✓ Trade decision logged successfully")
        print(f"  Trade ID: {trade_id}")
        print(f"  Symbol: {trade_record.symbol}")
        print(f"  Direction: {trade_record.direction}")
        print(f"  Structure: {trade_record.structure}")
        print(f"  Timestamp: {trade_record.timestamp}")
        print()
        print("Next steps:")
        print(f"1. Submit order to broker")
        print(f"2. Update execution details: PATCH /trades/decisions/{trade_id}/execution")
        print()

        return trade_id

    finally:
        db.close()


def log_execution_update_example(trade_id: str):
    """
    Example of updating execution details after broker response.

    Call this after you get a response from the broker with
    order ID and fill details.
    """
    from crud.trade_decision import update_trade_execution
    from schemas.trade_decision import TradeDecisionUpdateExecution

    execution_update = TradeDecisionUpdateExecution(
        order_id="abc123",
        entry_price=550.30,
        target_price=556.00,
        stop_price=548.00,
        slippage_bps=2.5,
        status="filled",
    )

    db = SessionLocal()
    try:
        updated_record = update_trade_execution(db, trade_id, execution_update)
        if updated_record:
            print("✓ Execution details updated successfully")
            print(f"  Order ID: {updated_record.order_id}")
            print(f"  Entry Price: {updated_record.entry_price}")
            print(f"  Status: {updated_record.status}")
        else:
            print(f"✗ Trade decision {trade_id} not found")

    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("GNOSIS Trade Decision Logging Example")
    print("=" * 60)
    print()

    # Step 1: Log trade decision
    trade_id = log_trade_decision_example()

    # Step 2: Update execution details (simulated)
    print()
    print("Simulating broker response...")
    print()
    log_execution_update_example(trade_id)
