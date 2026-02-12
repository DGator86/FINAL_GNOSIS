"""
GNOSIS V2.0 Configuration - All adaptive parameters centralized
Feature flags enable safe, incremental deployment
"""

# ============================================================================
# MASTER FEATURE FLAGS
# ============================================================================

GNOSIS_V2_CONFIG = {
    # Core feature flags
    "enabled": True,  # Master switch for all V2 functionality (ENABLED)
    "shadow_mode": False,  # Execute live decisions
    "adaptive_regimes": True,  # Use percentile-based regime classification
    "correlation_greeks": True,  # Portfolio-level Greek aggregation
    "ml_adaptation": False,  # Disable learning initially
    "slippage_modeling": True,  # Enhanced execution cost modeling
    # Regime Classification (Adaptive)
    "regime_config": {
        "vix_history_days": 756,  # 3 years for percentile calculation
        "percentile_boundaries": {
            "R1": (0, 20),
            "R2": (20, 50),
            "R3": (50, 80),
            "R4": (80, 95),
            "R5": (95, 100),
        },
        "bayesian_weights": {
            "R1": {"garch": 0.60, "hv20": 0.30, "hv60": 0.10},
            "R2": {"garch": 0.60, "hv20": 0.30, "hv60": 0.10},
            "R3": {"garch": 0.40, "hv20": 0.40, "hv60": 0.20},
            "R4": {"garch": 0.20, "hv20": 0.60, "hv60": 0.20},
            "R5": {"garch": 0.20, "hv20": 0.60, "hv60": 0.20},
        },
    },
    # Liquidity & Execution (Fixed slippage model)
    "liquidity_config": {
        "liquidity_tiers": {
            "tier_1": {"max_spread_pct": 0.02, "min_volume": 100, "min_oi": 500},
            "tier_2": {"max_spread_pct": 0.05, "min_volume": 50, "min_oi": 200},
            "tier_3": {"max_spread_pct": 0.10, "min_volume": 20, "min_oi": 50},
            "tier_4": {"max_spread_pct": 0.50, "min_volume": 1, "min_oi": 1},
        },
        "slippage_multipliers": {"tier_1": 0.15, "tier_2": 0.25, "tier_3": 0.40, "tier_4": 0.60},
        "tradeable_thresholds": {
            "tier_1": {"min_volume": 10, "max_spread_pct": 0.05},
            "tier_2": {"min_volume": 5, "max_spread_pct": 0.10},
            "tier_3": {"min_volume": 2, "max_spread_pct": 0.20},
            "tier_4": {"min_volume": 1, "max_spread_pct": 0.50},
        },
    },
    # Risk Management (Portfolio-level)
    "risk_limits": {
        "max_vega_per_100k": {
            "R1": 150,
            "R2": 350,
            "R3": 450,
            "R4": 200,
            "R5": 0,  # No short vol in crisis
        },
        "max_gamma_per_100k": {"R1": 3, "R2": 5, "R3": 6, "R4": 10, "R5": 15},
        "concentration_limits": {"max_sector_vega_pct": 0.25, "max_single_position_pct": 0.05},
    },
    # ML Adaptation (Shadow mode with safety constraints)
    "ml_adaptation_config": {
        "shadow_mode": True,  # Critical: log but don't apply changes
        "min_trades_for_adaptation": 20,
        "adaptation_rate": 0.10,  # 10% adjustment per cycle
        "max_drift_constraints": {
            "threshold_adjustment_max": 0.20,  # ±20% per cycle
            "six_month_drift_max": 0.50,  # ±50% over 6 months
        },
        "target_win_rates": {"R1": 0.65, "R2": 0.68, "R3": 0.70, "R4": 0.60, "R5": 0.55},
    },
    # Intelligent Orchestrator (Phase 6)
    "orchestrator": {
        "enabled": True,  # Enable intelligent stock/option selection
        "low_iv_threshold": 20,  # Below this → prefer stocks
        "high_iv_threshold": 40,  # Above this → prefer spreads
        "spread_iv_rank_threshold": 50,  # Min IV rank for spreads
        "stock_confidence_threshold": 0.70,  # Min confidence for stocks
        "option_confidence_threshold": 0.55,  # Min confidence for options
        "max_option_allocation": 0.30,  # Max 30% portfolio in options
    },
}
