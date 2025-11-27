"""
GNOSIS Volatility Trading Protocols
====================================

Complete implementation of precision entry and exit frameworks for volatility trading.

Modules:
--------
- edge_detection: Mathematical edge calculations (Vol Edge, IV Rank, Skew)
- regime_classification: Market regime detection (R1-R5)
- entry_protocols: Entry validation and execution framework
- exit_protocols: Systematic exit management
- position_sizing: Risk-based position sizing algorithms
- advanced_strategies: Advanced volatility trading strategies
- psychological_framework: Trading psychology management
"""

from .edge_detection import (
    calculate_vol_edge,
    calculate_iv_rank,
    calculate_skew,
    calculate_term_premium,
    VolEdgeScore,
)

from .regime_classification import (
    classify_regime,
    calculate_transition_risk,
    RegimeClassifier,
    Regime,
)

from .entry_protocols import (
    validate_entry,
    EntryValidator,
    EntryConditions,
)

from .exit_protocols import (
    calculate_exit_signal,
    ExitManager,
    ExitConditions,
)

from .position_sizing import (
    calculate_position_size,
    PositionSizer,
    GreekLimits,
)

from .advanced_strategies import (
    AdvancedStrategySelector,
    ADVANCED_STRATEGIES,
)

from .psychological_framework import (
    PsychologicalGuardrails,
    PSYCHOLOGICAL_DEMONS,
)

__all__ = [
    # Edge Detection
    'calculate_vol_edge',
    'calculate_iv_rank',
    'calculate_skew',
    'calculate_term_premium',
    'VolEdgeScore',

    # Regime Classification
    'classify_regime',
    'calculate_transition_risk',
    'RegimeClassifier',
    'Regime',

    # Entry Protocols
    'validate_entry',
    'EntryValidator',
    'EntryConditions',

    # Exit Protocols
    'calculate_exit_signal',
    'ExitManager',
    'ExitConditions',

    # Position Sizing
    'calculate_position_size',
    'PositionSizer',
    'GreekLimits',

    # Advanced Strategies
    'AdvancedStrategySelector',
    'ADVANCED_STRATEGIES',

    # Psychological Framework
    'PsychologicalGuardrails',
    'PSYCHOLOGICAL_DEMONS',
]

__version__ = '1.0.0'
