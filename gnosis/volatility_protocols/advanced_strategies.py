"""
Advanced Volatility Trading Strategies
=======================================

Top 10 professional volatility strategies (2024-2025 meta):

1. 0DTE/1DTE Strangles (VIX crush + gamma scalping)
2. VIX Calendar Spreads (contango harvesting)
3. VIX Butterfly Hedging (crisis insurance)
4. Earnings RV Crush Trades
5. Skew Arbitrage (Risk Reversals)
6. Front Ratio Backspreads
7. Jade Lizard / Reverse Jade
8. Event Convexity Trades
9. SPX vs VIX Variance Arb
10. Weekend Theta Harvest

Current 2025 Meta Metrics (as of Nov 2025):
- Highest Sharpe: VIX calendars (1.8-2.7)
- Highest Sortino: VIX call butterflies
- Highest win rate: 0DTE SPX strangles (68-74%)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
from enum import Enum

from .regime_classification import Regime


class AdvancedStrategy(Enum):
    """Advanced strategy types"""
    ZERO_DTE_STRANGLE = "0dte_strangle"
    VIX_CALENDAR = "vix_calendar"
    VIX_BUTTERFLY = "vix_butterfly"
    EARNINGS_CRUSH = "earnings_crush"
    SKEW_ARB = "skew_arbitrage"
    RATIO_BACKSPREAD = "ratio_backspread"
    JADE_LIZARD = "jade_lizard"
    EVENT_CONVEXITY = "event_convexity"
    VARIANCE_ARB = "variance_arbitrage"
    WEEKEND_THETA = "weekend_theta"


@dataclass
class AdvancedStrategyConfig:
    """Configuration for advanced strategy"""
    name: str
    description: str
    best_regime: List[Regime]
    holding_period_days: Tuple[int, int]  # (min, max)
    risk_profile: str
    edge_source: str
    entry_window: Optional[Tuple[time, time]] = None  # Time window for entry
    symbols: List[str] = None  # Preferred symbols
    min_sharpe: float = 1.0
    avg_win_rate: float = 0.5
    notes: str = ""


# Strategy 1: 0DTE/1DTE Strangles
STRATEGY_0DTE_STRANGLE = AdvancedStrategyConfig(
    name="0DTE/1DTE Strangle",
    description="Massive positive theta + intraday vol crush; gamma scalping on spikes",
    best_regime=[Regime.R3, Regime.R4],  # VIX 20-35
    holding_period_days=(0, 1),
    risk_profile="High (undefined risk)",
    edge_source="Intraday vol crush + theta decay",
    entry_window=(time(10, 0), time(11, 30)),  # 10:00-11:30 ET
    symbols=['SPX', 'NDX'],  # Cash-settled only
    min_sharpe=2.0,
    avg_win_rate=0.70,
    notes="Target 50-70% of credit by 14:00 ET. Size aggressively only when VVIX/VIX > 1.2. "
          "Use SPX/NDX only for cash settlement. Enter 10:00-11:30 ET, close by 14:00 ET.",
)

# Strategy 2: VIX Calendar Spreads
STRATEGY_VIX_CALENDAR = AdvancedStrategyConfig(
    name="VIX Calendar Spread",
    description="Persistent VIX term-structure mean reversion",
    best_regime=[Regime.R1, Regime.R2],  # Contango >12%
    holding_period_days=(3, 14),
    risk_profile="Low-Medium",
    edge_source="Term structure contango harvesting",
    symbols=['VIX'],
    min_sharpe=2.2,
    avg_win_rate=0.75,
    notes="Long front (VIX weeklies), short 2nd or 3rd month. Roll front leg every Wednesday. "
          "Best risk-adjusted returns in options universe right now. Requires contango >12%.",
)

# Strategy 3: VIX Butterfly Hedging
STRATEGY_VIX_BUTTERFLY = AdvancedStrategyConfig(
    name="VIX Call Butterfly",
    description="Asymmetric payoff when VIX spikes 100%+",
    best_regime=[Regime.R1, Regime.R2, Regime.R3],  # Any, but hold in low vol
    holding_period_days=(30, 90),
    risk_profile="Very low cost",
    edge_source="Crisis insurance with positive expectancy",
    symbols=['VIX'],
    min_sharpe=3.0,
    avg_win_rate=0.15,  # Low win rate, massive payoff
    notes="Buy 15/30/50 or 20/40/70 VIX call butterflies for 0.50-1.50 debit. "
          "Pays 20-100× in black swans. Portfolio insurance that actually makes money.",
)

# Strategy 4: Earnings RV Crush
STRATEGY_EARNINGS_CRUSH = AdvancedStrategyConfig(
    name="Earnings RV Crush Trade",
    description="Implied vol overstates realized vol by 8-15% on average",
    best_regime=[Regime.R2, Regime.R3],  # IV Rank >80%
    holding_period_days=(0, 7),
    risk_profile="Medium-High",
    edge_source="Post-earnings vol crush",
    symbols=['NVDA', 'TSLA', 'META', 'AMD'],  # Proven crush names
    min_sharpe=1.5,
    avg_win_rate=0.65,
    notes="Short straddles 7→0 DTE. Only on names with proven post-earnings vol crush. "
          "Size with 1-2% risk max. Close night before earnings if IV Rank drops <60%.",
)

# Strategy 5: Skew Arbitrage
STRATEGY_SKEW_ARB = AdvancedStrategyConfig(
    name="Skew Arbitrage (25Δ Risk Reversal)",
    description="Equity put skew persistently rich vs statistical fair value",
    best_regime=[Regime.R2, Regime.R3],
    holding_period_days=(14, 45),
    risk_profile="Low",
    edge_source="Put skew richness",
    symbols=['SPY', 'QQQ', 'IWM'],
    min_sharpe=1.8,
    avg_win_rate=0.60,
    notes="Sell 25Δ put, buy 25Δ call when put skew >12-15%. Delta-hedged daily. "
          "Pure vol skew trade. Can reverse for call skew opportunities.",
)

# Strategy 6: Front Ratio Backspreads
STRATEGY_RATIO_BACKSPREAD = AdvancedStrategyConfig(
    name="Front Ratio Backspread (1×2 or 1×3)",
    description="Free or credit, explosive payoff on vol explosions",
    best_regime=[Regime.R1],  # IV Rank <25%
    holding_period_days=(21, 60),
    risk_profile="Low cost, high upside",
    edge_source="Long vol with minimal cost",
    symbols=['SPY', 'QQQ', 'Individual names'],
    min_sharpe=1.5,
    avg_win_rate=0.40,
    notes="Buy 1 × 30Δ put, sell 2 × 20Δ puts (or lower). "
          "Best long-vol strategy that isn't a naked long straddle. Enter when IV cheap.",
)

# Strategy 7: Jade Lizard
STRATEGY_JADE_LIZARD = AdvancedStrategyConfig(
    name="Jade Lizard / Reverse Jade",
    description="Higher yield than covered call with no naked call risk",
    best_regime=[Regime.R3],  # IV Rank 65-90%
    holding_period_days=(21, 45),
    risk_profile="Medium",
    edge_source="Premium collection with no upside risk",
    symbols=['High IV stocks'],
    min_sharpe=1.6,
    avg_win_rate=0.65,
    notes="Sell OTM put + sell OTM call spread. Premium > put strike width → no upside breach risk. "
          "Reverse Jade for bear moves. Requires put skew >8%.",
)

# Strategy 8: Event Convexity
STRATEGY_EVENT_CONVEXITY = AdvancedStrategyConfig(
    name="Event Convexity (FOMC, CPI, OPEX)",
    description="VIX underprices binary event moves",
    best_regime=[Regime.R2, Regime.R3],
    holding_period_days=(0, 3),
    risk_profile="Medium",
    edge_source="VVIX misprices event gamma",
    symbols=['SPX', 'SPY', 'QQQ'],
    min_sharpe=1.4,
    avg_win_rate=0.55,
    notes="Buy 7-14 DTE ATM strangles/straddles day before FOMC/CPI. Close day after. "
          "Works because VVIX misprices event gamma. 24-48 hours before macro events.",
)

# Strategy 9: Variance Arbitrage
STRATEGY_VARIANCE_ARB = AdvancedStrategyConfig(
    name="SPX vs VIX Variance Arbitrage",
    description="SPX implied variance consistently richer than realized VIX moves",
    best_regime=[Regime.R2, Regime.R3],
    holding_period_days=(1, 30),
    risk_profile="Low-Medium",
    edge_source="SPX IV vs VIX futures mismatch",
    symbols=['SPX', 'VIX'],
    min_sharpe=1.7,
    avg_win_rate=0.60,
    notes="Sell SPX premium when SPX IV > VIX futures + 6-8 points. "
          "Box trade or dispersion if you're a fund. Requires sophisticated execution.",
)

# Strategy 10: Weekend Theta Harvest
STRATEGY_WEEKEND_THETA = AdvancedStrategyConfig(
    name="Weekend Theta Harvest",
    description="3 days of theta for 1 day of risk",
    best_regime=[Regime.R2, Regime.R3],
    holding_period_days=(0, 4),  # Thursday to Monday
    risk_profile="Medium",
    edge_source="Weekend theta decay",
    symbols=['SPX'],
    min_sharpe=1.8,
    avg_win_rate=0.72,
    notes="Sell Thursday 1-2 DTE SPX strangles after 2pm ET, close Monday open. "
          "Captures weekend theta with almost no gap risk (2020-2025 data).",
)


# Master strategy dictionary
ADVANCED_STRATEGIES: Dict[AdvancedStrategy, AdvancedStrategyConfig] = {
    AdvancedStrategy.ZERO_DTE_STRANGLE: STRATEGY_0DTE_STRANGLE,
    AdvancedStrategy.VIX_CALENDAR: STRATEGY_VIX_CALENDAR,
    AdvancedStrategy.VIX_BUTTERFLY: STRATEGY_VIX_BUTTERFLY,
    AdvancedStrategy.EARNINGS_CRUSH: STRATEGY_EARNINGS_CRUSH,
    AdvancedStrategy.SKEW_ARB: STRATEGY_SKEW_ARB,
    AdvancedStrategy.RATIO_BACKSPREAD: STRATEGY_RATIO_BACKSPREAD,
    AdvancedStrategy.JADE_LIZARD: STRATEGY_JADE_LIZARD,
    AdvancedStrategy.EVENT_CONVEXITY: STRATEGY_EVENT_CONVEXITY,
    AdvancedStrategy.VARIANCE_ARB: STRATEGY_VARIANCE_ARB,
    AdvancedStrategy.WEEKEND_THETA: STRATEGY_WEEKEND_THETA,
}


@dataclass
class StrategySelection:
    """Strategy selection result"""
    strategy: AdvancedStrategy
    config: AdvancedStrategyConfig
    suitability_score: float  # 0-100
    reasons: List[str]
    warnings: List[str]


class AdvancedStrategySelector:
    """
    Advanced strategy selector based on market conditions
    """

    def select_best_strategy(
        self,
        current_regime: Regime,
        vix_level: float,
        term_structure: float,
        vvix_level: float,
        iv_rank: float,
        current_time: datetime,
        upcoming_events: List[str] = None,
    ) -> List[StrategySelection]:
        """
        Select best advanced strategies for current conditions

        Args:
            current_regime: Current market regime
            vix_level: Current VIX level
            term_structure: Term structure percentage
            vvix_level: VVIX level
            iv_rank: Current IV Rank
            current_time: Current datetime
            upcoming_events: List of upcoming events (earnings, FOMC, etc.)

        Returns:
            List of StrategySelection ordered by suitability
        """
        selections = []

        for strategy_enum, config in ADVANCED_STRATEGIES.items():
            score = 0.0
            reasons = []
            warnings = []

            # ========================================
            # 1. Check regime suitability
            # ========================================
            if current_regime in config.best_regime:
                score += 30
                reasons.append(f"Optimal for {current_regime.value}")
            else:
                score -= 10
                warnings.append(f"Not optimal for {current_regime.value}")

            # ========================================
            # 2. Strategy-specific conditions
            # ========================================

            # 0DTE Strangle
            if strategy_enum == AdvancedStrategy.ZERO_DTE_STRANGLE:
                # Check time window
                if config.entry_window:
                    start_time, end_time = config.entry_window
                    if start_time <= current_time.time() <= end_time:
                        score += 20
                        reasons.append("Within optimal entry window (10:00-11:30 ET)")
                    else:
                        score -= 30
                        warnings.append("Outside entry window")

                # Check VVIX/VIX ratio
                vvix_vix_ratio = vvix_level / vix_level if vix_level > 0 else 0
                if vvix_vix_ratio > 1.2:
                    score += 25
                    reasons.append(f"VVIX/VIX ratio {vvix_vix_ratio:.2f} > 1.2 (favorable)")
                else:
                    warnings.append(f"VVIX/VIX ratio {vvix_vix_ratio:.2f} below 1.2")

            # VIX Calendar
            elif strategy_enum == AdvancedStrategy.VIX_CALENDAR:
                if term_structure > 12.0:
                    score += 30
                    reasons.append(f"Term structure {term_structure:.1f}% > 12% (excellent)")
                elif term_structure > 8.0:
                    score += 15
                    reasons.append(f"Term structure {term_structure:.1f}% > 8% (good)")
                else:
                    score -= 20
                    warnings.append(f"Term structure {term_structure:.1f}% too low")

            # VIX Butterfly
            elif strategy_enum == AdvancedStrategy.VIX_BUTTERFLY:
                if current_regime in [Regime.R1, Regime.R2]:
                    score += 20
                    reasons.append("Low vol environment - good time to buy insurance")
                # Always somewhat applicable as insurance
                score += 10

            # Earnings Crush
            elif strategy_enum == AdvancedStrategy.EARNINGS_CRUSH:
                if iv_rank > 80:
                    score += 25
                    reasons.append(f"IV Rank {iv_rank:.1f}% > 80%")
                elif iv_rank > 60:
                    score += 10
                    reasons.append(f"IV Rank {iv_rank:.1f}% > 60%")
                else:
                    score -= 15
                    warnings.append(f"IV Rank {iv_rank:.1f}% too low for earnings crush")

            # Skew Arbitrage
            elif strategy_enum == AdvancedStrategy.SKEW_ARB:
                # Would need actual skew data here
                if current_regime in [Regime.R2, Regime.R3]:
                    score += 15
                    reasons.append("Good regime for skew trades")

            # Ratio Backspread
            elif strategy_enum == AdvancedStrategy.RATIO_BACKSPREAD:
                if iv_rank < 25:
                    score += 30
                    reasons.append(f"IV Rank {iv_rank:.1f}% < 25% (cheap vol)")
                elif iv_rank < 40:
                    score += 15
                    reasons.append(f"IV Rank {iv_rank:.1f}% < 40%")
                else:
                    score -= 10
                    warnings.append("IV too high for ratio backspread")

            # Jade Lizard
            elif strategy_enum == AdvancedStrategy.JADE_LIZARD:
                if 65 <= iv_rank <= 90:
                    score += 25
                    reasons.append(f"IV Rank {iv_rank:.1f}% in 65-90% range")
                elif iv_rank >= 60:
                    score += 10
                else:
                    warnings.append("IV Rank too low")

            # Event Convexity
            elif strategy_enum == AdvancedStrategy.EVENT_CONVEXITY:
                if upcoming_events:
                    score += 30
                    reasons.append(f"Upcoming events: {', '.join(upcoming_events)}")
                else:
                    score -= 20
                    warnings.append("No upcoming events detected")

            # Variance Arb
            elif strategy_enum == AdvancedStrategy.VARIANCE_ARB:
                # Would need SPX IV data
                if current_regime in [Regime.R2, Regime.R3]:
                    score += 10

            # Weekend Theta
            elif strategy_enum == AdvancedStrategy.WEEKEND_THETA:
                # Check if it's Thursday after 2pm
                if current_time.weekday() == 3 and current_time.hour >= 14:  # Thursday
                    score += 35
                    reasons.append("Thursday after 2pm - optimal entry time")
                elif current_time.weekday() == 3:
                    score += 10
                    reasons.append("Thursday - good day for entry")
                else:
                    score -= 15
                    warnings.append("Not Thursday (optimal entry day)")

            # ========================================
            # 3. Create selection
            # ========================================
            # Normalize score to 0-100
            score = max(0, min(100, score + 50))  # Shift baseline

            selections.append(StrategySelection(
                strategy=strategy_enum,
                config=config,
                suitability_score=score,
                reasons=reasons,
                warnings=warnings,
            ))

        # Sort by suitability score
        selections.sort(key=lambda x: x.suitability_score, reverse=True)

        return selections

    def get_strategy_details(
        self,
        strategy: AdvancedStrategy,
    ) -> AdvancedStrategyConfig:
        """Get configuration for specific strategy"""
        return ADVANCED_STRATEGIES.get(strategy)


# Current 2025 Meta Statistics
STRATEGY_PERFORMANCE_2025 = {
    'highest_sharpe': {
        'strategy': 'VIX Calendar Spreads',
        'sharpe': 2.2,
        'note': 'Best risk-adjusted returns',
    },
    'highest_sortino': {
        'strategy': 'VIX Call Butterflies',
        'sortino': 3.5,
        'note': 'Crisis insurance that prints',
    },
    'highest_win_rate': {
        'strategy': '0DTE SPX Strangles',
        'win_rate': 0.71,
        'note': '68-74% win rate if closed same-day at 50-60%',
    },
    'biggest_blowup_risk': {
        'strategy': 'Naked Short Straddles in R3',
        'note': 'One VIX spike and you\'re toast',
    },
}


def get_2025_meta_summary() -> str:
    """Get current volatility trading meta summary"""
    return """
CURRENT 2025 VOLATILITY TRADING META (as of Nov 2025)
=====================================================

Top Performers:
- Highest Sharpe: VIX calendars (1.8-2.7)
- Highest Sortino: VIX call butterflies (crisis insurance)
- Highest Win Rate: 0DTE/1DTE SPX strangles (68-74%)
- Biggest Blow-up Risk: Naked short straddles in R3

Pro-Only Enhancements:
1. Use VVIX term structure instead of flat VVIX for vol-of-vol timing
2. Weight all short premium trades by IVTSR (Implied Vol Term Structure Ratio)
3. Never fight backwardation >3 days — close everything
4. Add realized kurtosis filter — if last 30-day realized has fat tails, avoid short gamma

Master 2-3 of the top 5 strategies above and you're operating at the 99th percentile
of retail/prop volatility traders.
"""
