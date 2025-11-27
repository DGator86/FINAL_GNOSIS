"""
Complete Volatility Trading Protocols Example
==============================================

Demonstrates the full workflow:
1. Edge Detection
2. Regime Classification
3. Entry Validation
4. Position Sizing
5. Exit Management
6. Psychological Guardrails
"""

from gnosis.volatility_protocols import (
    # Edge Detection
    calculate_vol_edge,
    calculate_skew,
    calculate_term_premium,

    # Regime
    RegimeClassifier,
    Regime,

    # Entry
    EntryValidator,
    EntryConditions,
    StrategyCategory,
    EventCalendar,
    DirectionalSignal,

    # Exit
    ExitManager,
    ExitConditions,
    ExitTrigger,
    ExitUrgency,

    # Position Sizing
    PositionSizer,
    PositionSizingInput,
    StrategyRiskProfile,
    GreekLimits,

    # Advanced Strategies
    AdvancedStrategySelector,
    AdvancedStrategy,

    # Psychology
    PsychologicalGuardrails,
    get_all_demons,
)

from datetime import datetime, timedelta
import numpy as np


def example_complete_workflow():
    """Complete workflow from analysis to execution"""

    print("=" * 80)
    print("GNOSIS VOLATILITY TRADING PROTOCOLS - COMPLETE EXAMPLE")
    print("=" * 80)

    # ========================================
    # STEP 1: MARKET DATA
    # ========================================
    print("\n📊 STEP 1: Market Data Collection")
    print("-" * 80)

    # Simulated market data
    market_data = {
        'symbol': 'SPY',
        'price': 450.00,
        'iv_current': 28.0,
        'rv_20day': 23.0,
        'iv_252_low': 15.0,
        'iv_252_high': 45.0,
        'vix': 25.0,
        'term_structure': 8.0,  # Contango
        'vvix': 115.0,
        'atm_iv': 28.0,
        'put_25delta_iv': 31.0,
        'call_25delta_iv': 26.5,
    }

    print(f"Symbol: {market_data['symbol']}")
    print(f"Price: ${market_data['price']:.2f}")
    print(f"VIX: {market_data['vix']:.2f}")
    print(f"Current IV: {market_data['iv_current']:.1f}%")

    # ========================================
    # STEP 2: EDGE DETECTION
    # ========================================
    print("\n🎯 STEP 2: Edge Detection")
    print("-" * 80)

    # Calculate Vol Edge
    vol_edge = calculate_vol_edge(
        iv_current=market_data['iv_current'],
        rv_20day=market_data['rv_20day'],
        iv_252_low=market_data['iv_252_low'],
        iv_252_high=market_data['iv_252_high'],
    )

    print(f"\nVol Edge Analysis:")
    print(f"  Vol Edge: {vol_edge.vol_edge:.2f}%")
    print(f"  IV Rank: {vol_edge.iv_rank:.1f}%")
    print(f"  IV Current: {vol_edge.iv_current:.1f}%")
    print(f"  RV 20-day: {vol_edge.rv_20day:.1f}%")

    # Check thresholds
    if vol_edge.meets_threshold('short_vol'):
        print(f"  ✓ Meets SHORT VOL threshold (>{15}%)")
    if vol_edge.in_optimal_range('short_vol'):
        print(f"  ✓ In OPTIMAL range for short vol (20-40%)")

    # Calculate Skew
    skew = calculate_skew(
        atm_iv=market_data['atm_iv'],
        put_25delta_iv=market_data['put_25delta_iv'],
        call_25delta_iv=market_data['call_25delta_iv'],
    )

    print(f"\nSkew Analysis:")
    print(f"  Put Skew: {skew.put_skew:.2f}%")
    print(f"  Call Skew: {skew.call_skew:.2f}%")

    if skew.jade_lizard_favorable():
        print(f"  ✓ Jade Lizard favorable (put skew >8%)")

    # ========================================
    # STEP 3: REGIME CLASSIFICATION
    # ========================================
    print("\n🌡️  STEP 3: Regime Classification")
    print("-" * 80)

    classifier = RegimeClassifier()
    regime = classifier.classify(
        vix_level=market_data['vix'],
        term_structure=market_data['term_structure'],
        vvix_level=market_data['vvix'],
    )

    print(f"\nRegime Classification:")
    print(f"  Regime: {regime.regime.value} ({regime.regime.name})")
    print(f"  Stability: {regime.stability_days} days")
    print(f"  Transition Risk: {regime.transition_risk:.1f}%")
    print(f"  Allows Short Vol: {regime.allows_short_vol}")
    print(f"  Favors Long Vol: {regime.favors_long_vol}")

    # Check for exit signals
    exit_signal = classifier.get_forced_exit_signal(regime)
    if exit_signal:
        print(f"\n⚠️  REGIME EXIT SIGNAL:")
        print(f"  Urgency: {exit_signal['urgency']}")
        print(f"  Action: {exit_signal['action']}")

    # ========================================
    # STEP 4: ADVANCED STRATEGY SELECTION
    # ========================================
    print("\n🎲 STEP 4: Advanced Strategy Selection")
    print("-" * 80)

    strategy_selector = AdvancedStrategySelector()
    strategies = strategy_selector.select_best_strategy(
        current_regime=regime.regime,
        vix_level=market_data['vix'],
        term_structure=market_data['term_structure'],
        vvix_level=market_data['vvix'],
        iv_rank=vol_edge.iv_rank,
        current_time=datetime.now(),
    )

    print(f"\nTop 5 Recommended Strategies:")
    for i, selection in enumerate(strategies[:5], 1):
        print(f"\n{i}. {selection.config.name} (Score: {selection.suitability_score:.0f}/100)")
        print(f"   Edge: {selection.config.edge_source}")
        print(f"   Holding Period: {selection.config.holding_period_days[0]}-{selection.config.holding_period_days[1]} days")
        if selection.reasons:
            print(f"   Reasons: {', '.join(selection.reasons[:2])}")

    # Select Iron Condor for this example
    selected_strategy = "Iron Condor"

    # ========================================
    # STEP 5: POSITION SIZING
    # ========================================
    print("\n💰 STEP 5: Position Sizing")
    print("-" * 80)

    account_value = 100000.0
    max_loss_per_contract = 200.0  # $5 wide spread - $3 credit = $2 loss × 100

    sizer = PositionSizer()

    sizing_input = PositionSizingInput(
        account_value=account_value,
        account_risk_budget_pct=0.03,  # 3% risk per trade
        max_loss_per_contract=max_loss_per_contract,
        strategy_risk_profile=StrategyRiskProfile.DEFINED_RISK,
        edge_confidence=0.8,  # Strong edge
        regime_stability=1.0,  # Regime is stable
        current_notional_exposure_pct=0.10,  # 10% already deployed
        projected_delta_per_contract=-5.0,
        projected_gamma_per_contract=-0.2,
        projected_vega_per_contract=-3.0,
        projected_theta_per_contract=8.0,
    )

    sizing_result = sizer.calculate(sizing_input)

    print(f"\nPosition Sizing Result:")
    print(f"  Contracts: {sizing_result.contracts}")
    print(f"  Total Risk: ${sizing_result.total_risk:,.2f}")
    print(f"  Risk %: {sizing_result.risk_pct_of_account:.2f}%")
    print(f"  Raw Contracts (before limits): {sizing_result.raw_contracts}")
    if sizing_result.limiting_factor != "none":
        print(f"  ⚠️  Limited by: {sizing_result.limiting_factor}")

    print(f"\nProjected Greeks (total position):")
    print(f"  Delta: {sizing_result.total_delta:.1f}")
    print(f"  Gamma: {sizing_result.total_gamma:.2f}")
    print(f"  Vega: {sizing_result.total_vega:.1f}")
    print(f"  Theta: ${sizing_result.total_theta:.2f}/day")

    # ========================================
    # STEP 6: ENTRY VALIDATION
    # ========================================
    print("\n✅ STEP 6: Entry Validation")
    print("-" * 80)

    # Build entry conditions
    entry_conditions = EntryConditions(
        vol_edge=vol_edge,
        skew=skew,
        regime=regime,
        spread_quality=2.5,  # 2.5% bid-ask spread
        open_interest=1500,
        daily_volume=800,
        asset_type='etf',
        strategy_category=StrategyCategory.NEUTRAL_VOL,
        strategy_name=selected_strategy,
        max_loss=sizing_result.total_risk,
        position_size=sizing_result.contracts,
        projected_delta=sizing_result.total_delta,
        projected_gamma=sizing_result.total_gamma,
        projected_vega=sizing_result.total_vega,
        projected_theta=sizing_result.total_theta,
        profit_target=sizing_result.total_risk * 0.5,  # 50% of max profit
        stop_loss=sizing_result.total_risk * 2.0,  # 2× credit
        time_exit_dte=21,
        regime_exit_trigger="VIX >30",
        account_risk_available=account_value * 0.05,  # 5% available
    )

    # Validate
    validator = EntryValidator()
    validation_result = validator.validate(entry_conditions)

    print(validation_result.get_summary())

    if validation_result.is_valid:
        print("\n🎉 ENTRY APPROVED - READY TO EXECUTE")
        print(f"\nTrade Summary:")
        print(f"  Strategy: {selected_strategy}")
        print(f"  Contracts: {sizing_result.contracts}")
        print(f"  Max Risk: ${sizing_result.total_risk:,.2f}")
        print(f"  Profit Target: ${entry_conditions.profit_target:,.2f}")
        print(f"  Stop Loss: ${entry_conditions.stop_loss:,.2f}")
    else:
        print("\n❌ ENTRY REJECTED - DO NOT TRADE")
        return

    # ========================================
    # STEP 7: SIMULATED POSITION MONITORING
    # ========================================
    print("\n\n📈 STEP 7: Position Monitoring & Exit Management")
    print("-" * 80)

    # Simulate position after 10 days
    print("\n[10 days later - position update]")

    # Position is up 50%
    current_price = 1.50  # Sold for $3.00, now worth $1.50
    current_pnl = (3.00 - current_price) * 100 * sizing_result.contracts
    current_pnl_pct = ((3.00 - current_price) / 3.00) * 100

    exit_conditions = ExitConditions(
        strategy_name=selected_strategy,
        is_credit_strategy=True,
        entry_price=3.00,
        current_price=current_price,
        entry_date=datetime.now() - timedelta(days=10),
        current_dte=25,
        current_pnl=current_pnl,
        current_pnl_pct=current_pnl_pct,
        entry_regime=Regime.R3,
        current_regime=Regime.R3,
        current_delta=-4.0 * sizing_result.contracts,
        current_gamma=-0.15 * sizing_result.contracts,
        current_vega=-2.0 * sizing_result.contracts,
        current_theta=6.0 * sizing_result.contracts,
        entry_iv_rank=vol_edge.iv_rank,
        current_iv_rank=65.0,  # Dropped from 75 to 65
        current_vol_edge=15.0,  # Still positive
    )

    exit_manager = ExitManager()
    exit_signals = exit_manager.evaluate_exit(exit_conditions)

    print(f"\nCurrent Position Status:")
    print(f"  P&L: ${current_pnl:,.2f} ({current_pnl_pct:.1f}%)")
    print(f"  DTE: {exit_conditions.current_dte}")
    print(f"  Current Price: ${current_price:.2f} (Entry: ${exit_conditions.entry_price:.2f})")

    if exit_signals:
        print(f"\n🔔 EXIT SIGNALS DETECTED ({len(exit_signals)}):")
        for signal in exit_signals:
            urgency_emoji = {
                ExitUrgency.ROUTINE: "📋",
                ExitUrgency.URGENT: "⚠️",
                ExitUrgency.IMMEDIATE: "🚨",
                ExitUrgency.EMERGENCY: "🆘",
            }.get(signal.urgency, "📋")

            print(f"\n{urgency_emoji} {signal.trigger.value.upper()}")
            print(f"   Urgency: {signal.urgency.value}")
            print(f"   Reason: {signal.reason}")
            print(f"   Action: {signal.recommended_action}")
    else:
        print("\nNo exit signals - continue monitoring")

    # ========================================
    # STEP 8: PSYCHOLOGICAL GUARDRAILS
    # ========================================
    print("\n\n🧠 STEP 8: Psychological Guardrails")
    print("-" * 80)

    guardrails = PsychologicalGuardrails()

    # Simulate win streak
    for _ in range(8):
        warning = guardrails.record_win()

    # Check status
    status = guardrails.get_psychological_status()

    print(f"\nPsychological Status:")
    print(f"  Can Trade: {status['can_trade']}")
    print(f"  Consecutive Wins: {status['consecutive_wins']}")
    print(f"  Consecutive Losses: {status['consecutive_losses']}")
    print(f"  Total Stop Losses: {status['total_stop_losses']}")

    if status['warnings']:
        print(f"\n⚠️  Active Warnings:")
        for warning in status['warnings']:
            print(f"  • {warning}")

    # Check regime paralysis
    should_act, message = guardrails.check_regime_paralysis(
        current_vix=market_data['vix'],
        has_short_vol_positions=True,
    )

    if should_act:
        print(f"\n🚨 REGIME ACTION REQUIRED:")
        print(f"  {message}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)

    print("\n✅ Successfully demonstrated:")
    print("  1. Edge detection (Vol Edge, IV Rank, Skew)")
    print("  2. Regime classification (R1-R5)")
    print("  3. Advanced strategy selection")
    print("  4. Risk-based position sizing")
    print("  5. Complete entry validation (12-point checklist)")
    print("  6. Systematic exit management")
    print("  7. Psychological guardrails")

    print("\n💡 Key Takeaways:")
    print("  • Math-based entry criteria (no gut feelings)")
    print("  • Regime-aware strategy selection")
    print("  • Systematic exit management (profit/loss/time/regime)")
    print("  • Psychological safeguards prevent common errors")
    print("  • Complete audit trail for every decision")

    print("\n" + "=" * 80)


def example_demon_fixes():
    """Demonstrate psychological demon fixes"""

    print("\n\n" + "=" * 80)
    print("PSYCHOLOGICAL DEMONS & FIXES")
    print("=" * 80)

    demons = get_all_demons()

    # Show top 3 most severe demons
    sorted_demons = sorted(
        demons.items(),
        key=lambda x: x[1].severity,
        reverse=True,
    )

    print("\nTop 3 Most Severe Demons:")

    for i, (demon_type, profile) in enumerate(sorted_demons[:3], 1):
        print(f"\n{i}. {profile.name} (Severity: {profile.severity}/10)")
        print(f"   Feels Like: {profile.what_it_feels_like[:100]}...")
        print(f"   Why It Kills: {profile.why_it_kills[:100]}...")
        print(f"   Fix: {profile.permanent_fix[:100]}...")


if __name__ == "__main__":
    # Run complete workflow
    example_complete_workflow()

    # Show demon fixes
    example_demon_fixes()

    print("\n\n✨ For full documentation, see: docs/VOLATILITY_TRADING_PROTOCOLS.md")
