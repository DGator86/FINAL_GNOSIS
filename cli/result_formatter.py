"""Human-readable formatter for pipeline results."""

from typing import Any, Dict, List, Optional
from schemas.core_schemas import (
    PipelineResult,
    HedgeSnapshot,
    LiquiditySnapshot,
    SentimentSnapshot,
    ElasticitySnapshot,
    AgentSuggestion,
    TradeIdea,
    ExpectedMove,
    PriceRange,
    MTFAnalysis,
    TimeframeSignal,
)


def format_pipeline_result(result: PipelineResult) -> str:
    """Format a PipelineResult into human-readable output."""
    lines = []

    # Header
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  PIPELINE RESULT: {result.symbol}")
    lines.append(f"  Time: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("=" * 80)

    # Section 1: Engine Snapshots (the raw data)
    lines.append("")
    lines.append("-" * 80)
    lines.append("  ENGINES (Raw Market Analysis)")
    lines.append("-" * 80)

    if result.hedge_snapshot:
        lines.extend(_format_hedge_snapshot(result.hedge_snapshot))

    if result.liquidity_snapshot:
        lines.extend(_format_liquidity_snapshot(result.liquidity_snapshot))

    if result.sentiment_snapshot:
        lines.extend(_format_sentiment_snapshot(result.sentiment_snapshot))

    if result.elasticity_snapshot:
        lines.extend(_format_elasticity_snapshot(result.elasticity_snapshot))

    # Section 1b: Multi-Timeframe Analysis (if available)
    if result.mtf_analysis:
        lines.append("")
        lines.append("-" * 80)
        lines.append("  MULTI-TIMEFRAME ANALYSIS (All Timeframes)")
        lines.append("-" * 80)
        lines.extend(_format_mtf_analysis(result.mtf_analysis))

    # Section 2: Agent Suggestions (what each agent thinks)
    if result.suggestions:
        lines.append("")
        lines.append("-" * 80)
        lines.append("  PRIMARY AGENTS (Individual Opinions)")
        lines.append("-" * 80)
        for suggestion in result.suggestions:
            lines.extend(_format_suggestion(suggestion))

    # Section 3: Composer Consensus (the combined view)
    if result.consensus:
        lines.append("")
        lines.append("-" * 80)
        lines.append("  COMPOSER (Combined Consensus)")
        lines.append("-" * 80)
        lines.extend(_format_consensus(result.consensus))

    # Section 4: Trade Ideas (actionable recommendations)
    if result.trade_ideas:
        lines.append("")
        lines.append("-" * 80)
        lines.append("  TRADE AGENT (Actionable Ideas)")
        lines.append("-" * 80)
        for i, idea in enumerate(result.trade_ideas, 1):
            lines.extend(_format_trade_idea(idea, i))

    # Section 5: Order Results (what actually happened)
    if result.order_results:
        lines.append("")
        lines.append("-" * 80)
        lines.append("  ORDER EXECUTION")
        lines.append("-" * 80)
        for order in result.order_results:
            status = "FILLED" if order.filled else "PENDING"
            lines.append(f"  [{status}] {order.order_id}")
            if order.filled_price:
                lines.append(f"    Filled at: ${order.filled_price:.2f}")

    # Footer
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def _format_hedge_snapshot(snap: HedgeSnapshot) -> List[str]:
    """Format hedge engine snapshot with plain English explanations."""
    lines = []
    lines.append("")
    lines.append("  [HEDGE ENGINE] - Options Flow & Dealer Positioning")

    # Energy Asymmetry - the key directional signal
    energy = snap.energy_asymmetry
    if energy > 20:
        energy_meaning = "STRONG BULLISH pressure from options market"
    elif energy > 5:
        energy_meaning = "Moderate bullish tilt"
    elif energy < -20:
        energy_meaning = "STRONG BEARISH pressure from options market"
    elif energy < -5:
        energy_meaning = "Moderate bearish tilt"
    else:
        energy_meaning = "Balanced / no clear direction"

    lines.append(f"    Energy Asymmetry: {energy:+.1f}")
    lines.append(f"      -> {energy_meaning}")

    # Movement Energy - volatility expectation
    movement = snap.movement_energy
    if movement > 50:
        move_meaning = "HIGH volatility expected - big moves likely"
    elif movement > 25:
        move_meaning = "Elevated volatility - moderate moves expected"
    else:
        move_meaning = "Low volatility - quiet market expected"

    lines.append(f"    Movement Energy: {movement:.0f}")
    lines.append(f"      -> {move_meaning}")

    # Regime
    regime = snap.regime.upper()
    regime_explanations = {
        "TRENDING_UP": "Market in uptrend mode",
        "TRENDING_DOWN": "Market in downtrend mode",
        "MEAN_REVERTING": "Market choppy, expect pullbacks to reverse",
        "HIGH_VOLATILITY": "Unstable, expect large swings",
        "LOW_VOLATILITY": "Quiet, range-bound market",
        "NEUTRAL": "No clear regime detected",
    }
    regime_meaning = regime_explanations.get(regime, f"Regime: {regime}")

    lines.append(f"    Regime: {regime}")
    lines.append(f"      -> {regime_meaning}")

    # Dealer positioning
    gamma = snap.dealer_gamma_sign
    if gamma > 0.3:
        gamma_meaning = "Dealers are LONG gamma (will dampen moves)"
    elif gamma < -0.3:
        gamma_meaning = "Dealers are SHORT gamma (will amplify moves!)"
    else:
        gamma_meaning = "Dealers neutral"

    lines.append(f"    Dealer Gamma: {gamma:+.2f}")
    lines.append(f"      -> {gamma_meaning}")

    lines.append(f"    Confidence: {snap.confidence:.0%}")

    return lines


def _format_liquidity_snapshot(snap: LiquiditySnapshot) -> List[str]:
    """Format liquidity engine snapshot."""
    lines = []
    lines.append("")
    lines.append("  [LIQUIDITY ENGINE] - Market Depth & Trading Conditions")

    score = snap.liquidity_score
    if score > 0.8:
        liq_meaning = "EXCELLENT liquidity - easy to trade"
    elif score > 0.6:
        liq_meaning = "Good liquidity - normal conditions"
    elif score > 0.4:
        liq_meaning = "Fair liquidity - may see some slippage"
    else:
        liq_meaning = "POOR liquidity - be careful with size"

    lines.append(f"    Liquidity Score: {score:.2f}")
    lines.append(f"      -> {liq_meaning}")

    spread = snap.bid_ask_spread
    if spread > 0:
        if spread < 0.01:
            spread_meaning = "Tight spread - low cost to trade"
        elif spread < 0.05:
            spread_meaning = "Normal spread"
        else:
            spread_meaning = "Wide spread - higher trading costs"
        lines.append(f"    Bid-Ask Spread: {spread:.4f}")
        lines.append(f"      -> {spread_meaning}")

    if snap.volume > 0:
        lines.append(f"    Volume: {snap.volume:,.0f}")

    return lines


def _format_sentiment_snapshot(snap: SentimentSnapshot) -> List[str]:
    """Format sentiment engine snapshot."""
    lines = []
    lines.append("")
    lines.append("  [SENTIMENT ENGINE] - News, Flow & Technical Signals")

    score = snap.sentiment_score
    if score > 0.3:
        sent_meaning = "BULLISH sentiment across sources"
    elif score > 0.1:
        sent_meaning = "Slightly bullish lean"
    elif score < -0.3:
        sent_meaning = "BEARISH sentiment across sources"
    elif score < -0.1:
        sent_meaning = "Slightly bearish lean"
    else:
        sent_meaning = "Neutral / mixed sentiment"

    lines.append(f"    Overall Sentiment: {score:+.2f}")
    lines.append(f"      -> {sent_meaning}")

    # Component breakdown
    lines.append(f"    Components:")
    lines.append(f"      News:      {snap.news_sentiment:+.2f}")
    lines.append(f"      Flow:      {snap.flow_sentiment:+.2f}")
    lines.append(f"      Technical: {snap.technical_sentiment:+.2f}")

    lines.append(f"    Confidence: {snap.confidence:.0%}")

    return lines


def _format_elasticity_snapshot(snap: ElasticitySnapshot) -> List[str]:
    """Format elasticity engine snapshot."""
    lines = []
    lines.append("")
    lines.append("  [ELASTICITY ENGINE] - Volatility & Trend Analysis")

    vol = snap.volatility
    if vol > 0.3:
        vol_meaning = "HIGH volatility - expect big moves"
    elif vol > 0.15:
        vol_meaning = "Moderate volatility - normal conditions"
    else:
        vol_meaning = "Low volatility - quiet market"

    lines.append(f"    Volatility: {vol:.1%}")
    lines.append(f"      -> {vol_meaning}")

    regime = snap.volatility_regime.upper()
    lines.append(f"    Vol Regime: {regime}")

    trend = snap.trend_strength
    if abs(trend) > 0.5:
        trend_dir = "BULLISH" if trend > 0 else "BEARISH"
        trend_meaning = f"Strong {trend_dir} trend"
    elif abs(trend) > 0.2:
        trend_dir = "bullish" if trend > 0 else "bearish"
        trend_meaning = f"Moderate {trend_dir} trend"
    else:
        trend_meaning = "No clear trend / choppy"

    lines.append(f"    Trend Strength: {trend:+.2f}")
    lines.append(f"      -> {trend_meaning}")

    return lines


def _format_suggestion(suggestion: AgentSuggestion) -> List[str]:
    """Format an agent suggestion."""
    lines = []

    # Direction with visual indicator
    direction = suggestion.direction.value.upper()
    if direction == "LONG":
        arrow = "[BUY]"
    elif direction == "SHORT":
        arrow = "[SELL]"
    else:
        arrow = "[HOLD]"

    lines.append("")
    lines.append(f"  {suggestion.agent_name}")
    lines.append(f"    Direction: {arrow} {direction}")
    lines.append(f"    Confidence: {suggestion.confidence:.0%}")

    if suggestion.reasoning:
        # Wrap reasoning text
        reasoning = suggestion.reasoning[:200] + "..." if len(suggestion.reasoning) > 200 else suggestion.reasoning
        lines.append(f"    Reasoning: {reasoning}")

    if suggestion.target_allocation > 0:
        lines.append(f"    Suggested Size: {suggestion.target_allocation:.1%} of portfolio")

    return lines


def _format_consensus(consensus: Dict[str, Any]) -> List[str]:
    """Format the composer consensus."""
    lines = []
    lines.append("")

    direction = consensus.get("direction", "neutral").upper()
    confidence = consensus.get("confidence", 0.5)
    reasoning = consensus.get("reasoning", "")

    if direction == "LONG":
        verdict = "BUY SIGNAL"
    elif direction == "SHORT":
        verdict = "SELL SIGNAL"
    else:
        verdict = "NO CLEAR SIGNAL"

    strength = ""
    if confidence > 0.8:
        strength = "(HIGH confidence)"
    elif confidence > 0.6:
        strength = "(moderate confidence)"
    else:
        strength = "(low confidence)"

    lines.append(f"    VERDICT: {verdict} {strength}")
    lines.append(f"    Direction: {direction}")
    lines.append(f"    Confidence: {confidence:.0%}")

    if reasoning:
        lines.append(f"    Summary: {reasoning}")

    # Show component weights if available
    weights = consensus.get("weights", {})
    if weights:
        lines.append(f"    Agent Contributions:")
        for agent, weight in weights.items():
            lines.append(f"      {agent}: {weight:.0%}")

    return lines


def _format_trade_idea(idea: TradeIdea, num: int) -> List[str]:
    """Format a trade idea."""
    lines = []
    lines.append("")

    direction = idea.direction.value.upper()
    if direction == "LONG":
        action = "BUY"
    elif direction == "SHORT":
        action = "SELL"
    else:
        action = "HOLD"

    lines.append(f"  Trade #{num}: {action} {idea.symbol}")
    lines.append(f"    Strategy: {idea.strategy_type.value.replace('_', ' ').title()}")
    lines.append(f"    Confidence: {idea.confidence:.0%}")

    if idea.entry_price:
        lines.append(f"    Entry Price: ${idea.entry_price:.2f}")

    if idea.stop_loss:
        lines.append(f"    Stop Loss: ${idea.stop_loss:.2f}")

    if idea.take_profit:
        lines.append(f"    Take Profit: ${idea.take_profit:.2f}")

    if idea.size > 0:
        lines.append(f"    Position Size: {idea.size:.1%} of portfolio")

    if idea.reasoning:
        reasoning = idea.reasoning[:200] + "..." if len(idea.reasoning) > 200 else idea.reasoning
        lines.append(f"    Rationale: {reasoning}")

    # Expected Move Range (Industry Standard)
    if idea.expected_move:
        lines.extend(_format_expected_move(idea.expected_move, idea.entry_price))

    # Options details if present
    if idea.options_request:
        opt = idea.options_request
        lines.append(f"    Options Order:")
        lines.append(f"      {opt.side.upper()} {opt.quantity}x {opt.option_type.upper()}")
        if hasattr(opt, 'strike') and opt.strike:
            lines.append(f"      Strike: ${opt.strike:.2f}")
        if hasattr(opt, 'expiration') and opt.expiration:
            lines.append(f"      Expiry: {opt.expiration}")

    return lines


def _format_expected_move(em: ExpectedMove, entry_price: Optional[float] = None) -> List[str]:
    """
    Format expected move with industry-standard probability ranges.

    Displays the same information professional traders see:
    - 1σ (68% probability): Most likely range
    - 2σ (95% probability): Extended range
    - 3σ (99.7% probability): Extreme/tail risk range
    """
    lines = []
    lines.append("")
    lines.append("    ┌─────────────────────────────────────────────────────────┐")
    lines.append("    │  EXPECTED PRICE MOVEMENT (Industry Standard)            │")
    lines.append("    └─────────────────────────────────────────────────────────┘")

    # Method and timeframe
    method_names = {
        "iv_based": "Implied Volatility (Options Market)",
        "atr_based": "Average True Range (Historical)",
        "regime_estimate": "Volatility Regime Estimate",
        "hybrid": "Hybrid (IV + ATR)",
    }
    method = method_names.get(em.calculation_method, em.calculation_method)
    lines.append(f"    Method: {method}")
    lines.append(f"    Timeframe: {em.timeframe}")

    # IV metrics if available
    if em.implied_volatility:
        lines.append(f"    Implied Volatility: {em.implied_volatility:.1%} annualized")
    if em.historical_volatility:
        lines.append(f"    Historical Volatility: {em.historical_volatility:.1%}")
    if em.iv_rank is not None:
        iv_rank_meaning = ""
        if em.iv_rank > 80:
            iv_rank_meaning = " (HIGH - options expensive)"
        elif em.iv_rank > 50:
            iv_rank_meaning = " (elevated)"
        elif em.iv_rank < 20:
            iv_rank_meaning = " (LOW - options cheap)"
        lines.append(f"    IV Rank: {em.iv_rank:.0f}th percentile{iv_rank_meaning}")

    lines.append("")
    lines.append("    PROBABILITY RANGES:")

    # 1-Sigma (68% probability)
    if em.one_sigma:
        lines.append("")
        lines.append("    68% Probability (1-sigma, most likely):")
        lines.append(f"      Range: ${em.one_sigma.lower:.2f} - ${em.one_sigma.upper:.2f}")
        move_pct = em.expected_move_pct if em.expected_move_pct else 0
        lines.append(f"      Expected Move: +/- {move_pct:.2f}%")
        lines.append(f"        -> Price will likely stay within this range")

    # 2-Sigma (95% probability)
    if em.two_sigma:
        lines.append("")
        lines.append("    95% Probability (2-sigma, extended range):")
        lines.append(f"      Range: ${em.two_sigma.lower:.2f} - ${em.two_sigma.upper:.2f}")
        if em.expected_move_pct:
            lines.append(f"      Expected Move: +/- {em.expected_move_pct * 2:.2f}%")
        lines.append(f"        -> Only 5% chance of moving outside this range")

    # 3-Sigma (99.7% probability)
    if em.three_sigma:
        lines.append("")
        lines.append("    99.7% Probability (3-sigma, tail risk):")
        lines.append(f"      Range: ${em.three_sigma.lower:.2f} - ${em.three_sigma.upper:.2f}")
        if em.expected_move_pct:
            lines.append(f"      Expected Move: +/- {em.expected_move_pct * 3:.2f}%")
        lines.append(f"        -> Extreme move territory (black swan)")

    # Directional probabilities
    if em.upside_probability and em.downside_probability:
        lines.append("")
        lines.append("    DIRECTIONAL BIAS:")
        if em.upside_probability > 0.55:
            lines.append(f"      Upside favored: {em.upside_probability:.0%} probability of moving higher")
        elif em.downside_probability > 0.55:
            lines.append(f"      Downside favored: {em.downside_probability:.0%} probability of moving lower")
        else:
            lines.append(f"      Neutral: {em.upside_probability:.0%} up / {em.downside_probability:.0%} down")

    # Visual price ladder
    if em.one_sigma and entry_price:
        lines.append("")
        lines.append("    PRICE LADDER:")
        lines.append(_create_price_ladder(em, entry_price))

    return lines


def _create_price_ladder(em: ExpectedMove, entry_price: float) -> str:
    """Create a visual ASCII price ladder showing probability zones."""
    lines = []

    # Build ladder from highest to lowest price
    prices = []

    if em.three_sigma:
        prices.append((em.three_sigma.upper, "3σ Upper (99.7%)", "---"))
    if em.two_sigma:
        prices.append((em.two_sigma.upper, "2σ Upper (95%)", "==="))
    if em.one_sigma:
        prices.append((em.one_sigma.upper, "1σ Upper (68%)", "###"))

    prices.append((entry_price, "ENTRY", ">>>"))

    if em.one_sigma:
        prices.append((em.one_sigma.lower, "1σ Lower (68%)", "###"))
    if em.two_sigma:
        prices.append((em.two_sigma.lower, "2σ Lower (95%)", "==="))
    if em.three_sigma:
        prices.append((em.three_sigma.lower, "3σ Lower (99.7%)", "---"))

    # Sort by price descending
    prices.sort(key=lambda x: x[0], reverse=True)

    # Create visual
    result = []
    for price, label, marker in prices:
        pct_from_entry = ((price - entry_price) / entry_price) * 100
        pct_str = f"{pct_from_entry:+.1f}%" if price != entry_price else "0.0%"
        result.append(f"      ${price:>8.2f} {marker} {label:<20} ({pct_str})")

    return "\n".join(result)


def _format_mtf_analysis(mtf: MTFAnalysis) -> List[str]:
    """
    Format multi-timeframe analysis showing each timeframe's signal.

    This is the key feature showing different determinations per timeframe:
    - 5m might show neutral
    - 1D might show long
    - etc.
    """
    lines = []
    lines.append("")

    # Overall summary
    overall_dir = mtf.overall_direction.upper()
    if overall_dir == "LONG":
        overall_indicator = "📈 BULLISH"
    elif overall_dir == "SHORT":
        overall_indicator = "📉 BEARISH"
    else:
        overall_indicator = "➡️  NEUTRAL"

    lines.append(f"    Overall Direction: {overall_indicator}")
    lines.append(f"    Overall Confidence: {mtf.overall_confidence:.0%}")
    lines.append(f"    Timeframe Alignment: {mtf.alignment_score:.0%}")

    if mtf.dominant_timeframe:
        lines.append(f"    Strongest Signal From: {mtf.dominant_timeframe}")

    # Alignment interpretation
    if mtf.alignment_score > 0.8:
        alignment_meaning = "Strong agreement - all timeframes aligned"
    elif mtf.alignment_score > 0.5:
        alignment_meaning = "Moderate agreement - most timeframes agree"
    elif mtf.alignment_score > 0.2:
        alignment_meaning = "Mixed signals - timeframes conflicting"
    else:
        alignment_meaning = "No consensus - highly conflicting signals"

    lines.append(f"      -> {alignment_meaning}")

    # Individual timeframe signals table
    lines.append("")
    lines.append("    ┌────────────┬───────────┬──────────┬───────────┬───────────────────────────────────┐")
    lines.append("    │ TIMEFRAME  │ DIRECTION │ STRENGTH │ CONFIDENCE│ REASONING                         │")
    lines.append("    ├────────────┼───────────┼──────────┼───────────┼───────────────────────────────────┤")

    for signal in mtf.signals:
        # Direction with visual indicator
        direction = signal.direction.upper()
        if direction == "LONG":
            dir_display = "🟢 LONG  "
        elif direction == "SHORT":
            dir_display = "🔴 SHORT "
        else:
            dir_display = "⚪ NEUTR "

        # Strength bar
        strength_val = signal.strength
        if strength_val > 0:
            strength_bar = f"+{strength_val:.2f}"
        else:
            strength_bar = f"{strength_val:.2f}"

        # Confidence
        conf = f"{signal.confidence:.0%}"

        # Reasoning (truncated to fit)
        reasoning = signal.reasoning[:33] + ".." if len(signal.reasoning) > 35 else signal.reasoning

        lines.append(
            f"    │ {signal.timeframe:<10} │ {dir_display} │ {strength_bar:>8} │ {conf:>9} │ {reasoning:<35} │"
        )

    lines.append("    └────────────┴───────────┴──────────┴───────────┴───────────────────────────────────┘")

    # Summary by direction
    long_tfs = [s.timeframe for s in mtf.signals if s.direction == "long"]
    short_tfs = [s.timeframe for s in mtf.signals if s.direction == "short"]
    neutral_tfs = [s.timeframe for s in mtf.signals if s.direction == "neutral"]

    lines.append("")
    lines.append("    SUMMARY BY DIRECTION:")
    if long_tfs:
        lines.append(f"      🟢 Bullish: {', '.join(long_tfs)}")
    if short_tfs:
        lines.append(f"      🔴 Bearish: {', '.join(short_tfs)}")
    if neutral_tfs:
        lines.append(f"      ⚪ Neutral: {', '.join(neutral_tfs)}")

    # Trading interpretation
    lines.append("")
    lines.append("    TRADING INTERPRETATION:")

    if mtf.alignment_score > 0.7 and mtf.overall_direction == "long":
        lines.append("      -> Strong buy setup: All timeframes confirm bullish trend")
        lines.append("         Consider long entries with trend")
    elif mtf.alignment_score > 0.7 and mtf.overall_direction == "short":
        lines.append("      -> Strong sell setup: All timeframes confirm bearish trend")
        lines.append("         Consider short entries with trend")
    elif mtf.alignment_score < 0.3:
        lines.append("      -> CONFLICTING SIGNALS: Timeframes disagree significantly")
        lines.append("         Reduce position size or wait for alignment")
    elif long_tfs and short_tfs:
        higher_tfs = ["4Hour", "1Day"]
        lower_tfs = ["1Min", "5Min", "15Min"]

        higher_long = any(tf in long_tfs for tf in higher_tfs)
        lower_short = any(tf in short_tfs for tf in lower_tfs)
        higher_short = any(tf in short_tfs for tf in higher_tfs)
        lower_long = any(tf in long_tfs for tf in lower_tfs)

        if higher_long and lower_short:
            lines.append("      -> Higher TFs bullish, lower TFs bearish")
            lines.append("         Potential pullback in larger uptrend - watch for reversal")
        elif higher_short and lower_long:
            lines.append("      -> Higher TFs bearish, lower TFs bullish")
            lines.append("         Potential bounce in larger downtrend - trade cautiously")
        else:
            lines.append("      -> Mixed signals across timeframes")
            lines.append("         Wait for better alignment or trade smaller size")
    else:
        lines.append("      -> Mostly neutral conditions")
        lines.append("         No clear trend - range-bound trading may apply")

    return lines
