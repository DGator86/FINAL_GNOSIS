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
