#!/usr/bin/env python3
"""
🎯 SUPER GNOSIS DHPE v3 - ENHANCED LIVE TRADING DASHBOARD
Real-time tracking of positions, trades, and analytics
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
from execution.broker_adapters.settings import get_alpaca_paper_setting
from main import build_pipeline, load_config
from schemas.core_schemas import PipelineResult

# Page config
st.set_page_config(
    page_title="Super Gnosis DHPE v3 Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .positive {
        color: #00cc00;
        font-weight: bold;
    }
    .negative {
        color: #ff3333;
        font-weight: bold;
    }
    .neutral {
        color: #666666;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_broker():
    """Get broker adapter (cached)."""
    try:
        paper_mode = get_alpaca_paper_setting()
        return AlpacaBrokerAdapter(paper=paper_mode)
    except Exception as e:
        st.error(f"Failed to connect to Alpaca: {e}")
        return None


@st.cache_data(ttl=60)
def load_ledger_data():
    """Load ledger data from JSONL file."""
    ledger_path = Path("data/ledger.jsonl")
    sqlite_path = ledger_path.with_suffix(".db")

    if sqlite_path.exists():
        try:
            return pd.read_sql("SELECT * FROM ledger", f"sqlite:///{sqlite_path}")
        except Exception:
            pass

    if not ledger_path.exists():
        return pd.DataFrame()

    records = []
    with open(ledger_path, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(records)


def run_pipeline_with_trace(symbol: str) -> Tuple[PipelineResult, List[Dict[str, Any]]]:
    """Run the pipeline with per-step tracing for UI visualization."""

    trace: List[Dict[str, Any]] = []

    def record_step(name: str, step_type: str, func):
        start = time.perf_counter()
        status = "success"
        error: str | None = None
        payload = None

        try:
            payload = func()
        except Exception as exc:  # pragma: no cover - defensive for UI
            status = "error"
            error = str(exc)

        duration = time.perf_counter() - start
        trace.append({
            "name": name,
            "type": step_type,
            "status": status,
            "duration": duration,
            "error": error,
        })
        return payload

    try:
        config = load_config()
        runner = build_pipeline(symbol, config)
    except Exception as exc:  # pragma: no cover - UI defensive guard
        trace.append({
            "name": "Pipeline Setup",
            "type": "setup",
            "status": "error",
            "duration": 0.0,
            "error": str(exc),
        })
        return PipelineResult(timestamp=datetime.now(timezone.utc), symbol=symbol), trace

    timestamp = datetime.now(timezone.utc)
    result = PipelineResult(timestamp=timestamp, symbol=symbol.upper())

    # Engines
    if "hedge" in runner.engines:
        result.hedge_snapshot = record_step(
            "Hedge Engine",
            "engine",
            lambda: runner.engines["hedge"].run(symbol, timestamp),
        )

    if "liquidity" in runner.engines:
        result.liquidity_snapshot = record_step(
            "Liquidity Engine",
            "engine",
            lambda: runner.engines["liquidity"].run(symbol, timestamp),
        )

    if "sentiment" in runner.engines:
        result.sentiment_snapshot = record_step(
            "Sentiment Engine",
            "engine",
            lambda: runner.engines["sentiment"].run(symbol, timestamp),
        )

    if "elasticity" in runner.engines:
        result.elasticity_snapshot = record_step(
            "Elasticity Engine",
            "engine",
            lambda: runner.engines["elasticity"].run(symbol, timestamp),
        )

    if runner.ml_engine:
        result.ml_snapshot = record_step(
            "ML Enhancement",
            "engine",
            lambda: runner.ml_engine.enhance(result, timestamp),
        )

    # Agents
    for agent_name, agent in runner.primary_agents.items():
        suggestion = record_step(
            agent_name.replace("_", " ").title(),
            "agent",
            lambda a=agent: a.suggest(result, timestamp),
        )
        if suggestion:
            result.suggestions.append(suggestion)

    if runner.composer and result.suggestions:
        result.consensus = record_step(
            "Composer", "agent", lambda: runner.composer.compose(result.suggestions, timestamp)
        )

    if runner.trade_agent:
        result.trade_ideas = record_step(
            "Trade Agent", "agent", lambda: runner.trade_agent.generate_ideas(result, timestamp)
        ) or []

    return result, trace


def format_currency(value):
    """Format value as currency."""
    return f"${value:,.2f}"


def format_percentage(value):
    """Format value as percentage."""
    return f"{value:+.2f}%"


def get_pnl_color(value):
    """Get color class for P&L."""
    if value > 0:
        return "positive"
    elif value < 0:
        return "negative"
    else:
        return "neutral"


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<div class="main-header">🎯 SUPER GNOSIS DHPE v3<br/>Live Trading Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=DHPE+v3", use_column_width=True)
        st.markdown("### ⚙️ Settings")
        
        auto_refresh = st.checkbox("Auto Refresh (5s)", value=False)
        show_debug = st.checkbox("Show Debug Info", value=False)
        
        st.markdown("---")
        st.markdown("### 📊 Quick Actions")

        symbol = st.text_input("Symbol", value="SPY")
        risk_level = st.slider("Risk Level", 0.1, 1.0, 0.5, 0.1)

        if st.button("🔍 Run Analysis", type="primary"):
            with st.spinner("Running pipeline..."):
                try:
                    config = load_config()
                    config["risk_level"] = risk_level
                    runner = build_pipeline(symbol, config)
                    result = runner.run_once(datetime.now())
                    st.session_state["last_result"] = result
                    st.success(f"✅ Analysis complete for {symbol}")
                    if result.hedge_snapshot and result.hedge_snapshot.movement_energy > 50:
                        st.warning("High Movement Energy Alert! Potential Squeeze.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        st.markdown("---")
        st.markdown("### 📝 System Status")
        broker = get_broker()
        if broker:
            st.success("✅ Alpaca Connected")
        else:
            st.error("❌ Alpaca Disconnected")
    
    # Main content
    broker = get_broker()
    
    if not broker:
        st.error("⚠️ Cannot connect to broker. Check your credentials.")
        return
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "💼 Positions", 
        "📈 Analytics",
        "📜 Trade History",
        "⚙️ Engine Metrics",
        "🧬 Pipeline Vision"
    ])
    
    # Core Universe list
    CORE_UNIVERSE = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMD", "MSFT", "AMZN", "META"]
    
    # TAB 1: Overview
    with tab1:
        st.markdown("## 💰 Account Overview")
        
        try:
            account = broker.get_account()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Portfolio Value",
                    value=format_currency(account.portfolio_value),
                    delta=format_currency(account.equity - account.last_equity)
                )
            
            with col2:
                st.metric(
                    label="Cash",
                    value=format_currency(account.cash),
                )
            
            with col3:
                st.metric(
                    label="Buying Power",
                    value=format_currency(account.buying_power),
                )
            
            with col4:
                day_pnl = account.equity - account.last_equity
                day_pnl_pct = (day_pnl / account.last_equity * 100) if account.last_equity > 0 else 0
                st.metric(
                    label="Today's P&L",
                    value=format_currency(day_pnl),
                    delta=format_percentage(day_pnl_pct)
                )
            
            # Account details
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📋 Account Details")
                st.markdown(f"""
                - **Account ID**: `{account.account_id}`
                - **Account Type**: Paper Trading
                - **Pattern Day Trader**: {'Yes' if account.pattern_day_trader else 'No'}
                """)
            
            with col2:
                st.markdown("### 📊 Capital Allocation")
                equity = account.equity
                cash_pct = (account.cash / equity * 100) if equity > 0 else 0
                invested_pct = 100 - cash_pct
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Cash', 'Invested'],
                    values=[cash_pct, invested_pct],
                    hole=.3,
                    marker=dict(colors=['#636EFA', '#00CC96'])
                )])
                fig.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading account: {e}")
    
    # TAB 2: Positions
    with tab2:
        st.markdown("## 💼 Current Positions")
        
        try:
            positions = broker.get_positions()
            
            if positions:
                # Summary metrics
                total_value = sum(p.market_value for p in positions)
                total_pnl = sum(p.unrealized_pnl for p in positions)
                total_pnl_pct = (total_pnl / (total_value - total_pnl) * 100) if (total_value - total_pnl) > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Positions", len(positions))
                with col2:
                    st.metric("Market Value", format_currency(total_value))
                with col3:
                    st.metric("Unrealized P&L", format_currency(total_pnl), delta=format_percentage(total_pnl_pct))
                
                st.markdown("---")
                
                # Positions table
                pos_data = []
                for p in positions:
                    pnl_pct = p.unrealized_pnl_pct * 100
                    pos_data.append({
                        'Symbol': p.symbol,
                        'Quantity': f"{p.quantity:,.0f}",
                        'Side': p.side.upper(),
                        'Avg Price': format_currency(p.avg_entry_price),
                        'Current Price': format_currency(p.current_price),
                        'Market Value': format_currency(p.market_value),
                        'Cost Basis': format_currency(p.cost_basis),
                        'Unrealized P&L': format_currency(p.unrealized_pnl),
                        'P&L %': format_percentage(pnl_pct),
                    })
                
                df = pd.DataFrame(pos_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # P&L chart
                st.markdown("### 📊 Position P&L Breakdown")
                pnl_fig = go.Figure(data=[
                    go.Bar(
                        x=[p.symbol for p in positions],
                        y=[p.unrealized_pnl for p in positions],
                        marker_color=['green' if p.unrealized_pnl >= 0 else 'red' for p in positions],
                        text=[format_currency(p.unrealized_pnl) for p in positions],
                        textposition='auto',
                    )
                ])
                pnl_fig.update_layout(
                    title="Unrealized P&L by Position",
                    xaxis_title="Symbol",
                    yaxis_title="P&L ($)",
                    height=400,
                )
                st.plotly_chart(pnl_fig, use_container_width=True)
            else:
                st.info("📭 No open positions")
        
        except Exception as e:
            st.error(f"Error loading positions: {e}")
    
    # TAB 3: Analytics
    with tab3:
        st.markdown("## 📈 Live Analytics")
        
        # Run analysis for selected symbol
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                config = load_config()
                runner = build_pipeline(symbol, config)
                result = runner.run_once(datetime.now())
                
                # Display engine outputs
                col1, col2 = st.columns(2)
                
                with col1:
                    if result.hedge_snapshot:
                        st.markdown("### 🎯 Hedge Engine v3.0")
                        h = result.hedge_snapshot
                        
                        st.metric("Elasticity", f"{h.elasticity:.2f}")
                        st.metric("Movement Energy", f"{h.movement_energy:.2f}")
                        st.metric("Energy Asymmetry", f"{h.energy_asymmetry:+.3f}")
                        st.metric("Dealer Gamma Sign", f"{h.dealer_gamma_sign:+.3f}")
                        
                        st.markdown(f"**Regime**: {h.regime.upper()}")
                        st.progress(h.confidence, text=f"Confidence: {h.confidence:.1%}")
                
                with col2:
                    if result.liquidity_snapshot:
                        st.markdown("### 💧 Liquidity")
                        l = result.liquidity_snapshot
                        st.metric("Liquidity Score", f"{l.liquidity_score:.3f}")
                        st.metric("Bid-Ask Spread", f"{l.bid_ask_spread:.4f}%")
                        st.metric("Impact Cost", f"{l.impact_cost:.4f}%")
                
                st.markdown("---")

                ledger_df = load_ledger_data()
                if not ledger_df.empty and "timestamp" in ledger_df.columns:
                    try:
                        ledger_df["timestamp"] = pd.to_datetime(ledger_df["timestamp"])
                        fig = px.line(
                            ledger_df,
                            x="timestamp",
                            y=ledger_df.get("elasticity", ledger_df.get("elasticity_snapshot.elasticity", None)),
                            title="Market Elasticity",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass

                col1, col2 = st.columns(2)
                
                with col1:
                    if result.sentiment_snapshot:
                        st.markdown("### 📰 Sentiment")
                        s = result.sentiment_snapshot
                        
                        # Sentiment gauge
                        sentiment_fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=s.sentiment_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Overall Sentiment"},
                            delta={'reference': 0},
                            gauge={
                                'axis': {'range': [-1, 1]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [-1, -0.3], 'color': "red"},
                                    {'range': [-0.3, 0.3], 'color': "lightgray"},
                                    {'range': [0.3, 1], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': s.sentiment_score
                                }
                            }
                        ))
                        sentiment_fig.update_layout(height=300)
                        st.plotly_chart(sentiment_fig, use_container_width=True)
                
                with col2:
                    if result.elasticity_snapshot:
                        st.markdown("### ⚡ Elasticity")
                        e = result.elasticity_snapshot
                        st.metric("Volatility", f"{e.volatility:.2%}")
                        st.metric("Regime", e.volatility_regime.upper())
                        st.metric("Trend Strength", f"{e.trend_strength:.3f}")
                
                # Agent suggestions
                if result.suggestions:
                    st.markdown("---")
                    st.markdown("### 🤖 Agent Suggestions")
                    
                    for sug in result.suggestions:
                        with st.expander(f"**{sug.agent_name}** - {sug.direction.value.upper()} ({sug.confidence:.1%})"):
                            st.write(sug.reasoning)
                
                # Consensus
                if result.consensus:
                    st.markdown("---")
                    st.markdown("### 🎯 Consensus")
                    c = result.consensus
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Direction", c['direction'].upper())
                    with col2:
                        st.metric("Confidence", f"{c['confidence']:.1%}")
                    with col3:
                        st.metric("Consensus Value", f"{c['consensus_value']:+.3f}")
                
            except Exception as e:
                st.error(f"Error running analysis: {e}")
    
    # TAB 4: Trade History
    with tab4:
        st.markdown("## 📜 Trade History")
        
        ledger_df = load_ledger_data()
        
        if not ledger_df.empty:
            st.markdown(f"### 📊 {len(ledger_df)} Pipeline Runs Recorded")
            
            # Show recent runs
            st.dataframe(
                ledger_df[['timestamp', 'symbol']].tail(10),
                use_container_width=True,
                hide_index=True
            )
            
            # Analysis over time
            if len(ledger_df) > 1:
                st.markdown("### 📈 Metrics Over Time")
                # Add charts here based on ledger data
        else:
            st.info("📭 No trade history yet. Run some analyses to populate this view.")
    
    # TAB 5: Engine + Agent Monitor
    with tab5:
        st.markdown("## ⚙️ Engine & Agent Monitor")
        st.write("Watch each engine and agent step as the pipeline processes market data.")

        if "trace_events" not in st.session_state:
            st.session_state.trace_events = []
            st.session_state.trace_result = None
            st.session_state.trace_symbol = symbol

        monitor_symbol = st.text_input(
            "Symbol to trace", value=st.session_state.trace_symbol, key="trace_symbol_input"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            run_trace = st.button("▶️ Run traced pipeline", type="primary")
        with col2:
            st.caption(
                "Uses the same pipeline configuration as the main system and shows durations per step."
            )

        if run_trace:
            st.session_state.trace_result, st.session_state.trace_events = run_pipeline_with_trace(
                monitor_symbol
            )
            st.session_state.trace_symbol = monitor_symbol

        if st.session_state.trace_events:
            events_df = pd.DataFrame(st.session_state.trace_events)
            success_count = (events_df["status"] == "success").sum()
            total_steps = len(events_df)
            total_duration = events_df["duration"].sum()

            st.markdown("### ⏱️ Run summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Steps completed", f"{success_count}/{total_steps}")
            with col2:
                st.metric("Total duration", f"{total_duration:.2f}s")
            with col3:
                st.metric("Latest symbol", st.session_state.trace_symbol.upper())

            progress = success_count / total_steps if total_steps else 0
            st.progress(progress, text=f"{progress:.0%} of steps succeeded")

            st.markdown("### 🔍 Detailed timeline")
            events_df = events_df.sort_values("type", ascending=False)
            events_df["duration_ms"] = (events_df["duration"] * 1000).round(1)
            events_df_display = events_df[["type", "name", "status", "duration_ms", "error"]]
            events_df_display.rename(
                columns={"type": "Component", "name": "Step", "duration_ms": "Duration (ms)"},
                inplace=True,
            )
            st.dataframe(events_df_display, hide_index=True, use_container_width=True)

            result = st.session_state.trace_result
            if result:
                st.markdown("### 🛰️ Latest snapshots")
                snapshot_cols = st.columns(4)

                with snapshot_cols[0]:
                    if result.hedge_snapshot:
                        st.metric("Elasticity", f"{result.hedge_snapshot.elasticity:.2f}")
                        st.metric(
                            "Energy Asymmetry",
                            f"{result.hedge_snapshot.energy_asymmetry:+.2f}",
                        )

                with snapshot_cols[1]:
                    if result.liquidity_snapshot:
                        st.metric(
                            "Liquidity Score", f"{result.liquidity_snapshot.liquidity_score:.3f}"
                        )
                        st.metric(
                            "Bid/Ask Spread", f"{result.liquidity_snapshot.bid_ask_spread:.4f}%"
                        )

                with snapshot_cols[2]:
                    if result.sentiment_snapshot:
                        st.metric(
                            "Sentiment Score", f"{result.sentiment_snapshot.sentiment_score:+.3f}"
                        )
                        st.metric(
                            "Confidence", f"{result.sentiment_snapshot.confidence:.1%}"
                        )

                with snapshot_cols[3]:
                    if result.elasticity_snapshot:
                        st.metric("Volatility", f"{result.elasticity_snapshot.volatility:.2%}")
                        st.metric("Trend", f"{result.elasticity_snapshot.trend_strength:.3f}")

                if result.suggestions:
                    st.markdown("### 🤖 Agent suggestions")
                    for sug in result.suggestions:
                        with st.expander(
                            f"{sug.agent_name}: {sug.direction.value.upper()} ({sug.confidence:.1%})"
                        ):
                            st.write(sug.reasoning)

                if result.consensus:
                    st.markdown("### 🎯 Consensus")
                    consensus_cols = st.columns(3)
                    with consensus_cols[0]:
                        st.metric("Direction", result.consensus.get("direction", "-"))
                    with consensus_cols[1]:
                        st.metric(
                            "Confidence", f"{result.consensus.get('confidence', 0):.1%}"
                        )
                    with consensus_cols[2]:
                        st.metric(
                            "Consensus Value",
                            f"{result.consensus.get('consensus_value', 0):+.3f}",
                        )
        else:
            st.info("Run a traced pipeline to visualize engine and agent activity.")
    
    # TAB 6: Pipeline Vision
    with tab6:
        st.markdown("## 🧬 Pipeline Vision (Deep Dive)")
        st.write("Inspect the internal state of the Gnosis Pipeline for any asset in the universe.")
        
        # 1. Universe Selection
        selected_ticker = st.selectbox("Select Ticker", CORE_UNIVERSE, index=0)
        
        # 2. Run Pipeline Button
        if st.button(f"🔍 Scan {selected_ticker}", type="primary"):
            with st.spinner(f"Running Gnosis Physics Engine on {selected_ticker}..."):
                try:
                    config = load_config()
                    runner = build_pipeline(selected_ticker, config)
                    result = runner.run_once(datetime.now())
                    
                    st.success(f"Pipeline executed successfully for {selected_ticker}")
                    
                    # 3. Main Metrics Grid
                    st.markdown("### 🧠 Core Physics (GMM)")
                    
                    phys = result.physics_snapshot
                    if phys:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Entropy (Chaos)", f"{phys.entropy:.2f}", help="Lower is better (Stable structure)")
                        with col2:
                            st.metric("Stiffness (Beta)", f"{phys.stiffness:.2f}", help="Resistance to flow")
                        with col3:
                            st.metric("P_up Probability", f"{phys.p_up:.1%}", help="Prob of Upward move")
                        with col4:
                            st.metric("Restoring Force", f"{phys.restoring_force:.2f}", help="Pull back to Equilibrium")
                    else:
                        st.warning("Physics Engine returned no data.")

                    # 4. Engine Cards
                    st.markdown("---")
                    st.markdown("### ⚙️ Engine Signals")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    
                    # Hedge Engine
                    with c1:
                        st.markdown("**🛡️ Hedge**")
                        if result.hedge_snapshot:
                            h = result.hedge_snapshot
                            st.write(f"Regime: `{h.regime}`")
                            st.write(f"Energy: `{h.movement_energy:.1f}`")
                            st.progress(h.confidence, text="Confidence")
                        else:
                            st.caption("No Data")

                    # Liquidity Engine
                    with c2:
                        st.markdown("**💧 Liquidity**")
                        if result.liquidity_snapshot:
                            l = result.liquidity_snapshot
                            st.write(f"Score: `{l.liquidity_score:.2f}`")
                            st.write(f"Spread: `{l.bid_ask_spread:.2%}`")
                        else:
                            st.caption("No Data")

                    # Sentiment Engine
                    with c3:
                        st.markdown("**📰 Sentiment**")
                        if result.sentiment_snapshot:
                            s = result.sentiment_snapshot
                            st.write(f"Score: `{s.sentiment_score:.2f}`")
                            st.write(f"News: `{s.news_sentiment:.2f}`")
                        else:
                            st.caption("No Data")

                    # Elasticity Engine
                    with c4:
                        st.markdown("**⚡ Elasticity**")
                        if result.elasticity_snapshot:
                            e = result.elasticity_snapshot
                            st.write(f"Vol Regime: `{e.volatility_regime}`")
                            st.write(f"Trend: `{e.trend_strength:.2f}`")
                        else:
                            st.caption("No Data")

                    # 5. Consensus & Action
                    st.markdown("---")
                    st.markdown("### 🎯 Final Consensus")
                    
                    if result.consensus:
                        c = result.consensus
                        direction = c.get('direction', 'neutral').upper()
                        conf = c.get('confidence', 0.0)
                        
                        # Big Banner
                        if direction == "LONG":
                            st.success(f"**BUY SIGNAL** ({conf:.1%})")
                        elif direction == "SHORT":
                            st.error(f"**SELL SIGNAL** ({conf:.1%})")
                        else:
                            st.info(f"**NEUTRAL / HOLD** ({conf:.1%})")
                            
                        # Agent breakdown
                        st.markdown("#### Agent Votes")
                        for sug in result.suggestions:
                            st.write(f"- **{sug.agent_name}**: {sug.direction.value.upper()} ({sug.confidence:.1%})")
                            st.caption(f"  Reason: {sug.reasoning}")
                            
                except Exception as e:
                    st.error(f"Failed to scan {selected_ticker}: {e}")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
