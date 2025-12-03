#!/usr/bin/env python3
"""
🎯 SUPER GNOSIS DHPE v3 - ENHANCED LIVE TRADING DASHBOARD
Real-time tracking of positions, trades, and analytics
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

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
    
    if not ledger_path.exists():
        return pd.DataFrame()
    
    records = []
    with open(ledger_path, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed JSON lines
                continue
    
    if not records:
        return pd.DataFrame()
    
    return pd.DataFrame(records)


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
        
        if st.button("🔍 Run Analysis", type="primary"):
            with st.spinner("Running pipeline..."):
                try:
                    config = load_config()
                    runner = build_pipeline(symbol, config)
                    result = runner.run_once(datetime.now())
                    st.success(f"✅ Analysis complete for {symbol}")
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "💼 Positions", 
        "📈 Analytics",
        "📜 Trade History",
        "⚙️ Engine Metrics"
    ])
    
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
    
    # TAB 5: Engine Metrics
    with tab5:
        st.markdown("## ⚙️ Engine Metrics")
        
        ledger_df = load_ledger_data()
        
        if not ledger_df.empty and len(ledger_df) > 0:
            st.markdown("### 📊 Historical Engine Performance")
            
            # Extract metrics from ledger
            st.info("Engine metrics tracking coming soon!")
        else:
            st.info("📭 No engine metrics yet. Run analyses to collect data.")
    
    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
