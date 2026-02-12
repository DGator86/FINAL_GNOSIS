#!/usr/bin/env python3
"""
Super Gnosis DHPE v4 - Enhanced Dashboard (REAL DATA)
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

# Page config
st.set_page_config(
    page_title="Super Gnosis DHPE v4",
    page_icon="🚀",
    layout="wide",
)

# Minimal Dark Theme CSS
st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    .main .block-container { padding: 1rem 2rem; max-width: 100%; }
    
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .header-banner h1 { color: white !important; margin: 0; font-size: 2rem; }
    .header-banner p { color: rgba(255,255,255,0.9) !important; margin: 0.5rem 0 0 0; }
    
    .metric-green { color: #3fb950; }
    .metric-red { color: #f85149; }
    
    .signal-buy { background: #3fb95020; color: #3fb950; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; }
    .signal-sell { background: #f8514920; color: #f85149; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; }
    .signal-hold { background: #d2992220; color: #d29922; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; }
    
    .status-online { color: #3fb950; }
    .status-offline { color: #f85149; }
    
    div[data-testid="stMetricValue"] { color: #f0f6fc; }
    div[data-testid="stMetricLabel"] { color: #8b949e; }
    
    .error-box { background: #f8514920; border: 1px solid #f85149; padding: 1rem; border-radius: 8px; color: #f85149; }
    .info-box { background: #58a6ff20; border: 1px solid #58a6ff; padding: 1rem; border-radius: 8px; color: #58a6ff; }
</style>
""", unsafe_allow_html=True)


# ============ HELPER FUNCTIONS ============

def fmt_currency(val):
    if val is None:
        return "$0.00"
    return f"${val:,.2f}"

def fmt_pct(val):
    if val is None:
        return "0.00%"
    return f"{val:+.2f}%"


# ============ REAL DATA FUNCTIONS ============

@st.cache_resource
def get_broker():
    """Get the Alpaca broker adapter."""
    try:
        from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
        from execution.broker_adapters.settings import get_alpaca_paper_setting
        broker = AlpacaBrokerAdapter(paper=get_alpaca_paper_setting())
        return broker
    except Exception as e:
        st.error(f"Failed to connect to Alpaca: {e}")
        return None


def get_real_account_data(broker):
    """Fetch real account data from Alpaca."""
    if not broker:
        return None
    
    try:
        account = broker.get_account()
        return {
            "portfolio_value": float(account.portfolio_value),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "last_equity": float(account.last_equity),
            "account_id": account.account_id,
            "pattern_day_trader": account.pattern_day_trader,
        }
    except Exception as e:
        st.error(f"Error fetching account: {e}")
        return None


def get_real_positions(broker):
    """Fetch real positions from Alpaca."""
    if not broker:
        return []
    
    try:
        positions = broker.get_positions()
        result = []
        for p in positions:
            result.append({
                "symbol": p.symbol,
                "qty": float(p.quantity),
                "entry": float(p.avg_entry_price),
                "current": float(p.current_price),
                "market_value": float(p.market_value),
                "cost_basis": float(p.cost_basis),
                "pnl": float(p.unrealized_pnl),
                "pnl_pct": float(p.unrealized_pnl_pct) * 100,
                "side": p.side,
            })
        return result
    except Exception as e:
        st.error(f"Error fetching positions: {e}")
        return []


def run_engine_analysis(symbol: str):
    """Run the actual trading engines on a symbol."""
    try:
        from main import build_pipeline, load_config
        config = load_config()
        runner = build_pipeline(symbol, config)
        result = runner.run_once(datetime.now())
        
        engines = {}
        
        # Hedge Engine
        if result.hedge_snapshot:
            h = result.hedge_snapshot
            engines["hedge"] = {
                "elasticity": h.elasticity,
                "energy": h.movement_energy,
                "asymmetry": h.energy_asymmetry,
                "dealer_gamma": h.dealer_gamma_sign,
                "regime": h.regime.upper() if hasattr(h, 'regime') else "NEUTRAL",
                "confidence": h.confidence,
            }
        
        # Liquidity Engine
        if result.liquidity_snapshot:
            l = result.liquidity_snapshot
            engines["liquidity"] = {
                "score": l.liquidity_score,
                "spread": l.bid_ask_spread,
                "impact": l.impact_cost,
            }
        
        # Sentiment Engine
        if result.sentiment_snapshot:
            s = result.sentiment_snapshot
            engines["sentiment"] = {
                "score": s.sentiment_score,
                "news": s.news_sentiment if hasattr(s, 'news_sentiment') else 0,
                "confidence": s.confidence,
            }
        
        # Elasticity Engine  
        if result.elasticity_snapshot:
            e = result.elasticity_snapshot
            engines["elasticity"] = {
                "volatility": e.volatility,
                "regime": e.volatility_regime.upper() if hasattr(e, 'volatility_regime') else "MODERATE",
                "trend": e.trend_strength,
            }
        
        # Agent Suggestions
        agents = []
        if result.suggestions:
            for sug in result.suggestions:
                agents.append({
                    "name": sug.agent_name,
                    "signal": sug.direction.value.upper(),
                    "conf": sug.confidence,
                    "reasoning": sug.reasoning,
                })
        
        # Consensus
        consensus = None
        if result.consensus:
            consensus = {
                "direction": result.consensus.get("direction", "HOLD").upper(),
                "confidence": result.consensus.get("confidence", 0),
                "value": result.consensus.get("consensus_value", 0),
            }
        
        return {
            "engines": engines,
            "agents": agents,
            "consensus": consensus,
            "success": True,
        }
    except Exception as e:
        return {
            "engines": {},
            "agents": [],
            "consensus": None,
            "success": False,
            "error": str(e),
        }


def load_portfolio_history():
    """Load portfolio history from ledger."""
    ledger_path = Path("data/ledger.jsonl")
    if not ledger_path.exists():
        return None
    
    records = []
    try:
        with open(ledger_path, 'r') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except:
                    continue
        if records:
            return pd.DataFrame(records)
    except:
        pass
    return None


# ============ MAIN APP ============

def main():
    # Header
    st.markdown("""
    <div class="header-banner">
        <h1>🚀 Super Gnosis DHPE v4</h1>
        <p>Live Trading Dashboard - Real Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get broker connection
    broker = get_broker()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        symbol = st.text_input("Symbol", value="SPY").upper()
        auto_refresh = st.checkbox("Auto Refresh (30s)")
        
        if st.button("🔄 Refresh Data"):
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🔌 Connection Status")
        
        if broker:
            st.markdown("✅ **Alpaca:** <span class='status-online'>Connected</span>", unsafe_allow_html=True)
            # Get account to verify
            account = get_real_account_data(broker)
            if account:
                st.markdown(f"📋 Account: `{account['account_id'][:8]}...`")
        else:
            st.markdown("❌ **Alpaca:** <span class='status-offline'>Disconnected</span>", unsafe_allow_html=True)
            st.markdown("<div class='error-box'>Check your .env file for ALPACA_API_KEY and ALPACA_SECRET_KEY</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f"🕐 Last Update: {datetime.now().strftime('%H:%M:%S')}")
    
    # Check if broker is connected
    if not broker:
        st.error("⚠️ Cannot connect to Alpaca. Please check your API credentials in .env file.")
        st.code("""
# Required in .env file:
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # for paper trading
        """)
        return
    
    # Get real data
    account = get_real_account_data(broker)
    positions = get_real_positions(broker)
    
    if not account:
        st.error("Failed to fetch account data")
        return
    
    day_pnl = account["equity"] - account["last_equity"]
    day_pnl_pct = (day_pnl / account["last_equity"]) * 100 if account["last_equity"] > 0 else 0
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "💼 Portfolio", "📈 Analytics", "🎯 Signals", "⚠️ Risk"])
    
    # ============ TAB 1: OVERVIEW ============
    with tab1:
        st.markdown("### 💰 Account Overview (Live)")
        
        # Top metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Portfolio Value", fmt_currency(account["portfolio_value"]))
        with col2:
            delta_color = "normal" if day_pnl >= 0 else "inverse"
            st.metric("Today's P&L", fmt_currency(day_pnl), fmt_pct(day_pnl_pct))
        with col3:
            st.metric("Cash", fmt_currency(account["cash"]))
        with col4:
            st.metric("Buying Power", fmt_currency(account["buying_power"]))
        with col5:
            st.metric("Open Positions", len(positions))
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📈 Portfolio Summary")
            
            if positions:
                total_invested = sum(p["market_value"] for p in positions)
                total_pnl = sum(p["pnl"] for p in positions)
                
                st.markdown(f"""
                - **Total Invested:** {fmt_currency(total_invested)}
                - **Unrealized P&L:** {fmt_currency(total_pnl)}
                - **Cash Available:** {fmt_currency(account['cash'])}
                """)
            else:
                st.info("No open positions. Portfolio is 100% cash.")
        
        with col2:
            # Allocation pie
            if positions:
                labels = [p["symbol"] for p in positions] + ["Cash"]
                values = [p["market_value"] for p in positions] + [account["cash"]]
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.5,
                    marker=dict(colors=['#667eea', '#3fb950', '#f85149', '#d29922', '#58a6ff', '#a371f7']),
                )])
                fig.update_layout(
                    title="Allocation",
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#8b949e'),
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("#### 💵 100% Cash")
    
    # ============ TAB 2: PORTFOLIO ============
    with tab2:
        st.markdown("### 💼 Current Positions (Live)")
        
        if positions:
            total_pnl = sum(p["pnl"] for p in positions)
            total_value = sum(p["market_value"] for p in positions)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Value", fmt_currency(total_value))
            with col2:
                pnl_pct = (total_pnl / total_value * 100) if total_value > 0 else 0
                st.metric("Total P&L", fmt_currency(total_pnl), fmt_pct(pnl_pct))
            with col3:
                winners = sum(1 for p in positions if p["pnl"] > 0)
                st.metric("Winners", f"{winners}/{len(positions)}")
            with col4:
                losers = sum(1 for p in positions if p["pnl"] < 0)
                st.metric("Losers", f"{losers}/{len(positions)}")
            
            st.markdown("---")
            
            # Positions table
            df = pd.DataFrame(positions)
            df_display = df[["symbol", "qty", "entry", "current", "market_value", "pnl", "pnl_pct"]].copy()
            df_display.columns = ["Symbol", "Qty", "Entry", "Current", "Value", "P&L", "P&L %"]
            df_display["Entry"] = df_display["Entry"].apply(lambda x: f"${x:.2f}")
            df_display["Current"] = df_display["Current"].apply(lambda x: f"${x:.2f}")
            df_display["Value"] = df_display["Value"].apply(lambda x: f"${x:.2f}")
            df_display["P&L"] = df_display["P&L"].apply(lambda x: f"${x:+,.2f}")
            df_display["P&L %"] = df_display["P&L %"].apply(lambda x: f"{x:+.2f}%")
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # P&L Bar Chart
            fig = go.Figure(data=[go.Bar(
                x=[p["symbol"] for p in positions],
                y=[p["pnl"] for p in positions],
                marker_color=['#3fb950' if p["pnl"] >= 0 else '#f85149' for p in positions],
                text=[f"${p['pnl']:+,.0f}" for p in positions],
                textposition='auto',
            )])
            fig.update_layout(
                title="Unrealized P&L by Position",
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#161b22',
                font=dict(color='#8b949e'),
                xaxis=dict(gridcolor='#30363d'),
                yaxis=dict(gridcolor='#30363d', tickformat='$,.0f'),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📭 No open positions. Your portfolio is 100% cash.")
    
    # ============ TAB 3: ANALYTICS ============
    with tab3:
        st.markdown(f"### 📈 Live Analytics: {symbol}")
        
        if st.button(f"🔍 Run Analysis on {symbol}", type="primary"):
            with st.spinner(f"Running engines on {symbol}..."):
                analysis = run_engine_analysis(symbol)
            
            if analysis["success"]:
                st.success("✅ Analysis complete!")
                st.session_state["last_analysis"] = analysis
                st.session_state["last_symbol"] = symbol
            else:
                st.error(f"❌ Analysis failed: {analysis.get('error', 'Unknown error')}")
        
        # Display last analysis if available
        if "last_analysis" in st.session_state:
            analysis = st.session_state["last_analysis"]
            analyzed_symbol = st.session_state.get("last_symbol", symbol)
            
            st.markdown(f"**Last analyzed:** {analyzed_symbol}")
            
            engines = analysis.get("engines", {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("#### 🛡️ Hedge Engine")
                if "hedge" in engines:
                    h = engines["hedge"]
                    st.metric("Elasticity", f"{h['elasticity']:.1f}")
                    st.metric("Movement Energy", f"{h['energy']:.2f}")
                    st.metric("Asymmetry", f"{h['asymmetry']:+.3f}")
                    regime = h.get('regime', 'NEUTRAL')
                    color = "#3fb950" if regime == "BULLISH" else "#f85149" if regime == "BEARISH" else "#d29922"
                    st.markdown(f"**Regime:** <span style='color:{color}'>{regime}</span>", unsafe_allow_html=True)
                else:
                    st.caption("No data")
            
            with col2:
                st.markdown("#### 💧 Liquidity")
                if "liquidity" in engines:
                    l = engines["liquidity"]
                    st.metric("Score", f"{l['score']:.3f}")
                    st.metric("Spread", f"{l['spread']*100:.3f}%")
                    st.metric("Impact", f"{l['impact']*100:.3f}%")
                else:
                    st.caption("No data")
            
            with col3:
                st.markdown("#### 📰 Sentiment")
                if "sentiment" in engines:
                    s = engines["sentiment"]
                    st.metric("Score", f"{s['score']:+.3f}")
                    st.metric("Confidence", f"{s['confidence']:.1%}")
                else:
                    st.caption("No data")
            
            with col4:
                st.markdown("#### ⚡ Elasticity")
                if "elasticity" in engines:
                    e = engines["elasticity"]
                    st.metric("Volatility", f"{e['volatility']*100:.1f}%")
                    st.metric("Trend", f"{e['trend']:+.3f}")
                    st.markdown(f"**Regime:** {e.get('regime', 'MODERATE')}")
                else:
                    st.caption("No data")
        else:
            st.info("👆 Click 'Run Analysis' to analyze a symbol with the trading engines.")
    
    # ============ TAB 4: SIGNALS ============
    with tab4:
        st.markdown("### 🎯 Agent Signals")
        
        if "last_analysis" in st.session_state:
            analysis = st.session_state["last_analysis"]
            agents = analysis.get("agents", [])
            consensus = analysis.get("consensus")
            
            if consensus:
                direction = consensus["direction"]
                conf = consensus["confidence"]
                
                if direction == "LONG" or direction == "BUY":
                    css_class = "signal-buy"
                    display_signal = "BUY"
                elif direction == "SHORT" or direction == "SELL":
                    css_class = "signal-sell"
                    display_signal = "SELL"
                else:
                    css_class = "signal-hold"
                    display_signal = "HOLD"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem; background: #161b22; border-radius: 12px; margin-bottom: 1rem;">
                    <div style="color: #8b949e; text-transform: uppercase; letter-spacing: 2px;">Consensus Signal</div>
                    <div style="margin: 1rem 0;"><span class="{css_class}" style="font-size: 2rem;">{display_signal}</span></div>
                    <div style="color: #f0f6fc;">Confidence: {conf:.1%} | {len(agents)} Agents</div>
                </div>
                """, unsafe_allow_html=True)
            
            if agents:
                st.markdown("#### Individual Agent Votes")
                cols = st.columns(min(len(agents), 5))
                for i, agent in enumerate(agents):
                    with cols[i % 5]:
                        sig = agent["signal"]
                        if sig in ["LONG", "BUY"]:
                            sig_class = "signal-buy"
                            sig_display = "BUY"
                        elif sig in ["SHORT", "SELL"]:
                            sig_class = "signal-sell"
                            sig_display = "SELL"
                        else:
                            sig_class = "signal-hold"
                            sig_display = "HOLD"
                        
                        st.markdown(f"""
                        <div style="background: #161b22; border-radius: 8px; padding: 1rem; text-align: center; margin-bottom: 0.5rem;">
                            <div style="color: #8b949e; font-size: 0.75rem;">{agent['name']}</div>
                            <div style="margin: 0.5rem 0;"><span class="{sig_class}">{sig_display}</span></div>
                            <div style="color: #f0f6fc; font-size: 1.1rem; font-weight: bold;">{agent['conf']:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("Reasoning"):
                            st.write(agent.get("reasoning", "No reasoning provided"))
            else:
                st.info("No agent signals available. Run an analysis first.")
        else:
            st.info("👆 Go to Analytics tab and run an analysis to see agent signals.")
    
    # ============ TAB 5: RISK ============
    with tab5:
        st.markdown("### ⚠️ Risk Overview")
        
        if positions:
            total_value = sum(p["market_value"] for p in positions)
            total_pnl = sum(p["pnl"] for p in positions)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Concentration risk - largest position
                if positions:
                    largest = max(positions, key=lambda p: p["market_value"])
                    concentration = (largest["market_value"] / account["portfolio_value"]) * 100
                    st.metric("Largest Position", f"{largest['symbol']}", f"{concentration:.1f}% of portfolio")
            
            with col2:
                # Cash ratio
                cash_pct = (account["cash"] / account["portfolio_value"]) * 100
                st.metric("Cash Ratio", f"{cash_pct:.1f}%")
            
            with col3:
                # Simple VaR estimate (2% of portfolio)
                var_estimate = account["portfolio_value"] * 0.02
                st.metric("Est. Daily VaR (2%)", fmt_currency(var_estimate))
            
            with col4:
                # P&L as % of portfolio
                pnl_impact = (total_pnl / account["portfolio_value"]) * 100
                st.metric("P&L Impact", fmt_pct(pnl_impact))
            
            st.markdown("---")
            
            # Position weights
            st.markdown("#### Position Weights")
            weights_data = []
            for p in positions:
                weight = (p["market_value"] / account["portfolio_value"]) * 100
                weights_data.append({"Symbol": p["symbol"], "Weight": weight, "Value": p["market_value"]})
            
            weights_df = pd.DataFrame(weights_data)
            weights_df["Weight"] = weights_df["Weight"].apply(lambda x: f"{x:.1f}%")
            weights_df["Value"] = weights_df["Value"].apply(lambda x: fmt_currency(x))
            st.dataframe(weights_df, use_container_width=True, hide_index=True)
        else:
            st.info("📭 No positions to analyze. Portfolio is 100% cash - no market risk exposure.")
    
    # Auto refresh
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
