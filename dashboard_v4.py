#!/usr/bin/env python3
"""
Super Gnosis DHPE v4 - Enhanced Dashboard (Lightweight)
"""

import json
import os
import sys
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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
    
    .metric-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-label { color: #8b949e; font-size: 0.8rem; text-transform: uppercase; }
    .metric-value { color: #f0f6fc; font-size: 1.5rem; font-weight: bold; }
    .metric-green { color: #3fb950; }
    .metric-red { color: #f85149; }
    
    .signal-buy { background: #3fb95020; color: #3fb950; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; }
    .signal-sell { background: #f8514920; color: #f85149; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; }
    .signal-hold { background: #d2992220; color: #d29922; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; }
    
    .status-online { color: #3fb950; }
    .status-offline { color: #f85149; }
    
    div[data-testid="stMetricValue"] { color: #f0f6fc; }
    div[data-testid="stMetricLabel"] { color: #8b949e; }
</style>
""", unsafe_allow_html=True)


# ============ HELPER FUNCTIONS ============

def fmt_currency(val):
    return f"${val:,.2f}"

def fmt_pct(val):
    return f"{val:+.2f}%"

def get_color(val):
    return "metric-green" if val >= 0 else "metric-red"


# ============ DATA FUNCTIONS ============

@st.cache_resource
def get_broker():
    try:
        from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
        from execution.broker_adapters.settings import get_alpaca_paper_setting
        return AlpacaBrokerAdapter(paper=get_alpaca_paper_setting())
    except:
        return None

def get_mock_data():
    return {
        "account": {
            "portfolio_value": 150000 + random.uniform(-5000, 5000),
            "cash": 45000,
            "buying_power": 90000,
            "equity": 150000,
            "last_equity": 148500,
        },
        "positions": [
            {"symbol": "SPY", "qty": 50, "entry": 580, "current": 595, "pnl": 750},
            {"symbol": "QQQ", "qty": 30, "entry": 510, "current": 505, "pnl": -150},
            {"symbol": "NVDA", "qty": 20, "entry": 140, "current": 148, "pnl": 160},
            {"symbol": "AAPL", "qty": 40, "entry": 195, "current": 192, "pnl": -120},
            {"symbol": "TSLA", "qty": 15, "entry": 250, "current": 265, "pnl": 225},
        ],
        "engines": {
            "hedge": {"elasticity": 1250.5, "energy": 3.2, "asymmetry": 0.15, "regime": "BULLISH"},
            "liquidity": {"score": 0.92, "spread": 0.012, "impact": 0.005},
            "sentiment": {"score": 0.35, "news": 0.42, "social": 0.28},
            "elasticity": {"volatility": 0.18, "regime": "MODERATE", "trend": 0.45},
        }
    }


# ============ MAIN APP ============

def main():
    # Header
    st.markdown("""
    <div class="header-banner">
        <h1>🚀 Super Gnosis DHPE v4</h1>
        <p>Premium AI-Powered Trading Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        symbol = st.text_input("Symbol", value="SPY").upper()
        auto_refresh = st.checkbox("Auto Refresh (10s)")
        
        st.markdown("---")
        st.markdown("### 🔌 Status")
        
        broker = get_broker()
        if broker:
            st.markdown("✅ Alpaca: <span class='status-online'>Connected</span>", unsafe_allow_html=True)
        else:
            st.markdown("❌ Alpaca: <span class='status-offline'>Offline</span>", unsafe_allow_html=True)
        
        st.markdown(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    
    # Get data
    data = get_mock_data()
    account = data["account"]
    positions = data["positions"]
    engines = data["engines"]
    
    day_pnl = account["equity"] - account["last_equity"]
    day_pnl_pct = (day_pnl / account["last_equity"]) * 100
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "💼 Portfolio", "📈 Analytics", "🎯 Signals", "⚠️ Risk"])
    
    # ============ TAB 1: OVERVIEW ============
    with tab1:
        # Top metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Portfolio Value", fmt_currency(account["portfolio_value"]))
        with col2:
            st.metric("Today's P&L", fmt_currency(day_pnl), fmt_pct(day_pnl_pct))
        with col3:
            st.metric("Cash", fmt_currency(account["cash"]))
        with col4:
            st.metric("Buying Power", fmt_currency(account["buying_power"]))
        with col5:
            st.metric("Positions", len(positions))
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Portfolio performance chart
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            values = np.cumsum(np.random.randn(30) * 1000) + account["portfolio_value"]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=values,
                fill='tozeroy',
                line=dict(color='#667eea', width=2),
                fillcolor='rgba(102, 126, 234, 0.2)',
            ))
            fig.update_layout(
                title="Portfolio Performance (30D)",
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#161b22',
                font=dict(color='#8b949e'),
                xaxis=dict(gridcolor='#30363d'),
                yaxis=dict(gridcolor='#30363d', tickformat='$,.0f'),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Allocation pie
            labels = [p["symbol"] for p in positions] + ["Cash"]
            values = [p["current"] * p["qty"] for p in positions] + [account["cash"]]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(colors=['#667eea', '#3fb950', '#f85149', '#d29922', '#58a6ff', '#a371f7']),
            )])
            fig.update_layout(
                title="Allocation",
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8b949e'),
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ============ TAB 2: PORTFOLIO ============
    with tab2:
        st.markdown("### 💼 Current Positions")
        
        total_pnl = sum(p["pnl"] for p in positions)
        total_value = sum(p["current"] * p["qty"] for p in positions)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Value", fmt_currency(total_value))
        with col2:
            st.metric("Total P&L", fmt_currency(total_pnl), fmt_pct(total_pnl/total_value*100 if total_value else 0))
        with col3:
            winners = sum(1 for p in positions if p["pnl"] > 0)
            st.metric("Win Rate", f"{winners}/{len(positions)}")
        
        st.markdown("---")
        
        # Positions table
        df = pd.DataFrame(positions)
        df.columns = ["Symbol", "Qty", "Entry", "Current", "P&L"]
        df["Entry"] = df["Entry"].apply(lambda x: f"${x:.2f}")
        df["Current"] = df["Current"].apply(lambda x: f"${x:.2f}")
        df["P&L"] = df["P&L"].apply(lambda x: f"${x:+.2f}")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # P&L Bar Chart
        fig = go.Figure(data=[go.Bar(
            x=[p["symbol"] for p in positions],
            y=[p["pnl"] for p in positions],
            marker_color=['#3fb950' if p["pnl"] >= 0 else '#f85149' for p in positions],
            text=[f"${p['pnl']:+.0f}" for p in positions],
            textposition='auto',
        )])
        fig.update_layout(
            title="P&L by Position",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#161b22',
            font=dict(color='#8b949e'),
            xaxis=dict(gridcolor='#30363d'),
            yaxis=dict(gridcolor='#30363d'),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ============ TAB 3: ANALYTICS ============
    with tab3:
        st.markdown(f"### 📈 Analytics: {symbol}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### 🛡️ Hedge Engine")
            st.metric("Elasticity", f"{engines['hedge']['elasticity']:.1f}")
            st.metric("Movement Energy", f"{engines['hedge']['energy']:.2f}")
            st.metric("Asymmetry", f"{engines['hedge']['asymmetry']:+.3f}")
            regime = engines['hedge']['regime']
            color = "#3fb950" if regime == "BULLISH" else "#f85149" if regime == "BEARISH" else "#d29922"
            st.markdown(f"**Regime:** <span style='color:{color}'>{regime}</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 💧 Liquidity")
            st.metric("Score", f"{engines['liquidity']['score']:.3f}")
            st.metric("Spread", f"{engines['liquidity']['spread']*100:.3f}%")
            st.metric("Impact", f"{engines['liquidity']['impact']*100:.3f}%")
        
        with col3:
            st.markdown("#### 📰 Sentiment")
            st.metric("Overall", f"{engines['sentiment']['score']:+.3f}")
            st.metric("News", f"{engines['sentiment']['news']:+.3f}")
            st.metric("Social", f"{engines['sentiment']['social']:+.3f}")
        
        with col4:
            st.markdown("#### ⚡ Elasticity")
            st.metric("Volatility", f"{engines['elasticity']['volatility']*100:.1f}%")
            st.metric("Trend", f"{engines['elasticity']['trend']:+.3f}")
            st.markdown(f"**Regime:** {engines['elasticity']['regime']}")
        
        st.markdown("---")
        
        # Sentiment Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=engines['sentiment']['score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment Score", 'font': {'color': '#8b949e'}},
            number={'font': {'color': '#f0f6fc'}},
            gauge={
                'axis': {'range': [-1, 1], 'tickcolor': '#30363d'},
                'bar': {'color': '#667eea'},
                'bgcolor': '#21262d',
                'bordercolor': '#30363d',
                'steps': [
                    {'range': [-1, -0.3], 'color': 'rgba(248, 81, 73, 0.3)'},
                    {'range': [-0.3, 0.3], 'color': 'rgba(210, 153, 34, 0.3)'},
                    {'range': [0.3, 1], 'color': 'rgba(63, 185, 80, 0.3)'},
                ],
            }
        ))
        fig.update_layout(
            height=250,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f0f6fc'),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ============ TAB 4: SIGNALS ============
    with tab4:
        st.markdown("### 🎯 Trading Signals")
        
        agents = [
            {"name": "Hedge Agent", "signal": "BUY", "conf": 0.78},
            {"name": "Liquidity Agent", "signal": "HOLD", "conf": 0.65},
            {"name": "Sentiment Agent", "signal": "BUY", "conf": 0.82},
            {"name": "Momentum Agent", "signal": "BUY", "conf": 0.71},
            {"name": "Mean Reversion", "signal": "SELL", "conf": 0.58},
        ]
        
        # Consensus
        buy_count = sum(1 for a in agents if a["signal"] == "BUY")
        sell_count = sum(1 for a in agents if a["signal"] == "SELL")
        
        if buy_count > sell_count:
            consensus = "BUY"
            css_class = "signal-buy"
        elif sell_count > buy_count:
            consensus = "SELL"
            css_class = "signal-sell"
        else:
            consensus = "HOLD"
            css_class = "signal-hold"
        
        avg_conf = sum(a["conf"] for a in agents) / len(agents)
        
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: #161b22; border-radius: 12px; margin-bottom: 1rem;">
            <div style="color: #8b949e; text-transform: uppercase; letter-spacing: 2px;">Consensus Signal</div>
            <div style="margin: 1rem 0;"><span class="{css_class}" style="font-size: 2rem;">{consensus}</span></div>
            <div style="color: #f0f6fc;">Confidence: {avg_conf:.1%} | {len(agents)} Agents</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Agent cards
        cols = st.columns(len(agents))
        for i, agent in enumerate(agents):
            with cols[i]:
                sig_class = f"signal-{agent['signal'].lower()}"
                st.markdown(f"""
                <div style="background: #161b22; border-radius: 8px; padding: 1rem; text-align: center;">
                    <div style="color: #8b949e; font-size: 0.8rem;">{agent['name']}</div>
                    <div style="margin: 0.5rem 0;"><span class="{sig_class}">{agent['signal']}</span></div>
                    <div style="color: #f0f6fc; font-size: 1.2rem; font-weight: bold;">{agent['conf']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # ============ TAB 5: RISK ============
    with tab5:
        st.markdown("### ⚠️ Risk Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Beta", "1.05")
        with col2:
            st.metric("Net Delta", "+0.32")
        with col3:
            st.metric("VaR (95%)", fmt_currency(account["portfolio_value"] * 0.035))
        with col4:
            st.metric("Max Drawdown", "-8.5%")
        
        st.markdown("---")
        
        # Correlation heatmap
        symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]
        corr = np.array([
            [1.00, 0.92, 0.75, 0.68, 0.45],
            [0.92, 1.00, 0.80, 0.72, 0.48],
            [0.75, 0.80, 1.00, 0.65, 0.38],
            [0.68, 0.72, 0.65, 1.00, 0.55],
            [0.45, 0.48, 0.38, 0.55, 1.00],
        ])
        
        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=symbols,
            y=symbols,
            colorscale=[[0, '#f85149'], [0.5, '#21262d'], [1, '#3fb950']],
            zmin=0, zmax=1,
            text=np.round(corr, 2),
            texttemplate='%{text}',
            textfont={'color': '#f0f6fc'},
        ))
        fig.update_layout(
            title="Correlation Matrix",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#161b22',
            font=dict(color='#8b949e'),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Auto refresh
    if auto_refresh:
        import time
        time.sleep(10)
        st.rerun()


if __name__ == "__main__":
    main()
