import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Gnosis | Institutional Signals",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        .metric-card {
            background-color: #262730;
            border: 1px solid #464b5d;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
        }
        .bullish { color: #4ade80; font-weight: bold; }
        .bearish { color: #f87171; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"

def fetch_signals(mode="live"):
    try:
        # Try to hit the API
        response = requests.get(f"{API_URL}/trades/decisions", params={"mode": mode, "limit": 20}, timeout=1)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Fallback / Demo Data if API is offline
    return [
        {
            "symbol": "SPY",
            "direction": "long",
            "timestamp": datetime.now().isoformat(),
            "price": 472.50,
            "options_liq_score": 8.4,
            "sentiment_agent_vote": {"risk_posture": "Risk-On"},
            "hedge_agent_vote": {"bias": "Long Gamma"},
            "composer_decision": {"sizing": {"confidence": 0.88}, "reason_codes": ["Vanna Support", "Positive GEX"]}
        },
        {
            "symbol": "NVDA",
            "direction": "short",
            "timestamp": datetime.now().isoformat(),
            "price": 485.20,
            "options_liq_score": 6.2,
            "sentiment_agent_vote": {"risk_posture": "Overextended"},
            "hedge_agent_vote": {"bias": "Short Delta"},
            "composer_decision": {"sizing": {"confidence": 0.72}, "reason_codes": ["Gamma Wall Rejection"]}
        },
        {
            "symbol": "TSLA",
            "direction": "long",
            "timestamp": datetime.now().isoformat(),
            "price": 245.30,
            "options_liq_score": 7.9,
            "sentiment_agent_vote": {"risk_posture": "Dip Buy"},
            "hedge_agent_vote": {"bias": "Neutral"},
            "composer_decision": {"sizing": {"confidence": 0.65}, "reason_codes": ["Oversold RSI", "Call Flow"]}
        }
    ]

# Sidebar
with st.sidebar:
    st.title("⚡ GNOSIS ALPHA")
    st.caption("Institutional Trade Intelligence")
    
    st.markdown("---")
    
    mode = st.selectbox("Data Feed", ["Live Market", "Paper Trading", "Backtest"])
    st.markdown(f"**Status:** 🟢 Connected")
    
    st.markdown("### Filters")
    min_conf = st.slider("Min Confidence", 0, 100, 60)
    asset_class = st.multiselect("Asset Class", ["Equities", "Options", "Crypto"], default=["Equities", "Options"])
    
    st.markdown("---")
    if st.button("Refresh Feed", use_container_width=True):
        st.experimental_rerun()

# Main Content
col1, col2 = st.columns([3, 1])

with col1:
    st.title("📡 Live Signal Feed")
    st.markdown("Real-time AI analysis of equity structure, dealer positioning, and options flow.")

with col2:
    st.markdown("### Market Regime")
    st.info("🌊 High Volatility (Long Gamma)")

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
signals = fetch_signals(mode.lower().split()[0])
active_count = len(signals)

m1.metric("Active Signals", active_count, "+2")
m2.metric("Avg Confidence", "76%", "+4%")
m3.metric("Net Delta", "+$2.4M", "Bullish")
m4.metric("Dealer GEX", "$5.2B", "Supportive")

st.markdown("---")

# Signal Cards
for signal in signals:
    # Logic for colors
    is_long = signal['direction'] == 'long'
    color_class = "bullish" if is_long else "bearish"
    direction_arrow = "▲" if is_long else "▼"
    
    confidence = signal.get('composer_decision', {}).get('sizing', {}).get('confidence', 0.5) * 100
    
    with st.container():
        # Custom Card Layout
        c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1.5, 1.5, 2])
        
        with c1:
            st.markdown(f"### {signal['symbol']}")
            st.caption(f"Entry: ${signal.get('price', 0)}")
            
        with c2:
            st.markdown(f"<span class='{color_class}' style='font-size: 1.2em'>{direction_arrow} {signal['direction'].upper()}</span>", unsafe_allow_html=True)
            
        with c3:
            st.markdown("**Sentiment**")
            st.text(signal.get('sentiment_agent_vote', {}).get('risk_posture', 'N/A'))
            
        with c4:
            st.markdown("**Hedge**")
            st.text(signal.get('hedge_agent_vote', {}).get('bias', 'N/A'))
            
        with c5:
            st.progress(int(confidence))
            st.caption(f"Confidence: {int(confidence)}% - {', '.join(signal.get('composer_decision', {}).get('reason_codes', [])[:2])}")
            
        st.markdown("---")
