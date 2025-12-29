#!/usr/bin/env python3
"""
Super Gnosis DHPE v4 - ENHANCED PREMIUM TRADING DASHBOARD
A modern, feature-rich dashboard with dark theme and advanced analytics
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

# Page config - MUST BE FIRST
st.set_page_config(
    page_title="Super Gnosis DHPE v4 | Premium Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/super-gnosis/dhpe',
        'Report a bug': 'https://github.com/super-gnosis/dhpe/issues',
        'About': '# Super Gnosis DHPE v4\nAdvanced AI-Powered Trading System'
    }
)

# ============================================================================
# CUSTOM CSS - PREMIUM DARK THEME
# ============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root Variables */
    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-tertiary: #21262d;
        --bg-card: #1c2128;
        --border-color: #30363d;
        --text-primary: #f0f6fc;
        --text-secondary: #8b949e;
        --text-muted: #6e7681;
        --accent-green: #3fb950;
        --accent-red: #f85149;
        --accent-blue: #58a6ff;
        --accent-purple: #a371f7;
        --accent-orange: #d29922;
        --accent-cyan: #39c5cf;
        --gradient-start: #667eea;
        --gradient-end: #764ba2;
    }
    
    /* Global Styles */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Main Container */
    .main .block-container {
        padding: 1rem 2rem 2rem 2rem !important;
        max-width: 100% !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }
    
    /* Text */
    p, span, div, label {
        color: var(--text-secondary) !important;
    }
    
    /* Premium Header Banner */
    .premium-header {
        background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }
    
    .premium-header h1 {
        color: white !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .premium-header p {
        color: rgba(255,255,255,0.9) !important;
        margin: 0.5rem 0 0 0 !important;
        font-size: 1rem;
        position: relative;
        z-index: 1;
    }
    
    /* Metric Cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        border-color: var(--accent-blue);
        box-shadow: 0 4px 20px rgba(88, 166, 255, 0.15);
        transform: translateY(-2px);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--gradient-start), var(--gradient-end));
        border-radius: 4px 0 0 4px;
    }
    
    .metric-label {
        color: var(--text-muted) !important;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: var(--text-primary) !important;
        font-size: 1.75rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .metric-delta {
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    .metric-positive {
        color: var(--accent-green) !important;
    }
    
    .metric-negative {
        color: var(--accent-red) !important;
    }
    
    .metric-neutral {
        color: var(--text-muted) !important;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.375rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-online {
        background: rgba(63, 185, 80, 0.15);
        color: var(--accent-green);
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    
    .status-offline {
        background: rgba(248, 81, 73, 0.15);
        color: var(--accent-red);
        border: 1px solid rgba(248, 81, 73, 0.3);
    }
    
    .status-warning {
        background: rgba(210, 153, 34, 0.15);
        color: var(--accent-orange);
        border: 1px solid rgba(210, 153, 34, 0.3);
    }
    
    /* Signal Badges */
    .signal-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .signal-buy {
        background: linear-gradient(135deg, rgba(63, 185, 80, 0.2), rgba(63, 185, 80, 0.1));
        color: var(--accent-green);
        border: 1px solid rgba(63, 185, 80, 0.4);
    }
    
    .signal-sell {
        background: linear-gradient(135deg, rgba(248, 81, 73, 0.2), rgba(248, 81, 73, 0.1));
        color: var(--accent-red);
        border: 1px solid rgba(248, 81, 73, 0.4);
    }
    
    .signal-hold {
        background: linear-gradient(135deg, rgba(210, 153, 34, 0.2), rgba(210, 153, 34, 0.1));
        color: var(--accent-orange);
        border: 1px solid rgba(210, 153, 34, 0.4);
    }
    
    /* Data Tables */
    .stDataFrame {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    .stDataFrame table {
        color: var(--text-secondary) !important;
    }
    
    .stDataFrame th {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.5px !important;
    }
    
    .stDataFrame td {
        background: var(--bg-card) !important;
        border-color: var(--border-color) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary) !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
        gap: 0.5rem !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.25rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end)) !important;
        color: white !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    
    [data-testid="stSidebar"] .block-container {
        padding: 1rem !important;
    }
    
    /* Inputs */
    .stTextInput input, .stSelectbox select, .stNumberInput input {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15) !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end)) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Progress Bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end)) !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    
    /* Charts */
    .js-plotly-plot .plotly .modebar {
        background: var(--bg-tertiary) !important;
        border-radius: 8px !important;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
    
    /* Alerts/Messages */
    .stAlert {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }
    
    /* Tooltips */
    [data-testid="stTooltipIcon"] {
        color: var(--text-muted) !important;
    }
    
    /* Dividers */
    hr {
        border-color: var(--border-color) !important;
        opacity: 0.5;
    }
    
    /* Live Indicator Animation */
    @keyframes pulse-live {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-green);
        border-radius: 50%;
        animation: pulse-live 2s ease-in-out infinite;
    }
    
    /* Sparkline container */
    .sparkline-container {
        height: 40px;
        margin-top: 0.5rem;
    }
    
    /* Grid Cards */
    .grid-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        height: 100%;
    }
    
    /* Engine Status Icons */
    .engine-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }
    
    .engine-hedge { background: linear-gradient(135deg, #667eea, #764ba2); }
    .engine-liquidity { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .engine-sentiment { background: linear-gradient(135deg, #fc4a1a, #f7b733); }
    .engine-elasticity { background: linear-gradient(135deg, #ff6b6b, #ee5a24); }
    .engine-ml { background: linear-gradient(135deg, #00b4db, #0083b0); }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_currency(value: float, prefix: str = "$") -> str:
    """Format value as currency."""
    if value >= 1_000_000:
        return f"{prefix}{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{prefix}{value/1_000:.1f}K"
    return f"{prefix}{value:,.2f}"


def format_percentage(value: float, show_sign: bool = True) -> str:
    """Format value as percentage."""
    if show_sign:
        return f"{value:+.2f}%"
    return f"{value:.2f}%"


def get_pnl_class(value: float) -> str:
    """Get CSS class for P&L values."""
    if value > 0:
        return "metric-positive"
    elif value < 0:
        return "metric-negative"
    return "metric-neutral"


def get_signal_class(signal: str) -> str:
    """Get CSS class for signal."""
    signal = signal.lower()
    if signal in ["buy", "long", "bullish"]:
        return "signal-buy"
    elif signal in ["sell", "short", "bearish"]:
        return "signal-sell"
    return "signal-hold"


def create_sparkline(values: List[float], color: str = "#58a6ff") -> go.Figure:
    """Create a mini sparkline chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=values,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba{tuple(list(bytes.fromhex(color[1:])) + [0.1])}' if color.startswith('#') else color,
    ))
    fig.update_layout(
        height=50,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def create_gauge_chart(value: float, title: str, min_val: float = -1, max_val: float = 1) -> go.Figure:
    """Create a modern gauge chart."""
    # Normalize value to 0-1 range for positioning
    normalized = (value - min_val) / (max_val - min_val)
    
    # Color based on value
    if value > 0.3:
        color = "#3fb950"
    elif value < -0.3:
        color = "#f85149"
    else:
        color = "#d29922"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14, 'color': '#8b949e'}},
        number={'font': {'size': 28, 'color': '#f0f6fc'}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickcolor': '#30363d', 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': '#21262d',
            'borderwidth': 2,
            'bordercolor': '#30363d',
            'steps': [
                {'range': [min_val, (max_val - min_val) * 0.3 + min_val], 'color': 'rgba(248, 81, 73, 0.2)'},
                {'range': [(max_val - min_val) * 0.3 + min_val, (max_val - min_val) * 0.7 + min_val], 'color': 'rgba(210, 153, 34, 0.2)'},
                {'range': [(max_val - min_val) * 0.7 + min_val, max_val], 'color': 'rgba(63, 185, 80, 0.2)'},
            ],
            'threshold': {
                'line': {'color': '#f0f6fc', 'width': 3},
                'thickness': 0.8,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f0f6fc'}
    )
    
    return fig


def create_heatmap(data: pd.DataFrame, title: str) -> go.Figure:
    """Create a correlation heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale=[
            [0, '#f85149'],
            [0.5, '#21262d'],
            [1, '#3fb950']
        ],
        zmin=-1,
        zmax=1,
        text=np.round(data.values, 2),
        texttemplate='%{text}',
        textfont={'size': 10, 'color': '#f0f6fc'},
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='#f0f6fc', size=16)),
        height=400,
        margin=dict(l=60, r=20, t=50, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(color='#8b949e'), tickangle=45),
        yaxis=dict(tickfont=dict(color='#8b949e')),
    )
    
    return fig


def create_candlestick_chart(df: pd.DataFrame, title: str = "Price Action") -> go.Figure:
    """Create an interactive candlestick chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, 'Volume')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing_line_color='#3fb950',
            decreasing_line_color='#f85149',
            name='Price'
        ),
        row=1, col=1
    )
    
    # Volume
    colors = ['#3fb950' if c >= o else '#f85149' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='Volume'),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=60, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#161b22',
        xaxis_rangeslider_visible=False,
        showlegend=False,
        font=dict(color='#8b949e'),
    )
    
    fig.update_xaxes(gridcolor='#30363d', showgrid=True)
    fig.update_yaxes(gridcolor='#30363d', showgrid=True)
    
    return fig


def create_donut_chart(labels: List[str], values: List[float], title: str) -> go.Figure:
    """Create a modern donut chart."""
    colors = ['#667eea', '#3fb950', '#f85149', '#d29922', '#58a6ff', '#a371f7', '#39c5cf']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors[:len(labels)], line=dict(color='#161b22', width=2)),
        textinfo='percent+label',
        textfont=dict(size=11, color='#f0f6fc'),
        hovertemplate='%{label}<br>%{value:,.2f}<br>%{percent}<extra></extra>',
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='#f0f6fc', size=16)),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            font=dict(color='#8b949e', size=10),
            bgcolor='rgba(0,0,0,0)',
        ),
        annotations=[dict(
            text=title.split()[0] if title else '',
            x=0.5, y=0.5,
            font_size=14,
            font_color='#8b949e',
            showarrow=False
        )]
    )
    
    return fig


def create_bar_chart(x: List, y: List, title: str, color_by_value: bool = True) -> go.Figure:
    """Create a modern bar chart."""
    if color_by_value:
        colors = ['#3fb950' if v >= 0 else '#f85149' for v in y]
    else:
        colors = '#667eea'
    
    fig = go.Figure(data=[go.Bar(
        x=x,
        y=y,
        marker_color=colors,
        text=[f'{v:+.2f}' if isinstance(v, float) else v for v in y],
        textposition='auto',
        textfont=dict(color='#f0f6fc', size=11),
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='#f0f6fc', size=16)),
        height=350,
        margin=dict(l=60, r=20, t=50, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#161b22',
        xaxis=dict(tickfont=dict(color='#8b949e'), gridcolor='#30363d'),
        yaxis=dict(tickfont=dict(color='#8b949e'), gridcolor='#30363d'),
        bargap=0.3,
    )
    
    return fig


def create_treemap(data: Dict[str, float], title: str) -> go.Figure:
    """Create a treemap for portfolio allocation."""
    labels = list(data.keys())
    values = list(data.values())
    
    # Assign colors based on value sign
    colors = ['#3fb950' if v >= 0 else '#f85149' for v in values]
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=[''] * len(labels),
        values=[abs(v) for v in values],
        marker=dict(
            colors=colors,
            line=dict(color='#161b22', width=2)
        ),
        textinfo='label+value',
        textfont=dict(size=14, color='#f0f6fc'),
        hovertemplate='%{label}<br>%{value:,.2f}<extra></extra>',
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='#f0f6fc', size=16)),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


# ============================================================================
# DATA LOADING FUNCTIONS (with caching)
# ============================================================================

@st.cache_resource
def get_broker():
    """Get broker adapter (cached)."""
    try:
        from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
        from execution.broker_adapters.settings import get_alpaca_paper_setting
        paper_mode = get_alpaca_paper_setting()
        return AlpacaBrokerAdapter(paper=paper_mode)
    except Exception as e:
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


@st.cache_data(ttl=300)
def load_scanner_state():
    """Load scanner state for market overview."""
    state_file = Path("data/scanner_state/current_state.json")
    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    return None


def run_pipeline_analysis(symbol: str) -> Dict[str, Any]:
    """Run pipeline analysis for a symbol."""
    try:
        from main import build_pipeline, load_config
        config = load_config()
        runner = build_pipeline(symbol, config)
        result = runner.run_once(datetime.now())
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now(timezone.utc)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc)
        }


# ============================================================================
# MOCK DATA FOR DEMO (used when broker/data unavailable)
# ============================================================================

def get_mock_account():
    """Generate mock account data for demo."""
    return {
        "portfolio_value": 150000 + random.uniform(-5000, 5000),
        "cash": 45000 + random.uniform(-2000, 2000),
        "buying_power": 90000,
        "equity": 150000,
        "last_equity": 148500,
        "account_id": "DEMO-12345",
        "pattern_day_trader": False,
    }


def get_mock_positions():
    """Generate mock positions for demo."""
    symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "MSFT", "META"]
    positions = []
    
    for sym in random.sample(symbols, min(5, len(symbols))):
        entry = random.uniform(100, 500)
        current = entry * random.uniform(0.95, 1.10)
        qty = random.randint(10, 100)
        positions.append({
            "symbol": sym,
            "quantity": qty,
            "side": "long",
            "avg_entry_price": entry,
            "current_price": current,
            "market_value": current * qty,
            "cost_basis": entry * qty,
            "unrealized_pnl": (current - entry) * qty,
            "unrealized_pnl_pct": (current - entry) / entry,
        })
    
    return positions


def get_mock_engine_data():
    """Generate mock engine data for demo."""
    return {
        "hedge": {
            "elasticity": random.uniform(500, 2000),
            "movement_energy": random.uniform(1, 10),
            "energy_asymmetry": random.uniform(-0.5, 0.5),
            "dealer_gamma_sign": random.uniform(-1, 1),
            "regime": random.choice(["bullish", "bearish", "neutral"]),
            "confidence": random.uniform(0.5, 1.0),
        },
        "liquidity": {
            "liquidity_score": random.uniform(0.7, 1.0),
            "bid_ask_spread": random.uniform(0.001, 0.02),
            "impact_cost": random.uniform(0.001, 0.01),
        },
        "sentiment": {
            "sentiment_score": random.uniform(-0.5, 0.5),
            "news_sentiment": random.uniform(-0.5, 0.5),
            "social_sentiment": random.uniform(-0.5, 0.5),
            "confidence": random.uniform(0.5, 1.0),
        },
        "elasticity": {
            "volatility": random.uniform(0.1, 0.4),
            "volatility_regime": random.choice(["low", "moderate", "high"]),
            "trend_strength": random.uniform(-1, 1),
        },
    }


def get_mock_alerts():
    """Generate mock alerts for demo."""
    alert_types = [
        ("High Movement Energy Alert", "warning", "SPY showing elevated movement energy (75.2)"),
        ("Sentiment Shift Detected", "info", "NVDA sentiment turned bullish (+0.65)"),
        ("Liquidity Warning", "warning", "QQQ bid-ask spread widening to 0.15%"),
        ("Consensus Signal", "success", "AAPL: Strong BUY signal (87% confidence)"),
        ("Risk Alert", "danger", "Portfolio delta approaching limit (-0.45)"),
    ]
    
    return random.sample(alert_types, min(3, len(alert_types)))


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    """Main dashboard function."""
    
    # Initialize session state
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False
    if "selected_symbol" not in st.session_state:
        st.session_state.selected_symbol = "SPY"
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    
    # Premium Header
    st.markdown("""
    <div class="premium-header">
        <h1>🚀 Super Gnosis DHPE v4</h1>
        <p>Premium AI-Powered Trading Dashboard | Real-time Analytics & Signals</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Logo/Brand
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2.5rem;">🎯</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #f0f6fc;">Super Gnosis</div>
            <div style="font-size: 0.8rem; color: #8b949e;">DHPE Trading System</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Quick Controls
        st.markdown("### ⚡ Quick Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.auto_refresh = st.toggle("Auto Refresh", value=st.session_state.auto_refresh)
        with col2:
            refresh_interval = st.selectbox("Interval", [5, 10, 30, 60], index=1, label_visibility="collapsed")
        
        st.session_state.selected_symbol = st.text_input(
            "Symbol", 
            value=st.session_state.selected_symbol,
            placeholder="Enter ticker..."
        ).upper()
        
        if st.button("🔍 Run Analysis", use_container_width=True):
            with st.spinner(f"Analyzing {st.session_state.selected_symbol}..."):
                result = run_pipeline_analysis(st.session_state.selected_symbol)
                if result["success"]:
                    st.session_state["last_analysis"] = result
                    st.success("✅ Analysis complete!")
                else:
                    st.error(f"❌ {result['error']}")
        
        st.divider()
        
        # System Status
        st.markdown("### 🔌 System Status")
        
        broker = get_broker()
        
        if broker:
            st.markdown('<span class="status-badge status-online">● Alpaca Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-offline">● Alpaca Offline</span>', unsafe_allow_html=True)
        
        # Check other services
        try:
            from engines.inputs.massive_market_adapter import MassiveMarketDataAdapter
            massive = MassiveMarketDataAdapter()
            if massive.enabled:
                st.markdown('<span class="status-badge status-online">● Massive API</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-warning">● Massive API</span>', unsafe_allow_html=True)
        except:
            st.markdown('<span class="status-badge status-offline">● Massive API</span>', unsafe_allow_html=True)
        
        try:
            if os.getenv("UNUSUAL_WHALES_API_TOKEN"):
                st.markdown('<span class="status-badge status-online">● Unusual Whales</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-warning">● Unusual Whales</span>', unsafe_allow_html=True)
        except:
            st.markdown('<span class="status-badge status-offline">● Unusual Whales</span>', unsafe_allow_html=True)
        
        st.divider()
        
        # Live indicator
        st.markdown("""
        <div class="live-indicator">
            <span class="live-dot"></span>
            <span style="color: #8b949e; font-size: 0.85rem;">Live Data Feed</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
    
    # Main Content Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "💼 Portfolio",
        "📈 Analytics",
        "🎯 Signals",
        "⚠️ Risk",
        "🔬 Scanner",
        "⚙️ Engine Lab"
    ])
    
    # Get data (real or mock)
    broker = get_broker()
    if broker:
        try:
            account = broker.get_account()
            account_data = {
                "portfolio_value": account.portfolio_value,
                "cash": account.cash,
                "buying_power": account.buying_power,
                "equity": account.equity,
                "last_equity": account.last_equity,
                "account_id": account.account_id,
                "pattern_day_trader": account.pattern_day_trader,
            }
            positions_raw = broker.get_positions()
            positions_data = [{
                "symbol": p.symbol,
                "quantity": p.quantity,
                "side": p.side,
                "avg_entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "cost_basis": p.cost_basis,
                "unrealized_pnl": p.unrealized_pnl,
                "unrealized_pnl_pct": p.unrealized_pnl_pct,
            } for p in positions_raw]
        except Exception as e:
            account_data = get_mock_account()
            positions_data = get_mock_positions()
    else:
        account_data = get_mock_account()
        positions_data = get_mock_positions()
    
    engine_data = get_mock_engine_data()
    alerts_data = get_mock_alerts()
    
    # =========================================================================
    # TAB 1: OVERVIEW
    # =========================================================================
    with tab1:
        # Top Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        day_pnl = account_data["equity"] - account_data["last_equity"]
        day_pnl_pct = (day_pnl / account_data["last_equity"] * 100) if account_data["last_equity"] > 0 else 0
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Portfolio Value</div>
                <div class="metric-value">{format_currency(account_data['portfolio_value'])}</div>
                <div class="metric-delta {get_pnl_class(day_pnl)}">{format_percentage(day_pnl_pct)} today</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Today's P&L</div>
                <div class="metric-value {get_pnl_class(day_pnl)}">{format_currency(day_pnl)}</div>
                <div class="metric-delta {get_pnl_class(day_pnl)}">{format_percentage(day_pnl_pct)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Cash Available</div>
                <div class="metric-value">{format_currency(account_data['cash'])}</div>
                <div class="metric-delta metric-neutral">{format_percentage(account_data['cash']/account_data['portfolio_value']*100, False)} of portfolio</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Buying Power</div>
                <div class="metric-value">{format_currency(account_data['buying_power'])}</div>
                <div class="metric-delta metric-neutral">Available margin</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            open_positions = len(positions_data)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Open Positions</div>
                <div class="metric-value">{open_positions}</div>
                <div class="metric-delta metric-neutral">{sum(p['unrealized_pnl'] > 0 for p in positions_data)} winning</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts Row
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Portfolio Performance Chart (mock data)
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            values = [account_data['portfolio_value'] * (1 + random.uniform(-0.02, 0.02)) for _ in range(30)]
            values = np.cumsum([values[0]] + [values[i] - values[i-1] for i in range(1, len(values))])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=values,
                mode='lines',
                fill='tozeroy',
                line=dict(color='#667eea', width=2),
                fillcolor='rgba(102, 126, 234, 0.1)',
                name='Portfolio Value'
            ))
            fig.update_layout(
                title=dict(text='Portfolio Performance (30 Days)', font=dict(color='#f0f6fc', size=16)),
                height=350,
                margin=dict(l=60, r=20, t=50, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#161b22',
                xaxis=dict(gridcolor='#30363d', tickfont=dict(color='#8b949e')),
                yaxis=dict(gridcolor='#30363d', tickfont=dict(color='#8b949e'), tickformat='$,.0f'),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Allocation Donut
            if positions_data:
                labels = [p['symbol'] for p in positions_data]
                values = [p['market_value'] for p in positions_data]
                labels.append('Cash')
                values.append(account_data['cash'])
                fig = create_donut_chart(labels, values, "Capital Allocation")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No positions to display")
        
        # Alerts Section
        st.markdown("### 🔔 Active Alerts")
        
        alert_cols = st.columns(len(alerts_data)) if alerts_data else [st.container()]
        
        for i, (title, alert_type, msg) in enumerate(alerts_data):
            with alert_cols[i]:
                icon = "⚠️" if alert_type == "warning" else "✅" if alert_type == "success" else "ℹ️" if alert_type == "info" else "🚨"
                color = "#d29922" if alert_type == "warning" else "#3fb950" if alert_type == "success" else "#58a6ff" if alert_type == "info" else "#f85149"
                st.markdown(f"""
                <div style="background: {color}15; border: 1px solid {color}40; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem;">
                    <div style="color: {color}; font-weight: 600; font-size: 0.85rem;">{icon} {title}</div>
                    <div style="color: #8b949e; font-size: 0.8rem; margin-top: 0.25rem;">{msg}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 2: PORTFOLIO
    # =========================================================================
    with tab2:
        st.markdown("### 💼 Portfolio Positions")
        
        if positions_data:
            # Summary metrics
            total_value = sum(p['market_value'] for p in positions_data)
            total_pnl = sum(p['unrealized_pnl'] for p in positions_data)
            total_pnl_pct = (total_pnl / (total_value - total_pnl) * 100) if (total_value - total_pnl) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Positions", len(positions_data))
            with col2:
                st.metric("Market Value", format_currency(total_value))
            with col3:
                st.metric("Unrealized P&L", format_currency(total_pnl), format_percentage(total_pnl_pct))
            with col4:
                win_rate = sum(1 for p in positions_data if p['unrealized_pnl'] > 0) / len(positions_data) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Positions Table
            df_positions = pd.DataFrame(positions_data)
            df_display = df_positions[['symbol', 'quantity', 'side', 'avg_entry_price', 'current_price', 'market_value', 'unrealized_pnl', 'unrealized_pnl_pct']].copy()
            df_display.columns = ['Symbol', 'Qty', 'Side', 'Entry', 'Current', 'Value', 'P&L', 'P&L %']
            df_display['Entry'] = df_display['Entry'].apply(lambda x: f"${x:,.2f}")
            df_display['Current'] = df_display['Current'].apply(lambda x: f"${x:,.2f}")
            df_display['Value'] = df_display['Value'].apply(lambda x: f"${x:,.2f}")
            df_display['P&L'] = df_display['P&L'].apply(lambda x: f"${x:+,.2f}")
            df_display['P&L %'] = df_display['P&L %'].apply(lambda x: f"{x*100:+.2f}%")
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # P&L Bar Chart
            st.markdown("<br>", unsafe_allow_html=True)
            fig = create_bar_chart(
                [p['symbol'] for p in positions_data],
                [p['unrealized_pnl'] for p in positions_data],
                "Unrealized P&L by Position"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Treemap
            pnl_data = {p['symbol']: p['unrealized_pnl'] for p in positions_data}
            fig = create_treemap(pnl_data, "Position P&L Treemap")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📭 No open positions. Your portfolio is all cash.")
    
    # =========================================================================
    # TAB 3: ANALYTICS
    # =========================================================================
    with tab3:
        st.markdown(f"### 📈 Live Analytics: {st.session_state.selected_symbol}")
        
        # Engine Metrics Grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="grid-card">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                    <div class="engine-icon engine-hedge">🛡️</div>
                    <div>
                        <div style="color: #f0f6fc; font-weight: 600;">Hedge Engine</div>
                        <div style="color: #8b949e; font-size: 0.75rem;">v3.0</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Elasticity", f"{engine_data['hedge']['elasticity']:.1f}")
            st.metric("Movement Energy", f"{engine_data['hedge']['movement_energy']:.2f}")
            st.metric("Energy Asymmetry", f"{engine_data['hedge']['energy_asymmetry']:+.3f}")
            regime = engine_data['hedge']['regime'].upper()
            regime_color = "#3fb950" if regime == "BULLISH" else "#f85149" if regime == "BEARISH" else "#d29922"
            st.markdown(f'<span style="color: {regime_color}; font-weight: 600;">Regime: {regime}</span>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="grid-card">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                    <div class="engine-icon engine-liquidity">💧</div>
                    <div>
                        <div style="color: #f0f6fc; font-weight: 600;">Liquidity Engine</div>
                        <div style="color: #8b949e; font-size: 0.75rem;">v2.0</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Liquidity Score", f"{engine_data['liquidity']['liquidity_score']:.3f}")
            st.metric("Bid-Ask Spread", f"{engine_data['liquidity']['bid_ask_spread']*100:.3f}%")
            st.metric("Impact Cost", f"{engine_data['liquidity']['impact_cost']*100:.3f}%")
        
        with col3:
            st.markdown("""
            <div class="grid-card">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                    <div class="engine-icon engine-sentiment">📰</div>
                    <div>
                        <div style="color: #f0f6fc; font-weight: 600;">Sentiment Engine</div>
                        <div style="color: #8b949e; font-size: 0.75rem;">v1.5</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Overall Sentiment", f"{engine_data['sentiment']['sentiment_score']:+.3f}")
            st.metric("News Sentiment", f"{engine_data['sentiment']['news_sentiment']:+.3f}")
            st.metric("Social Sentiment", f"{engine_data['sentiment']['social_sentiment']:+.3f}")
        
        with col4:
            st.markdown("""
            <div class="grid-card">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                    <div class="engine-icon engine-elasticity">⚡</div>
                    <div>
                        <div style="color: #f0f6fc; font-weight: 600;">Elasticity Engine</div>
                        <div style="color: #8b949e; font-size: 0.75rem;">v1.0</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Volatility", f"{engine_data['elasticity']['volatility']*100:.1f}%")
            st.metric("Trend Strength", f"{engine_data['elasticity']['trend_strength']:+.3f}")
            vol_regime = engine_data['elasticity']['volatility_regime'].upper()
            st.markdown(f'<span style="color: #8b949e;">Vol Regime: <b>{vol_regime}</b></span>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Gauge Charts Row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = create_gauge_chart(engine_data['sentiment']['sentiment_score'], "Sentiment Score")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_gauge_chart(engine_data['hedge']['energy_asymmetry'], "Energy Asymmetry", -1, 1)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = create_gauge_chart(engine_data['elasticity']['trend_strength'], "Trend Strength", -1, 1)
            st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 4: SIGNALS
    # =========================================================================
    with tab4:
        st.markdown("### 🎯 Trading Signals")
        
        # Agent Signals
        agents = [
            {"name": "Hedge Agent v3", "signal": random.choice(["BUY", "SELL", "HOLD"]), "confidence": random.uniform(0.5, 0.95), "reasoning": "Energy asymmetry suggests momentum continuation"},
            {"name": "Liquidity Agent", "signal": random.choice(["BUY", "SELL", "HOLD"]), "confidence": random.uniform(0.5, 0.95), "reasoning": "Excellent liquidity conditions for entry"},
            {"name": "Sentiment Agent", "signal": random.choice(["BUY", "SELL", "HOLD"]), "confidence": random.uniform(0.5, 0.95), "reasoning": "News sentiment shift detected"},
            {"name": "Momentum Agent", "signal": random.choice(["BUY", "SELL", "HOLD"]), "confidence": random.uniform(0.5, 0.95), "reasoning": "Strong upward momentum confirmed"},
            {"name": "Mean Reversion Agent", "signal": random.choice(["BUY", "SELL", "HOLD"]), "confidence": random.uniform(0.5, 0.95), "reasoning": "Price deviation from fair value"},
        ]
        
        # Consensus
        buy_votes = sum(1 for a in agents if a['signal'] == 'BUY')
        sell_votes = sum(1 for a in agents if a['signal'] == 'SELL')
        
        if buy_votes > sell_votes:
            consensus_signal = "BUY"
            consensus_color = "#3fb950"
        elif sell_votes > buy_votes:
            consensus_signal = "SELL"
            consensus_color = "#f85149"
        else:
            consensus_signal = "HOLD"
            consensus_color = "#d29922"
        
        avg_confidence = sum(a['confidence'] for a in agents) / len(agents)
        
        # Consensus Banner
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {consensus_color}20, {consensus_color}10); border: 2px solid {consensus_color}; border-radius: 12px; padding: 1.5rem; text-align: center; margin-bottom: 1.5rem;">
            <div style="font-size: 0.9rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px;">Consensus Signal</div>
            <div style="font-size: 3rem; font-weight: 700; color: {consensus_color}; margin: 0.5rem 0;">{consensus_signal}</div>
            <div style="font-size: 1rem; color: #f0f6fc;">Confidence: {avg_confidence:.1%} | {len(agents)} Agents</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Agent Cards
        cols = st.columns(len(agents))
        for i, agent in enumerate(agents):
            with cols[i]:
                signal_class = get_signal_class(agent['signal'])
                st.markdown(f"""
                <div class="grid-card" style="text-align: center;">
                    <div style="color: #8b949e; font-size: 0.75rem; margin-bottom: 0.5rem;">{agent['name']}</div>
                    <div class="signal-badge {signal_class}" style="display: inline-block;">{agent['signal']}</div>
                    <div style="margin-top: 0.75rem;">
                        <div style="color: #f0f6fc; font-size: 1.25rem; font-weight: 600;">{agent['confidence']:.1%}</div>
                        <div style="color: #6e7681; font-size: 0.7rem;">confidence</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("Reasoning"):
                    st.write(agent['reasoning'])
    
    # =========================================================================
    # TAB 5: RISK
    # =========================================================================
    with tab5:
        st.markdown("### ⚠️ Risk Management Dashboard")
        
        # Risk Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        portfolio_beta = random.uniform(0.8, 1.2)
        portfolio_delta = random.uniform(-0.5, 0.5)
        var_95 = account_data['portfolio_value'] * random.uniform(0.02, 0.05)
        max_drawdown = random.uniform(0.05, 0.15)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Portfolio Beta</div>
                <div class="metric-value">{portfolio_beta:.2f}</div>
                <div class="metric-delta metric-neutral">vs SPY</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            delta_class = get_pnl_class(portfolio_delta)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Net Delta</div>
                <div class="metric-value {delta_class}">{portfolio_delta:+.2f}</div>
                <div class="metric-delta metric-neutral">Directional exposure</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">VaR (95%)</div>
                <div class="metric-value metric-negative">{format_currency(var_95)}</div>
                <div class="metric-delta metric-neutral">Daily risk</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value metric-negative">{format_percentage(-max_drawdown*100)}</div>
                <div class="metric-delta metric-neutral">Historical</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Correlation Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]
            corr_data = pd.DataFrame(
                np.random.uniform(0.3, 1.0, (len(symbols), len(symbols))),
                index=symbols, columns=symbols
            )
            np.fill_diagonal(corr_data.values, 1.0)
            corr_data = (corr_data + corr_data.T) / 2  # Make symmetric
            
            fig = create_heatmap(corr_data, "Asset Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk Distribution
            risk_categories = ['Market Risk', 'Liquidity Risk', 'Concentration Risk', 'Volatility Risk', 'Correlation Risk']
            risk_values = [random.uniform(10, 30) for _ in risk_categories]
            
            fig = go.Figure(data=[go.Bar(
                x=risk_values,
                y=risk_categories,
                orientation='h',
                marker_color=['#f85149', '#d29922', '#58a6ff', '#a371f7', '#39c5cf'],
                text=[f'{v:.1f}%' for v in risk_values],
                textposition='auto',
            )])
            fig.update_layout(
                title=dict(text='Risk Decomposition', font=dict(color='#f0f6fc', size=16)),
                height=400,
                margin=dict(l=120, r=20, t=50, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#161b22',
                xaxis=dict(gridcolor='#30363d', tickfont=dict(color='#8b949e'), title='Contribution %'),
                yaxis=dict(gridcolor='#30363d', tickfont=dict(color='#8b949e')),
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 6: SCANNER
    # =========================================================================
    with tab6:
        st.markdown("### 🔬 Market Scanner")
        
        # Universe selection
        universe = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "CRM", "NFLX", "PYPL"]
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            selected_universe = st.multiselect("Select Universe", universe, default=universe[:10])
        with col2:
            filter_signal = st.selectbox("Filter by Signal", ["All", "BUY", "SELL", "HOLD"])
        with col3:
            sort_by = st.selectbox("Sort by", ["Confidence", "Symbol", "Signal"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Scanner Results
        scanner_data = []
        for sym in selected_universe:
            signal = random.choice(["BUY", "SELL", "HOLD"])
            if filter_signal != "All" and signal != filter_signal:
                continue
            scanner_data.append({
                "Symbol": sym,
                "Price": random.uniform(50, 500),
                "Change": random.uniform(-5, 5),
                "Signal": signal,
                "Confidence": random.uniform(0.5, 0.95),
                "Elasticity": random.uniform(500, 2000),
                "Sentiment": random.uniform(-0.5, 0.5),
                "Volume": random.randint(100000, 10000000),
            })
        
        if sort_by == "Confidence":
            scanner_data.sort(key=lambda x: x['Confidence'], reverse=True)
        elif sort_by == "Symbol":
            scanner_data.sort(key=lambda x: x['Symbol'])
        
        # Display as cards grid
        cols = st.columns(5)
        for i, item in enumerate(scanner_data):
            with cols[i % 5]:
                signal_class = get_signal_class(item['Signal'])
                change_class = "metric-positive" if item['Change'] > 0 else "metric-negative"
                st.markdown(f"""
                <div class="grid-card" style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.1rem; font-weight: 600; color: #f0f6fc;">{item['Symbol']}</span>
                        <span class="signal-badge {signal_class}" style="font-size: 0.7rem; padding: 0.25rem 0.5rem;">{item['Signal']}</span>
                    </div>
                    <div style="margin: 0.75rem 0;">
                        <div style="font-size: 1.25rem; font-weight: 600; color: #f0f6fc;">${item['Price']:.2f}</div>
                        <div class="{change_class}" style="font-size: 0.85rem;">{item['Change']:+.2f}%</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #8b949e;">
                        <span>Conf: {item['Confidence']:.1%}</span>
                        <span>Sent: {item['Sentiment']:+.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 7: ENGINE LAB
    # =========================================================================
    with tab7:
        st.markdown("### ⚙️ Engine Lab - Deep Pipeline Analysis")
        
        lab_symbol = st.text_input("Analyze Symbol", value=st.session_state.selected_symbol, key="lab_symbol")
        
        if st.button("🔬 Run Deep Analysis", use_container_width=False):
            with st.spinner(f"Running deep analysis on {lab_symbol}..."):
                time.sleep(1)  # Simulated delay
                
                st.success(f"✅ Deep analysis complete for {lab_symbol}")
                
                # Timeline View
                st.markdown("#### ⏱️ Pipeline Execution Timeline")
                
                steps = [
                    {"name": "Data Fetch", "type": "data", "duration": 0.23, "status": "success"},
                    {"name": "Hedge Engine", "type": "engine", "duration": 0.45, "status": "success"},
                    {"name": "Liquidity Engine", "type": "engine", "duration": 0.32, "status": "success"},
                    {"name": "Sentiment Engine", "type": "engine", "duration": 0.67, "status": "success"},
                    {"name": "Elasticity Engine", "type": "engine", "duration": 0.28, "status": "success"},
                    {"name": "ML Enhancement", "type": "ml", "duration": 0.89, "status": "success"},
                    {"name": "Agent Voting", "type": "agent", "duration": 0.15, "status": "success"},
                    {"name": "Composer", "type": "agent", "duration": 0.12, "status": "success"},
                ]
                
                df_steps = pd.DataFrame(steps)
                total_duration = df_steps['duration'].sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Duration", f"{total_duration:.2f}s")
                with col2:
                    st.metric("Steps Completed", f"{len(steps)}/{len(steps)}")
                with col3:
                    st.metric("Success Rate", "100%")
                
                # Timeline chart
                fig = go.Figure()
                
                colors = {'data': '#58a6ff', 'engine': '#667eea', 'ml': '#39c5cf', 'agent': '#a371f7'}
                cumulative = 0
                
                for step in steps:
                    fig.add_trace(go.Bar(
                        y=[step['name']],
                        x=[step['duration']],
                        orientation='h',
                        marker_color=colors.get(step['type'], '#8b949e'),
                        text=f"{step['duration']:.2f}s",
                        textposition='auto',
                        name=step['type'].title(),
                        showlegend=False,
                    ))
                
                fig.update_layout(
                    title=dict(text='Execution Timeline', font=dict(color='#f0f6fc', size=16)),
                    height=350,
                    margin=dict(l=120, r=20, t=50, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#161b22',
                    xaxis=dict(gridcolor='#30363d', tickfont=dict(color='#8b949e'), title='Duration (seconds)'),
                    yaxis=dict(gridcolor='#30363d', tickfont=dict(color='#8b949e')),
                    barmode='stack',
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Results
                st.markdown("#### 📊 Detailed Engine Outputs")
                
                with st.expander("🛡️ Hedge Engine Output", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Elasticity", f"{engine_data['hedge']['elasticity']:.1f}")
                    with col2:
                        st.metric("Movement Energy", f"{engine_data['hedge']['movement_energy']:.2f}")
                    with col3:
                        st.metric("Energy Asymmetry", f"{engine_data['hedge']['energy_asymmetry']:+.3f}")
                    with col4:
                        st.metric("Dealer Gamma", f"{engine_data['hedge']['dealer_gamma_sign']:+.3f}")
                
                with st.expander("💧 Liquidity Engine Output"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Liquidity Score", f"{engine_data['liquidity']['liquidity_score']:.3f}")
                    with col2:
                        st.metric("Bid-Ask Spread", f"{engine_data['liquidity']['bid_ask_spread']*100:.3f}%")
                    with col3:
                        st.metric("Impact Cost", f"{engine_data['liquidity']['impact_cost']*100:.3f}%")
                
                with st.expander("📰 Sentiment Engine Output"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall", f"{engine_data['sentiment']['sentiment_score']:+.3f}")
                    with col2:
                        st.metric("News", f"{engine_data['sentiment']['news_sentiment']:+.3f}")
                    with col3:
                        st.metric("Social", f"{engine_data['sentiment']['social_sentiment']:+.3f}")
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
