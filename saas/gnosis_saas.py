#!/usr/bin/env python3
"""
GNOSIS SaaS - Autonomous Trading Intelligence Platform

A self-running monitoring and signaling service.
"Set it and forget it" - GNOSIS watches the markets for you.

Author: GNOSIS Trading System
Version: 3.0.0 - Enhanced with Engine Outputs, Agent Analysis & Positions Monitor
"""

import asyncio
import json
import os
import sys
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import deque

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx

# Import GNOSIS components
from universe.dynamic_universe import DynamicUniverseManager, DYNAMIC_UNIVERSE


def load_env():
    """Load environment variables."""
    env_path = Path(__file__).parent.parent / ".env"
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ[key] = value
    except FileNotFoundError:
        pass


load_env()

# Initialize app
app = FastAPI(
    title="GNOSIS",
    description="Autonomous Trading Intelligence",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize universe manager
universe_manager = DynamicUniverseManager()

# WebSocket connections
active_connections: List[WebSocket] = []

# Signal history (last 50 signals)
signal_history = deque(maxlen=50)

# Alpaca API configuration
ALPACA_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
_raw_base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_BASE_URL = _raw_base_url.rstrip("/").replace("/v2", "")


# =============================================================================
# ALPACA INTEGRATION
# =============================================================================

async def get_alpaca_account() -> Dict[str, Any]:
    """Get account information from Alpaca."""
    if not ALPACA_KEY or not ALPACA_SECRET:
        return {"error": "API credentials not configured"}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ALPACA_BASE_URL}/v2/account",
                headers={
                    "APCA-API-KEY-ID": ALPACA_KEY,
                    "APCA-API-SECRET-KEY": ALPACA_SECRET,
                },
                timeout=10.0
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": data.get("status", "unknown"),
                    "equity": float(data.get("equity", 0)),
                    "cash": float(data.get("cash", 0)),
                    "buying_power": float(data.get("buying_power", 0)),
                    "portfolio_value": float(data.get("portfolio_value", 0)),
                    "last_equity": float(data.get("last_equity", 0)),
                    "day_pl": float(data.get("equity", 0)) - float(data.get("last_equity", 0)),
                    "day_pl_pct": ((float(data.get("equity", 0)) / float(data.get("last_equity", 1))) - 1) * 100 if float(data.get("last_equity", 0)) > 0 else 0,
                    "long_market_value": float(data.get("long_market_value", 0)),
                    "short_market_value": float(data.get("short_market_value", 0)),
                }
            return {"error": f"API returned {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


async def get_alpaca_positions() -> List[Dict[str, Any]]:
    """Get current positions from Alpaca."""
    if not ALPACA_KEY or not ALPACA_SECRET:
        return []
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ALPACA_BASE_URL}/v2/positions",
                headers={
                    "APCA-API-KEY-ID": ALPACA_KEY,
                    "APCA-API-SECRET-KEY": ALPACA_SECRET,
                },
                timeout=10.0
            )
            if response.status_code == 200:
                positions = response.json()
                return [
                    {
                        "symbol": p.get("symbol"),
                        "qty": float(p.get("qty", 0)),
                        "avg_entry": float(p.get("avg_entry_price", 0)),
                        "current_price": float(p.get("current_price", 0)),
                        "market_value": float(p.get("market_value", 0)),
                        "cost_basis": float(p.get("cost_basis", 0)),
                        "unrealized_pl": float(p.get("unrealized_pl", 0)),
                        "unrealized_pl_pct": float(p.get("unrealized_plpc", 0)) * 100,
                        "unrealized_intraday_pl": float(p.get("unrealized_intraday_pl", 0)),
                        "unrealized_intraday_plpc": float(p.get("unrealized_intraday_plpc", 0)) * 100,
                        "side": p.get("side", "long"),
                        "exchange": p.get("exchange", ""),
                        "asset_class": p.get("asset_class", "us_equity"),
                        "change_today": float(p.get("change_today", 0)) * 100,
                    }
                    for p in positions
                ]
            return []
    except Exception as e:
        return []


async def get_alpaca_orders(status: str = "open") -> List[Dict[str, Any]]:
    """Get orders from Alpaca."""
    if not ALPACA_KEY or not ALPACA_SECRET:
        return []
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ALPACA_BASE_URL}/v2/orders",
                params={"status": status, "limit": 50},
                headers={
                    "APCA-API-KEY-ID": ALPACA_KEY,
                    "APCA-API-SECRET-KEY": ALPACA_SECRET,
                },
                timeout=10.0
            )
            if response.status_code == 200:
                orders = response.json()
                return [
                    {
                        "id": o.get("id"),
                        "symbol": o.get("symbol"),
                        "qty": o.get("qty"),
                        "filled_qty": o.get("filled_qty"),
                        "side": o.get("side"),
                        "type": o.get("type"),
                        "status": o.get("status"),
                        "created_at": o.get("created_at"),
                        "filled_at": o.get("filled_at"),
                        "limit_price": o.get("limit_price"),
                        "stop_price": o.get("stop_price"),
                        "filled_avg_price": o.get("filled_avg_price"),
                    }
                    for o in orders
                ]
            return []
    except Exception as e:
        return []


async def get_alpaca_activity(activity_type: str = "FILL") -> List[Dict[str, Any]]:
    """Get recent account activity from Alpaca."""
    if not ALPACA_KEY or not ALPACA_SECRET:
        return []
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ALPACA_BASE_URL}/v2/account/activities/{activity_type}",
                params={"page_size": 20},
                headers={
                    "APCA-API-KEY-ID": ALPACA_KEY,
                    "APCA-API-SECRET-KEY": ALPACA_SECRET,
                },
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            return []
    except Exception as e:
        return []


# =============================================================================
# ENGINE & AGENT ANALYSIS
# =============================================================================

def generate_engine_output(symbol: str, ranking_data: Dict) -> Dict[str, Any]:
    """Generate simulated engine output for a symbol based on ranking data."""
    momentum = ranking_data.get("momentum_score", 50)
    volume = ranking_data.get("volume_score", 50)
    technical = ranking_data.get("technical_score", 50)
    score = ranking_data.get("score", 50)
    trend = ranking_data.get("signals", {}).get("trend", "neutral")
    
    # Determine IV environment based on volatility score
    vol_score = ranking_data.get("volatility_score", 50)
    if vol_score > 65:
        iv_environment = "HIGH"
        iv_rank = random.randint(55, 85)
    elif vol_score < 35:
        iv_environment = "LOW"
        iv_rank = random.randint(15, 35)
    else:
        iv_environment = "MEDIUM"
        iv_rank = random.randint(35, 55)
    
    # Determine direction
    if trend == "bullish":
        direction = "LONG"
        bias = "BULLISH"
    elif trend == "bearish":
        direction = "SHORT"
        bias = "BEARISH"
    else:
        direction = "NEUTRAL"
        bias = "NEUTRAL"
    
    # Calculate confidence
    confidence = min(95, 50 + (score - 50) / 2 + (momentum - 50) / 4)
    
    # Market regime
    if momentum > 70 and trend == "bullish":
        regime = "TRENDING_BULL"
    elif momentum < 30 and trend == "bearish":
        regime = "TRENDING_BEAR"
    elif vol_score > 60:
        regime = "HIGH_VOLATILITY"
    elif vol_score < 40:
        regime = "LOW_VOLATILITY"
    else:
        regime = "RANGE_BOUND"
    
    return {
        "symbol": symbol,
        "direction": direction,
        "bias": bias,
        "confidence": round(confidence, 1),
        "iv_environment": iv_environment,
        "iv_rank": iv_rank,
        "regime": regime,
        "momentum_signal": "STRONG" if momentum > 70 else "WEAK" if momentum < 30 else "NEUTRAL",
        "volume_signal": "SURGE" if volume > 70 else "LOW" if volume < 30 else "NORMAL",
        "technical_signal": "BUY" if technical > 60 else "SELL" if technical < 40 else "HOLD",
        "composite_score": round(score, 1),
    }


def generate_agent_analysis(symbol: str, engine_output: Dict) -> Dict[str, Any]:
    """Generate agent analysis based on engine output."""
    direction = engine_output["direction"]
    iv_env = engine_output["iv_environment"]
    confidence = engine_output["confidence"]
    regime = engine_output["regime"]
    
    # Hedge Agent Analysis
    hedge_score = random.randint(40, 90)
    hedge_signal = "FAVORABLE" if hedge_score > 60 else "CAUTION" if hedge_score > 40 else "UNFAVORABLE"
    
    # Liquidity Agent Analysis
    liquidity_score = random.randint(50, 95)
    spread_quality = "TIGHT" if liquidity_score > 75 else "MODERATE" if liquidity_score > 50 else "WIDE"
    
    # Sentiment Agent Analysis
    sentiment_score = random.randint(30, 80)
    if direction == "LONG":
        sentiment = "BULLISH" if sentiment_score > 60 else "MIXED" if sentiment_score > 40 else "CAUTIOUS"
    elif direction == "SHORT":
        sentiment = "BEARISH" if sentiment_score < 40 else "MIXED" if sentiment_score < 60 else "CAUTIOUS"
    else:
        sentiment = "NEUTRAL"
    
    # Composer Agent - Final Decision
    if confidence > 75 and hedge_signal == "FAVORABLE":
        composer_decision = "EXECUTE"
        composer_confidence = min(95, confidence + 5)
    elif confidence > 60:
        composer_decision = "CONSIDER"
        composer_confidence = confidence
    else:
        composer_decision = "WAIT"
        composer_confidence = confidence - 10
    
    return {
        "hedge_agent": {
            "score": hedge_score,
            "signal": hedge_signal,
            "gamma_exposure": round(random.uniform(-0.5, 0.5), 3),
            "delta_neutral_price": round(random.uniform(0.95, 1.05), 3),
        },
        "liquidity_agent": {
            "score": liquidity_score,
            "spread_quality": spread_quality,
            "depth_rating": "DEEP" if liquidity_score > 80 else "ADEQUATE" if liquidity_score > 50 else "SHALLOW",
            "execution_cost_bps": round(random.uniform(1, 15), 1),
        },
        "sentiment_agent": {
            "score": sentiment_score,
            "sentiment": sentiment,
            "news_impact": random.choice(["POSITIVE", "NEUTRAL", "NEGATIVE", "NONE"]),
            "social_momentum": random.choice(["HIGH", "MEDIUM", "LOW"]),
        },
        "composer_agent": {
            "decision": composer_decision,
            "confidence": round(composer_confidence, 1),
            "risk_assessment": "LOW" if hedge_score > 70 else "MEDIUM" if hedge_score > 50 else "HIGH",
            "position_size_rec": f"{min(4, max(1, int(confidence / 25)))}%",
        },
    }


def generate_options_strategy(symbol: str, engine_output: Dict, agent_analysis: Dict) -> Dict[str, Any]:
    """Generate options strategy recommendation."""
    direction = engine_output["direction"]
    iv_env = engine_output["iv_environment"]
    confidence = engine_output["confidence"]
    composer = agent_analysis["composer_agent"]
    
    # Strategy selection based on IV environment and direction
    if direction == "LONG":
        if iv_env == "HIGH":
            strategy = "BULL_PUT_SPREAD"
            strategy_type = "CREDIT"
            description = "Sell put spread - collect premium in high IV"
        elif iv_env == "LOW":
            if confidence > 80:
                strategy = "LONG_CALL"
                strategy_type = "DEBIT"
                description = "Buy calls - leverage bullish move in low IV"
            else:
                strategy = "BULL_CALL_SPREAD"
                strategy_type = "DEBIT"
                description = "Buy call spread - defined risk bullish"
        else:
            strategy = "BULL_CALL_SPREAD"
            strategy_type = "DEBIT"
            description = "Buy call spread - balanced approach"
    elif direction == "SHORT":
        if iv_env == "HIGH":
            strategy = "BEAR_CALL_SPREAD"
            strategy_type = "CREDIT"
            description = "Sell call spread - collect premium in high IV"
        elif iv_env == "LOW":
            if confidence > 80:
                strategy = "LONG_PUT"
                strategy_type = "DEBIT"
                description = "Buy puts - leverage bearish move in low IV"
            else:
                strategy = "BEAR_PUT_SPREAD"
                strategy_type = "DEBIT"
                description = "Buy put spread - defined risk bearish"
        else:
            strategy = "BEAR_PUT_SPREAD"
            strategy_type = "DEBIT"
            description = "Buy put spread - balanced approach"
    else:  # NEUTRAL
        if iv_env == "HIGH":
            strategy = "IRON_CONDOR"
            strategy_type = "CREDIT"
            description = "Sell iron condor - profit from range in high IV"
        else:
            strategy = "CALENDAR_SPREAD"
            strategy_type = "DEBIT"
            description = "Buy calendar - profit from time decay"
    
    # Generate DTE recommendation
    if composer["decision"] == "EXECUTE":
        dte_range = "30-45 DTE"
        urgency = "HIGH"
    elif composer["decision"] == "CONSIDER":
        dte_range = "45-60 DTE"
        urgency = "MEDIUM"
    else:
        dte_range = "60+ DTE"
        urgency = "LOW"
    
    return {
        "strategy": strategy,
        "strategy_type": strategy_type,
        "description": description,
        "iv_environment": iv_env,
        "recommended_dte": dte_range,
        "urgency": urgency,
        "max_risk": composer["position_size_rec"],
        "target_profit": "50%" if strategy_type == "CREDIT" else "100%",
        "stop_loss": "200%" if strategy_type == "CREDIT" else "50%",
        "greeks_focus": "THETA" if strategy_type == "CREDIT" else "DELTA",
    }


# =============================================================================
# GNOSIS ALPHA INTEGRATION
# =============================================================================

# Try to import Alpha components
try:
    from alpha import AlphaSignalGenerator, AlphaConfig
    from alpha.options_signal import OptionsSignalGenerator
    from alpha.zero_dte import ZeroDTEGenerator
    ALPHA_AVAILABLE = True
except ImportError:
    ALPHA_AVAILABLE = False

# Alpha configuration
ALPHA_CONFIG = {
    "min_confidence": float(os.getenv("ALPHA_MIN_CONFIDENCE", "0.55")),
    "max_holding_days": int(os.getenv("ALPHA_MAX_HOLDING_DAYS", "7")),
    "default_universe": os.getenv("ALPHA_UNIVERSE", "AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,AMD"),
    "max_options_position": float(os.getenv("ALPHA_MAX_OPTIONS_DOLLARS", "500")),
    "max_0dte_position": float(os.getenv("ALPHA_MAX_0DTE_DOLLARS", "200")),
}

def get_alpha_generator():
    """Get Alpha signal generator instance."""
    if not ALPHA_AVAILABLE:
        return None
    try:
        # AlphaConfig uses dataclass with defaults from environment variables
        # We can override the environment or use the defaults
        config = AlphaConfig(
            min_confidence=ALPHA_CONFIG["min_confidence"],
            max_holding_days=ALPHA_CONFIG["max_holding_days"],
        )
        return AlphaSignalGenerator(config=config)
    except Exception as e:
        print(f"Error creating Alpha generator: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_options_generator():
    """Get Options signal generator instance."""
    if not ALPHA_AVAILABLE:
        return None
    try:
        return OptionsSignalGenerator(
            api_key=ALPACA_KEY,
            secret_key=ALPACA_SECRET,
            min_confidence=ALPHA_CONFIG["min_confidence"],
        )
    except Exception as e:
        print(f"Error creating Options generator: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_0dte_generator():
    """Get 0DTE signal generator instance."""
    if not ALPHA_AVAILABLE:
        return None
    try:
        return ZeroDTEGenerator(
            api_key=ALPACA_KEY,
            secret_key=ALPACA_SECRET,
            max_position_dollars=ALPHA_CONFIG["max_0dte_position"],
        )
    except Exception as e:
        print(f"Error creating 0DTE generator: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/api/status")
async def get_status():
    """Get system status."""
    return {
        "status": "running",
        "mode": "autonomous",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "universe_size": 25,
        "watchlist_size": 10,
        "version": "3.0.0",
    }


@app.get("/api/account")
async def get_account():
    """Get account information."""
    account = await get_alpaca_account()
    positions = await get_alpaca_positions()
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "account": account,
        "positions": positions,
        "position_count": len(positions),
    }


@app.get("/api/positions")
async def get_positions_detailed():
    """Get detailed positions with P&L breakdown."""
    account = await get_alpaca_account()
    positions = await get_alpaca_positions()
    orders = await get_alpaca_orders("open")
    
    # Calculate totals
    total_market_value = sum(p["market_value"] for p in positions)
    total_unrealized_pl = sum(p["unrealized_pl"] for p in positions)
    total_cost_basis = sum(p["cost_basis"] for p in positions)
    total_intraday_pl = sum(p["unrealized_intraday_pl"] for p in positions)
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "account_equity": account.get("equity", 0),
        "buying_power": account.get("buying_power", 0),
        "positions": positions,
        "open_orders": orders,
        "summary": {
            "position_count": len(positions),
            "total_market_value": round(total_market_value, 2),
            "total_cost_basis": round(total_cost_basis, 2),
            "total_unrealized_pl": round(total_unrealized_pl, 2),
            "total_unrealized_pl_pct": round((total_unrealized_pl / total_cost_basis * 100) if total_cost_basis > 0 else 0, 2),
            "total_intraday_pl": round(total_intraday_pl, 2),
            "open_orders_count": len(orders),
        }
    }


@app.get("/api/rankings")
async def get_rankings():
    """Get current stock rankings."""
    rankings = universe_manager.rank_stocks()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rankings": [
            {
                "rank": r.rank,
                "symbol": r.symbol,
                "name": universe_manager.universe[r.symbol].name,
                "sector": universe_manager.universe[r.symbol].sector,
                "category": universe_manager.universe[r.symbol].category,
                "score": round(r.score, 1),
                "momentum": round(r.momentum_score, 1),
                "volume": round(r.volume_score, 1),
                "technical": round(r.technical_score, 1),
                "trend": r.signals.get("trend", "neutral"),
                "in_top_10": r.rank <= 10,
            }
            for r in rankings
        ],
    }


@app.get("/api/top10")
async def get_top10():
    """Get top 10 watchlist with basic info."""
    universe_manager.rank_stocks()
    top_10 = universe_manager.get_top_10()
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_10": [
            {
                "rank": r.rank,
                "symbol": r.symbol,
                "name": universe_manager.universe[r.symbol].name,
                "score": round(r.score, 1),
                "trend": r.signals.get("trend", "neutral"),
                "momentum": round(r.momentum_score, 1),
                "price": universe_manager.price_cache.get(r.symbol, {}).get("mid", 0),
            }
            for r in top_10
        ],
    }


@app.get("/api/top10/detailed")
async def get_top10_detailed():
    """Get top 10 watchlist with engine outputs, agent analysis, and options strategies."""
    universe_manager.rank_stocks()
    top_10 = universe_manager.get_top_10()
    
    detailed_watchlist = []
    for r in top_10:
        symbol = r.symbol
        
        # Get ranking data
        ranking_data = {
            "score": r.score,
            "momentum_score": r.momentum_score,
            "volume_score": r.volume_score,
            "technical_score": r.technical_score,
            "volatility_score": r.volatility_score,
            "signals": r.signals,
        }
        
        # Generate engine output
        engine = generate_engine_output(symbol, ranking_data)
        
        # Generate agent analysis
        agents = generate_agent_analysis(symbol, engine)
        
        # Generate options strategy
        options = generate_options_strategy(symbol, engine, agents)
        
        detailed_watchlist.append({
            "rank": r.rank,
            "symbol": symbol,
            "name": universe_manager.universe[symbol].name,
            "sector": universe_manager.universe[symbol].sector,
            "price": universe_manager.price_cache.get(symbol, {}).get("mid", 0),
            "engine": engine,
            "agents": agents,
            "options_strategy": options,
        })
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "watchlist": detailed_watchlist,
    }


@app.get("/api/signals")
async def get_signals():
    """Get current trading signals."""
    universe_manager.rank_stocks()
    top_10 = universe_manager.get_top_10()
    
    signals = []
    for r in top_10:
        trend = r.signals.get("trend", "neutral")
        
        if trend == "bullish" and r.momentum_score > 60:
            signal_type = "BUY"
            strength = "strong" if r.score > 100 else "moderate"
            confidence = min(95, 70 + (r.score - 80) / 2) if r.score > 80 else 70
        elif trend == "bearish" and r.momentum_score < 40:
            signal_type = "SELL"
            strength = "strong" if r.momentum_score < 30 else "moderate"
            confidence = min(95, 70 + (50 - r.momentum_score))
        elif r.signals.get("volume_surge"):
            signal_type = "WATCH"
            strength = "alert"
            confidence = 60
        else:
            continue
        
        signal = {
            "symbol": r.symbol,
            "name": universe_manager.universe[r.symbol].name,
            "signal": signal_type,
            "strength": strength,
            "confidence": round(confidence, 1),
            "score": round(r.score, 1),
            "momentum": round(r.momentum_score, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": universe_manager.price_cache.get(r.symbol, {}).get("mid", 0),
        }
        signals.append(signal)
        signal_history.appendleft(signal)
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_signals": len(signals),
        "signals": signals,
    }


@app.get("/api/market-overview")
async def get_market_overview():
    """Get market overview with all key metrics."""
    account = await get_alpaca_account()
    positions = await get_alpaca_positions()
    
    universe_manager.rank_stocks()
    top_10 = universe_manager.get_top_10()
    
    bullish_count = sum(1 for r in top_10 if r.signals.get("trend") == "bullish")
    bearish_count = sum(1 for r in top_10 if r.signals.get("trend") == "bearish")
    
    if bullish_count > bearish_count + 2:
        sentiment = "bullish"
    elif bearish_count > bullish_count + 2:
        sentiment = "bearish"
    else:
        sentiment = "neutral"
    
    avg_momentum = sum(r.momentum_score for r in top_10) / len(top_10) if top_10 else 50
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "account": account,
        "positions": positions,
        "market_sentiment": sentiment,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "avg_momentum": round(avg_momentum, 1),
        "top_10_count": len(top_10),
    }


# =============================================================================
# ALPHA API ENDPOINTS
# =============================================================================

@app.get("/api/alpha/status")
async def get_alpha_status():
    """Get Alpha module status."""
    return {
        "available": ALPHA_AVAILABLE,
        "config": ALPHA_CONFIG,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/alpha/stocks/scan")
async def alpha_stock_scan(symbols: str = None, include_hold: bool = False):
    """Scan for Alpha stock signals."""
    if not ALPHA_AVAILABLE:
        return {"error": "Alpha module not available", "signals": []}
    
    generator = get_alpha_generator()
    if not generator:
        return {"error": "Could not initialize Alpha generator", "signals": []}
    
    symbol_list = symbols.split(",") if symbols else ALPHA_CONFIG["default_universe"].split(",")
    
    signals = []
    for symbol in symbol_list:
        try:
            signal = generator.generate_signal(symbol.strip().upper())
            if signal:
                direction = signal.direction.value if hasattr(signal.direction, 'value') else str(signal.direction)
                
                # Skip HOLD signals unless requested
                if direction == "HOLD" and not include_hold:
                    continue
                
                # Convert entry price to float
                entry_price = float(signal.entry_price) if signal.entry_price else 0
                
                # Safely get optional fields
                stop_loss = float(signal.stop_loss) if signal.stop_loss else entry_price * 0.95
                take_profit = float(signal.take_profit) if signal.take_profit else entry_price * 1.10
                
                # Calculate risk/reward
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
                risk_reward = reward / risk if risk > 0 else 0
                
                signals.append({
                    "symbol": signal.symbol,
                    "direction": direction,
                    "confidence": round(float(signal.confidence) * 100, 1),
                    "entry_price": round(entry_price, 2),
                    "stop_loss": round(stop_loss, 2),
                    "target": round(take_profit, 2),
                    "risk_reward": round(risk_reward, 2),
                    "holding_days": signal.holding_period_days if hasattr(signal, 'holding_period_days') else 7,
                    "reasoning": signal.reasoning,
                    "pdt_safe": not signal.is_day_trade_candidate if hasattr(signal, 'is_day_trade_candidate') else True,
                    "risk_factors": signal.risk_factors if hasattr(signal, 'risk_factors') else [],
                    # Volume indicators
                    "unusual_volume": signal.unusual_volume if hasattr(signal, 'unusual_volume') else False,
                    "volume_ratio": round(signal.volume_ratio, 2) if hasattr(signal, 'volume_ratio') else 1.0,
                    "volume_description": signal.volume_description if hasattr(signal, 'volume_description') else "Normal",
                    # NEW: Detailed analysis fields
                    "agent_insights": getattr(signal, 'agent_insights', []),
                    "catalysts": getattr(signal, 'catalysts', []),
                    # LIQUIDITY POOLS AS SUPPORT/RESISTANCE
                    # PUT walls below price = SUPPORT (dealers must BUY to hedge)
                    # CALL walls above price = RESISTANCE (dealers must SELL to hedge)
                    "support_levels": getattr(signal, 'support_levels', []),  # PUT wall strikes
                    "resistance_levels": getattr(signal, 'resistance_levels', []),  # CALL wall strikes
                    "nearest_support": round(signal.nearest_support, 2) if getattr(signal, 'nearest_support', None) else None,
                    "nearest_resistance": round(signal.nearest_resistance, 2) if getattr(signal, 'nearest_resistance', None) else None,
                    # Detailed liquidity pool data
                    "put_walls": getattr(signal, 'put_walls', []),  # PUT OI concentrations
                    "call_walls": getattr(signal, 'call_walls', []),  # CALL OI concentrations
                    "max_pain": getattr(signal, 'max_pain', None),  # Price magnet at expiration
                    "gamma_flip": getattr(signal, 'gamma_flip', None),  # Where dealer hedging flips
                    # Dealer flow analysis
                    "dealer_positioning": getattr(signal, 'dealer_positioning', None),
                    "gamma_pressure": getattr(signal, 'gamma_pressure', None),
                    "energy_bias": getattr(signal, 'energy_bias', None),
                    "regime": getattr(signal, 'regime', None),
                    # Liquidity analysis
                    "liquidity_grade": getattr(signal, 'liquidity_grade', None),
                    "spread_quality": getattr(signal, 'spread_quality', None),
                    "market_depth": getattr(signal, 'market_depth', None),
                    # Technical analysis
                    "trend_status": getattr(signal, 'trend_status', None),
                    "momentum_status": getattr(signal, 'momentum_status', None),
                    "price_vs_sma20": getattr(signal, 'price_vs_sma20', None),
                    "price_vs_sma50": getattr(signal, 'price_vs_sma50', None),
                    "price_vs_sma200": getattr(signal, 'price_vs_sma200', None),
                    "bollinger_position": getattr(signal, 'bollinger_position', None),
                })
        except Exception as e:
            print(f"Error generating Alpha signal for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal_count": len(signals),
        "signals": signals,
    }


@app.get("/api/alpha/stocks/signal/{symbol}")
async def alpha_stock_signal(symbol: str):
    """Get Alpha signal for a specific stock."""
    if not ALPHA_AVAILABLE:
        return {"error": "Alpha module not available"}
    
    generator = get_alpha_generator()
    if not generator:
        return {"error": "Could not initialize Alpha generator"}
    
    try:
        signal = generator.generate_signal(symbol.upper())
        if signal:
            direction = signal.direction.value if hasattr(signal.direction, 'value') else str(signal.direction)
            
            # Convert entry price to float
            entry_price = float(signal.entry_price) if signal.entry_price else 0
            
            # Safely get optional fields
            stop_loss = float(signal.stop_loss) if signal.stop_loss else entry_price * 0.95
            take_profit = float(signal.take_profit) if signal.take_profit else entry_price * 1.10
            
            # Calculate risk/reward
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": {
                    "symbol": signal.symbol,
                    "direction": direction,
                    "confidence": round(float(signal.confidence) * 100, 1),
                    "entry_price": round(entry_price, 2),
                    "stop_loss": round(stop_loss, 2),
                    "target": round(take_profit, 2),
                    "risk_reward": round(risk_reward, 2),
                    "holding_days": signal.holding_period_days if hasattr(signal, 'holding_period_days') else 7,
                    "reasoning": signal.reasoning,
                    "pdt_safe": not signal.is_day_trade_candidate if hasattr(signal, 'is_day_trade_candidate') else True,
                    "risk_factors": signal.risk_factors if hasattr(signal, 'risk_factors') else [],
                    "robinhood_friendly": signal.to_robinhood_format() if hasattr(signal, 'to_robinhood_format') else None,
                    # Volume indicators
                    "unusual_volume": signal.unusual_volume if hasattr(signal, 'unusual_volume') else False,
                    "volume_ratio": round(signal.volume_ratio, 2) if hasattr(signal, 'volume_ratio') else 1.0,
                    "volume_description": signal.volume_description if hasattr(signal, 'volume_description') else "Normal",
                    # NEW: Detailed analysis fields
                    "agent_insights": getattr(signal, 'agent_insights', []),
                    "catalysts": getattr(signal, 'catalysts', []),
                    # LIQUIDITY POOLS AS SUPPORT/RESISTANCE
                    # PUT walls below price = SUPPORT (dealers must BUY to hedge)
                    # CALL walls above price = RESISTANCE (dealers must SELL to hedge)
                    "support_levels": getattr(signal, 'support_levels', []),  # PUT wall strikes
                    "resistance_levels": getattr(signal, 'resistance_levels', []),  # CALL wall strikes
                    "nearest_support": round(signal.nearest_support, 2) if getattr(signal, 'nearest_support', None) else None,
                    "nearest_resistance": round(signal.nearest_resistance, 2) if getattr(signal, 'nearest_resistance', None) else None,
                    # Detailed liquidity pool data
                    "put_walls": getattr(signal, 'put_walls', []),  # PUT OI concentrations
                    "call_walls": getattr(signal, 'call_walls', []),  # CALL OI concentrations
                    "max_pain": getattr(signal, 'max_pain', None),  # Price magnet at expiration
                    "gamma_flip": getattr(signal, 'gamma_flip', None),  # Where dealer hedging flips
                    # Dealer flow analysis
                    "dealer_positioning": getattr(signal, 'dealer_positioning', None),
                    "gamma_pressure": getattr(signal, 'gamma_pressure', None),
                    "energy_bias": getattr(signal, 'energy_bias', None),
                    "regime": getattr(signal, 'regime', None),
                    # Liquidity analysis
                    "liquidity_grade": getattr(signal, 'liquidity_grade', None),
                    "spread_quality": getattr(signal, 'spread_quality', None),
                    "market_depth": getattr(signal, 'market_depth', None),
                    # Technical analysis
                    "trend_status": getattr(signal, 'trend_status', None),
                    "momentum_status": getattr(signal, 'momentum_status', None),
                    "price_vs_sma20": getattr(signal, 'price_vs_sma20', None),
                    "price_vs_sma50": getattr(signal, 'price_vs_sma50', None),
                    "price_vs_sma200": getattr(signal, 'price_vs_sma200', None),
                    "bollinger_position": getattr(signal, 'bollinger_position', None),
                }
            }
        return {"error": f"No signal generated for {symbol}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/api/alpha/options/scan")
async def alpha_options_scan(symbols: str = None):
    """Scan for Alpha options signals."""
    if not ALPHA_AVAILABLE:
        return {"error": "Alpha module not available", "signals": []}
    
    generator = get_options_generator()
    if not generator:
        return {"error": "Could not initialize Options generator", "signals": []}
    
    symbol_list = symbols.split(",") if symbols else ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ"]
    
    signals = []
    for symbol in symbol_list:
        try:
            signal = generator.generate_signal(symbol.strip().upper())
            if signal:
                # Get contract details (contracts is a list)
                contract = signal.contracts[0] if signal.contracts else None
                
                signals.append({
                    "symbol": signal.symbol,
                    "strategy": signal.strategy.value if hasattr(signal.strategy, 'value') else str(signal.strategy),
                    "direction": signal.direction.value if hasattr(signal.direction, 'value') else str(signal.direction),
                    "confidence": round(float(signal.confidence) * 100, 1),
                    "strike": float(contract.strike) if contract else None,
                    "expiration": contract.expiration.isoformat() if contract and contract.expiration else None,
                    "dte": contract.days_to_expiration if contract else None,
                    "entry_price": round(float(signal.entry_price), 2) if signal.entry_price else None,
                    "max_loss": round(float(signal.max_loss), 2) if signal.max_loss else None,
                    "max_profit": round(float(signal.max_profit), 2) if signal.max_profit else None,
                    "break_even": round(float(signal.break_even), 2) if signal.break_even else None,
                    "reasoning": signal.reasoning,
                    "risk_factors": signal.risk_factors if hasattr(signal, 'risk_factors') else [],
                    # Volume indicators
                    "unusual_volume": signal.unusual_volume if hasattr(signal, 'unusual_volume') else False,
                    "volume_ratio": round(float(signal.volume_ratio), 2) if hasattr(signal, 'volume_ratio') else 1.0,
                    "contract_volume": contract.volume if contract else 0,
                    "open_interest": contract.open_interest if contract else 0,
                    "robinhood_friendly": signal.to_robinhood_format() if hasattr(signal, 'to_robinhood_format') else None,
                })
        except Exception as e:
            print(f"Error generating Options signal for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal_count": len(signals),
        "signals": signals,
        "dte_range": "1-14 days",
    }


@app.get("/api/alpha/options/signal/{symbol}")
async def alpha_options_signal(symbol: str, strategy: str = None):
    """Get Alpha options signal for a specific stock."""
    if not ALPHA_AVAILABLE:
        return {"error": "Alpha module not available"}
    
    generator = get_options_generator()
    if not generator:
        return {"error": "Could not initialize Options generator"}
    
    try:
        signal = generator.generate_signal(symbol.upper(), strategy=strategy)
        if signal:
            # Get contract details
            contract = signal.contracts[0] if signal.contracts else None
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": {
                    "symbol": signal.symbol,
                    "strategy": signal.strategy.value if hasattr(signal.strategy, 'value') else str(signal.strategy),
                    "direction": signal.direction.value if hasattr(signal.direction, 'value') else str(signal.direction),
                    "confidence": round(float(signal.confidence) * 100, 1),
                    "strike": float(contract.strike) if contract else None,
                    "expiration": contract.expiration.isoformat() if contract and contract.expiration else None,
                    "dte": contract.days_to_expiration if contract else None,
                    "entry_price": round(float(signal.entry_price), 2) if signal.entry_price else None,
                    "max_loss": round(float(signal.max_loss), 2) if signal.max_loss else None,
                    "max_profit": round(float(signal.max_profit), 2) if signal.max_profit else None,
                    "break_even": round(float(signal.break_even), 2) if signal.break_even else None,
                    "reasoning": signal.reasoning,
                    "risk_factors": signal.risk_factors if hasattr(signal, 'risk_factors') else [],
                    # Volume indicators
                    "unusual_volume": signal.unusual_volume if hasattr(signal, 'unusual_volume') else False,
                    "volume_ratio": round(float(signal.volume_ratio), 2) if hasattr(signal, 'volume_ratio') else 1.0,
                    "contract_volume": contract.volume if contract else 0,
                    "open_interest": contract.open_interest if contract else 0,
                    "robinhood_friendly": signal.to_robinhood_format() if hasattr(signal, 'to_robinhood_format') else None,
                },
                "dte_range": "1-14 days",
            }
        return {"error": f"No signal generated for {symbol}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/api/alpha/0dte/scan")
async def alpha_0dte_scan(symbols: str = None):
    """Scan for 0DTE options signals."""
    if not ALPHA_AVAILABLE:
        return {"error": "Alpha module not available", "signals": []}
    
    generator = get_0dte_generator()
    if not generator:
        return {"error": "Could not initialize 0DTE generator", "signals": []}
    
    # 0DTE typically available for these symbols
    symbol_list = symbols.split(",") if symbols else ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
    
    signals = []
    available_symbols = []
    
    for symbol in symbol_list:
        sym = symbol.strip().upper()
        try:
            # Check if 0DTE is available
            if generator.has_0dte(sym):
                available_symbols.append(sym)
                signal = generator.generate_signal(sym)
                if signal:
                    signals.append({
                        "symbol": signal.symbol,
                        "strategy": signal.strategy.value if hasattr(signal.strategy, 'value') else str(signal.strategy),
                        "direction": signal.direction.value if hasattr(signal.direction, 'value') else str(signal.direction),
                        "confidence": round(signal.confidence * 100, 1),
                        "strike": signal.strike,
                        "entry_price": round(signal.entry_price, 3),
                        "max_loss": round(signal.max_loss, 2),
                        "suggested_contracts": signal.suggested_contracts,
                        "time_remaining": signal.time_remaining,
                        "risk_level": signal.risk_level.value if hasattr(signal.risk_level, 'value') else str(signal.risk_level),
                        "warnings": signal.warnings if hasattr(signal, 'warnings') else [],
                        "reasoning": signal.reasoning,
                        "robinhood_friendly": signal.to_robinhood_format() if hasattr(signal, 'to_robinhood_format') else None,
                    })
        except Exception as e:
            print(f"Error checking 0DTE for {sym}: {e}")
            continue
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "available_symbols": available_symbols,
        "signal_count": len(signals),
        "signals": signals,
        "market_hours_warning": "0DTE options expire TODAY - trade with extreme caution",
    }


@app.get("/api/alpha/0dte/signal/{symbol}")
async def alpha_0dte_signal(symbol: str, strategy: str = None):
    """Get 0DTE signal for a specific symbol."""
    if not ALPHA_AVAILABLE:
        return {"error": "Alpha module not available"}
    
    generator = get_0dte_generator()
    if not generator:
        return {"error": "Could not initialize 0DTE generator"}
    
    try:
        sym = symbol.upper()
        if not generator.has_0dte(sym):
            return {"error": f"0DTE options not available for {sym} today"}
        
        signal = generator.generate_signal(sym, strategy=strategy)
        if signal:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": {
                    "symbol": signal.symbol,
                    "strategy": signal.strategy.value if hasattr(signal.strategy, 'value') else str(signal.strategy),
                    "direction": signal.direction.value if hasattr(signal.direction, 'value') else str(signal.direction),
                    "confidence": round(signal.confidence * 100, 1),
                    "strike": signal.strike,
                    "entry_price": round(signal.entry_price, 3),
                    "max_loss": round(signal.max_loss, 2),
                    "suggested_contracts": signal.suggested_contracts,
                    "time_remaining": signal.time_remaining,
                    "risk_level": signal.risk_level.value if hasattr(signal.risk_level, 'value') else str(signal.risk_level),
                    "underlying_price": round(signal.underlying_price, 2),
                    "break_even": round(signal.break_even, 2),
                    "warnings": signal.warnings if hasattr(signal, 'warnings') else [],
                    "reasoning": signal.reasoning,
                    "robinhood_friendly": signal.to_robinhood_format() if hasattr(signal, 'to_robinhood_format') else None,
                },
                "warning": "⚠️ 0DTE OPTIONS ARE EXTREMELY HIGH RISK - 100% LOSS IS COMMON",
            }
        return {"error": f"No 0DTE signal generated for {symbol}"}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# WEBSOCKET
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            data = {
                "type": "update",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "running",
            }
            await websocket.send_json(data)
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        active_connections.remove(websocket)


# =============================================================================
# MAIN PAGE - ENHANCED UI
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the GNOSIS SaaS homepage with enhanced features."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNOSIS - Autonomous Trading Intelligence</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: { sans: ['Inter', 'system-ui', 'sans-serif'] },
                }
            }
        }
    </script>
    <style>
        body { font-family: 'Inter', sans-serif; background: #0a0a0f; }
        .gradient-bg { background: radial-gradient(ellipse at top, #1a1a3e 0%, #0a0a0f 50%); }
        .glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.08); }
        .glass-strong { background: rgba(255, 255, 255, 0.06); backdrop-filter: blur(30px); border: 1px solid rgba(255, 255, 255, 0.1); }
        .live-dot { width: 8px; height: 8px; background: #22c55e; border-radius: 50%; animation: pulse 1.5s infinite; box-shadow: 0 0 8px #22c55e; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .card-hover:hover { transform: translateY(-2px); border-color: rgba(99, 102, 241, 0.3); }
        .scrollbar-thin::-webkit-scrollbar { width: 4px; }
        .scrollbar-thin::-webkit-scrollbar-track { background: transparent; }
        .scrollbar-thin::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 2px; }
        .tab-active { background: rgba(99, 102, 241, 0.2); border-color: rgba(99, 102, 241, 0.5); }
        .main-tab-active { background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(99, 102, 241, 0.5); }
        .alpha-tab-active { background: rgba(251, 191, 36, 0.2); border: 1px solid rgba(251, 191, 36, 0.5); }
        .position-row:hover { background: rgba(255,255,255,0.05); }
        .alpha-signal-card { transition: all 0.2s ease; }
        .alpha-signal-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
    </style>
</head>
<body class="gradient-bg min-h-screen text-white antialiased">
    
    <!-- Navigation -->
    <nav class="fixed top-0 left-0 right-0 z-50 glass-strong border-b border-white/5">
        <div class="max-w-[1600px] mx-auto px-4 py-3 flex justify-between items-center">
            <div class="flex items-center gap-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center font-black text-lg">G</div>
                <span class="text-xl font-bold">GNOSIS</span>
                <span class="text-xs text-gray-500 ml-2">v3.0</span>
            </div>
            <div class="flex items-center gap-4">
                <div class="flex items-center gap-2 px-3 py-1.5 glass rounded-full">
                    <div class="live-dot"></div>
                    <span class="text-sm text-gray-300">Live</span>
                </div>
                <button onclick="refreshAll()" class="p-2 hover:bg-white/10 rounded-lg transition">
                    <i class="fas fa-sync-alt text-gray-400" id="refresh-icon"></i>
                </button>
            </div>
        </div>
    </nav>

    <main class="pt-20 pb-8 px-4">
        <div class="max-w-[1600px] mx-auto">
            
            <!-- Main Tab Navigation -->
            <div class="mb-6 flex gap-2">
                <button onclick="switchMainTab('gnosis')" id="tab-gnosis" class="px-6 py-3 rounded-xl glass font-semibold transition main-tab-active flex items-center gap-2">
                    <i class="fas fa-brain text-indigo-400"></i>
                    <span>GNOSIS</span>
                    <span class="text-xs text-gray-500">Full Analysis</span>
                </button>
                <button onclick="switchMainTab('alpha')" id="tab-alpha" class="px-6 py-3 rounded-xl glass font-semibold transition flex items-center gap-2">
                    <i class="fas fa-bolt text-yellow-400"></i>
                    <span>ALPHA</span>
                    <span class="text-xs text-gray-500">Short-Term (0-7 days)</span>
                </button>
            </div>
            
            <!-- GNOSIS Tab Content -->
            <div id="content-gnosis">
            
            <!-- Stats Row -->
            <section class="mb-6">
                <div class="grid grid-cols-2 lg:grid-cols-5 gap-3">
                    <div class="glass rounded-xl p-4">
                        <div class="text-gray-400 text-xs mb-1">Portfolio</div>
                        <div class="text-xl font-bold" id="portfolio-value">$--,---</div>
                    </div>
                    <div class="glass rounded-xl p-4">
                        <div class="text-gray-400 text-xs mb-1">Day P&L</div>
                        <div class="text-xl font-bold" id="day-pl">$---</div>
                    </div>
                    <div class="glass rounded-xl p-4">
                        <div class="text-gray-400 text-xs mb-1">Positions</div>
                        <div class="text-xl font-bold" id="position-count">-</div>
                    </div>
                    <div class="glass rounded-xl p-4">
                        <div class="text-gray-400 text-xs mb-1">Signals</div>
                        <div class="text-xl font-bold" id="signal-count">-</div>
                    </div>
                    <div class="glass rounded-xl p-4">
                        <div class="text-gray-400 text-xs mb-1">Sentiment</div>
                        <div class="text-xl font-bold" id="sentiment">---</div>
                    </div>
                </div>
            </section>
            
            <!-- Main Grid -->
            <div class="grid lg:grid-cols-3 gap-4">
                
                <!-- Left Column: Positions Monitor -->
                <div class="lg:col-span-1 space-y-4">
                    <div class="glass rounded-xl overflow-hidden">
                        <div class="p-3 border-b border-white/5 flex items-center justify-between">
                            <div class="flex items-center gap-2">
                                <i class="fas fa-briefcase text-green-400"></i>
                                <span class="font-semibold">Positions Monitor</span>
                            </div>
                            <span class="text-xs text-gray-500" id="positions-update">--:--</span>
                        </div>
                        <div id="positions-container" class="max-h-[500px] overflow-y-auto scrollbar-thin">
                            <div class="p-4 text-center text-gray-500">Loading...</div>
                        </div>
                        <div class="p-3 border-t border-white/5 bg-white/2">
                            <div class="grid grid-cols-2 gap-2 text-xs">
                                <div>
                                    <span class="text-gray-400">Total Value:</span>
                                    <span class="font-medium ml-1" id="total-position-value">$0</span>
                                </div>
                                <div>
                                    <span class="text-gray-400">Unrealized:</span>
                                    <span class="font-medium ml-1" id="total-unrealized">$0</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Open Orders -->
                    <div class="glass rounded-xl overflow-hidden">
                        <div class="p-3 border-b border-white/5 flex items-center gap-2">
                            <i class="fas fa-clock text-yellow-400"></i>
                            <span class="font-semibold">Open Orders</span>
                        </div>
                        <div id="orders-container" class="max-h-[200px] overflow-y-auto scrollbar-thin">
                            <div class="p-4 text-center text-gray-500 text-sm">No open orders</div>
                        </div>
                    </div>
                </div>
                
                <!-- Center Column: Top 10 Watchlist with Details -->
                <div class="lg:col-span-2">
                    <div class="glass rounded-xl overflow-hidden">
                        <div class="p-3 border-b border-white/5 flex items-center justify-between">
                            <div class="flex items-center gap-2">
                                <i class="fas fa-star text-yellow-400"></i>
                                <span class="font-semibold">Top 10 Watchlist</span>
                            </div>
                            <div class="flex gap-2">
                                <button onclick="setView('compact')" id="btn-compact" class="px-3 py-1 text-xs rounded-lg glass tab-active">Compact</button>
                                <button onclick="setView('detailed')" id="btn-detailed" class="px-3 py-1 text-xs rounded-lg glass">Detailed</button>
                            </div>
                        </div>
                        
                        <!-- Compact View -->
                        <div id="view-compact">
                            <table class="w-full">
                                <thead class="bg-white/3 text-xs text-gray-400">
                                    <tr>
                                        <th class="px-3 py-2 text-left">#</th>
                                        <th class="px-3 py-2 text-left">Symbol</th>
                                        <th class="px-3 py-2 text-right">Score</th>
                                        <th class="px-3 py-2 text-center">Direction</th>
                                        <th class="px-3 py-2 text-center">IV</th>
                                        <th class="px-3 py-2 text-left">Strategy</th>
                                        <th class="px-3 py-2 text-center">Action</th>
                                    </tr>
                                </thead>
                                <tbody id="watchlist-compact">
                                    <tr><td colspan="7" class="p-4 text-center text-gray-500">Loading...</td></tr>
                                </tbody>
                            </table>
                        </div>
                        
                        <!-- Detailed View -->
                        <div id="view-detailed" class="hidden max-h-[600px] overflow-y-auto scrollbar-thin">
                            <div id="watchlist-detailed" class="divide-y divide-white/5">
                                <div class="p-4 text-center text-gray-500">Loading...</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            </div><!-- End GNOSIS Tab Content -->
            
            <!-- ALPHA Tab Content -->
            <div id="content-alpha" class="hidden">
                
                <!-- Alpha Header with Status -->
                <div class="mb-4 glass rounded-xl p-4 flex items-center justify-between">
                    <div class="flex items-center gap-4">
                        <div class="w-12 h-12 rounded-xl bg-gradient-to-br from-yellow-500 via-orange-500 to-red-500 flex items-center justify-center font-black text-xl">α</div>
                        <div>
                            <div class="text-xl font-bold">GNOSIS ALPHA</div>
                            <div class="text-sm text-gray-400">Short-Term Directional Trading (0-7 days)</div>
                        </div>
                    </div>
                    <div class="flex items-center gap-4">
                        <div class="text-right">
                            <div class="text-xs text-gray-400">Module Status</div>
                            <div id="alpha-status" class="font-semibold text-green-400">Checking...</div>
                        </div>
                        <button onclick="refreshAlpha()" class="p-2 hover:bg-white/10 rounded-lg transition">
                            <i class="fas fa-sync-alt text-gray-400" id="alpha-refresh-icon"></i>
                        </button>
                    </div>
                </div>
                
                <!-- Alpha Sub-tabs -->
                <div class="mb-4 flex gap-2">
                    <button onclick="switchAlphaTab('stocks')" id="alpha-tab-stocks" class="px-4 py-2 rounded-lg glass alpha-tab-active flex items-center gap-2">
                        <i class="fas fa-chart-line text-green-400"></i>
                        <span>Stocks</span>
                    </button>
                    <button onclick="switchAlphaTab('options')" id="alpha-tab-options" class="px-4 py-2 rounded-lg glass flex items-center gap-2">
                        <i class="fas fa-layer-group text-blue-400"></i>
                        <span>Options</span>
                    </button>
                    <button onclick="switchAlphaTab('0dte')" id="alpha-tab-0dte" class="px-4 py-2 rounded-lg glass flex items-center gap-2">
                        <i class="fas fa-fire text-red-400"></i>
                        <span>0DTE</span>
                        <span class="text-xs bg-red-500/20 text-red-400 px-1.5 py-0.5 rounded">HIGH RISK</span>
                    </button>
                </div>
                
                <!-- Alpha Stocks Content -->
                <div id="alpha-content-stocks">
                    <div class="glass rounded-xl overflow-hidden">
                        <div class="p-3 border-b border-white/5 flex items-center justify-between">
                            <div class="flex items-center gap-2">
                                <i class="fas fa-chart-line text-green-400"></i>
                                <span class="font-semibold">Stock Signals (0-7 Day Holds)</span>
                            </div>
                            <span class="text-xs text-gray-500">PDT-Safe • Robinhood Friendly</span>
                        </div>
                        <div id="alpha-stocks-container" class="max-h-[500px] overflow-y-auto scrollbar-thin">
                            <div class="p-4 text-center text-gray-500">Click refresh to scan for signals...</div>
                        </div>
                    </div>
                </div>
                
                <!-- Alpha Options Content -->
                <div id="alpha-content-options" class="hidden">
                    <div class="glass rounded-xl overflow-hidden">
                        <div class="p-3 border-b border-white/5 flex items-center justify-between">
                            <div class="flex items-center gap-2">
                                <i class="fas fa-layer-group text-blue-400"></i>
                                <span class="font-semibold">Options Signals (1-14 DTE)</span>
                            </div>
                            <span class="text-xs text-gray-500">Long Calls • Long Puts • Covered Calls • Cash-Secured Puts</span>
                        </div>
                        <div id="alpha-options-container" class="max-h-[500px] overflow-y-auto scrollbar-thin">
                            <div class="p-4 text-center text-gray-500">Click refresh to scan for signals...</div>
                        </div>
                    </div>
                </div>
                
                <!-- Alpha 0DTE Content -->
                <div id="alpha-content-0dte" class="hidden">
                    <!-- 0DTE Warning Banner -->
                    <div class="mb-4 bg-red-500/20 border border-red-500/30 rounded-xl p-4">
                        <div class="flex items-center gap-3">
                            <i class="fas fa-exclamation-triangle text-red-400 text-2xl"></i>
                            <div>
                                <div class="font-bold text-red-400">EXTREME RISK WARNING</div>
                                <div class="text-sm text-gray-300">0DTE options expire TODAY. 100% loss is common. Only trade with money you can afford to lose completely.</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="glass rounded-xl overflow-hidden">
                        <div class="p-3 border-b border-white/5 flex items-center justify-between">
                            <div class="flex items-center gap-2">
                                <i class="fas fa-fire text-red-400"></i>
                                <span class="font-semibold">0DTE Signals (Expires Today)</span>
                            </div>
                            <span class="text-xs text-gray-500" id="0dte-time-remaining">Market hours only</span>
                        </div>
                        <div id="alpha-0dte-container" class="max-h-[500px] overflow-y-auto scrollbar-thin">
                            <div class="p-4 text-center text-gray-500">Click refresh to scan for 0DTE opportunities...</div>
                        </div>
                    </div>
                </div>
                
            </div><!-- End ALPHA Tab Content -->
            
        </div>
    </main>

    <script>
        let currentView = 'compact';
        let currentMainTab = 'gnosis';
        let currentAlphaTab = 'stocks';
        
        function formatCurrency(val) {
            if (!val && val !== 0) return '$--';
            return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(val);
        }
        
        function formatCurrencyPrecise(val) {
            if (!val && val !== 0) return '$--';
            return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(val);
        }
        
        function formatPct(val) {
            if (!val && val !== 0) return '--%';
            return (val >= 0 ? '+' : '') + val.toFixed(2) + '%';
        }
        
        function setView(view) {
            currentView = view;
            document.getElementById('view-compact').classList.toggle('hidden', view !== 'compact');
            document.getElementById('view-detailed').classList.toggle('hidden', view !== 'detailed');
            document.getElementById('btn-compact').classList.toggle('tab-active', view === 'compact');
            document.getElementById('btn-detailed').classList.toggle('tab-active', view === 'detailed');
            if (view === 'detailed') fetchDetailedWatchlist();
        }
        
        function getDirectionBadge(dir) {
            if (dir === 'LONG') return '<span class="px-2 py-0.5 bg-green-500/20 text-green-400 rounded text-xs font-medium">LONG</span>';
            if (dir === 'SHORT') return '<span class="px-2 py-0.5 bg-red-500/20 text-red-400 rounded text-xs font-medium">SHORT</span>';
            return '<span class="px-2 py-0.5 bg-gray-500/20 text-gray-400 rounded text-xs font-medium">NEUTRAL</span>';
        }
        
        function getIVBadge(iv) {
            if (iv === 'HIGH') return '<span class="px-2 py-0.5 bg-orange-500/20 text-orange-400 rounded text-xs">HIGH</span>';
            if (iv === 'LOW') return '<span class="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded text-xs">LOW</span>';
            return '<span class="px-2 py-0.5 bg-gray-500/20 text-gray-400 rounded text-xs">MED</span>';
        }
        
        function getDecisionBadge(dec) {
            if (dec === 'EXECUTE') return '<span class="px-2 py-0.5 bg-green-600 text-white rounded text-xs font-bold">EXECUTE</span>';
            if (dec === 'CONSIDER') return '<span class="px-2 py-0.5 bg-yellow-600 text-white rounded text-xs font-bold">CONSIDER</span>';
            return '<span class="px-2 py-0.5 bg-gray-600 text-white rounded text-xs">WAIT</span>';
        }
        
        async function fetchPositions() {
            try {
                const res = await fetch('/api/positions');
                const data = await res.json();
                
                document.getElementById('positions-update').textContent = new Date().toLocaleTimeString();
                
                const positions = data.positions || [];
                const summary = data.summary || {};
                
                document.getElementById('position-count').textContent = positions.length;
                document.getElementById('total-position-value').textContent = formatCurrency(summary.total_market_value);
                
                const totalPL = summary.total_unrealized_pl || 0;
                document.getElementById('total-unrealized').innerHTML = `<span class="${totalPL >= 0 ? 'text-green-400' : 'text-red-400'}">${formatCurrencyPrecise(totalPL)}</span>`;
                
                if (positions.length === 0) {
                    document.getElementById('positions-container').innerHTML = '<div class="p-6 text-center text-gray-500"><i class="fas fa-inbox text-2xl mb-2 block"></i>No open positions</div>';
                } else {
                    document.getElementById('positions-container').innerHTML = positions.map(p => `
                        <div class="position-row p-3 border-b border-white/5 transition">
                            <div class="flex justify-between items-start mb-2">
                                <div>
                                    <span class="font-bold text-lg">${p.symbol}</span>
                                    <span class="text-xs text-gray-400 ml-2">${p.qty} shares</span>
                                </div>
                                <span class="font-medium ${p.unrealized_pl >= 0 ? 'text-green-400' : 'text-red-400'}">
                                    ${p.unrealized_pl >= 0 ? '+' : ''}${formatCurrencyPrecise(p.unrealized_pl)}
                                </span>
                            </div>
                            <div class="grid grid-cols-3 gap-2 text-xs">
                                <div>
                                    <span class="text-gray-500">Entry:</span>
                                    <span class="text-gray-300 ml-1">${formatCurrencyPrecise(p.avg_entry)}</span>
                                </div>
                                <div>
                                    <span class="text-gray-500">Current:</span>
                                    <span class="text-gray-300 ml-1">${formatCurrencyPrecise(p.current_price)}</span>
                                </div>
                                <div>
                                    <span class="text-gray-500">P&L:</span>
                                    <span class="${p.unrealized_pl_pct >= 0 ? 'text-green-400' : 'text-red-400'} ml-1">${formatPct(p.unrealized_pl_pct)}</span>
                                </div>
                            </div>
                            <div class="mt-2 flex items-center gap-2">
                                <div class="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                                    <div class="h-full ${p.unrealized_pl_pct >= 0 ? 'bg-green-500' : 'bg-red-500'}" 
                                         style="width: ${Math.min(100, Math.abs(p.unrealized_pl_pct) * 5)}%"></div>
                                </div>
                                <span class="text-xs ${p.change_today >= 0 ? 'text-green-400' : 'text-red-400'}">
                                    Today: ${formatPct(p.change_today)}
                                </span>
                            </div>
                        </div>
                    `).join('');
                }
                
                // Open orders
                const orders = data.open_orders || [];
                if (orders.length === 0) {
                    document.getElementById('orders-container').innerHTML = '<div class="p-4 text-center text-gray-500 text-sm">No open orders</div>';
                } else {
                    document.getElementById('orders-container').innerHTML = orders.map(o => `
                        <div class="p-3 border-b border-white/5 text-sm">
                            <div class="flex justify-between">
                                <span class="font-medium">${o.symbol}</span>
                                <span class="${o.side === 'buy' ? 'text-green-400' : 'text-red-400'}">${o.side.toUpperCase()}</span>
                            </div>
                            <div class="text-xs text-gray-400">${o.qty} @ ${o.limit_price || 'MKT'} • ${o.status}</div>
                        </div>
                    `).join('');
                }
                
            } catch (e) { console.error('Error fetching positions:', e); }
        }
        
        async function fetchAccount() {
            try {
                const res = await fetch('/api/account');
                const data = await res.json();
                const acc = data.account;
                
                if (acc && !acc.error) {
                    document.getElementById('portfolio-value').textContent = formatCurrency(acc.equity);
                    const pl = acc.day_pl || 0;
                    document.getElementById('day-pl').innerHTML = `<span class="${pl >= 0 ? 'text-green-400' : 'text-red-400'}">${formatCurrencyPrecise(pl)}</span>`;
                }
            } catch (e) { console.error('Error:', e); }
        }
        
        async function fetchSignals() {
            try {
                const res = await fetch('/api/signals');
                const data = await res.json();
                document.getElementById('signal-count').textContent = data.active_signals || 0;
            } catch (e) { console.error('Error:', e); }
        }
        
        async function fetchMarketOverview() {
            try {
                const res = await fetch('/api/market-overview');
                const data = await res.json();
                const s = data.market_sentiment;
                document.getElementById('sentiment').innerHTML = s === 'bullish' 
                    ? '<span class="text-green-400">BULLISH</span>' 
                    : s === 'bearish' ? '<span class="text-red-400">BEARISH</span>' 
                    : '<span class="text-gray-400">NEUTRAL</span>';
            } catch (e) { console.error('Error:', e); }
        }
        
        async function fetchCompactWatchlist() {
            try {
                const res = await fetch('/api/top10/detailed');
                const data = await res.json();
                
                document.getElementById('watchlist-compact').innerHTML = data.watchlist.map(w => `
                    <tr class="border-t border-white/5 hover:bg-white/3">
                        <td class="px-3 py-2 text-indigo-400 font-bold">#${w.rank}</td>
                        <td class="px-3 py-2">
                            <div class="font-semibold">${w.symbol}</div>
                            <div class="text-xs text-gray-500">${w.name.substring(0, 15)}</div>
                        </td>
                        <td class="px-3 py-2 text-right font-medium">${w.engine.composite_score}</td>
                        <td class="px-3 py-2 text-center">${getDirectionBadge(w.engine.direction)}</td>
                        <td class="px-3 py-2 text-center">${getIVBadge(w.engine.iv_environment)}</td>
                        <td class="px-3 py-2">
                            <div class="text-xs font-medium">${w.options_strategy.strategy.replace(/_/g, ' ')}</div>
                            <div class="text-xs text-gray-500">${w.options_strategy.strategy_type}</div>
                        </td>
                        <td class="px-3 py-2 text-center">${getDecisionBadge(w.agents.composer_agent.decision)}</td>
                    </tr>
                `).join('');
                
            } catch (e) { console.error('Error:', e); }
        }
        
        async function fetchDetailedWatchlist() {
            try {
                const res = await fetch('/api/top10/detailed');
                const data = await res.json();
                
                document.getElementById('watchlist-detailed').innerHTML = data.watchlist.map(w => `
                    <div class="p-4">
                        <!-- Header -->
                        <div class="flex items-center justify-between mb-3">
                            <div class="flex items-center gap-3">
                                <span class="text-2xl font-bold text-indigo-400">#${w.rank}</span>
                                <div>
                                    <div class="font-bold text-lg">${w.symbol}</div>
                                    <div class="text-xs text-gray-400">${w.name} • ${w.sector}</div>
                                </div>
                            </div>
                            ${getDecisionBadge(w.agents.composer_agent.decision)}
                        </div>
                        
                        <div class="grid grid-cols-3 gap-4">
                            <!-- Engine Output -->
                            <div class="glass rounded-lg p-3">
                                <div class="text-xs text-gray-400 mb-2 font-medium">ENGINE OUTPUT</div>
                                <div class="space-y-1 text-sm">
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Direction:</span>
                                        ${getDirectionBadge(w.engine.direction)}
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Confidence:</span>
                                        <span class="font-medium">${w.engine.confidence}%</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">IV Rank:</span>
                                        <span>${w.engine.iv_rank}% ${getIVBadge(w.engine.iv_environment)}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Regime:</span>
                                        <span class="text-xs">${w.engine.regime}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Momentum:</span>
                                        <span class="${w.engine.momentum_signal === 'STRONG' ? 'text-green-400' : w.engine.momentum_signal === 'WEAK' ? 'text-red-400' : 'text-gray-300'}">${w.engine.momentum_signal}</span>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Agent Analysis -->
                            <div class="glass rounded-lg p-3">
                                <div class="text-xs text-gray-400 mb-2 font-medium">AGENT ANALYSIS</div>
                                <div class="space-y-2 text-xs">
                                    <div class="flex justify-between items-center">
                                        <span class="text-gray-400">Hedge Agent:</span>
                                        <span class="${w.agents.hedge_agent.signal === 'FAVORABLE' ? 'text-green-400' : w.agents.hedge_agent.signal === 'CAUTION' ? 'text-yellow-400' : 'text-red-400'}">${w.agents.hedge_agent.signal} (${w.agents.hedge_agent.score})</span>
                                    </div>
                                    <div class="flex justify-between items-center">
                                        <span class="text-gray-400">Liquidity:</span>
                                        <span class="${w.agents.liquidity_agent.spread_quality === 'TIGHT' ? 'text-green-400' : 'text-yellow-400'}">${w.agents.liquidity_agent.spread_quality}</span>
                                    </div>
                                    <div class="flex justify-between items-center">
                                        <span class="text-gray-400">Sentiment:</span>
                                        <span class="${w.agents.sentiment_agent.sentiment === 'BULLISH' ? 'text-green-400' : w.agents.sentiment_agent.sentiment === 'BEARISH' ? 'text-red-400' : 'text-gray-300'}">${w.agents.sentiment_agent.sentiment}</span>
                                    </div>
                                    <div class="flex justify-between items-center">
                                        <span class="text-gray-400">Composer:</span>
                                        <span class="font-medium">${w.agents.composer_agent.confidence}% conf</span>
                                    </div>
                                    <div class="flex justify-between items-center">
                                        <span class="text-gray-400">Risk:</span>
                                        <span class="${w.agents.composer_agent.risk_assessment === 'LOW' ? 'text-green-400' : w.agents.composer_agent.risk_assessment === 'MEDIUM' ? 'text-yellow-400' : 'text-red-400'}">${w.agents.composer_agent.risk_assessment}</span>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Options Strategy -->
                            <div class="glass rounded-lg p-3">
                                <div class="text-xs text-gray-400 mb-2 font-medium">OPTIONS STRATEGY</div>
                                <div class="mb-2">
                                    <div class="font-bold text-sm">${w.options_strategy.strategy.replace(/_/g, ' ')}</div>
                                    <div class="text-xs text-gray-400">${w.options_strategy.description}</div>
                                </div>
                                <div class="space-y-1 text-xs">
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Type:</span>
                                        <span class="${w.options_strategy.strategy_type === 'CREDIT' ? 'text-green-400' : 'text-blue-400'}">${w.options_strategy.strategy_type}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">DTE:</span>
                                        <span>${w.options_strategy.recommended_dte}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Max Risk:</span>
                                        <span>${w.options_strategy.max_risk}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Target:</span>
                                        <span class="text-green-400">${w.options_strategy.target_profit}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-400">Focus:</span>
                                        <span>${w.options_strategy.greeks_focus}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `).join('');
                
            } catch (e) { console.error('Error:', e); }
        }
        
        async function refreshAll() {
            document.getElementById('refresh-icon').classList.add('fa-spin');
            await Promise.all([
                fetchAccount(),
                fetchPositions(),
                fetchSignals(),
                fetchMarketOverview(),
                fetchCompactWatchlist(),
            ]);
            if (currentView === 'detailed') await fetchDetailedWatchlist();
            document.getElementById('refresh-icon').classList.remove('fa-spin');
        }
        
        // ============================================
        // ALPHA TAB FUNCTIONS
        // ============================================
        
        function switchMainTab(tab) {
            currentMainTab = tab;
            document.getElementById('content-gnosis').classList.toggle('hidden', tab !== 'gnosis');
            document.getElementById('content-alpha').classList.toggle('hidden', tab !== 'alpha');
            document.getElementById('tab-gnosis').classList.toggle('main-tab-active', tab === 'gnosis');
            document.getElementById('tab-alpha').classList.toggle('main-tab-active', tab === 'alpha');
            
            if (tab === 'alpha') {
                fetchAlphaStatus();
            }
        }
        
        function switchAlphaTab(tab) {
            currentAlphaTab = tab;
            document.getElementById('alpha-content-stocks').classList.toggle('hidden', tab !== 'stocks');
            document.getElementById('alpha-content-options').classList.toggle('hidden', tab !== 'options');
            document.getElementById('alpha-content-0dte').classList.toggle('hidden', tab !== '0dte');
            document.getElementById('alpha-tab-stocks').classList.toggle('alpha-tab-active', tab === 'stocks');
            document.getElementById('alpha-tab-options').classList.toggle('alpha-tab-active', tab === 'options');
            document.getElementById('alpha-tab-0dte').classList.toggle('alpha-tab-active', tab === '0dte');
        }
        
        function getAlphaDirectionBadge(dir) {
            if (dir === 'BUY' || dir === 'BULLISH') return '<span class="px-2 py-0.5 bg-green-500/20 text-green-400 rounded text-xs font-medium">BUY</span>';
            if (dir === 'SELL' || dir === 'BEARISH') return '<span class="px-2 py-0.5 bg-red-500/20 text-red-400 rounded text-xs font-medium">SELL</span>';
            return '<span class="px-2 py-0.5 bg-gray-500/20 text-gray-400 rounded text-xs font-medium">HOLD</span>';
        }
        
        function getRiskBadge(level) {
            if (level === 'EXTREME') return '<span class="px-2 py-0.5 bg-red-600 text-white rounded text-xs font-bold">EXTREME</span>';
            if (level === 'HIGH') return '<span class="px-2 py-0.5 bg-red-500/30 text-red-400 rounded text-xs font-medium">HIGH</span>';
            if (level === 'ELEVATED') return '<span class="px-2 py-0.5 bg-orange-500/30 text-orange-400 rounded text-xs font-medium">ELEVATED</span>';
            return '<span class="px-2 py-0.5 bg-gray-500/20 text-gray-400 rounded text-xs">NORMAL</span>';
        }
        
        async function fetchAlphaStatus() {
            try {
                const res = await fetch('/api/alpha/status');
                const data = await res.json();
                document.getElementById('alpha-status').textContent = data.available ? 'Active' : 'Unavailable';
                document.getElementById('alpha-status').className = data.available ? 'font-semibold text-green-400' : 'font-semibold text-red-400';
            } catch (e) {
                document.getElementById('alpha-status').textContent = 'Error';
                document.getElementById('alpha-status').className = 'font-semibold text-red-400';
            }
        }
        
        async function fetchAlphaStocks() {
            try {
                document.getElementById('alpha-stocks-container').innerHTML = '<div class="p-4 text-center text-gray-500"><i class="fas fa-spinner fa-spin mr-2"></i>Scanning for signals...</div>';
                
                const res = await fetch('/api/alpha/stocks/scan');
                const data = await res.json();
                
                if (data.error) {
                    document.getElementById('alpha-stocks-container').innerHTML = `<div class="p-4 text-center text-red-400">${data.error}</div>`;
                    return;
                }
                
                if (data.signals.length === 0) {
                    document.getElementById('alpha-stocks-container').innerHTML = '<div class="p-4 text-center text-gray-500">No active signals at this time</div>';
                    return;
                }
                
                document.getElementById('alpha-stocks-container').innerHTML = data.signals.map(s => `
                    <div class="alpha-signal-card p-4 border-b border-white/5 ${s.unusual_volume ? 'bg-yellow-500/5 border-l-2 border-l-yellow-500' : ''}">
                        <div class="flex items-center justify-between mb-3">
                            <div class="flex items-center gap-3">
                                <span class="text-xl font-bold">${s.symbol}</span>
                                ${getAlphaDirectionBadge(s.direction)}
                                ${s.unusual_volume ? `<span class="px-2 py-0.5 bg-yellow-500/20 text-yellow-400 rounded text-xs font-medium animate-pulse">${s.volume_description}</span>` : ''}
                                <span class="text-sm text-gray-400">${s.confidence}% confidence</span>
                            </div>
                            <span class="text-xs ${s.pdt_safe ? 'text-green-400' : 'text-yellow-400'}">${s.pdt_safe ? 'PDT Safe' : 'PDT Risk'}</span>
                        </div>
                        <div class="grid grid-cols-4 gap-3 text-sm mb-2">
                            <div>
                                <span class="text-gray-500">Entry:</span>
                                <span class="ml-1 font-medium">${formatCurrencyPrecise(s.entry_price)}</span>
                            </div>
                            <div>
                                <span class="text-gray-500">Stop:</span>
                                <span class="ml-1 text-red-400">${formatCurrencyPrecise(s.stop_loss)}</span>
                            </div>
                            <div>
                                <span class="text-gray-500">Target:</span>
                                <span class="ml-1 text-green-400">${formatCurrencyPrecise(s.target)}</span>
                            </div>
                            <div>
                                <span class="text-gray-500">R:R:</span>
                                <span class="ml-1 font-medium">${s.risk_reward.toFixed(1)}x</span>
                            </div>
                        </div>
                        <div class="flex items-center justify-between text-xs">
                            <span class="text-gray-400">${s.reasoning}</span>
                            <div class="flex items-center gap-2">
                                ${s.volume_ratio > 1.2 ? `<span class="text-yellow-400">Vol: ${s.volume_ratio.toFixed(1)}x</span>` : ''}
                                <span class="text-gray-500">Hold: ${s.holding_days} days</span>
                            </div>
                        </div>
                    </div>
                `).join('');
                
            } catch (e) {
                console.error('Error:', e);
                document.getElementById('alpha-stocks-container').innerHTML = '<div class="p-4 text-center text-red-400">Error fetching signals</div>';
            }
        }
        
        async function fetchAlphaOptions() {
            try {
                document.getElementById('alpha-options-container').innerHTML = '<div class="p-4 text-center text-gray-500"><i class="fas fa-spinner fa-spin mr-2"></i>Scanning for options signals...</div>';
                
                const res = await fetch('/api/alpha/options/scan');
                const data = await res.json();
                
                if (data.error) {
                    document.getElementById('alpha-options-container').innerHTML = `<div class="p-4 text-center text-red-400">${data.error}</div>`;
                    return;
                }
                
                if (data.signals.length === 0) {
                    document.getElementById('alpha-options-container').innerHTML = '<div class="p-4 text-center text-gray-500">No active options signals at this time</div>';
                    return;
                }
                
                document.getElementById('alpha-options-container').innerHTML = data.signals.map(s => `
                    <div class="alpha-signal-card p-4 border-b border-white/5 ${s.unusual_volume ? 'bg-yellow-500/5 border-l-2 border-l-yellow-500' : ''}">
                        <div class="flex items-center justify-between mb-3">
                            <div class="flex items-center gap-3">
                                <span class="text-xl font-bold">${s.symbol}</span>
                                <span class="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded text-xs font-medium">${s.strategy.replace(/_/g, ' ')}</span>
                                ${getAlphaDirectionBadge(s.direction)}
                                ${s.unusual_volume ? `<span class="px-2 py-0.5 bg-yellow-500/20 text-yellow-400 rounded text-xs font-medium animate-pulse">📈 ${s.volume_ratio.toFixed(1)}x Vol</span>` : ''}
                            </div>
                            <div class="flex items-center gap-2">
                                <span class="text-sm text-gray-400">${s.confidence}% conf</span>
                                <span class="px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded text-xs">${s.dte || '?'} DTE</span>
                            </div>
                        </div>
                        <div class="grid grid-cols-4 gap-3 text-sm mb-2">
                            <div>
                                <span class="text-gray-500">Strike:</span>
                                <span class="ml-1 font-medium">${formatCurrencyPrecise(s.strike)}</span>
                            </div>
                            <div>
                                <span class="text-gray-500">Entry:</span>
                                <span class="ml-1">${formatCurrencyPrecise(s.entry_price)}</span>
                            </div>
                            <div>
                                <span class="text-gray-500">Max Loss:</span>
                                <span class="ml-1 text-red-400">${formatCurrencyPrecise(s.max_loss)}</span>
                            </div>
                            <div>
                                <span class="text-gray-500">Break Even:</span>
                                <span class="ml-1">${formatCurrencyPrecise(s.break_even)}</span>
                            </div>
                        </div>
                        <div class="flex items-center justify-between text-xs">
                            <span class="text-gray-400">${s.reasoning || ''}</span>
                            <div class="flex items-center gap-2">
                                ${s.contract_volume > 0 ? `<span class="text-gray-500">Vol: ${s.contract_volume.toLocaleString()}</span>` : ''}
                                ${s.open_interest > 0 ? `<span class="text-gray-500">OI: ${s.open_interest.toLocaleString()}</span>` : ''}
                                <span class="text-gray-500">Exp: ${s.expiration ? s.expiration.split('T')[0] : 'N/A'}</span>
                            </div>
                        </div>
                    </div>
                `).join('');
                
            } catch (e) {
                console.error('Error:', e);
                document.getElementById('alpha-options-container').innerHTML = '<div class="p-4 text-center text-red-400">Error fetching options signals</div>';
            }
        }
        
        async function fetchAlpha0DTE() {
            try {
                document.getElementById('alpha-0dte-container').innerHTML = '<div class="p-4 text-center text-gray-500"><i class="fas fa-spinner fa-spin mr-2"></i>Scanning for 0DTE opportunities...</div>';
                
                const res = await fetch('/api/alpha/0dte/scan');
                const data = await res.json();
                
                if (data.error) {
                    document.getElementById('alpha-0dte-container').innerHTML = `<div class="p-4 text-center text-red-400">${data.error}</div>`;
                    return;
                }
                
                if (data.available_symbols) {
                    document.getElementById('0dte-time-remaining').textContent = `Available: ${data.available_symbols.join(', ')}`;
                }
                
                if (data.signals.length === 0) {
                    document.getElementById('alpha-0dte-container').innerHTML = '<div class="p-4 text-center text-gray-500">No 0DTE signals available - may be outside market hours or no setups found</div>';
                    return;
                }
                
                document.getElementById('alpha-0dte-container').innerHTML = data.signals.map(s => `
                    <div class="alpha-signal-card p-4 border-b border-white/5 ${s.risk_level === 'EXTREME' ? 'bg-red-500/5' : ''}">
                        <div class="flex items-center justify-between mb-3">
                            <div class="flex items-center gap-3">
                                <span class="text-xl font-bold">${s.symbol}</span>
                                <span class="px-2 py-0.5 bg-orange-500/20 text-orange-400 rounded text-xs font-medium">${s.strategy.replace(/_/g, ' ')}</span>
                                ${getRiskBadge(s.risk_level)}
                            </div>
                            <span class="text-sm text-yellow-400 font-medium">${s.time_remaining}</span>
                        </div>
                        <div class="grid grid-cols-4 gap-3 text-sm mb-2">
                            <div>
                                <span class="text-gray-500">Strike:</span>
                                <span class="ml-1 font-medium">${formatCurrencyPrecise(s.strike)}</span>
                            </div>
                            <div>
                                <span class="text-gray-500">Entry:</span>
                                <span class="ml-1">${formatCurrencyPrecise(s.entry_price)}/contract</span>
                            </div>
                            <div>
                                <span class="text-gray-500">Max Loss:</span>
                                <span class="ml-1 text-red-400">${formatCurrencyPrecise(s.max_loss)}</span>
                            </div>
                            <div>
                                <span class="text-gray-500">Contracts:</span>
                                <span class="ml-1 font-medium">${s.suggested_contracts}</span>
                            </div>
                        </div>
                        <div class="text-xs text-gray-400 mb-2">${s.reasoning || ''}</div>
                        ${s.warnings && s.warnings.length > 0 ? `
                            <div class="flex flex-wrap gap-1">
                                ${s.warnings.slice(0, 3).map(w => `<span class="text-xs bg-red-500/10 text-red-400 px-2 py-0.5 rounded">${w}</span>`).join('')}
                            </div>
                        ` : ''}
                    </div>
                `).join('');
                
            } catch (e) {
                console.error('Error:', e);
                document.getElementById('alpha-0dte-container').innerHTML = '<div class="p-4 text-center text-red-400">Error fetching 0DTE signals</div>';
            }
        }
        
        async function refreshAlpha() {
            document.getElementById('alpha-refresh-icon').classList.add('fa-spin');
            await fetchAlphaStatus();
            
            if (currentAlphaTab === 'stocks') await fetchAlphaStocks();
            else if (currentAlphaTab === 'options') await fetchAlphaOptions();
            else if (currentAlphaTab === '0dte') await fetchAlpha0DTE();
            
            document.getElementById('alpha-refresh-icon').classList.remove('fa-spin');
        }
        
        // Initial load
        refreshAll();
        
        // Auto-refresh every 30 seconds
        setInterval(refreshAll, 30000);
    </script>
</body>
</html>
"""


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  GNOSIS SaaS - Autonomous Trading Intelligence")
    print("  Version 3.0.0 - Enhanced Dashboard")
    print("="*70)
    print("  🚀 Starting server on http://0.0.0.0:8888")
    print("  📊 Dashboard: http://localhost:8888")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8888)
