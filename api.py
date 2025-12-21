"""
Super Gnosis Trading System - FastAPI Application Entry Point

Main API server with:
- REST endpoints for trading operations
- WebSocket streaming for real-time data
- Prometheus metrics endpoint
- Health checks
- Trading Hub integration (connects all trading components)

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, HTMLResponse
from loguru import logger

# Import routers
from routers import ml_trades_router, trade_decisions_router, options_greeks_router
from routers.websocket_api import router as websocket_router, start_websocket_publisher, stop_websocket_publisher

# Import utilities
from utils.metrics import metrics, record_api_request
from utils.redis_cache import cache, initialize_cache, close_cache

# Import Trading Hub
from integration import get_trading_hub, start_trading_hub, stop_trading_hub, HubConfig


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Super Gnosis Trading System...")
    
    # Initialize cache
    await initialize_cache()
    logger.info("Cache initialized")
    
    # Start WebSocket publisher
    await start_websocket_publisher()
    logger.info("WebSocket publisher started")
    
    # Start Trading Hub (central integration layer)
    hub_config = HubConfig(
        enable_paper_trading=os.getenv("ENABLE_PAPER_TRADING", "false").lower() == "true",
        enable_websocket=True,
        enable_anomaly_detection=True,
        enable_flow_scanner=True,
        enable_notifications=os.getenv("ENABLE_NOTIFICATIONS", "false").lower() == "true",
        enable_greeks_hedger=True,
        enable_portfolio_analytics=True,
        auto_hedge_enabled=os.getenv("AUTO_HEDGE_ENABLED", "false").lower() == "true",
    )
    
    try:
        await start_trading_hub(hub_config)
        logger.info("Trading Hub started")
    except Exception as e:
        logger.warning(f"Trading Hub startup warning (non-fatal): {e}")
    
    logger.info("Super Gnosis Trading System started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Super Gnosis Trading System...")
    
    await stop_trading_hub()
    await stop_websocket_publisher()
    await close_cache()
    
    logger.info("Shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Super Gnosis Trading System",
    description="""
    Institutional-grade options trading platform with:
    
    - **Real-time Greeks calculation** and portfolio analysis
    - **Multi-leg options** execution (spreads, iron condors, etc.)
    - **ML-powered** trade signals and optimization
    - **WebSocket streaming** for live data
    - **Prometheus metrics** for monitoring
    
    ## Features
    
    - Options Greeks Calculator API
    - Portfolio Optimizer (Markowitz, Risk Parity, Black-Litterman)
    - ML Trade Dataset Generation
    - Real-time Position Tracking
    - Risk Management and Safety Controls
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# =============================================================================
# Middleware
# =============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add timing header and record metrics."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Record API metrics
    record_api_request(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code,
        latency=process_time,
    )
    
    return response


# =============================================================================
# Include Routers
# =============================================================================

app.include_router(options_greeks_router)
app.include_router(ml_trades_router)
app.include_router(trade_decisions_router)
app.include_router(websocket_router)


# =============================================================================
# Health & Metrics Endpoints
# =============================================================================

@app.get("/health", tags=["system"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "cache": cache.get_stats(),
    }


@app.get("/ready", tags=["system"])
async def readiness_check() -> Dict[str, Any]:
    """Readiness check for Kubernetes."""
    return {
        "ready": True,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/metrics", response_class=PlainTextResponse, tags=["system"])
async def prometheus_metrics() -> str:
    """Prometheus metrics endpoint."""
    return metrics.get_prometheus_output()


@app.get("/", tags=["system"])
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {
        "name": "Super Gnosis Trading System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "metrics": "/metrics",
        "websocket": "/ws/stream",
        "dashboard": "/dashboard",
        "hub_status": "/hub/status",
    }


# =============================================================================
# Trading Hub Endpoints
# =============================================================================

@app.get("/hub/status", tags=["trading"])
async def hub_status() -> Dict[str, Any]:
    """Get Trading Hub status and metrics."""
    hub = get_trading_hub()
    return hub.get_status()


@app.post("/hub/alert", tags=["trading"])
async def send_hub_alert(
    alert_type: str,
    title: str,
    message: str,
    symbol: str = None,
    priority: str = "medium",
) -> Dict[str, Any]:
    """Send an alert through the Trading Hub.
    
    Args:
        alert_type: Type of alert (e.g., 'manual', 'test')
        title: Alert title
        message: Alert message
        symbol: Optional symbol
        priority: Alert priority (low, medium, high, critical)
    """
    from integration import TradingAlert, AlertPriority
    
    priority_map = {
        "low": AlertPriority.LOW,
        "medium": AlertPriority.MEDIUM,
        "high": AlertPriority.HIGH,
        "critical": AlertPriority.CRITICAL,
    }
    
    hub = get_trading_hub()
    
    alert = TradingAlert(
        alert_type=alert_type,
        priority=priority_map.get(priority, AlertPriority.MEDIUM),
        title=title,
        message=message,
        symbol=symbol,
        source="api",
    )
    
    hub._dispatch_alert(alert)
    
    return {"status": "sent", "alert": alert.to_dict()}


@app.post("/hub/market-data", tags=["trading"])
async def process_market_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process market data through anomaly detection.
    
    Args:
        data: Market data point with symbol, price, volume, etc.
    """
    hub = get_trading_hub()
    await hub.process_market_data(data)
    return {"status": "processed"}


@app.post("/hub/options-flow", tags=["trading"])
async def process_options_flow(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process options trade through flow scanner.
    
    Args:
        data: Options trade data
    """
    hub = get_trading_hub()
    await hub.process_options_flow(data)
    return {"status": "processed"}


@app.post("/hub/portfolio", tags=["trading"])
async def update_portfolio(data: Dict[str, Any]) -> Dict[str, Any]:
    """Update portfolio state in the hub.
    
    Args:
        data: Portfolio state data
    """
    hub = get_trading_hub()
    hub.update_portfolio_state(data)
    return {"status": "updated"}


@app.post("/hub/greeks", tags=["trading"])
async def update_greeks(data: Dict[str, Any]) -> Dict[str, Any]:
    """Update portfolio Greeks and check hedging needs.
    
    Args:
        data: Greeks data (delta, gamma, theta, vega)
    """
    hub = get_trading_hub()
    await hub.update_portfolio_greeks(data)
    return {"status": "updated"}


@app.get("/dashboard", response_class=HTMLResponse, tags=["ui"])
async def trading_dashboard() -> str:
    """Serve the trading dashboard HTML."""
    try:
        from dashboard.trading_dashboard import TradingDashboard
        dashboard = TradingDashboard()
        return dashboard.generate_html()
    except Exception as e:
        logger.error(f"Dashboard generation error: {e}")
        return f"<html><body><h1>Dashboard Error</h1><p>{str(e)}</p></body></html>"


@app.get("/analytics", tags=["trading"])
async def portfolio_analytics() -> Dict[str, Any]:
    """Get portfolio analytics report."""
    hub = get_trading_hub()
    
    if hub.portfolio_analytics:
        return hub.portfolio_analytics.generate_full_report()
    
    return {"error": "Portfolio analytics not available"}


# =============================================================================
# Run with Uvicorn (for development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
