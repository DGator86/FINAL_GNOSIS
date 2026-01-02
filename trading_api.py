"""
Super Gnosis Trading System - FastAPI Application Entry Point

Main API server with:
- REST endpoints for trading operations
- WebSocket streaming for real-time data
- Prometheus metrics endpoint
- Health checks
- Trading Hub integration (connects all trading components)
- API Key and JWT authentication
- Rate limiting

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.openapi.utils import get_openapi
from loguru import logger

# Import routers
from routers import ml_trades_router, trade_decisions_router, options_greeks_router
from routers.websocket_api import router as websocket_router, start_websocket_publisher, stop_websocket_publisher
from routers.auth import router as auth_router

# Import utilities
from utils.metrics import metrics, record_api_request
from utils.redis_cache import cache, initialize_cache, close_cache

# Import Trading Hub
from integration import get_trading_hub, start_trading_hub, stop_trading_hub, HubConfig

# Import middleware
from middleware.auth import (
    get_current_user,
    optional_auth,
    require_api_key,
    require_permission,
    APIUser,
    Permission,
)
from middleware.rate_limiter import (
    get_rate_limiter,
    rate_limit,
    RateLimitConfig,
)


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
    
    ## Authentication
    
    This API supports two authentication methods:
    
    1. **API Key**: Include `X-API-Key` header with your API key
    2. **JWT Token**: Include `Authorization: Bearer <token>` header
    
    Contact admin for API key generation.
    
    ## Rate Limits
    
    - Default: 60 requests/minute, 1000 requests/hour
    - Rate limits vary by role (admin, trader, viewer)
    - Exceeding limits results in 429 Too Many Requests
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# Custom OpenAPI schema with security
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API Key authentication",
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token authentication",
        },
    }
    
    # Add global security (optional)
    openapi_schema["security"] = [
        {"ApiKeyAuth": []},
        {"BearerAuth": []},
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

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


# Request timing and rate limiting middleware
@app.middleware("http")
async def add_timing_and_rate_limit(request: Request, call_next):
    """Add timing header, rate limiting, and record metrics."""
    start_time = time.time()
    
    # Rate limiting (skip for exempt endpoints)
    limiter = get_rate_limiter()
    exempt_endpoints = ["/health", "/ready", "/docs", "/redoc", "/openapi.json", "/"]
    
    if request.url.path not in exempt_endpoints:
        # Get user from authentication if available
        user_id = None
        role = None
        
        allowed, limit_info = await limiter.check_rate_limit(request, user_id, role)
        
        if not allowed:
            from fastapi.responses import JSONResponse
            headers = limiter.get_headers(limit_info)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded. Retry after {limit_info.get('retry_after', 60)} seconds.",
                    "retry_after": limit_info.get("retry_after", 60),
                },
                headers=headers,
            )
    
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

app.include_router(auth_router)
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
async def hub_status(
    user: Optional[APIUser] = Depends(optional_auth),
) -> Dict[str, Any]:
    """Get Trading Hub status and metrics.
    
    Authentication optional - returns limited info if not authenticated.
    """
    hub = get_trading_hub()
    status = hub.get_status()
    
    # If not authenticated, return limited info
    if not user:
        return {
            "state": status.get("state"),
            "message": "Authenticate for full status",
        }
    
    return status


@app.get("/rate-limits", tags=["system"])
async def rate_limit_stats() -> Dict[str, Any]:
    """Get rate limiter statistics."""
    limiter = get_rate_limiter()
    return limiter.get_stats()


@app.post("/hub/alert", tags=["trading"])
async def send_hub_alert(
    alert_type: str,
    title: str,
    message: str,
    symbol: str = None,
    priority: str = "medium",
    user: APIUser = Depends(require_permission(Permission.ALERTS)),
) -> Dict[str, Any]:
    """Send an alert through the Trading Hub.
    
    Requires ALERTS permission.
    
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
        source=f"api:{user.user_id}",
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
