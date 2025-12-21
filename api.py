"""
Super Gnosis Trading System - FastAPI Application Entry Point

Main API server with:
- REST endpoints for trading operations
- WebSocket streaming for real-time data
- Prometheus metrics endpoint
- Health checks

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
from fastapi.responses import PlainTextResponse
from loguru import logger

# Import routers
from routers import ml_trades_router, trade_decisions_router, options_greeks_router
from routers.websocket_api import router as websocket_router, start_websocket_publisher, stop_websocket_publisher

# Import utilities
from utils.metrics import metrics, record_api_request
from utils.redis_cache import cache, initialize_cache, close_cache


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
    
    logger.info("Super Gnosis Trading System started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Super Gnosis Trading System...")
    
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
    }


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
