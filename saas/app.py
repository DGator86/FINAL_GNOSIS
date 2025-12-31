"""FastAPI control plane for managing Super Gnosis as a SaaS."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from saas.services import (
    get_health_snapshot,
    load_recent_ledger_entries,
    run_pipeline_once,
    watchlist_overview,
)

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="Super Gnosis SaaS Control Plane", version="1.0.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class RunRequest(BaseModel):
    """Request payload for triggering a pipeline run."""

    symbol: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Render the SaaS dashboard."""

    health = get_health_snapshot()
    ledger = load_recent_ledger_entries(limit=10)
    watchlist = watchlist_overview()

    return templates.TemplateResponse(
        "saas/dashboard.html",
        {
            "request": request,
            "health": health,
            "ledger": ledger,
            "watchlist": watchlist,
        },
    )


@app.get("/tickers", response_class=HTMLResponse)
async def tickers_view(request: Request) -> HTMLResponse:
    """Render the dedicated tickers viewer page."""

    watchlist = watchlist_overview()
    current_time = datetime.now().strftime("%I:%M:%S %p")

    return templates.TemplateResponse(
        "saas/tickers.html",
        {
            "request": request,
            "watchlist": watchlist,
            "current_time": current_time,
        },
    )


@app.get("/api/health", response_model=Dict[str, object])
async def api_health() -> Dict[str, object]:
    """Return configuration + ledger status."""

    return get_health_snapshot()


@app.get("/api/watchlist", response_model=Dict[str, object])
async def api_watchlist() -> Dict[str, object]:
    """Return the active watchlist."""

    return watchlist_overview()


@app.get("/api/trades", response_model=List[Dict[str, object]])
async def api_trades(limit: int = 20) -> List[Dict[str, object]]:
    """Return recent ledger entries (if any)."""

    return load_recent_ledger_entries(limit=limit)


@app.post("/api/run")
async def api_run(request: RunRequest) -> Dict[str, object]:
    """Trigger a pipeline iteration for the requested symbol."""

    result = run_pipeline_once(request.symbol)

    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Pipeline failed."))

    return result


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    import uvicorn

    uvicorn.run("saas.app:app", host="0.0.0.0", port=8000, reload=True)
