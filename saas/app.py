"""FastAPI control plane for managing Super Gnosis as a SaaS."""

from __future__ import annotations

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
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="Super Gnosis SaaS", version="1.1.0")
# Serve static assets for the marketing site
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "saas" / "static")), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class RunRequest(BaseModel):
    """Request payload for triggering a pipeline run."""

    symbol: str


@app.get("/", response_class=HTMLResponse)
async def marketing_home(request: Request) -> HTMLResponse:
    """Public marketing landing page with CTA to app."""

    health = get_health_snapshot()

    return templates.TemplateResponse(
        "marketing/index.html",
        {
            "request": request,
            "health": health,
        },
    )


@app.get("/app", response_class=HTMLResponse)
async def app_dashboard(request: Request) -> HTMLResponse:
    """Render the SaaS dashboard (app)."""

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


class WaitlistRequest(BaseModel):
    email: str
    name: str | None = None


@app.post("/api/waitlist")
async def api_waitlist(req: WaitlistRequest) -> Dict[str, object]:
    """Collect early-access signups into data/waitlist.jsonl."""
    from pathlib import Path
    import json
    from datetime import datetime, timezone

    out = Path("data/waitlist.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "email": req.email.strip(),
        "name": (req.name or "").strip() or None,
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "marketing",
    }
    with out.open("a") as f:
        f.write(json.dumps(payload) + "\n")
    return {"ok": True}


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    import uvicorn

    uvicorn.run("saas.app:app", host="0.0.0.0", port=8000, reload=True)
