# Super Gnosis SaaS Control Plane

A lightweight FastAPI-powered control panel that wraps the DHPE v3 pipeline with a SaaS-style web experience.

## Features
- Health snapshot covering configuration, watchlist availability, and ledger history
- One-click pipeline trigger with live result summary
- Read-only watchlist view to confirm active coverage
- Recent ledger history preview for quick audit trails

## Running the app
```bash
uvicorn saas.app:app --host 0.0.0.0 --port 8000 --reload
```

Then open http://localhost:8000 to access the control plane.

## API endpoints
- `GET /api/health` – System readiness + metadata
- `GET /api/watchlist` – Active symbols loaded from the universe loader
- `GET /api/trades?limit=20` – Recent ledger entries (if available)
- `POST /api/run` – Trigger a single pipeline iteration for a symbol
