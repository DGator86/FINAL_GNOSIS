# Super Gnosis Trading System

Institutional-grade options trading platform with ML-powered signals.

## 🚀 Quick Start

### 1. Start the Backend (API)
The backend runs the Gnosis engine and exposes the data via REST API.

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn web_api:app --reload --host 0.0.0.0 --port 8000
```

Access the API docs at: http://localhost:8000/docs

### 2. Start the SaaS Frontend (Website)
The frontend is a modern Next.js dashboard for viewing trade signals.

```bash
cd saas_frontend

# Install Node dependencies
npm install

# Start the dev server
npm run dev
```

Access the dashboard at: http://localhost:3000

## 🏗 Architecture

- **Backend**: Python (FastAPI, SQLAlchemy, PyTorch)
  - Handles market data ingestion
  - Runs ML models
  - Generates trade signals
- **Frontend**: Next.js (React, TailwindCSS)
  - Displays real-time signals
  - User dashboard
  - Signal filtering
- **Database**: PostgreSQL
  - Stores `trade_decisions` (The signals)
  - Stores `saas_users` (User profiles)

## 🔑 Key Features

- **Live Signal Feed**: Real-time stream of AI trade ideas.
- **Advanced Metrics**: View Sentiment, Liquidity, and Hedge agent votes.
- **Multi-Tenancy**: Built-in support for multiple users (Free/Pro tiers).
