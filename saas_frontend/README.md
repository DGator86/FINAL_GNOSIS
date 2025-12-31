# GNOSIS SaaS Frontend

A modern, professional analysis dashboard for the GNOSIS Trading System - similar to Unusual Whales but focused on the unique tenets of the GNOSIS trading intelligence platform.

## 🎯 Features

### Dashboard
- **System Health Monitoring** - Real-time status of trading engines and API connections
- **Performance Metrics** - Track confidence scores, active symbols, and trading activity
- **Live Analytics** - Visual representation of system performance
- **Recent Activity Feed** - Latest pipeline executions and results

### Watchlist
- **Active Universe** - View all symbols being monitored by the system
- **Search & Filter** - Quickly find specific symbols
- **Real-time Updates** - Auto-refresh to see the latest watchlist

### Trade History
- **Complete Ledger** - All pipeline executions with detailed results
- **Confidence Scores** - Visual indicators for signal quality
- **Regime Classification** - Market regime for each analysis
- **Direction & Strategy** - Recommended direction and options strategies
- **Filterable Views** - View last 20, 50, 100, or 500 trades

### Pipeline Execution
- **Manual Triggers** - Run analysis on any symbol on-demand
- **Real-time Results** - Instant feedback on pipeline execution
- **Multi-Engine Analysis** - See results from Physics, PENTA, Sentiment engines
- **Execution History** - Track recent manual runs

### Settings
- **API Configuration** - Configure backend connection
- **Dashboard Preferences** - Customize refresh intervals and notifications
- **System Information** - View version and component details

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- GNOSIS backend running (default: `http://localhost:8000`)

### Installation

```bash
# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Edit .env to configure your backend URL (optional)
nano .env
```

### Development

```bash
# Start development server
npm run dev
```

The application will be available at `http://localhost:5173/`

### Build for Production

```bash
# Build optimized production bundle
npm run build

# Preview production build
npm run preview
```

## 🏗️ Technology Stack

- **React 18** - Modern UI library
- **Vite** - Lightning-fast build tool
- **React Router** - Client-side routing
- **Tailwind CSS** - Utility-first styling
- **Recharts** - Data visualization
- **Lucide React** - Beautiful icons
- **Axios** - HTTP client for API calls

## 📁 Project Structure

```
saas_frontend/
├── src/
│   ├── api/
│   │   └── client.js          # API client and methods
│   ├── components/
│   │   └── Layout.jsx         # Main layout with navigation
│   ├── pages/
│   │   ├── Dashboard.jsx      # System dashboard
│   │   ├── Watchlist.jsx      # Symbol watchlist
│   │   ├── TradeHistory.jsx   # Trade ledger
│   │   ├── Pipeline.jsx       # Pipeline execution
│   │   └── Settings.jsx       # App settings
│   ├── App.jsx                # Main app component
│   ├── main.jsx               # Entry point
│   └── index.css              # Global styles
├── public/                    # Static assets
├── .env.example               # Environment template
├── tailwind.config.js         # Tailwind configuration
├── vite.config.js             # Vite configuration
└── package.json               # Dependencies
```

## 🔌 API Integration

The frontend connects to the GNOSIS backend API with the following endpoints:

### GET `/api/health`
Returns system health status and configuration

**Response:**
```json
{
  "ok": true,
  "config_loaded": true,
  "watchlist_size": 25,
  "ledger_size": 150
}
```

### GET `/api/watchlist`
Returns active symbols being monitored

**Response:**
```json
{
  "symbols": ["SPY", "QQQ", "AAPL", "MSFT", ...],
  "source": "universe_loader"
}
```

### GET `/api/trades?limit=20`
Returns recent pipeline executions

**Response:**
```json
[
  {
    "symbol": "SPY",
    "timestamp": "2025-12-31T12:00:00Z",
    "confidence": 0.85,
    "regime": "STRONG_BULL",
    "direction": "Bullish",
    "strategy": "Bull Call Spread"
  },
  ...
]
```

### POST `/api/run`
Triggers pipeline execution for a symbol

**Request:**
```json
{
  "symbol": "AAPL"
}
```

**Response:**
```json
{
  "ok": true,
  "message": "Pipeline executed successfully",
  "analysis": {
    "confidence": 0.78,
    "regime": "BULL",
    "direction": "Bullish",
    "strategy": "Long Call"
  }
}
```

## 🎨 Design Philosophy

The dashboard is designed with the following principles:

1. **Dark Theme** - Optimized for extended viewing sessions
2. **Data Density** - Maximum information with minimal clutter
3. **Visual Hierarchy** - Important metrics stand out
4. **Real-time Updates** - Auto-refresh for live data
5. **Professional Aesthetic** - Similar to institutional trading platforms

## 🔧 Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
VITE_API_URL=http://localhost:8000
VITE_APP_NAME=GNOSIS Trading Intelligence
VITE_APP_VERSION=1.0.0
```

### Backend Connection

The frontend expects the backend to be running on `http://localhost:8000` by default. To change this:

1. Update `VITE_API_URL` in `.env`
2. Restart the development server

## 📊 GNOSIS System Overview

The frontend visualizes data from the GNOSIS Trading System, which includes:

### Physics Engine
- Price-as-Particle modeling
- Mass (market cap), Velocity (momentum), Energy (volume)
- Kinetic and potential energy calculations

### PENTA Liquidity Analysis
- **Wyckoff** - Volume Spread Analysis
- **ICT** - Fair Value Gaps, Order Blocks
- **Order Flow** - Cumulative Volume Delta
- **Supply/Demand** - Zone identification
- **Liquidity Concepts** - Pool detection

### Sentiment Engine
- Technical indicators (RSI, MACD, Momentum)
- News sentiment
- Options flow analysis
- Social media sentiment

### Hedge Engine
- Dealer positioning
- Greek exposures (Gamma, Vanna, Charm)
- Elasticity and movement energy

## 🚀 Deployment

### Production Build

```bash
npm run build
```

The build output will be in the `dist/` directory. Deploy this to any static hosting service:

- **Vercel**: `vercel deploy`
- **Netlify**: Drag & drop `dist/` folder
- **AWS S3**: Upload `dist/` contents
- **Nginx**: Serve `dist/` directory

### Docker

Create a `Dockerfile`:

```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

Build and run:

```bash
docker build -t gnosis-frontend .
docker run -p 80:80 gnosis-frontend
```

## 📝 License

This project is proprietary. All rights reserved.

## 🤝 Support

For issues or questions, please contact the development team or create an issue in the repository.

---

**Last Updated:** 2025-12-31
