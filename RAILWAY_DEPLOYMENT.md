# Railway Deployment Guide

Deploy Super Gnosis Trading System to Railway in minutes.

---

## Quick Start

### 1. Create Railway Account

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Create a new project

### 2. Deploy from GitHub

1. In Railway dashboard, click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Connect your repository
4. Railway will auto-detect the `Dockerfile` and `railway.toml`

### 3. Add Services (Optional but Recommended)

**PostgreSQL Database:**
1. Click **"+ New"** in your project
2. Select **"Database"** → **"PostgreSQL"**
3. Railway auto-injects `DATABASE_URL`

**Redis Cache:**
1. Click **"+ New"** in your project
2. Select **"Database"** → **"Redis"**
3. Railway auto-injects `REDIS_URL`

> **Note:** The app works without these - it falls back to SQLite/memory cache

### 4. Configure Environment Variables

Click on your service → **"Variables"** tab → Add these:

**Required:**
```
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
JWT_SECRET_KEY=generate-a-secure-random-string
```

**Optional (for free tier, leave empty):**
```
UNUSUAL_WHALES_API_TOKEN=
MASSIVE_API_ENABLED=false
POLYGON_API_KEY=your_free_polygon_key
```

See `.env.railway` for full list of variables.

### 5. Deploy

Railway auto-deploys on push. Manual deploy:
1. Click **"Deploy"** button
2. Watch build logs
3. Visit your app at the generated URL

---

## Architecture on Railway

```
┌─────────────────────────────────────────────────┐
│                  Railway Project                 │
├─────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────┐  ┌───────────┐ │
│  │   Web App    │  │ Postgres │  │   Redis   │ │
│  │  (Gnosis)    │──│   (DB)   │  │  (Cache)  │ │
│  │  Port: $PORT │  │  5432    │  │   6379    │ │
│  └──────────────┘  └──────────┘  └───────────┘ │
│         ↓                                       │
│  ┌──────────────────────────────────────────┐  │
│  │            External APIs                  │  │
│  │  • Alpaca (trading/data)                 │  │
│  │  • Yahoo Finance (fallback)              │  │
│  │  • Polygon.io (optional)                 │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## Pricing Tiers

| Tier | Cost | Resources | Best For |
|------|------|-----------|----------|
| **Starter** | $5 credit/mo | 500 hours | Testing, paper trading |
| **Hobby** | $5/mo | Unlimited hours | Personal use |
| **Pro** | $20/mo | Team features | Production |

**Database costs (additional):**
- PostgreSQL: ~$5/mo for small instance
- Redis: ~$5/mo for small instance

---

## Minimal Free Deployment

Deploy without databases (uses in-memory fallbacks):

1. Deploy just the app (no Postgres/Redis)
2. Set environment variables
3. Features that work:
   - API endpoints ✓
   - Paper trading ✓
   - Market data ✓
   - Greeks calculation ✓
   - In-memory caching ✓

4. Features that need DB:
   - Trade history persistence ✗
   - User accounts ✗
   - Long-term analytics ✗

---

## Environment Variables Reference

### Required
| Variable | Description | Example |
|----------|-------------|---------|
| `ALPACA_API_KEY` | Alpaca paper trading key | `PKxxx...` |
| `ALPACA_SECRET_KEY` | Alpaca secret key | `xxx...` |
| `JWT_SECRET_KEY` | JWT signing key | Random 32+ chars |

### Automatically Set by Railway
| Variable | Description |
|----------|-------------|
| `PORT` | Port to listen on |
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |

### Optional
| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `production` | Environment mode |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ALPACA_BASE_URL` | Paper API | Trading endpoint |
| `MASSIVE_API_ENABLED` | `false` | Enable Massive.com |

---

## Health Checks

Railway monitors `/health` endpoint:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "cache": {"backend": "redis", ...}
}
```

---

## Troubleshooting

### Build Fails
- Check `requirements.txt` has all dependencies
- Verify Dockerfile syntax
- Check build logs for specific errors

### App Crashes on Start
- Verify all required env vars are set
- Check `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` are valid
- View deploy logs in Railway dashboard

### Database Connection Errors
- Ensure PostgreSQL service is added
- Check `DATABASE_URL` is auto-populated
- Railway uses `postgres://` - our code handles conversion

### Redis Connection Errors
- App automatically falls back to memory cache
- If Redis added, check `REDIS_URL` is set
- View logs for connection status

### Health Check Fails
- Increase `healthcheckTimeout` in `railway.toml`
- Check app starts within timeout period
- Verify `/health` endpoint returns 200

---

## Local Testing (Railway-like)

Test with Railway-like environment locally:

```bash
# Set environment
export PORT=8000
export DATABASE_URL=postgresql://user:pass@localhost:5432/gnosis
export REDIS_URL=redis://localhost:6379

# Run
uvicorn web_api:app --host 0.0.0.0 --port $PORT
```

Or use Docker:

```bash
docker build -t gnosis .
docker run -p 8000:8000 \
  -e ALPACA_API_KEY=xxx \
  -e ALPACA_SECRET_KEY=xxx \
  gnosis
```

---

## Monitoring

### View Logs
Railway Dashboard → Your Service → **"Logs"** tab

### Metrics
- Built-in `/metrics` endpoint (Prometheus format)
- Railway provides basic CPU/memory metrics

### Alerts
Configure alerts in Railway dashboard for:
- Deployment failures
- Health check failures
- Resource limits

---

## Scaling

### Horizontal Scaling
In `railway.toml`:
```toml
[deploy]
numReplicas = 2  # Run 2 instances
```

### Vertical Scaling
Adjust in Railway dashboard:
- Memory: 512MB → 2GB
- CPU: Shared → Dedicated

---

## CI/CD

Railway auto-deploys on:
- Push to main branch
- Pull request (preview environments)

Configure in Settings → Deployments:
- Auto-deploy: On/Off
- Deploy on PR: On/Off
- Branch filters

---

## Quick Commands

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link project
railway link

# Deploy
railway up

# View logs
railway logs

# Open shell
railway shell

# Set variable
railway variables set KEY=value
```
