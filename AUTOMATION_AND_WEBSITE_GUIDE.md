# GNOSIS Automation & Website Deployment Guide

## Overview
This guide walks you through:
1. **Automated Execution** - Running GNOSIS on a schedule using n8n, cron, or cloud schedulers
2. **Professional Website** - Deploying a production-ready web interface
3. **Production Deployment** - Hosting options and best practices

---

## Part 1: Automated Execution Setup

### Option A: n8n Workflow Automation (Recommended)

**What is n8n?**
n8n is a fair-code workflow automation platform (like Zapier but self-hosted) that can trigger GNOSIS on schedules, webhooks, or events.

#### Step 1: Install n8n

**Cloud Option (Easiest)**:
```bash
# Sign up at n8n.cloud (free tier available)
# https://n8n.cloud
```

**Self-Hosted Option**:
```bash
# Docker (recommended)
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  docker.n8n.io/n8nio/n8n

# OR npm
npm install n8n -g
n8n start
```

#### Step 2: Create GNOSIS Execution Workflow

**Workflow Structure**:
```
[Cron Trigger] → [HTTP Request to GNOSIS API] → [Error Handler] → [Notification]
```

**n8n Workflow JSON** (import this into n8n):

1. **Schedule Trigger** - Run every 5 minutes during market hours
   - Node: "Schedule Trigger"
   - Cron: `*/5 9-16 * * 1-5` (Every 5 min, 9am-4pm, Mon-Fri EST)

2. **Execute GNOSIS Analysis**
   - Node: "HTTP Request"
   - Method: POST
   - URL: `http://your-server:8000/api/trades/scan-and-execute`
   - Headers: `{"Authorization": "Bearer YOUR_API_KEY"}`
   - Body:
     ```json
     {
       "symbols": ["SPY", "QQQ", "AAPL"],
       "mode": "paper",
       "max_positions": 5
     }
     ```

3. **Check Results**
   - Node: "IF" (conditional)
   - Condition: `{{ $json.status === "success" }}`

4. **Success Path** - Log results
   - Node: "Set" → Store results to database/logs

5. **Error Path** - Alert on failures
   - Node: "Send Email" or "Telegram" notification

#### Step 3: Alternative - Direct Script Execution

If you prefer running Python directly:

**n8n Execute Command Node**:
```bash
# SSH into your server
cd /home/user/FINAL_GNOSIS
source venv/bin/activate
python main.py scan-opportunities --top 10 --execute
```

**Full n8n Node Config**:
```json
{
  "name": "Run GNOSIS Trading Loop",
  "type": "n8n-nodes-base.executeCommand",
  "parameters": {
    "command": "cd /home/user/FINAL_GNOSIS && source venv/bin/activate && python main.py multi-symbol-loop --top 5 --duration 300"
  }
}
```

---

### Option B: Cron Jobs (Traditional)

**Edit crontab**:
```bash
crontab -e
```

**Add GNOSIS schedules**:
```cron
# Run trading loop every 5 minutes during market hours (9:30am-4pm EST)
*/5 9-16 * * 1-5 cd /home/user/FINAL_GNOSIS && /usr/bin/python3 main.py scan-opportunities --top 10 >> /var/log/gnosis/trading.log 2>&1

# Daily portfolio summary at 4:30pm
30 16 * * 1-5 cd /home/user/FINAL_GNOSIS && /usr/bin/python3 scripts/daily_summary.py

# Weekly performance report (Sundays at noon)
0 12 * * 0 cd /home/user/FINAL_GNOSIS && /usr/bin/python3 scripts/weekly_report.py
```

**Market Hours Aware Cron** (better approach):
```bash
# Create wrapper script: /home/user/FINAL_GNOSIS/scripts/market_hours_wrapper.sh
#!/bin/bash
HOUR=$(date +%H)
DAY=$(date +%u)  # 1-7 (Mon-Sun)

# Only run Mon-Fri (1-5) between 9am-4pm
if [ "$DAY" -le 5 ] && [ "$HOUR" -ge 9 ] && [ "$HOUR" -le 16 ]; then
    cd /home/user/FINAL_GNOSIS
    source venv/bin/activate
    python main.py multi-symbol-loop --top 5 --duration 300
fi
```

```cron
# Run every 5 minutes, script checks market hours
*/5 * * * * /home/user/FINAL_GNOSIS/scripts/market_hours_wrapper.sh
```

---

### Option C: Cloud Schedulers

#### AWS EventBridge + Lambda
```yaml
# serverless.yml
service: gnosis-automation

functions:
  runTrading:
    handler: lambda_handler.run_gnosis
    timeout: 900  # 15 minutes
    events:
      - schedule:
          rate: cron(*/5 9-16 ? * MON-FRI *)
          enabled: true
    environment:
      ALPACA_API_KEY: ${env:ALPACA_API_KEY}
      ALPACA_SECRET_KEY: ${env:ALPACA_SECRET_KEY}
```

**Lambda Handler** (`lambda_handler.py`):
```python
import subprocess
import os

def run_gnosis(event, context):
    os.chdir('/opt/gnosis')
    result = subprocess.run(
        ['python3', 'main.py', 'scan-opportunities', '--top', '10'],
        capture_output=True,
        text=True
    )
    return {
        'statusCode': 200,
        'body': result.stdout
    }
```

#### Google Cloud Scheduler + Cloud Run
```bash
# Deploy GNOSIS as Cloud Run service
gcloud run deploy gnosis-api \
  --source . \
  --region us-east1 \
  --allow-unauthenticated

# Create scheduler job
gcloud scheduler jobs create http gnosis-trading \
  --schedule="*/5 9-16 * * 1-5" \
  --uri="https://gnosis-api-xxx.run.app/api/trades/scan-and-execute" \
  --http-method=POST \
  --headers="Authorization=Bearer YOUR_API_KEY"
```

#### Azure Logic Apps
1. Create Logic App
2. Add Recurrence Trigger (every 5 minutes)
3. Add Condition (check market hours)
4. Add HTTP action (call GNOSIS API)
5. Add error handling

---

### Option D: Systemd Service (Background Daemon)

**Create service file**: `/etc/systemd/system/gnosis-trading.service`
```ini
[Unit]
Description=GNOSIS Autonomous Trading System
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/user/FINAL_GNOSIS
Environment="PATH=/home/user/FINAL_GNOSIS/venv/bin:/usr/bin"
ExecStart=/home/user/FINAL_GNOSIS/venv/bin/python run_trading_daemon.py
Restart=always
RestartSec=10

# Safety limits
MemoryLimit=2G
CPUQuota=150%

# Logging
StandardOutput=append:/var/log/gnosis/trading.log
StandardError=append:/var/log/gnosis/error.log

[Install]
WantedBy=multi-user.target
```

**Enable and start**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable gnosis-trading
sudo systemctl start gnosis-trading
sudo systemctl status gnosis-trading

# View logs
journalctl -u gnosis-trading -f
```

---

## Part 2: Professional Website Deployment

### Architecture Options

You have 3 main components to deploy:
1. **FastAPI Backend** (`api.py`) - REST API + WebSocket
2. **React Frontend** (`saas_frontend/`) - Modern SaaS UI
3. **Streamlit Dashboard** (`dashboard.py`) - Real-time analytics (optional)

---

### Deployment Option 1: Single VPS (DigitalOcean, Linode, Vultr)

**Recommended Stack**:
- Ubuntu 22.04 LTS
- Nginx (reverse proxy)
- PM2 (process manager)
- SSL via Let's Encrypt

#### Step-by-Step Setup

**1. Provision Server**:
```bash
# DigitalOcean Droplet (recommended: $12/mo - 2GB RAM, 1 vCPU)
# OR Linode, Vultr, AWS Lightsail

# Initial setup
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv nodejs npm nginx certbot python3-certbot-nginx
```

**2. Clone and Setup GNOSIS**:
```bash
cd /opt
sudo git clone https://github.com/YOUR_USERNAME/FINAL_GNOSIS.git gnosis
sudo chown -R $USER:$USER /opt/gnosis
cd /opt/gnosis

# Python backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
nano .env  # Add your API keys
```

**3. Build React Frontend**:
```bash
cd /opt/gnosis/saas_frontend
npm install
npm run build
# Creates /opt/gnosis/saas_frontend/dist
```

**4. Configure Nginx**:

**/etc/nginx/sites-available/gnosis**:
```nginx
# Frontend (React SPA)
server {
    listen 80;
    server_name gnosis.yourdomain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name gnosis.yourdomain.com;

    # SSL certificates (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/gnosis.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/gnosis.yourdomain.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Serve React frontend
    root /opt/gnosis/saas_frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # API backend
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # WebSocket endpoint
    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }

    # Static assets caching
    location ~* \.(jpg|jpeg|png|gif|ico|css|js|svg|woff|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}

# Streamlit Dashboard (optional)
server {
    listen 443 ssl http2;
    server_name dashboard.gnosis.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/gnosis.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/gnosis.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

**Enable site**:
```bash
sudo ln -s /etc/nginx/sites-available/gnosis /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**5. SSL Certificate**:
```bash
sudo certbot --nginx -d gnosis.yourdomain.com -d dashboard.gnosis.yourdomain.com
```

**6. Start Backend with PM2**:

**ecosystem.config.js** (already exists in your repo):
```javascript
module.exports = {
  apps: [
    {
      name: 'gnosis-api',
      script: 'venv/bin/uvicorn',
      args: 'api:app --host 0.0.0.0 --port 8000 --workers 4',
      cwd: '/opt/gnosis',
      interpreter: 'none',
      env: {
        APP_ENV: 'production',
        LOG_LEVEL: 'INFO'
      }
    },
    {
      name: 'gnosis-trading',
      script: 'venv/bin/python',
      args: 'run_trading_daemon.py',
      cwd: '/opt/gnosis',
      interpreter: 'none',
      autorestart: true,
      max_restarts: 10
    },
    {
      name: 'gnosis-dashboard',
      script: 'venv/bin/streamlit',
      args: 'run dashboard.py --server.port 8501 --server.address 0.0.0.0',
      cwd: '/opt/gnosis',
      interpreter: 'none'
    }
  ]
};
```

**Start services**:
```bash
npm install -g pm2
cd /opt/gnosis
pm2 start ecosystem.config.js
pm2 save
pm2 startup  # Enable auto-start on boot
```

**7. Firewall**:
```bash
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

**8. DNS Configuration**:
```
A Record: gnosis.yourdomain.com → YOUR_SERVER_IP
A Record: dashboard.gnosis.yourdomain.com → YOUR_SERVER_IP
```

---

### Deployment Option 2: Docker Compose (Easiest)

**Use existing docker-compose.yml**:
```bash
cd /home/user/FINAL_GNOSIS
docker-compose up -d
```

**Your existing `docker-compose.yml` includes**:
- GNOSIS API
- Trading daemon
- Streamlit dashboard
- (Add frontend service)

**Add React frontend to docker-compose.yml**:
```yaml
services:
  frontend:
    build:
      context: ./saas_frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - api
    environment:
      - VITE_API_URL=http://api:8000

  api:
    # ... existing config

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - /etc/letsencrypt:/etc/letsencrypt
    depends_on:
      - frontend
      - api
```

**Create `saas_frontend/Dockerfile`**:
```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

---

### Deployment Option 3: Cloud Platforms (PaaS)

#### Vercel (Frontend) + Railway (Backend)

**Frontend on Vercel** (Free tier):
```bash
cd saas_frontend
npm install -g vercel
vercel login
vercel --prod

# Set environment variable in Vercel dashboard:
# VITE_API_URL = https://your-backend.railway.app
```

**Backend on Railway** (Free tier):
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up

# Set environment variables in Railway dashboard:
# - ALPACA_API_KEY
# - ALPACA_SECRET_KEY
# - All .env variables
```

#### Render.com (All-in-One)

**Create `render.yaml`**:
```yaml
services:
  - type: web
    name: gnosis-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: ALPACA_API_KEY
        sync: false
      - key: ALPACA_SECRET_KEY
        sync: false

  - type: web
    name: gnosis-frontend
    env: static
    buildCommand: cd saas_frontend && npm install && npm run build
    staticPublishPath: saas_frontend/dist
    routes:
      - type: rewrite
        source: /api/*
        destination: https://gnosis-api.onrender.com/api/*

  - type: worker
    name: gnosis-trading
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python run_trading_daemon.py
```

Push to GitHub, connect to Render, and deploy.

---

### Deployment Option 4: Kubernetes (Enterprise)

**Use existing Kubernetes configs**:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

---

## Part 3: Production Checklist

### Pre-Launch Checklist

- [ ] **Environment Configuration**
  - [ ] `.env` file configured with production credentials
  - [ ] `ALPACA_BASE_URL` set to paper or live API
  - [ ] `APP_ENV=production`
  - [ ] `ENABLE_TRADING=true` (after testing)
  - [ ] All API keys secured (not in git)

- [ ] **Security**
  - [ ] SSL certificate installed
  - [ ] API authentication enabled
  - [ ] Rate limiting configured
  - [ ] CORS properly restricted
  - [ ] Firewall rules set
  - [ ] Secrets in environment variables (not code)

- [ ] **Testing**
  - [ ] Run `pytest tests/ -v` (all 920+ tests pass)
  - [ ] Paper trading tested for 1+ week
  - [ ] API endpoints tested
  - [ ] Frontend connects to backend
  - [ ] WebSocket streaming works
  - [ ] Error handling tested

- [ ] **Monitoring**
  - [ ] Logging configured (`/var/log/gnosis/`)
  - [ ] Prometheus metrics enabled
  - [ ] Error alerting set up (email, Telegram, Discord)
  - [ ] Uptime monitoring (UptimeRobot, Pingdom)
  - [ ] Performance monitoring (New Relic, DataDog)

- [ ] **Automation**
  - [ ] Scheduled execution configured (n8n, cron, etc.)
  - [ ] Market hours validation
  - [ ] Auto-restart on failures
  - [ ] Daily/weekly reports scheduled

- [ ] **Backups**
  - [ ] Database backups automated
  - [ ] Trade ledger backed up
  - [ ] Configuration files versioned
  - [ ] Disaster recovery plan

- [ ] **Compliance**
  - [ ] Terms of Service created
  - [ ] Privacy Policy added
  - [ ] Disclaimer (not financial advice)
  - [ ] User authentication (if multi-tenant)
  - [ ] Data retention policies

---

## Part 4: Recommended Architecture

### Best Practice Setup

```
                              ┌─────────────────┐
                              │   Cloudflare    │
                              │  (CDN + DDoS)   │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │      Nginx      │
                              │  (Load Balance) │
                              └────┬──────┬─────┘
                                   │      │
                   ┌───────────────┘      └───────────────┐
                   │                                       │
          ┌────────▼──────────┐                  ┌────────▼─────────┐
          │  React Frontend   │                  │  FastAPI Backend │
          │    (Vercel/S3)    │                  │   (Docker/PM2)   │
          └───────────────────┘                  └────────┬─────────┘
                                                          │
                                              ┌───────────┼──────────┐
                                              │           │          │
                                    ┌─────────▼───┐  ┌───▼─────┐ ┌─▼──────┐
                                    │ Trading Bot │  │  Redis  │ │PostGres│
                                    │  (PM2/K8s)  │  │ (Cache) │ │  (DB)  │
                                    └─────────────┘  └─────────┘ └────────┘
                                              │
                                    ┌─────────▼────────┐
                                    │  Alpaca Markets  │
                                    │   (Execution)    │
                                    └──────────────────┘
```

**Services**:
1. **Frontend**: Vercel (free) or S3 + CloudFront
2. **Backend API**: DigitalOcean Droplet ($12/mo) or Railway ($5-20/mo)
3. **Trading Bot**: Same server as API or separate worker
4. **Database**: PostgreSQL (for trade history, managed service recommended)
5. **Cache**: Redis (for real-time data)
6. **CDN**: Cloudflare (free tier, DDoS protection)
7. **Monitoring**: Sentry (errors) + Grafana (metrics)

**Monthly Cost Estimate**:
- **Minimal**: $12/mo (DigitalOcean + Vercel free tier)
- **Professional**: $50/mo (Managed DB, Redis, monitoring)
- **Enterprise**: $200+/mo (HA setup, Kubernetes, multi-region)

---

## Part 5: Quick Start Commands

### Local Development
```bash
# Terminal 1: Backend API
cd /home/user/FINAL_GNOSIS
source venv/bin/activate
python api.py

# Terminal 2: Frontend
cd saas_frontend
npm run dev

# Terminal 3: Trading Bot (paper mode)
cd /home/user/FINAL_GNOSIS
source venv/bin/activate
ENABLE_TRADING=true python run_trading_daemon.py

# Terminal 4: Dashboard
streamlit run dashboard.py
```

Access:
- Frontend: http://localhost:5173
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

### Production Commands
```bash
# Deploy with Docker
docker-compose up -d

# Deploy with PM2
pm2 start ecosystem.config.js

# Check status
pm2 status
pm2 logs gnosis-api
docker-compose logs -f

# Update deployment
git pull
docker-compose down && docker-compose up -d --build
# OR
pm2 restart all
```

---

## Part 6: Domain & Branding

### Recommended Domain Structure

**Primary Domain**: `gnosisai.trading` or `supergnosis.ai`

**Subdomains**:
- `app.gnosisai.trading` - Main SaaS application
- `api.gnosisai.trading` - API endpoint
- `dashboard.gnosisai.trading` - Streamlit dashboard
- `docs.gnosisai.trading` - Documentation
- `status.gnosisai.trading` - Status page

### Website Pages to Create

1. **Landing Page** (`/`)
   - Hero: "Institutional-Grade AI Trading System"
   - Features: Hedge analysis, multi-agent AI, options strategies
   - Pricing tiers
   - CTA: Sign up / Request demo

2. **Product Page** (`/product`)
   - Deep dive into engines
   - Real-time analytics showcase
   - Performance metrics
   - Case studies

3. **Pricing** (`/pricing`)
   - Free tier: Paper trading, basic analytics
   - Pro: Live trading, advanced features ($99/mo)
   - Enterprise: Custom limits, API access ($499/mo)

4. **Documentation** (`/docs`)
   - API reference
   - Getting started guide
   - Strategy explanations
   - FAQ

5. **Dashboard** (`/app`)
   - Login-protected
   - React SaaS frontend
   - Real-time portfolio view

6. **Legal** (`/legal`)
   - Terms of Service
   - Privacy Policy
   - Disclaimer: "Not financial advice"

---

## Next Steps

**For n8n Automation**:
1. Set up n8n instance (cloud or self-hosted)
2. Import the workflow template above
3. Configure API credentials
4. Test with paper trading
5. Enable production schedule

**For Website**:
1. Choose deployment platform (recommended: Vercel + Railway)
2. Set up domain and DNS
3. Deploy backend to Railway/DigitalOcean
4. Deploy frontend to Vercel
5. Configure SSL and environment variables
6. Test end-to-end
7. Launch!

**Need Help?**
Let me know which path you want to take, and I can:
- Generate n8n workflow JSON files
- Create deployment scripts
- Set up CI/CD pipeline
- Configure monitoring and alerts
- Build landing page components

Would you like me to start with any specific part?
