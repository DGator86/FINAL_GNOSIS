# 🚀 Getting Started with GNOSIS Automation & Website

Welcome! This guide will help you quickly set up automated trading and deploy a professional website for GNOSIS.

---

## 📚 Documentation Overview

I've created comprehensive guides for you:

1. **[AUTOMATION_AND_WEBSITE_GUIDE.md](AUTOMATION_AND_WEBSITE_GUIDE.md)** ⭐ **START HERE**
   - Complete guide covering automation (n8n, cron, cloud) and website deployment
   - Detailed architecture options and best practices
   - Step-by-step instructions for all deployment scenarios

2. **[DEPLOYMENT_QUICKSTART.md](DEPLOYMENT_QUICKSTART.md)** ⚡ **QUICK REFERENCE**
   - 5-minute local setup
   - Docker deployment commands
   - Common troubleshooting

3. **[PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md)** ✅ **PRE-LAUNCH**
   - Complete pre-deployment checklist
   - Security hardening steps
   - Go-live procedures

---

## 🎯 Choose Your Path

### Path 1: Quick Local Test (10 minutes)
**Goal**: Run GNOSIS locally to see how it works

```bash
# 1. Setup environment
cp .env.example .env
nano .env  # Add your Alpaca API keys

# 2. Install dependencies
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Test run
python main.py run-once --symbol SPY

# 4. Start dashboard
streamlit run dashboard.py
```

👉 **Next**: Try the automation setup

---

### Path 2: Docker Deployment (20 minutes)
**Goal**: Run everything in containers

```bash
# 1. Configure environment
cp .env.example .env
nano .env  # Add API keys

# 2. Start services
docker-compose -f docker-compose.full-stack.yml up -d

# 3. Check status
docker-compose logs -f
```

**Access**:
- Frontend: http://localhost:5173
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

👉 **Next**: Set up automation (n8n)

---

### Path 3: n8n Automation (30 minutes)
**Goal**: Automate GNOSIS to run on schedule

```bash
# 1. Install n8n (Docker)
docker run -d --restart unless-stopped \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  docker.n8n.io/n8nio/n8n

# 2. Access n8n
# Open: http://localhost:5678

# 3. Import workflow
# - Go to Workflows → Import
# - Upload: n8n-workflow-gnosis.json
# - Configure credentials

# 4. Test workflow
# Click "Execute Workflow"

# 5. Activate
# Toggle "Active" switch
```

👉 **Next**: Deploy to production

---

### Path 4: Production VPS Deployment (2 hours)
**Goal**: Deploy to a real server with domain

**Prerequisites**:
- VPS server (DigitalOcean, Linode, etc.)
- Domain name
- Alpaca API keys

**Quick Deploy**:
```bash
# On your VPS
curl -o deploy.sh https://raw.githubusercontent.com/YOUR_REPO/FINAL_GNOSIS/main/scripts/quick_deploy.sh
chmod +x deploy.sh
./deploy.sh
```

**OR Manual** (see AUTOMATION_AND_WEBSITE_GUIDE.md, Part 2, Option 1)

👉 **Next**: Configure SSL and go live

---

## 🛠️ What's Been Created for You

### Automation Files
- **`n8n-workflow-gnosis.json`** - Pre-configured n8n workflow
  - Triggers every 5 minutes during market hours
  - Executes GNOSIS trading logic
  - Sends notifications on success/failure

- **`scripts/market_hours_wrapper.sh`** - Cron-friendly wrapper
  - Only runs during US market hours
  - Logs execution
  - Production-ready

### Docker Configuration
- **`docker-compose.full-stack.yml`** - Complete deployment
  - Frontend (React SaaS)
  - Backend (FastAPI)
  - Trading Bot
  - Dashboard (Streamlit)
  - Redis cache
  - Nginx reverse proxy
  - Prometheus + Grafana (monitoring)

### Frontend Setup
- **`saas_frontend/Dockerfile`** - Multi-stage build
- **`saas_frontend/nginx.conf`** - Optimized serving config

### Nginx Configuration
- **`nginx/conf.d/gnosis.conf`** - Reverse proxy config
  - API routing
  - WebSocket support
  - SSL ready
  - Security headers

---

## 📋 Quick Reference Commands

### Development
```bash
# Single analysis (no trading)
python main.py run-once --symbol SPY

# Start API server
python api.py

# Run dashboard
streamlit run dashboard.py

# Paper trading loop (5 min)
ENABLE_TRADING=true python main.py multi-symbol-loop --top 5 --duration 300
```

### Docker
```bash
# Start all services
docker-compose -f docker-compose.full-stack.yml up -d

# With trading enabled
docker-compose -f docker-compose.full-stack.yml --profile trading up -d

# View logs
docker-compose logs -f api

# Restart
docker-compose restart api
```

### Automation
```bash
# Test market hours wrapper
./scripts/market_hours_wrapper.sh

# Add to crontab
crontab -e
# Add: */5 * * * * /home/user/FINAL_GNOSIS/scripts/market_hours_wrapper.sh
```

### Production
```bash
# Start with PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup

# View status
pm2 status
pm2 logs gnosis-api

# Nginx
sudo nginx -t
sudo systemctl reload nginx
```

---

## 🔑 Environment Variables

**Minimum Required** (`.env`):
```bash
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

**Recommended**:
```bash
APP_ENV=production
LOG_LEVEL=INFO
ENABLE_TRADING=false  # Set true when ready
MAX_DAILY_LOSS=5000
MAX_POSITIONS=10
TRADING_SYMBOLS=SPY,QQQ,AAPL,NVDA,MSFT
```

**Full list**: See PRODUCTION_CHECKLIST.md

---

## 🌐 Deployment Options Comparison

| Option | Cost | Complexity | Time | Best For |
|--------|------|------------|------|----------|
| **Local** | Free | Low | 10 min | Testing |
| **Docker (Local)** | Free | Medium | 20 min | Development |
| **n8n Cloud** | $20/mo | Low | 30 min | Quick automation |
| **VPS (DigitalOcean)** | $12/mo | Medium | 2 hrs | Production |
| **Cloud Run (GCP)** | ~$5/mo | Medium | 1 hr | Serverless |
| **Kubernetes** | $50+/mo | High | 4 hrs | Enterprise |

**Recommendation**: Start with Docker locally, then VPS for production.

---

## 🎬 Step-by-Step Walkthrough

### 1️⃣ Test Locally (30 minutes)
- [ ] Clone repository
- [ ] Copy `.env.example` to `.env`
- [ ] Add Alpaca API keys
- [ ] Install Python dependencies
- [ ] Run single analysis: `python main.py run-once --symbol SPY`
- [ ] Start dashboard: `streamlit run dashboard.py`
- [ ] Review results

### 2️⃣ Set Up Automation (1 hour)
- [ ] Choose automation method:
  - [ ] **n8n** (recommended for beginners)
  - [ ] **Cron** (simple, reliable)
  - [ ] **Cloud scheduler** (AWS EventBridge, GCP Scheduler)
- [ ] Import/configure workflow
- [ ] Test execution
- [ ] Enable scheduling

### 3️⃣ Deploy Website (2-4 hours)
- [ ] Choose hosting:
  - [ ] **VPS** (DigitalOcean, Linode) - recommended
  - [ ] **PaaS** (Vercel + Railway) - easiest
  - [ ] **Docker Compose** - self-hosted
- [ ] Build frontend: `cd saas_frontend && npm run build`
- [ ] Configure Nginx
- [ ] Set up SSL (Let's Encrypt)
- [ ] Deploy backend
- [ ] Test end-to-end

### 4️⃣ Paper Trading (1 week)
- [ ] Set `ENABLE_TRADING=true`
- [ ] Keep `ALPACA_BASE_URL=https://paper-api.alpaca.markets`
- [ ] Monitor dashboard daily
- [ ] Review trades in Alpaca dashboard
- [ ] Adjust confidence thresholds if needed

### 5️⃣ Production (if ready)
- [ ] Review PRODUCTION_CHECKLIST.md
- [ ] Complete all checklist items
- [ ] Switch to live Alpaca API (optional)
- [ ] Monitor closely for first week
- [ ] Set up alerts and monitoring

---

## 🚨 Important Safety Notes

1. **Start with Paper Trading**
   - GNOSIS defaults to paper trading
   - Test for at least 1 week before live trading
   - Review all trades manually initially

2. **Set Conservative Limits**
   ```bash
   MAX_DAILY_LOSS=5000      # Stop if down $5,000 in a day
   MAX_POSITIONS=10         # Maximum concurrent positions
   MAX_POSITION_SIZE=0.10   # Max 10% of portfolio per position
   ```

3. **Monitor Closely**
   - Check dashboard multiple times per day initially
   - Review error logs daily
   - Validate trades in Alpaca dashboard

4. **Legal Disclaimer**
   - Add to website: "Not financial advice"
   - Include Terms of Service
   - Privacy Policy if collecting user data

---

## 🆘 Troubleshooting

### API Keys Not Working
```bash
# Verify keys are in .env
cat .env | grep ALPACA

# Test connection
python -c "from alpaca_trade_api import REST; api = REST(); print(api.get_account())"
```

### n8n Workflow Not Triggering
- Check time zone (should be America/New_York for US markets)
- Verify workflow is "Active"
- Check execution history for errors
- Test manually with "Execute Workflow"

### Frontend Can't Connect to API
```bash
# Check CORS settings
cat .env | grep CORS_ORIGINS

# Should include frontend URL:
CORS_ORIGINS=http://localhost:5173,https://yourdomain.com
```

### Nginx 502 Bad Gateway
```bash
# Check if API is running
curl http://localhost:8000/health

# Check nginx logs
sudo tail -f /var/log/nginx/error.log

# Restart services
pm2 restart all
sudo systemctl restart nginx
```

**More**: See DEPLOYMENT_QUICKSTART.md "Troubleshooting" section

---

## 📚 Additional Resources

### Documentation
- **Architecture**: `ARCHITECTURE_OVERVIEW.md`
- **Alpaca Integration**: `ALPACA_INTEGRATION.md`
- **Quickstart**: `QUICKSTART.md`
- **Full Guides**: `*.md` files in repo

### External Links
- [Alpaca API Docs](https://alpaca.markets/docs/)
- [n8n Documentation](https://docs.n8n.io/)
- [Docker Docs](https://docs.docker.com/)
- [Nginx Guide](https://nginx.org/en/docs/)

### Community
- GitHub Issues: Report bugs
- Discussions: Share strategies
- Discord/Slack: Real-time help (if available)

---

## ✅ Next Steps

1. **Read** `AUTOMATION_AND_WEBSITE_GUIDE.md` - comprehensive guide
2. **Test** locally - verify everything works
3. **Automate** - set up n8n or cron
4. **Deploy** - choose hosting and go live
5. **Monitor** - watch closely for first week

---

## 💡 Tips for Success

✅ **Start Small** - Test with 1-2 symbols first (SPY, QQQ)

✅ **Be Patient** - Let paper trading run for at least 1 week

✅ **Monitor Closely** - Check dashboard daily for first month

✅ **Stay Conservative** - High confidence thresholds (0.6+) initially

✅ **Backup Everything** - Automate backups from day 1

✅ **Document Changes** - Keep notes on config adjustments

✅ **Stay Compliant** - Add legal disclaimers to website

---

## 🎉 You're Ready!

You now have everything you need to:
- ✅ Automate GNOSIS trading
- ✅ Deploy a professional website
- ✅ Run safely in production

**Questions?** Check the comprehensive guides or create a GitHub issue.

**Ready to deploy?** Start with Path 1 (Quick Local Test) above!

---

**Happy Trading! 🚀📈**

*Remember: Past performance is not indicative of future results. Trade responsibly.*
