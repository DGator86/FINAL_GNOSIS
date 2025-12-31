# GNOSIS Deployment Quick Start

## 🚀 5-Minute Local Setup

### Step 1: Environment Setup
```bash
cd /home/user/FINAL_GNOSIS

# Copy environment template
cp .env.example .env

# Edit with your credentials
nano .env
```

**Required in `.env`**:
```bash
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ENABLE_TRADING=false  # Set to true when ready
```

### Step 2: Install Dependencies
```bash
# Python backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend (optional)
cd saas_frontend
npm install
```

### Step 3: Test the System
```bash
# Run a single analysis (no trading)
python main.py run-once --symbol SPY

# Start the API
python api.py &

# Start the dashboard
streamlit run dashboard.py
```

Access at:
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

---

## 🐳 Docker Deployment (Recommended)

### Option 1: Basic Docker Setup
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 2: Full Stack with Frontend
```bash
# Start everything (frontend + backend + monitoring)
docker-compose -f docker-compose.full-stack.yml up -d

# With trading bot enabled
docker-compose -f docker-compose.full-stack.yml --profile trading up -d

# With dashboard
docker-compose -f docker-compose.full-stack.yml --profile dashboard up -d

# Full production setup
docker-compose -f docker-compose.full-stack.yml \
  --profile trading \
  --profile dashboard \
  --profile monitoring \
  --profile production \
  up -d
```

---

## 🌐 Production VPS Deployment

### Prerequisites
- Ubuntu 22.04 server
- Domain name (e.g., gnosis.yourdomain.com)
- DNS configured

### Quick Deploy Script
```bash
#!/bin/bash
# Run this on your VPS

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.11 python3.11-venv nodejs npm nginx certbot python3-certbot-nginx git

# Clone repository
cd /opt
sudo git clone https://github.com/YOUR_USERNAME/FINAL_GNOSIS.git gnosis
sudo chown -R $USER:$USER /opt/gnosis
cd /opt/gnosis

# Setup Python
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup environment
cp .env.example .env
nano .env  # Add your API keys

# Build frontend
cd saas_frontend
npm install
npm run build
cd ..

# Install PM2
npm install -g pm2

# Start services
pm2 start ecosystem.config.js
pm2 save
pm2 startup

# Setup Nginx
sudo cp nginx/conf.d/gnosis.conf /etc/nginx/sites-available/gnosis
sudo ln -s /etc/nginx/sites-available/gnosis /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get SSL certificate
sudo certbot --nginx -d gnosis.yourdomain.com

# Setup firewall
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

---

## 🤖 n8n Automation Setup

### 1. Install n8n
```bash
# Docker (easiest)
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  docker.n8n.io/n8nio/n8n

# OR cloud: https://n8n.cloud
```

### 2. Import Workflow
1. Open n8n at http://localhost:5678
2. Go to Workflows → Import
3. Upload `n8n-workflow-gnosis.json`
4. Configure credentials:
   - Set your GNOSIS API URL
   - Add API authentication token
   - Configure notification endpoints (email/Telegram)

### 3. Enable Workflow
1. Click "Activate" toggle
2. Test with "Execute Workflow" button
3. Monitor executions in n8n dashboard

---

## 📋 Deployment Checklist

### Pre-Production
- [ ] Alpaca API keys configured (paper trading)
- [ ] Test single analysis: `python main.py run-once --symbol SPY`
- [ ] Verify API works: `curl http://localhost:8000/health`
- [ ] Test paper trading for 1+ week
- [ ] Review all 920+ tests pass: `pytest tests/ -v`

### Production Launch
- [ ] Switch to live Alpaca API (if desired)
- [ ] Set `ENABLE_TRADING=true`
- [ ] Configure position limits in `.env`
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure error alerts (email, Telegram, Discord)
- [ ] SSL certificate installed
- [ ] Firewall configured
- [ ] Backups automated
- [ ] Terms of Service and disclaimers added to website

### Post-Launch
- [ ] Monitor first week closely
- [ ] Review daily P&L reports
- [ ] Check error logs daily
- [ ] Validate trades in Alpaca dashboard
- [ ] Fine-tune confidence thresholds

---

## 🔧 Common Commands

### Development
```bash
# Run analysis
python main.py run-once --symbol SPY

# Start API
python api.py

# Run tests
pytest tests/ -v

# Check logs
tail -f logs/gnosis.log
```

### Docker
```bash
# View logs
docker-compose logs -f api
docker-compose logs -f trading-bot

# Restart service
docker-compose restart api

# Rebuild
docker-compose up -d --build

# Clean up
docker-compose down -v
```

### PM2
```bash
# Status
pm2 status

# Logs
pm2 logs gnosis-api
pm2 logs gnosis-trading

# Restart
pm2 restart all
pm2 restart gnosis-api

# Monitor
pm2 monit
```

### Systemd (if using service)
```bash
# Status
sudo systemctl status gnosis-trading

# Start/Stop
sudo systemctl start gnosis-trading
sudo systemctl stop gnosis-trading

# Logs
journalctl -u gnosis-trading -f
```

---

## 🚨 Troubleshooting

### "No module named 'X'"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "ALPACA_API_KEY not found"
```bash
# Ensure .env exists and is loaded
ls -la .env
cat .env | grep ALPACA
```

### API not responding
```bash
# Check if running
ps aux | grep python
curl http://localhost:8000/health

# Check logs
tail -f logs/api.log
```

### Frontend can't connect to API
```bash
# Check CORS settings in .env
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Restart API
pm2 restart gnosis-api
```

### Trading not executing
```bash
# Check environment variable
echo $ENABLE_TRADING  # Should be "true"

# Check market hours
date  # Must be Mon-Fri 9:30 AM - 4:00 PM EST

# Check logs
tail -f logs/trading.log
```

---

## 📚 Next Steps

1. **Read the full guide**: `AUTOMATION_AND_WEBSITE_GUIDE.md`
2. **Customize strategies**: Edit `config/config.yaml`
3. **Set up monitoring**: Configure Grafana dashboards
4. **Create landing page**: Customize `saas_frontend/src/`
5. **Production deployment**: Follow VPS or cloud platform guide

## 💡 Tips

- **Start with paper trading** - Run for at least 1 week before considering live trading
- **Monitor closely** - Check dashboard daily for first month
- **Adjust confidence thresholds** - Start conservative (0.6+), tune based on performance
- **Use automation carefully** - Test n8n workflows thoroughly before production
- **Backup regularly** - Automate backups of trade ledger and configs
- **Stay compliant** - Add proper disclaimers ("not financial advice")

## 🆘 Need Help?

- Check logs: `logs/` directory
- API docs: http://localhost:8000/docs
- Full guide: `AUTOMATION_AND_WEBSITE_GUIDE.md`
- Architecture: `ARCHITECTURE_OVERVIEW.md`
- Issues: Create GitHub issue with logs attached
