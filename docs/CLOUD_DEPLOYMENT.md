# Cloud Deployment Guide

Get your trading system running in the cloud with unrestricted network access.

## 🚀 Quick Deploy Options

### Option 1: AWS EC2 (Recommended - Most Control)

**Cost:** ~$10-30/month for t3.small or t3.medium

**Steps:**

1. **Launch an EC2 instance:**
   - Go to AWS Console → EC2 → Launch Instance
   - Choose: Ubuntu 22.04 LTS
   - Instance type: t3.small (2 vCPU, 2GB RAM) or larger
   - Create/select key pair for SSH
   - Security group: Allow SSH (port 22) from your IP
   - Storage: 20GB minimum

2. **Connect to your instance:**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **Install Docker:**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   exit  # Log out and back in
   ```

4. **Clone and run your code:**
   ```bash
   git clone https://github.com/DGator86/FINAL_GNOSIS.git
   cd FINAL_GNOSIS

   # Create .env file with your credentials
   nano .env
   # Paste your API keys (Alpaca + Unusual Whales)

   # Start trading!
   docker-compose up -d

   # View logs
   docker-compose logs -f
   ```

---

### Option 2: DigitalOcean Droplet (Easiest)

**Cost:** $6-12/month for Basic or Regular droplet

**Steps:**

1. **Create a droplet:**
   - Log into DigitalOcean → Create Droplet
   - Choose: Ubuntu 22.04
   - Plan: Basic ($6/mo) or Regular ($12/mo)
   - Add your SSH key
   - Create

2. **SSH into droplet:**
   ```bash
   ssh root@your-droplet-ip
   ```

3. **Quick setup:**
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com | sh

   # Clone your repo
   git clone https://github.com/DGator86/FINAL_GNOSIS.git
   cd FINAL_GNOSIS

   # Add credentials
   cat > .env << 'EOF'
   ALPACA_API_KEY=PKDGAH5CJM4G3RZ2NP5WQNH22U
   ALPACA_SECRET_KEY=EfW43tDsmhWgvJkucKhJL3bsXmKyu5Kt1B3WxTFcuHEq
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   UNUSUAL_WHALES_API_TOKEN=8932cd23-72b3-4f74-9848-13f9103b9df5
   ENABLE_TRADING=true
   EOF

   # Start trading
   docker-compose up -d
   ```

---

### Option 3: Railway.app (Simplest - Auto Deploy)

**Cost:** $5/month for Hobby plan

**Steps:**

1. **Push to GitHub** (if not already):
   ```bash
   git add .
   git commit -m "Ready for cloud deployment"
   git push
   ```

2. **Deploy on Railway:**
   - Go to https://railway.app
   - Click "Start a New Project" → "Deploy from GitHub repo"
   - Select FINAL_GNOSIS
   - Add environment variables in Railway dashboard:
     - `ALPACA_API_KEY`: PKDGAH5CJM4G3RZ2NP5WQNH22U
     - `ALPACA_SECRET_KEY`: EfW43tDsmhWgvJkucKhJL3bsXmKyu5Kt1B3WxTFcuHEq
     - `ALPACA_BASE_URL`: https://paper-api.alpaca.markets
     - `UNUSUAL_WHALES_API_TOKEN`: 8932cd23-72b3-4f74-9848-13f9103b9df5
   - Set start command: `python start_trading_now.py`
   - Deploy!

---

### Option 4: Google Cloud Run (Serverless)

**Cost:** Free tier available, ~$5-15/month for continuous running

**Steps:**

1. **Install gcloud CLI** (on your local machine)

2. **Build and deploy:**
   ```bash
   # From your FINAL_GNOSIS directory
   gcloud run deploy gnosis-trading \
     --source . \
     --platform managed \
     --region us-central1 \
     --set-env-vars ALPACA_API_KEY=PKDGAH5CJM4G3RZ2NP5WQNH22U \
     --set-env-vars ALPACA_SECRET_KEY=EfW43tDsmhWgvJkucKhJL3bsXmKyu5Kt1B3WxTFcuHEq \
     --set-env-vars ALPACA_BASE_URL=https://paper-api.alpaca.markets \
     --set-env-vars UNUSUAL_WHALES_API_TOKEN=8932cd23-72b3-4f74-9848-13f9103b9df5 \
     --min-instances 1
   ```

---

### Option 5: Run on Your Home Computer/Raspberry Pi

If you have a computer that stays on:

```bash
# Clone repo
git clone https://github.com/DGator86/FINAL_GNOSIS.git
cd FINAL_GNOSIS

# Create .env file
cat > .env << 'EOF'
ALPACA_API_KEY=PKDGAH5CJM4G3RZ2NP5WQNH22U
ALPACA_SECRET_KEY=EfW43tDsmhWgvJkucKhJL3bsXmKyu5Kt1B3WxTFcuHEq
ALPACA_BASE_URL=https://paper-api.alpaca.markets
UNUSUAL_WHALES_API_TOKEN=8932cd23-72b3-4f74-9848-13f9103b9df5
ENABLE_TRADING=true
EOF

# Install Python dependencies
pip install -r requirements.txt

# Run!
python start_trading_now.py
```

---

## 🔄 Choosing Which Launcher to Use

Once deployed, you can run different modes:

| Script | What It Does | Best For |
|--------|-------------|----------|
| `start_trading_now.py` | Simple paper trading with 5 symbols | Quick start, testing |
| `start_scanner_trading.py` | Multi-timeframe scanner + trading | Day trading, full features |
| `start_full_trading_system.py` | Complete system with all engines | Production trading |
| `start_dynamic_trading.py` | Dynamic universe (25+ symbols) | Options trading, broad market |
| `start_with_dashboard.py` | Web dashboard on port 8080 | Monitoring, analysis |

---

## 📊 Monitoring Your Bot

After deployment:

```bash
# View live logs
docker-compose logs -f trading-bot

# Check running containers
docker ps

# Restart
docker-compose restart

# Stop
docker-compose down
```

Or if running directly with Python:
```bash
# Run in background
nohup python start_trading_now.py > trading.log 2>&1 &

# View logs
tail -f trading.log

# Stop (find process ID)
ps aux | grep python
kill <PID>
```

---

## ✅ Verification

Your bot is working when you see:

```
✅ ALPACA CONNECTION SUCCESSFUL:
   Account ID: ...
   Balance: $...
   Buying Power: $...

⏰ MARKET STATUS:
   Market Open: ✅ YES - TRADING ACTIVE

🚀 TRADING SYSTEM ACTIVE - Monitoring markets...
```

---

## 💡 My Recommendation

**Start with DigitalOcean** ($6/mo Droplet):
- Simplest setup
- Full control
- No network restrictions
- Easy to SSH and debug
- Can upgrade anytime

**Command Summary:**
```bash
# One-time setup (5 minutes)
ssh root@your-droplet-ip
curl -fsSL https://get.docker.com | sh
git clone https://github.com/DGator86/FINAL_GNOSIS.git
cd FINAL_GNOSIS
nano .env  # paste credentials
docker-compose up -d

# Done! Trading is live.
```

---

## 🔒 Security Notes

1. **Never commit .env to GitHub** (already in .gitignore)
2. **Use paper trading first** to verify everything works
3. **Set up alerts** for when your bot stops running
4. **Monitor daily** - check logs and positions

---

## Need Help?

If you run into issues:
1. Check logs: `docker-compose logs -f`
2. Verify network: `curl https://paper-api.alpaca.markets/v2/clock`
3. Test credentials: `python -c "from alpaca.trading.client import TradingClient; print('OK')"`
