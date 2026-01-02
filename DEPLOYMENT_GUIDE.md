# GNOSIS Deployment Guide

## ✅ RECOMMENDED: Railway.app

Railway is the best platform for GNOSIS because it:
- Supports all Python dependencies (no size limits)
- Runs as a persistent service (not serverless)
- Has generous free tier ($5/month credit)
- Auto-deploys from GitHub
- Simple setup (5 minutes)

---

## 🚀 Deploy to Railway (Step-by-Step)

### Step 1: Sign Up
1. Go to https://railway.app
2. Click "Start a New Project"
3. Sign in with GitHub

### Step 2: Deploy from GitHub
1. Click "Deploy from GitHub repo"
2. Select `FINAL_GNOSIS` repository
3. Railway will auto-detect it's a Python app

### Step 3: Add Environment Variables (Optional)
If you want real trading data, add these in Railway dashboard:
- `ALPACA_API_KEY` - Your Alpaca API key
- `ALPACA_SECRET_KEY` - Your Alpaca secret
- `ALPACA_BASE_URL` - `https://paper-api.alpaca.markets`

### Step 4: Deploy!
1. Click "Deploy"
2. Railway will:
   - Install all dependencies from requirements.txt
   - Start GNOSIS service
   - Provide a public URL

### Step 5: Access Your Dashboard
- Railway will give you a URL like: `https://gnosis-production.up.railway.app`
- Open it to see your GNOSIS dashboard!

---

## 📊 Expected Build Time
- First deploy: 3-5 minutes (installing dependencies)
- Subsequent deploys: 1-2 minutes

---

## 🔧 Troubleshooting

### Build fails with memory error
In Railway dashboard:
1. Go to Settings
2. Increase memory limit to 2GB

### Dependencies take too long
This is normal for first build. Railway caches dependencies for future builds.

### Port issues
Railway automatically detects port 8888 from your code. No config needed!

---

## 💰 Cost Estimate

**Free Tier:**
- $5/month credit (no credit card required)
- Enough for development/testing

**Paid Plans (if needed):**
- Starter: $5/month
- Includes $5 usage credit
- Pay only for resources used
- Typical GNOSIS usage: ~$3-8/month

---

## Alternative: Render.com

If Railway doesn't work, try Render:

### Deploy to Render
1. Go to https://render.com
2. New > Web Service
3. Connect GitHub repo
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python3 scripts/gnosis_service.py`
   - **Environment**: Python 3
5. Add environment variables (same as Railway)
6. Deploy

**Free Tier**: Yes (with limitations - sleeps after 15 min inactivity)

---

## Not Recommended: Vercel

Vercel has these limitations for GNOSIS:
- ❌ 250MB function size limit (our dependencies are ~300MB+)
- ❌ Serverless only (no persistent processes)
- ❌ Function timeouts (10-60s)
- ❌ Not suitable for real-time trading systems

A minimal Vercel version is available (`api/vercel_app.py`) but with **very limited functionality** (mock data only).

---

## ✅ Deployment Checklist

- [x] Railway config files created (`railway.json`, `Procfile`)
- [x] Requirements.txt optimized
- [ ] Sign up for Railway account
- [ ] Connect GitHub repository
- [ ] Add environment variables (optional)
- [ ] Deploy and verify
- [ ] Access dashboard at provided URL
- [ ] Configure API keys for live trading data

---

## 📝 Post-Deployment

After deploying, you can:
1. View logs in Railway dashboard
2. Monitor API health at `/api/status`
3. Access rankings at `/api/rankings`
4. View top 10 picks at `/api/top10`
5. See live dashboard at your Railway URL

---

## 🆘 Need Help?

- Railway docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- GNOSIS issues: https://github.com/DGator86/FINAL_GNOSIS/issues

---

**Ready to deploy? Go to https://railway.app and click "Start a New Project"!** 🚀
