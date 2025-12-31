# GNOSIS Production Deployment Checklist

## 🎯 Overview

Use this checklist to ensure a safe, secure, and reliable production deployment of GNOSIS.

---

## Phase 1: Pre-Deployment (Development)

### ✅ Code Readiness
- [ ] All tests passing: `pytest tests/ -v` (920+ tests)
- [ ] No critical security vulnerabilities: `pip-audit`
- [ ] Code reviewed and approved
- [ ] Git repository up to date
- [ ] Latest changes pushed to main branch
- [ ] Version tagged (e.g., `v1.0.0`)

### ✅ Configuration
- [ ] `.env.example` updated with all required variables
- [ ] `config/config.yaml` reviewed and tuned
- [ ] API rate limits configured
- [ ] Position size limits set appropriately
- [ ] Max daily loss limits configured
- [ ] Trading symbols list finalized

### ✅ Paper Trading Validation
- [ ] Paper trading tested for minimum 1 week
- [ ] Win rate reviewed (should be >50%)
- [ ] Max drawdown acceptable
- [ ] No unexpected errors in logs
- [ ] Trade execution latency acceptable
- [ ] All strategies tested on multiple symbols

---

## Phase 2: Infrastructure Setup

### ✅ Server Provisioning
- [ ] VPS/Cloud instance provisioned (min: 2GB RAM, 2 vCPU)
- [ ] Ubuntu 22.04 LTS installed
- [ ] Server hardened (SSH key auth, disable password login)
- [ ] Firewall configured (UFW or cloud firewall)
- [ ] Monitoring agent installed (optional: DataDog, New Relic)

### ✅ Domain & DNS
- [ ] Domain purchased and configured
- [ ] DNS A records created:
  - [ ] `gnosis.yourdomain.com` → Server IP
  - [ ] `api.gnosis.yourdomain.com` → Server IP (optional)
  - [ ] `dashboard.gnosis.yourdomain.com` → Server IP (optional)
- [ ] DNS propagation verified (use `dig` or `nslookup`)

### ✅ Software Installation
- [ ] Python 3.11+ installed
- [ ] Node.js 18+ and npm installed
- [ ] Nginx installed and configured
- [ ] PM2 installed globally: `npm install -g pm2`
- [ ] Certbot installed for SSL
- [ ] Git installed
- [ ] Docker & Docker Compose (if using containers)

### ✅ SSL/TLS Configuration
- [ ] SSL certificate obtained (Let's Encrypt or commercial)
- [ ] Certificate auto-renewal configured
- [ ] HTTPS redirect configured in Nginx
- [ ] SSL grade verified (A+ on SSL Labs)
- [ ] HSTS header enabled

---

## Phase 3: Application Deployment

### ✅ Code Deployment
- [ ] Repository cloned to `/opt/gnosis` (or appropriate location)
- [ ] Correct branch checked out (main/production)
- [ ] File permissions set correctly
- [ ] Python virtual environment created
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Frontend built: `cd saas_frontend && npm run build`

### ✅ Environment Configuration
- [ ] `.env` file created from template
- [ ] All API keys configured:
  - [ ] Alpaca API key and secret
  - [ ] Unusual Whales API token (optional)
  - [ ] Massive.com API credentials (optional)
- [ ] JWT secret key generated (secure random string)
- [ ] Redis URL configured
- [ ] Database connection string configured (if applicable)
- [ ] Log level set appropriately (`INFO` for production)
- [ ] APP_ENV set to `production`
- [ ] File permissions secured: `chmod 600 .env`

### ✅ Secrets Management
- [ ] API keys NOT committed to Git
- [ ] `.env` added to `.gitignore`
- [ ] Secrets stored in environment variables (not hardcoded)
- [ ] Consider using secret management (AWS Secrets Manager, Vault)

### ✅ Database Setup (if applicable)
- [ ] PostgreSQL/MySQL installed or managed service configured
- [ ] Database created
- [ ] Migrations run
- [ ] Database backups configured
- [ ] Connection pooling configured

### ✅ Service Configuration
- [ ] PM2 ecosystem file configured (`ecosystem.config.js`)
- [ ] OR Systemd service file created
- [ ] Services started and verified
- [ ] Auto-restart on failure enabled
- [ ] Auto-start on boot enabled

---

## Phase 4: Web Server & Networking

### ✅ Nginx Configuration
- [ ] Site config created in `/etc/nginx/sites-available/`
- [ ] Symlink created in `/etc/nginx/sites-enabled/`
- [ ] Nginx config tested: `sudo nginx -t`
- [ ] Nginx reloaded: `sudo systemctl reload nginx`
- [ ] Reverse proxy working (API, WebSocket, frontend)
- [ ] Static file serving configured
- [ ] Gzip compression enabled
- [ ] Cache headers configured

### ✅ Firewall Rules
- [ ] SSH (port 22) allowed from trusted IPs only
- [ ] HTTP (port 80) allowed
- [ ] HTTPS (port 443) allowed
- [ ] All other ports blocked
- [ ] Firewall enabled: `sudo ufw enable`
- [ ] Rules tested and verified

### ✅ CORS & Security Headers
- [ ] CORS origins configured correctly in `.env`
- [ ] Security headers added to Nginx:
  - [ ] `X-Frame-Options: SAMEORIGIN`
  - [ ] `X-Content-Type-Options: nosniff`
  - [ ] `X-XSS-Protection: 1; mode=block`
  - [ ] `Strict-Transport-Security` (HSTS)
  - [ ] `Referrer-Policy`

---

## Phase 5: Monitoring & Logging

### ✅ Application Logging
- [ ] Log directory created: `/var/log/gnosis/`
- [ ] Log rotation configured (`logrotate`)
- [ ] Log levels appropriate (INFO for production, DEBUG for troubleshooting)
- [ ] Sensitive data NOT logged (API keys, secrets)
- [ ] Logs accessible but protected (proper permissions)

### ✅ System Monitoring
- [ ] Prometheus configured (if using)
- [ ] Grafana dashboards created (if using)
- [ ] Server metrics monitored (CPU, RAM, disk)
- [ ] Application metrics exposed (`/metrics` endpoint)
- [ ] Metrics endpoint access restricted (IP whitelist)

### ✅ Error Tracking
- [ ] Error alerting configured (email, Telegram, Discord)
- [ ] Critical errors trigger immediate notification
- [ ] Error logs reviewed daily
- [ ] Sentry or similar error tracking (optional)

### ✅ Uptime Monitoring
- [ ] External uptime monitoring configured (UptimeRobot, Pingdom)
- [ ] Health check endpoint monitored: `/health`
- [ ] Alert on downtime configured
- [ ] Response time monitored

### ✅ Trading Monitoring
- [ ] Daily P&L reports configured
- [ ] Position tracking enabled
- [ ] Trade execution logs reviewed
- [ ] Slippage monitored
- [ ] Order fill rate tracked

---

## Phase 6: Security Hardening

### ✅ Server Security
- [ ] OS packages up to date: `apt update && apt upgrade`
- [ ] Automatic security updates enabled
- [ ] SSH hardened:
  - [ ] Key-based auth only
  - [ ] Root login disabled
  - [ ] Custom SSH port (optional)
  - [ ] Fail2ban installed
- [ ] Unnecessary services disabled
- [ ] Rootkits scanned (rkhunter, chkrootkit)

### ✅ Application Security
- [ ] API authentication enabled
- [ ] Rate limiting configured
- [ ] Input validation enabled
- [ ] SQL injection protection (parameterized queries)
- [ ] XSS protection enabled
- [ ] CSRF protection enabled
- [ ] No debug mode in production
- [ ] No test/dev routes exposed

### ✅ Network Security
- [ ] DDoS protection enabled (Cloudflare, AWS Shield)
- [ ] Web Application Firewall (WAF) configured (optional)
- [ ] IP-based access restrictions (for sensitive endpoints)
- [ ] VPN for admin access (optional)

### ✅ Compliance & Legal
- [ ] Terms of Service created and linked
- [ ] Privacy Policy created and linked
- [ ] Disclaimer added ("Not financial advice")
- [ ] Cookie consent (GDPR compliance if applicable)
- [ ] Data retention policy defined
- [ ] User data handling compliant with regulations

---

## Phase 7: Automation Setup

### ✅ Scheduled Tasks
- [ ] Cron jobs configured OR n8n workflows created
- [ ] Market hours validation in place
- [ ] Trading loop scheduled (every 5 minutes during market hours)
- [ ] Daily summary reports scheduled
- [ ] Weekly performance reports scheduled

### ✅ n8n Configuration (if using)
- [ ] n8n instance deployed (cloud or self-hosted)
- [ ] Workflow imported: `n8n-workflow-gnosis.json`
- [ ] API credentials configured
- [ ] Notification channels configured
- [ ] Workflow tested and activated
- [ ] Error handling configured

### ✅ Backup Automation
- [ ] Database backups scheduled (daily)
- [ ] Trade ledger backups scheduled
- [ ] Configuration backups scheduled
- [ ] Backup retention policy set (30 days recommended)
- [ ] Backup restoration tested
- [ ] Offsite backup storage configured (S3, Google Cloud Storage)

---

## Phase 8: Testing & Validation

### ✅ Functionality Testing
- [ ] Health check endpoint: `curl https://yourdomain.com/health`
- [ ] API accessible: `curl https://yourdomain.com/api/health`
- [ ] API docs accessible: `https://yourdomain.com/docs`
- [ ] Frontend loads correctly
- [ ] WebSocket connections working
- [ ] User authentication working (if applicable)

### ✅ Performance Testing
- [ ] Load testing performed (Apache Bench, k6)
- [ ] Response times acceptable (<200ms for API)
- [ ] Concurrent user handling verified
- [ ] Database query performance optimized
- [ ] Redis caching working

### ✅ Trading Validation
- [ ] Manual trade execution tested
- [ ] Automated trading tested (paper mode first)
- [ ] Position limits enforced
- [ ] Stop loss triggers working
- [ ] Order execution latency acceptable (<1s)
- [ ] Portfolio tracking accurate

### ✅ Disaster Recovery Testing
- [ ] Server failure scenario tested
- [ ] Database restore tested
- [ ] Backup restoration verified
- [ ] Recovery time objective (RTO) acceptable
- [ ] Recovery point objective (RPO) acceptable

---

## Phase 9: Go-Live

### ✅ Pre-Launch
- [ ] All checklist items above completed
- [ ] Final code review
- [ ] Stakeholder approval
- [ ] Launch plan communicated
- [ ] Rollback plan prepared

### ✅ Launch
- [ ] Set `ENABLE_TRADING=true` (if ready for live trading)
- [ ] Switch to live Alpaca API (if moving from paper trading)
- [ ] Enable scheduled tasks/workflows
- [ ] Monitor closely for first 24 hours
- [ ] Review first trades manually

### ✅ Post-Launch
- [ ] Monitor error logs continuously
- [ ] Check dashboard every 2 hours (first day)
- [ ] Verify trades in Alpaca dashboard
- [ ] Review P&L daily
- [ ] User feedback collected (if applicable)
- [ ] Performance metrics reviewed

---

## Phase 10: Ongoing Maintenance

### ✅ Daily Tasks
- [ ] Review error logs
- [ ] Check P&L and positions
- [ ] Verify trading executed as expected
- [ ] Monitor system resources (CPU, RAM, disk)

### ✅ Weekly Tasks
- [ ] Review performance metrics
- [ ] Check backup integrity
- [ ] Update dependencies (if needed)
- [ ] Review and optimize strategies

### ✅ Monthly Tasks
- [ ] Security audit
- [ ] Performance tuning
- [ ] Cost optimization review
- [ ] User analytics review (if applicable)
- [ ] SSL certificate renewal check

### ✅ Quarterly Tasks
- [ ] Major version updates
- [ ] Comprehensive security audit
- [ ] Disaster recovery drill
- [ ] Strategy backtest with recent data

---

## 🚨 Emergency Procedures

### Trading Halt Procedure
```bash
# Immediately stop trading
pm2 stop gnosis-trading
# OR
docker-compose stop trading-bot
# OR
sudo systemctl stop gnosis-trading

# Disable automation
# - Pause n8n workflow
# - Comment out cron jobs

# Review recent trades
python scripts/review_recent_trades.py
```

### Rollback Procedure
```bash
# Stop services
pm2 stop all

# Checkout previous version
git checkout <previous-tag>

# Reinstall dependencies
source venv/bin/activate
pip install -r requirements.txt

# Restart services
pm2 restart all
```

### Data Breach Response
1. Immediately rotate all API keys
2. Review access logs
3. Notify affected users (if applicable)
4. Investigate breach source
5. Implement additional security measures
6. Document incident

---

## ✅ Sign-Off

**Deployment Lead**: _________________ Date: _________

**Technical Review**: _________________ Date: _________

**Security Review**: _________________ Date: _________

**Final Approval**: _________________ Date: _________

---

## 📋 Environment Variables Reference

Required `.env` variables for production:

```bash
# === REQUIRED ===
ALPACA_API_KEY=pk_live_xxxxx
ALPACA_SECRET_KEY=sk_live_xxxxx
ALPACA_BASE_URL=https://api.alpaca.markets  # or paper-api.alpaca.markets

# === APPLICATION ===
APP_ENV=production
LOG_LEVEL=INFO
DEBUG=false
ENABLE_TRADING=true

# === SECURITY ===
JWT_SECRET_KEY=<64-char-random-string>
CORS_ORIGINS=https://gnosis.yourdomain.com

# === TRADING CONFIG ===
TRADING_SYMBOLS=SPY,QQQ,AAPL,NVDA,MSFT
MAX_DAILY_LOSS=5000
MAX_POSITIONS=10
MAX_POSITION_SIZE=0.10

# === INFRASTRUCTURE ===
REDIS_URL=redis://localhost:6379/0
API_HOST=0.0.0.0
API_PORT=8000

# === OPTIONAL ===
UNUSUAL_WHALES_API_TOKEN=xxxxx
TELEGRAM_BOT_TOKEN=xxxxx
TELEGRAM_CHAT_ID=xxxxx
SENTRY_DSN=xxxxx
```

Generate JWT secret:
```bash
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

---

**Last Updated**: 2025-12-31
**Version**: 1.0.0
