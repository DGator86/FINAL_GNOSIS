# 🎯 Super Gnosis DHPE v3 - Dashboard Guide

## Overview

The Enhanced Live Trading Dashboard provides real-time monitoring and analytics for your trading system.

---

## 🚀 Quick Start

### Start the Dashboard

**Option 1: Using the launcher script**
```bash
./start_dashboard.sh
```

**Option 2: Direct Streamlit command**
```bash
streamlit run dashboard.py
```

**Option 3: Custom port**
```bash
streamlit run dashboard.py --server.port 8502
```

The dashboard will automatically open in your browser at `http://localhost:8501`

---

## 📊 Dashboard Features

### Tab 1: Overview 💰
**Account Summary**
- Portfolio value with daily P&L
- Cash balance
- Buying power
- Today's performance

**Visualizations**
- Capital allocation pie chart
- Account details
- Real-time metrics

### Tab 2: Positions 💼
**Position Tracking**
- All open positions
- Quantity, entry price, current price
- Unrealized P&L per position
- Cost basis and market value

**Analytics**
- P&L breakdown bar chart
- Position summary metrics
- Side (long/short) indicators

### Tab 3: Analytics 📈
**Live Market Analysis**
- Hedge Engine v3.0 metrics
  - Elasticity
  - Movement energy
  - Energy asymmetry
  - Dealer gamma sign
  - Regime detection

- Liquidity Engine
  - Liquidity score
  - Bid-ask spread
  - Impact cost

- Sentiment Engine
  - Interactive gauge chart
  - News, flow, technical breakdown

- Elasticity Engine
  - Volatility metrics
  - Regime classification
  - Trend strength

**Agent Intelligence**
- Individual agent suggestions
- Reasoning for each recommendation
- Confidence levels

**Consensus View**
- Weighted consensus direction
- Overall confidence
- Number of agents contributing

### Tab 4: Trade History 📜
**Historical Data**
- Pipeline execution history
- Timestamps and symbols
- Previous analyses

### Tab 5: Engine Metrics ⚙️
**Performance Tracking**
- Engine-level metrics over time
- Historical performance
- System statistics

---

## ⚙️ Settings & Controls

### Sidebar Features

**Connection Status**
- ✅ Alpaca Connected / ❌ Disconnected

**Auto Refresh**
- Enable for live updates (5-second intervals)
- Ideal for monitoring active trading

**Quick Actions**
- Symbol input for instant analysis
- Run Analysis button for on-demand execution

**Debug Mode**
- Toggle for additional diagnostic information

---

## 🎨 Visual Features

### Color Coding
- 🟢 **Green**: Positive P&L, bullish signals
- 🔴 **Red**: Negative P&L, bearish signals
- 🔵 **Blue**: Neutral positions
- ⚪ **Gray**: Inactive or neutral states

### Interactive Charts
- Hover for detailed information
- Zoom and pan capabilities
- Export chart data

### Responsive Design
- Wide layout for maximum screen usage
- Adaptive column widths
- Mobile-friendly (basic support)

---

## 📡 Live Data Sources

### Alpaca Integration
- **Real-time account data** (updated on refresh)
- **Position tracking** with live prices
- **Order status** monitoring

### DHPE Engines
- **Hedge Engine v3.0**: Elasticity calculations
- **Liquidity Engine**: Market microstructure
- **Sentiment Engine**: Multi-source analysis
- **Elasticity Engine**: Volatility regime

### Ledger Storage
- JSONL format for historical data
- Automatic timestamping
- Query-friendly structure

---

## 🔧 Configuration

### Environment Variables
Make sure these are set in your `.env`:
```bash
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
UNUSUAL_WHALES_API_KEY=your_token  # Optional
```

### Dashboard Settings
Located in `config/config.yaml`:
```yaml
tracking:
  ledger_path: "data/ledger.jsonl"
  log_level: "INFO"
```

---

## 💡 Usage Tips

### For Active Trading
1. ✅ Enable **Auto Refresh** in sidebar
2. 📊 Keep **Positions** tab open
3. 👀 Monitor P&L changes
4. 🚨 Watch for agent consensus shifts

### For Analysis
1. 📈 Use **Analytics** tab
2. 🔍 Enter different symbols
3. 📊 Compare engine metrics
4. 🤖 Review agent reasoning

### For Review
1. 📜 Check **Trade History**
2. 📊 Review **Engine Metrics**
3. 📈 Analyze performance trends
4. 🎯 Identify improvement areas

---

## 🚨 Troubleshooting

### Dashboard Won't Start
```bash
# Install missing dependencies
pip install streamlit plotly pandas

# Try a different port
streamlit run dashboard.py --server.port 8502
```

### Can't Connect to Alpaca
- Check `.env` file has correct credentials
- Verify internet connection
- Ensure API keys are valid (not expired)
- Check Alpaca status page

### No Data Showing
- Run at least one analysis: `python main.py run-once --symbol SPY`
- Check `data/ledger.jsonl` exists
- Verify broker connection in sidebar

### Slow Performance
- Disable auto-refresh when not needed
- Clear browser cache
- Restart dashboard
- Check system resources

---

## 🎯 Advanced Features

### Custom Symbols
Enter any tradeable symbol in the sidebar:
- Stocks: `AAPL`, `TSLA`, `NVDA`
- ETFs: `SPY`, `QQQ`, `IWM`
- Indices: `^GSPC`, `^DJI`, `^IXIC`

### Multiple Monitors
Run multiple dashboard instances:
```bash
# Terminal 1
streamlit run dashboard.py --server.port 8501

# Terminal 2
streamlit run dashboard.py --server.port 8502
```

### Remote Access
Configure for network access:
```bash
streamlit run dashboard.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Then access from any device on your network:
```
http://your-ip-address:8501
```

---

## 📊 Key Metrics Explained

### Elasticity
- **High (>1000)**: Market resists price movement
- **Low (<500)**: Market moves easily
- **Interpretation**: Measures dealer hedge pressure

### Movement Energy
- **High**: Expensive to move price
- **Low**: Cheap to move price
- **Usage**: Identifies breakout opportunities

### Energy Asymmetry
- **Positive**: Bullish bias (easier to go up)
- **Negative**: Bearish bias (easier to go down)
- **Zero**: Neutral (symmetric)

### Dealer Gamma Sign
- **Positive**: Dealers long gamma (stabilizing)
- **Negative**: Dealers short gamma (destabilizing)
- **Magnitude**: Strength of positioning

### Liquidity Score
- **>0.8**: Excellent liquidity
- **0.5-0.8**: Good liquidity
- **<0.5**: Poor liquidity

---

## 🔒 Security Notes

### Best Practices
- ✅ Use paper trading account for testing
- ✅ Never share screenshots with API keys visible
- ✅ Keep `.env` file secure (gitignored)
- ✅ Use environment variables, not hardcoded keys

### Production Use
- 🔐 Enable authentication (Streamlit Cloud feature)
- 🌐 Use HTTPS for remote access
- 🔑 Rotate API keys regularly
- 📝 Monitor access logs

---

## 🆕 Future Enhancements

### Planned Features
- [ ] Historical performance charts
- [ ] Advanced filtering and search
- [ ] Export data to CSV/Excel
- [ ] Alert notifications
- [ ] Multi-account support
- [ ] Dark mode theme
- [ ] Mobile app
- [ ] Email/SMS alerts
- [ ] Strategy backtesting UI
- [ ] Risk management dashboard

---

## 📞 Support

### Documentation
- Main README: `README.md`
- Celebration Doc: `CELEBRATION.md`
- Quick Reference: `QUICK_REFERENCE.md`

### Testing
```bash
# Test API connections
python test_api_connections.py

# Test pipeline
python main.py run-once --symbol SPY --dry-run

# Test dashboard (opens browser)
streamlit run dashboard.py
```

### Community
- GitHub Issues: Report bugs or request features
- GitHub Discussions: Ask questions
- Pull Requests: Contribute improvements

---

## 🎉 Enjoy Your Dashboard!

The Enhanced Live Trading Dashboard gives you **complete visibility** into your trading system.

**Monitor. Analyze. Trade. Win.** 🚀

---

*Built with Streamlit, Plotly, and ❤️*  
*Part of Super Gnosis DHPE v3*
