"""
GNOSIS Quick Start - Live Trading with Alpaca
Run this script to start trading!
"""

import sys
import signal
from datetime import datetime

from live_trading_engine import GnosisLiveTradingEngine


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Shutting down GNOSIS...")
    if "engine" in globals():
        engine.stop()
    sys.exit(0)


def main() -> None:
    """Main function to start live trading"""

    print(
        """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║        🚀 GNOSIS LIVE TRADING ENGINE 🚀                  ║
    ║                                                           ║
    ║     Advanced ML-Powered Algorithmic Trading System        ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    )

    # Configuration
    config = {
        # Trading symbols
        "symbols": ["SPY", "QQQ", "IWM"],  # ETFs to trade

        # Trading parameters
        "timeframe": "1Hour",
        "lookback_bars": 100,
        "trading_interval": 3600,  # Analyze every hour

        # Position sizing
        "max_position_value": 5000,  # Max $5k per position
        "use_bracket_orders": True,

        # Agent configuration
        "agent_config": {
            "enabled_agents": [
                "ml_forecasting",
                "regime_detection",
                "risk_management",
                "sentiment_analysis",
            ],

            "agent_weights": {
                "ml_forecasting": 0.35,
                "regime_detection": 0.25,
                "risk_management": 0.25,
                "sentiment_analysis": 0.15,
            },

            "aggregation_strategy": "weighted_average",
            "min_agent_agreement": 0.5,

            # Risk Management Config
            "risk_management_config": {
                "initial_capital": 100000,
                "max_portfolio_risk": 0.02,
                "max_position_size": 0.10,
                "max_drawdown": 0.15,
                "max_daily_loss": 0.05,
                "max_daily_trades": 10,
            },
        },
        "paper_mode": True,
    }

    # Initialize trading engine
    print("\n📊 Initializing GNOSIS Trading Engine...")
    global engine
    engine = GnosisLiveTradingEngine(config)

    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Start trading
    print("\n✅ Configuration loaded")
    print(f"📈 Trading symbols: {config['symbols']}")
    print(f"⏱️  Analysis interval: {config['trading_interval']}s")
    print(f"💰 Max position value: ${config['max_position_value']:,}")

    input("\n▶️  Press ENTER to start live trading (Ctrl+C to stop)...")

    engine.start()

    # Keep main thread alive
    print("\n🔴 GNOSIS is now LIVE! Press Ctrl+C to stop.\n")

    while True:
        try:
            # Print status every 5 minutes
            import time

            time.sleep(300)

            # Print performance summary
            summary = engine.get_performance_summary()
            print(f"\n{'=' * 60}")
            print(f"Status Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'=' * 60}")
            if summary:
                print(f"Portfolio Value: ${summary['account']['portfolio_value']:,.2f}")
                print(f"Cash: ${summary['account']['cash']:,.2f}")
                print(f"Unrealized P&L: ${summary['total_unrealized_pl']:,.2f}")
                print(f"Total Trades: {summary['total_trades']}")
            else:
                print("No performance data available yet.")
            print(f"{'=' * 60}\n")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
