#!/usr/bin/env python3
"""
GNOSIS Full Universe Trading System

Trades the ENTIRE universe of opportunities, not just a single symbol.

Features:
1. Dynamic Universe Scanning - Continuously ranks and selects top opportunities
2. Multi-Symbol Pipeline - Runs analysis on all active symbols in parallel
3. Adaptive Attitude - Adjusts strategy per symbol based on volatility regime
4. Full Position Lifecycle - Entry, monitoring, trailing stops, exits
5. Portfolio-Level Risk - Max positions, daily loss limits, correlation checks
6. ML Enhancement - LSTM, regime similarity, anomaly detection
7. Feedback Loop - Adaptation agent adjusts parameters based on performance

Usage:
    python start_full_system.py                     # Default: Top 10 symbols, adaptive
    python start_full_system.py --top 25            # Trade top 25 opportunities
    python start_full_system.py --scan-interval 60  # Scan universe every 60 seconds
    python start_full_system.py --attitude scalper  # Force scalping mode
    python start_full_system.py --dry-run           # No execution

Author: GNOSIS Trading System
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)
logger.add(
    "logs/gnosis_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
)


def print_banner():
    """Print startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     ██████╗ ███╗   ██╗ ██████╗ ███████╗██╗███████╗                            ║
║    ██╔════╝ ████╗  ██║██╔═══██╗██╔════╝██║██╔════╝                            ║
║    ██║  ███╗██╔██╗ ██║██║   ██║███████╗██║███████╗                            ║
║    ██║   ██║██║╚██╗██║██║   ██║╚════██║██║╚════██║                            ║
║    ╚██████╔╝██║ ╚████║╚██████╔╝███████║██║███████║                            ║
║     ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚═╝╚══════╝                            ║
║                                                                               ║
║            🌐 FULL UNIVERSE AUTONOMOUS TRADING SYSTEM 🌐                      ║
║                         PAPER MODE                                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_environment() -> Dict[str, bool]:
    """Check required environment variables and dependencies."""
    checks = {}
    checks["ALPACA_API_KEY"] = bool(os.getenv("ALPACA_API_KEY"))
    checks["ALPACA_SECRET_KEY"] = bool(os.getenv("ALPACA_SECRET_KEY"))
    checks["UNUSUAL_WHALES_API_KEY"] = bool(os.getenv("UNUSUAL_WHALES_API_KEY"))

    try:
        import alpaca_trade_api
        checks["alpaca-py"] = True
    except ImportError:
        checks["alpaca-py"] = False

    try:
        import pandas
        checks["pandas"] = True
    except ImportError:
        checks["pandas"] = False

    return checks


def print_system_status(checks: Dict[str, bool]) -> bool:
    """Print system status check results."""
    print("\n" + "=" * 80)
    print("  SYSTEM STATUS CHECK")
    print("=" * 80)

    required_ok = all([
        checks.get("ALPACA_API_KEY"),
        checks.get("ALPACA_SECRET_KEY"),
        checks.get("alpaca-py"),
        checks.get("pandas"),
    ])

    for name, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")

    print("=" * 80)

    if not required_ok:
        print("  ❌ CRITICAL: Missing required components!")
        return False

    print("  ✅ All required components ready")
    print("=" * 80)
    return True


class UniverseTradingSystem:
    """
    Full Universe Trading System.

    Manages:
    - Dynamic universe scanning
    - Multi-symbol pipeline execution
    - Autonomous position lifecycle per symbol
    - Portfolio-level risk management
    - ML enhancement
    - Feedback and adaptation
    """

    # Default universe for fallback
    DEFAULT_UNIVERSE = [
        # Major ETFs
        "SPY", "QQQ", "IWM", "DIA", "VXX",
        # Mega-cap tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        # Finance
        "JPM", "BAC", "GS", "MS", "WFC",
        # Other sectors
        "XLE", "XLF", "XLK", "XLV", "XLI",
        # High options volume
        "AMD", "COIN", "MARA", "RIOT", "PLTR",
    ]

    def __init__(
        self,
        top_n: int = 10,
        scan_interval: int = 300,
        trade_interval: int = 30,
        attitude: str = "auto",
        dry_run: bool = False,
        skip_ml: bool = False,
        max_positions: int = 10,
        max_daily_loss_pct: float = 5.0,
    ):
        self.top_n = top_n
        self.scan_interval = scan_interval
        self.trade_interval = trade_interval
        self.attitude = attitude
        self.dry_run = dry_run
        self.skip_ml = skip_ml
        self.max_positions = max_positions
        self.max_daily_loss_pct = max_daily_loss_pct

        # Components (initialized later)
        self.config = None
        self.adapters: Dict[str, Any] = {}
        self.engines: Dict[str, Any] = {}
        self.agents: Dict[str, Any] = {}
        self.composer = None
        self.scanner = None
        self.ml_engine = None
        self.feedback: Dict[str, Any] = {}
        self.autonomous_trader = None

        # State
        self.active_symbols: List[str] = []
        self.symbol_pipelines: Dict[str, Any] = {}
        self.running = False
        self.last_scan_time = 0
        self.iteration = 0

        # Thread safety
        self.lock = threading.Lock()

    def initialize(self):
        """Initialize all system components."""
        print("\n" + "=" * 80)
        print("  INITIALIZING FULL UNIVERSE TRADING SYSTEM")
        print("=" * 80)

        # Load config
        self._load_config()

        # Initialize components in order
        self._init_data_sources()
        self._init_engines()
        self._init_scanner()
        if not self.skip_ml:
            self._init_ml_components()
        self._init_agents()
        self._init_composer()
        self._init_feedback_system()
        self._init_autonomous_trader()

        print("\n" + "=" * 80)
        print("  ✅ ALL COMPONENTS INITIALIZED")
        print("=" * 80)

    def _load_config(self):
        """Load configuration."""
        try:
            from config.config_loader import load_config
            self.config = load_config()
            print("  ✅ Configuration loaded")
        except Exception as e:
            logger.warning(f"Using default config: {e}")
            from types import SimpleNamespace
            self.config = SimpleNamespace(
                engines=SimpleNamespace(hedge={}, liquidity={}, sentiment={}, elasticity={}),
                agents=SimpleNamespace(hedge={}, liquidity={}, sentiment={}, trade={}, composer={}),
                scanner=SimpleNamespace(model_dump=lambda: {}),
                watchlist={},
                adaptation={},
            )

    def _init_data_sources(self):
        """Initialize all data source adapters."""
        from engines.inputs.adapter_factory import (
            create_market_data_adapter,
            create_options_adapter,
            create_broker_adapter,
        )

        print("\n📡 Initializing Data Sources...")

        # Market data
        self.adapters["market"] = create_market_data_adapter(prefer_real=True)
        print(f"  ✅ Market Data: {type(self.adapters['market']).__name__}")

        # Options data
        try:
            self.adapters["options"] = create_options_adapter(prefer_real=True)
            print(f"  ✅ Options Data: {type(self.adapters['options']).__name__}")
        except Exception as e:
            logger.warning(f"Options adapter unavailable: {e}")
            self.adapters["options"] = None

        # Broker (Alpaca paper trading)
        if not self.dry_run:
            try:
                self.adapters["broker"] = create_broker_adapter(paper=True, prefer_real=True)
                account = self.adapters["broker"].get_account()
                print(f"  ✅ Broker: Alpaca Paper Trading")
                print(f"     Account: {account.account_id}")
                print(f"     Portfolio: ${account.portfolio_value:,.2f}")
                print(f"     Buying Power: ${account.buying_power:,.2f}")
            except Exception as e:
                logger.error(f"Failed to connect to broker: {e}")
                raise
        else:
            print("  ⚠️  Broker: DRY-RUN MODE (no execution)")
            self.adapters["broker"] = None

    def _init_engines(self):
        """Initialize all DHPE engines."""
        from engines.hedge.hedge_engine_v3 import HedgeEngineV3
        from engines.liquidity.liquidity_engine_v1 import LiquidityEngineV1
        from engines.sentiment.sentiment_engine_v1 import SentimentEngineV1
        from engines.elasticity.elasticity_engine_v1 import ElasticityEngineV1
        from engines.sentiment.processors import (
            TechnicalSentimentProcessor,
            FlowSentimentProcessor,
        )

        print("\n⚙️  Initializing Engines (DHPE)...")

        engine_config = self.config.engines if hasattr(self.config, 'engines') else SimpleNamespace()

        # Hedge Engine
        hedge_cfg = engine_config.hedge.model_dump() if hasattr(engine_config.hedge, 'model_dump') else {}
        self.engines["hedge"] = HedgeEngineV3(
            options_adapter=self.adapters.get("options"),
            config=hedge_cfg,
        )
        print("  ✅ Hedge Engine V3")

        # Liquidity Engine
        liq_cfg = engine_config.liquidity.model_dump() if hasattr(engine_config.liquidity, 'model_dump') else {}
        self.engines["liquidity"] = LiquidityEngineV1(
            market_adapter=self.adapters["market"],
            config=liq_cfg,
        )
        print("  ✅ Liquidity Engine V1")

        # Sentiment Processors
        sent_cfg = engine_config.sentiment.model_dump() if hasattr(engine_config.sentiment, 'model_dump') else {}
        processors = [TechnicalSentimentProcessor(self.adapters["market"], sent_cfg)]
        if self.adapters.get("options"):
            try:
                processors.append(FlowSentimentProcessor(sent_cfg))
            except Exception:
                pass

        self.engines["sentiment"] = SentimentEngineV1(
            processors=processors,
            config=sent_cfg,
        )
        print("  ✅ Sentiment Engine V1")

        # Elasticity Engine
        elast_cfg = engine_config.elasticity.model_dump() if hasattr(engine_config.elasticity, 'model_dump') else {}
        self.engines["elasticity"] = ElasticityEngineV1(
            market_adapter=self.adapters["market"],
            config=elast_cfg,
        )
        print("  ✅ Elasticity Engine V1")

        # MTF Processor
        self.engines["mtf"] = TechnicalSentimentProcessor(self.adapters["market"], sent_cfg)
        print("  ✅ MTF Processor (1Min-1Day)")

    def _init_scanner(self):
        """Initialize the opportunity scanner."""
        from engines.scanner import OpportunityScanner

        print("\n🔍 Initializing Universe Scanner...")

        self.scanner = OpportunityScanner(
            hedge_engine=self.engines["hedge"],
            liquidity_engine=self.engines["liquidity"],
            sentiment_engine=self.engines["sentiment"],
            elasticity_engine=self.engines["elasticity"],
            options_adapter=self.adapters.get("options"),
            market_adapter=self.adapters["market"],
        )
        print(f"  ✅ Opportunity Scanner (scanning top {self.top_n} from universe)")

    def _init_ml_components(self):
        """Initialize ML enhancement components."""
        print("\n🤖 Initializing ML Components...")

        try:
            from engines.ml.enhancement_engine import MLEnhancementEngine
            from engines.ml.forecasting import KatsForecasterAdapter
            from engines.ml.similarity import FaissRegimeRetriever
            from engines.ml.anomaly import AnomalyDetector

            forecaster = None
            try:
                forecaster = KatsForecasterAdapter(
                    market_adapter=self.adapters["market"],
                    forecast_horizon=5,
                    min_history=30,
                )
                print("  ✅ LSTM Forecaster")
            except Exception as e:
                logger.debug(f"LSTM unavailable: {e}")

            similarity = None
            try:
                similarity = FaissRegimeRetriever(max_history=500, k=5)
                print("  ✅ FAISS Regime Retriever")
            except Exception as e:
                logger.debug(f"FAISS unavailable: {e}")

            anomaly = None
            try:
                anomaly = AnomalyDetector(contamination=0.05, warmup=25)
                print("  ✅ Anomaly Detector")
            except Exception as e:
                logger.debug(f"Anomaly detector unavailable: {e}")

            self.ml_engine = MLEnhancementEngine(
                market_adapter=self.adapters["market"],
                forecaster=forecaster,
                similarity_retriever=similarity,
                anomaly_detector=anomaly,
                rl_evaluator=None,
            )
            print("  ✅ ML Enhancement Engine")

        except ImportError as e:
            logger.warning(f"ML components unavailable: {e}")
            print("  ⚠️  ML components not available (optional)")

    def _init_agents(self):
        """Initialize all primary agents."""
        from agents.hedge_agent_v3 import HedgeAgentV3
        from agents.liquidity_agent_v1 import LiquidityAgentV1
        from agents.sentiment_agent_v1 import SentimentAgentV1

        print("\n🧠 Initializing Primary Agents...")

        agent_config = self.config.agents if hasattr(self.config, 'agents') else SimpleNamespace()

        hedge_cfg = agent_config.hedge.model_dump() if hasattr(agent_config.hedge, 'model_dump') else {}
        self.agents["primary_hedge"] = HedgeAgentV3(config=hedge_cfg)
        print("  ✅ Hedge Agent V3")

        liq_cfg = agent_config.liquidity.model_dump() if hasattr(agent_config.liquidity, 'model_dump') else {}
        self.agents["primary_liquidity"] = LiquidityAgentV1(config=liq_cfg)
        print("  ✅ Liquidity Agent V1")

        sent_cfg = agent_config.sentiment.model_dump() if hasattr(agent_config.sentiment, 'model_dump') else {}
        self.agents["primary_sentiment"] = SentimentAgentV1(config=sent_cfg)
        print("  ✅ Sentiment Agent V1")

    def _init_composer(self):
        """Initialize the Composer Agent."""
        from agents.composer.composer_agent_v1 import ComposerAgentV1

        print("\n🎼 Initializing Composer...")

        agent_config = self.config.agents if hasattr(self.config, 'agents') else SimpleNamespace()
        composer_cfg = agent_config.composer.model_dump() if hasattr(agent_config.composer, 'model_dump') else {}
        self.composer = ComposerAgentV1(config=composer_cfg)
        print("  ✅ Composer Agent V1")

    def _init_feedback_system(self):
        """Initialize feedback and adaptation components."""
        from feedback.tracking_agent import TrackingAgent
        from feedback.adaptation_agent import AdaptationAgent
        from ledger.ledger_store import LedgerStore

        print("\n📊 Initializing Feedback System...")

        # Ledger Store
        ledger_path = Path("data/ledger.jsonl")
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.feedback["ledger"] = LedgerStore(path=str(ledger_path))
        print(f"  ✅ Ledger Store ({ledger_path})")

        # Tracking Agent
        self.feedback["tracking"] = TrackingAgent()
        print("  ✅ Tracking Agent")

        # Adaptation Agent
        adaptation_cfg = self.config.adaptation if hasattr(self.config, 'adaptation') else {}
        self.feedback["adaptation"] = AdaptationAgent(
            config=adaptation_cfg,
            state_path="data/adaptation_state.json",
        )
        print("  ✅ Adaptation Agent")

    def _init_autonomous_trader(self):
        """Initialize the Autonomous Trader."""
        from execution.autonomous_trader import AutonomousTrader, TraderConfig
        from execution.trading_attitude import TradingAttitude
        from execution.trailing_stop_manager import TrailingStopConfig
        from execution.exit_rules_engine import ExitRulesConfig

        print("\n🤖 Initializing Autonomous Trader...")

        # Parse attitude
        enable_adaptive = self.attitude.lower() == "auto"
        attitude_map = {
            "scalper": TradingAttitude.SCALPER,
            "day_trader": TradingAttitude.DAY_TRADER,
            "swing_trader": TradingAttitude.SWING_TRADER,
            "position_trader": TradingAttitude.POSITION_TRADER,
            "auto": TradingAttitude.DAY_TRADER,
        }
        selected_attitude = attitude_map.get(self.attitude.lower(), TradingAttitude.DAY_TRADER)

        trader_config = TraderConfig(
            max_positions=self.max_positions,
            max_position_size_pct=10.0,
            max_daily_trades=50,
            max_daily_loss_pct=self.max_daily_loss_pct,
            min_confidence=0.5,
            min_mtf_alignment=0.4,
            default_stop_pct=1.0,
            default_trailing_pct=0.5,
            default_target_1_pct=0.5,
            default_target_2_pct=1.0,
            default_max_hold_minutes=60,
            paper_mode=True,
            default_attitude=selected_attitude,
            enable_adaptive_attitude=enable_adaptive,
        )

        trailing_config = TrailingStopConfig(
            trailing_pct=0.5,
            initial_stop_pct=1.0,
            breakeven_trigger_pct=0.5,
            tighten_after_target_1=True,
        )

        exit_config = ExitRulesConfig(
            max_hold_minutes=60,
            target_1_pct=0.5,
            target_2_pct=1.0,
            exit_on_signal_reversal=True,
        )

        self.autonomous_trader = AutonomousTrader(
            broker_adapter=self.adapters.get("broker"),
            config=trader_config,
            trailing_config=trailing_config,
            exit_config=exit_config,
        )

        if not enable_adaptive:
            self.autonomous_trader.set_attitude_override(selected_attitude)

        attitude_desc = "ADAPTIVE" if enable_adaptive else selected_attitude.value.upper()
        print(f"  ✅ Autonomous Trader ({attitude_desc})")
        print(f"     Max Positions: {self.max_positions}")
        print(f"     Max Daily Loss: {self.max_daily_loss_pct}%")

    def build_pipeline_for_symbol(self, symbol: str) -> Any:
        """Build a pipeline runner for a specific symbol."""
        from engines.orchestration.pipeline_runner import PipelineRunner
        from trade.trade_agent_v1 import TradeAgentV1
        from watchlist.adaptive_watchlist import AdaptiveWatchlist

        agent_config = self.config.agents if hasattr(self.config, 'agents') else SimpleNamespace()
        trade_cfg = agent_config.trade.model_dump() if hasattr(agent_config.trade, 'model_dump') else {}

        trade_agent = TradeAgentV1(
            options_adapter=self.adapters.get("options"),
            market_adapter=self.adapters["market"],
            config=trade_cfg,
            broker=self.adapters.get("broker"),
        )

        watchlist = AdaptiveWatchlist(
            config=self.config.watchlist if hasattr(self.config, 'watchlist') else {}
        )

        runner = PipelineRunner(
            symbol=symbol,
            engines=self.engines,
            primary_agents=self.agents,
            composer=self.composer,
            trade_agent=trade_agent,
            ledger_store=self.feedback["ledger"],
            config=self.config,
            watchlist=watchlist,
            tracking_agent=self.feedback["tracking"],
            adaptation_agent=self.feedback["adaptation"],
            auto_execute=False,  # We handle execution via AutonomousTrader
            ml_engine=self.ml_engine,
        )

        return runner

    def scan_universe(self) -> List[str]:
        """Scan universe and return top N symbols."""
        print("\n🔍 Scanning Universe for Opportunities...")

        try:
            # Get dynamic universe
            from engines.scanner import get_dynamic_universe
            scanner_cfg = self.config.scanner.model_dump() if hasattr(self.config.scanner, 'model_dump') else {}
            universe = get_dynamic_universe(scanner_cfg, self.top_n * 3)  # Get more for filtering
        except Exception as e:
            logger.warning(f"Dynamic universe unavailable: {e}, using default")
            universe = self.DEFAULT_UNIVERSE

        # Scan for opportunities
        scan_result = self.scanner.scan(universe, top_n=self.top_n)

        # Extract symbols
        symbols = [opp.symbol for opp in scan_result.opportunities]

        print(f"  ✅ Scanned {scan_result.symbols_scanned} symbols in {scan_result.scan_duration_seconds:.1f}s")
        print(f"  📊 Top {len(symbols)} Opportunities:")

        for i, opp in enumerate(scan_result.opportunities[:self.top_n], 1):
            direction_icon = "🟢" if opp.direction == "long" else "🔴" if opp.direction == "short" else "⚪"
            print(f"     {i:2}. {opp.symbol:<6} {direction_icon} {opp.opportunity_type:<15} "
                  f"Score: {opp.score:.3f} Conf: {opp.confidence:.0%}")

        return symbols

    def run_pipeline_for_symbol(self, symbol: str, timestamp: datetime) -> Any:
        """Run pipeline for a single symbol."""
        try:
            # Get or create pipeline for this symbol
            if symbol not in self.symbol_pipelines:
                self.symbol_pipelines[symbol] = self.build_pipeline_for_symbol(symbol)

            runner = self.symbol_pipelines[symbol]
            result = runner.run_once(timestamp)
            return result
        except Exception as e:
            logger.error(f"Pipeline error for {symbol}: {e}")
            return None

    def run_iteration(self, timestamp: datetime):
        """Run a single trading iteration across all active symbols."""
        self.iteration += 1
        current_time = time.time()

        print(f"\n{'═' * 80}")
        print(f"  ITERATION {self.iteration} | {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"{'═' * 80}")

        # Re-scan universe if interval elapsed
        if current_time - self.last_scan_time >= self.scan_interval:
            self.active_symbols = self.scan_universe()
            self.last_scan_time = current_time

            # Clean up old pipelines for symbols no longer active
            active_set = set(self.active_symbols)
            for symbol in list(self.symbol_pipelines.keys()):
                if symbol not in active_set:
                    del self.symbol_pipelines[symbol]

        if not self.active_symbols:
            print("  ⚠️  No active symbols to trade")
            return

        # Run pipelines for all active symbols in parallel
        print(f"\n📊 Running Pipelines for {len(self.active_symbols)} Symbols...")

        all_trade_ideas = []
        results = {}

        with ThreadPoolExecutor(max_workers=min(10, len(self.active_symbols))) as executor:
            futures = {
                executor.submit(self.run_pipeline_for_symbol, symbol, timestamp): symbol
                for symbol in self.active_symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result:
                        results[symbol] = result
                        if result.trade_ideas:
                            all_trade_ideas.extend(result.trade_ideas)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

        # Report pipeline results
        print(f"\n📈 Pipeline Results:")
        for symbol, result in results.items():
            direction = "—"
            confidence = 0
            if result.consensus:
                direction = result.consensus.get("direction", "neutral")
                confidence = result.consensus.get("confidence", 0)

            direction_icon = "🟢" if direction == "long" else "🔴" if direction == "short" else "⚪"
            ideas = len(result.trade_ideas) if result.trade_ideas else 0
            print(f"   {symbol:<6} {direction_icon} {direction:<7} Conf: {confidence:.0%} | Ideas: {ideas}")

        # Process trade ideas through autonomous trader
        if all_trade_ideas and not self.dry_run:
            print(f"\n💰 Processing {len(all_trade_ideas)} Trade Ideas...")

            for idea in all_trade_ideas:
                # Get the pipeline result for this symbol
                pipeline_result = results.get(idea.symbol)
                if not pipeline_result:
                    continue

                # Process through autonomous trader
                opened = self.autonomous_trader.process_trade_ideas(
                    [idea],
                    pipeline_result,
                    timestamp,
                )

                if opened:
                    for pos in opened:
                        print(f"   ✅ OPENED: {pos.side.upper()} {pos.quantity} {pos.symbol} @ ${pos.entry_price:.2f}")

        # Monitor existing positions
        if not self.dry_run:
            print(f"\n👁️  Monitoring Positions...")
            monitor_result = self.autonomous_trader.monitor_positions(timestamp)

            if monitor_result["exits_executed"] > 0:
                print(f"   🔔 Executed {monitor_result['exits_executed']} exits")
            if monitor_result["stops_updated"] > 0:
                print(f"   📊 Updated {monitor_result['stops_updated']} trailing stops")

        # Show dashboard
        print(self.autonomous_trader.get_dashboard_summary())

    def run(self):
        """Run the main trading loop."""
        self.running = True

        def signal_handler(sig, frame):
            print("\n\n⚠️  Shutdown signal received...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print("\n" + "=" * 80)
        print("  🚀 FULL UNIVERSE TRADING SYSTEM ACTIVE")
        print("=" * 80)
        print(f"  Universe Size: Top {self.top_n} opportunities")
        print(f"  Scan Interval: {self.scan_interval} seconds")
        print(f"  Trade Interval: {self.trade_interval} seconds")
        print(f"  Max Positions: {self.max_positions}")
        print(f"  Mode: {'DRY-RUN' if self.dry_run else 'PAPER TRADING'}")
        print(f"  Attitude: {self.attitude.upper()}")
        print("=" * 80)
        print("  Press Ctrl+C to stop")
        print("=" * 80 + "\n")

        try:
            while self.running:
                timestamp = datetime.now(timezone.utc)

                try:
                    self.run_iteration(timestamp)
                except Exception as e:
                    logger.error(f"Iteration error: {e}")
                    print(f"  ❌ Error: {e}")

                if self.running:
                    time.sleep(self.trade_interval)

        finally:
            self.shutdown()

    def shutdown(self):
        """Shutdown the system gracefully."""
        print("\n" + "=" * 80)
        print("  🛑 SHUTTING DOWN")
        print("=" * 80)

        # Close all positions
        if not self.dry_run and self.autonomous_trader:
            positions = self.autonomous_trader.position_manager.positions
            if positions:
                print(f"  Closing {len(positions)} open positions...")
                self.autonomous_trader.close_all_positions("shutdown")

        # Final summary
        if self.autonomous_trader:
            print(self.autonomous_trader.get_dashboard_summary())

        print("\n  ✅ Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GNOSIS Full Universe Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python start_full_system.py                         # Default: Top 10, adaptive
    python start_full_system.py --top 25                # Trade top 25 opportunities
    python start_full_system.py --scan-interval 60      # Scan every 60 seconds
    python start_full_system.py --attitude scalper      # Force scalping mode
    python start_full_system.py --attitude swing_trader # Swing trading mode
    python start_full_system.py --max-positions 20      # Allow 20 concurrent positions
    python start_full_system.py --dry-run               # No execution
        """,
    )

    parser.add_argument(
        "--top", "-n",
        type=int,
        default=10,
        help="Number of top symbols to trade (default: 10)",
    )
    parser.add_argument(
        "--scan-interval",
        type=int,
        default=300,
        help="Seconds between universe scans (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--trade-interval",
        type=int,
        default=30,
        help="Seconds between trade iterations (default: 30)",
    )
    parser.add_argument(
        "--attitude", "-a",
        default="auto",
        choices=["auto", "scalper", "day_trader", "swing_trader", "position_trader"],
        help="Trading attitude (default: auto/adaptive)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=10,
        help="Maximum concurrent positions (default: 10)",
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=5.0,
        help="Maximum daily loss percentage before halt (default: 5.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without executing trades",
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip ML component initialization",
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Environment check
    checks = check_environment()
    if not print_system_status(checks):
        sys.exit(1)

    # Create and run system
    try:
        system = UniverseTradingSystem(
            top_n=args.top,
            scan_interval=args.scan_interval,
            trade_interval=args.trade_interval,
            attitude=args.attitude,
            dry_run=args.dry_run,
            skip_ml=args.skip_ml,
            max_positions=args.max_positions,
            max_daily_loss_pct=args.max_daily_loss,
        )

        system.initialize()
        system.run()

    except KeyboardInterrupt:
        print("\n\n  ⚠️  Interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"\n  ❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
