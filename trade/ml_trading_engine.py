"""
ML-Integrated Trading Engine - Combines ML Pipeline with Live/Paper Trading

This module provides a complete ML-driven trading engine that:
1. Uses ML pipeline for signal generation and position sizing
2. Integrates comprehensive safety controls
3. Supports both paper and live trading
4. Provides real-time performance tracking

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
import threading

from loguru import logger

# ML Integration
from ml import (
    create_integrated_pipeline,
    IntegratedMLPipeline,
    MLPipelineConfig,
    MLTradeDecision,
    MarketRegime,
    OptimizationConfig,
    OptimizationStage,
)

# Safety Controls
from trade.trading_safety import (
    TradingSafetyManager,
    SafetyConfig,
    TradeValidationResult,
    SafetyStatus,
    CircuitBreakerState,
    create_safety_manager,
)

# Trading Components
from trade.elite_trade_agent import (
    EliteTradeAgent,
    TradeProposal,
    MarketContext,
    create_elite_trade_agent,
)
from trade.position_lifecycle_manager import (
    PositionLifecycleManager,
    PositionMetrics,
    LifecycleDecision,
    create_lifecycle_manager,
)

from schemas.core_schemas import (
    DirectionEnum,
    OrderResult,
    OrderStatus,
    TradeIdea,
    PipelineResult,
)


class MLTradingState(str, Enum):
    """ML Trading engine states."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    SAFETY_HALT = "safety_halt"
    ERROR = "error"


@dataclass
class MLTradingConfig:
    """Configuration for ML-integrated trading."""
    # Trading mode
    paper_mode: bool = True
    dry_run: bool = False
    
    # ML settings
    ml_preset: str = "balanced"  # conservative, balanced, aggressive
    use_ml_signals: bool = True
    use_ml_position_sizing: bool = True
    ml_confidence_threshold: float = 0.60
    
    # Safety settings
    max_daily_loss_usd: float = 5000.0
    max_positions: int = 10
    max_position_size_pct: float = 0.04
    
    # Scanning
    scan_interval_seconds: int = 60
    position_check_interval_seconds: int = 30
    
    # Symbols
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"])
    
    # Risk management
    default_stop_loss_pct: float = 0.02
    default_take_profit_pct: float = 0.04
    trailing_stop_enabled: bool = True


@dataclass
class MLPosition:
    """Position tracked by ML trading engine."""
    symbol: str
    side: str  # "long" or "short"
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    
    # ML metadata
    ml_confidence: float = 0.0
    ml_regime: str = ""
    ml_signal_strength: str = ""
    
    # P&L
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    
    # Risk levels
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop: Optional[float] = None
    
    # Status
    order_id: str = ""
    is_closing: bool = False


@dataclass
class MLTradingStats:
    """Statistics for ML trading session."""
    start_time: datetime
    start_equity: float
    current_equity: float = 0.0
    
    # ML metrics
    ml_signals_generated: int = 0
    ml_signals_approved: int = 0
    ml_signals_blocked: int = 0
    
    # Trade metrics
    orders_placed: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    
    # Safety metrics
    safety_blocks: int = 0
    circuit_breaker_triggers: int = 0
    
    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Performance
    winning_trades: int = 0
    losing_trades: int = 0
    avg_ml_confidence: float = 0.0
    
    # Regime
    regime_changes: int = 0
    current_regime: str = ""


class MLTradingEngine:
    """
    Complete ML-integrated trading engine.
    
    Combines:
    - IntegratedMLPipeline for signal generation
    - TradingSafetyManager for risk controls
    - EliteTradeAgent for strategy selection
    - Real-time position management
    """
    
    def __init__(
        self,
        config: Optional[MLTradingConfig] = None,
        broker: Any = None,
        market_adapter: Any = None,
        options_adapter: Any = None,
    ):
        """
        Initialize the ML trading engine.
        
        Args:
            config: Trading configuration
            broker: Broker adapter
            market_adapter: Market data adapter
            options_adapter: Options data adapter
        """
        self.config = config or MLTradingConfig()
        self.broker = broker
        self.market_adapter = market_adapter
        self.options_adapter = options_adapter
        
        # State
        self.state = MLTradingState.INITIALIZING
        self.running = False
        self.paused = False
        self._thread: Optional[threading.Thread] = None
        
        # Components (initialized in _initialize)
        self.ml_pipeline: Optional[IntegratedMLPipeline] = None
        self.safety_manager: Optional[TradingSafetyManager] = None
        self.trade_agent: Optional[EliteTradeAgent] = None
        self.lifecycle_manager: Optional[PositionLifecycleManager] = None
        
        # Positions
        self.positions: Dict[str, MLPosition] = {}
        
        # Statistics
        self.stats: Optional[MLTradingStats] = None
        
        # Callbacks
        self.on_ml_signal: Optional[Callable] = None
        self.on_trade_executed: Optional[Callable] = None
        self.on_safety_event: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None
        
        # Regime tracking
        self._last_regime: Optional[MarketRegime] = None
        
        logger.info(
            f"MLTradingEngine created | "
            f"paper={self.config.paper_mode} | "
            f"dry_run={self.config.dry_run} | "
            f"ml_preset={self.config.ml_preset}"
        )
    
    def _initialize(self) -> bool:
        """Initialize all components."""
        try:
            logger.info("Initializing ML Trading Engine components...")
            
            # Get portfolio value
            portfolio_value = self._get_portfolio_value()
            
            # Initialize ML Pipeline
            ml_config = MLPipelineConfig(
                preset=self.config.ml_preset,
                use_lstm=True,
                use_feature_builder=True,
                use_regime_detection=True,
            )
            self.ml_pipeline = IntegratedMLPipeline(
                config=ml_config,
                market_adapter=self.market_adapter,
            )
            logger.info(f"ML Pipeline initialized with preset={self.config.ml_preset}")
            
            # Initialize Safety Manager
            safety_config = SafetyConfig(
                max_daily_loss_usd=self.config.max_daily_loss_usd,
                max_positions=self.config.max_positions,
                max_position_size_pct=self.config.max_position_size_pct,
            )
            self.safety_manager = TradingSafetyManager(
                config=safety_config,
                portfolio_value=portfolio_value,
                on_safety_event=self._handle_safety_event,
            )
            logger.info("Safety Manager initialized")
            
            # Initialize broker if needed
            if not self.broker and not self.config.dry_run:
                self._initialize_broker()
            
            # Initialize trade agent
            self.trade_agent = create_elite_trade_agent(
                options_adapter=self.options_adapter,
                market_adapter=self.market_adapter,
                broker=self.broker,
            )
            logger.info("EliteTradeAgent initialized")
            
            # Initialize lifecycle manager
            self.lifecycle_manager = create_lifecycle_manager(
                profit_target_pct=self.config.default_take_profit_pct * 100,
                stop_loss_pct=self.config.default_stop_loss_pct * 100,
            )
            logger.info("PositionLifecycleManager initialized")
            
            # Initialize stats
            self.stats = MLTradingStats(
                start_time=datetime.now(timezone.utc),
                start_equity=portfolio_value,
                current_equity=portfolio_value,
            )
            
            self.state = MLTradingState.READY
            logger.info("ML Trading Engine initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.state = MLTradingState.ERROR
            return False
    
    def _initialize_broker(self):
        """Initialize broker connection."""
        try:
            from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
            from execution.broker_adapters.settings import get_alpaca_paper_setting
            
            paper_mode = get_alpaca_paper_setting()
            if not paper_mode and self.config.paper_mode:
                logger.error("REFUSING TO RUN IN LIVE MODE when paper_mode=True")
                raise ValueError("Paper mode mismatch")
            
            self.broker = AlpacaBrokerAdapter(paper=self.config.paper_mode)
            logger.info(f"Broker initialized: paper={self.config.paper_mode}")
            
        except Exception as e:
            logger.warning(f"Could not initialize broker: {e}")
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        if self.broker:
            try:
                account = self.broker.get_account()
                return account.equity
            except Exception:
                pass
        return 100000.0  # Default
    
    def start(self, blocking: bool = True):
        """
        Start the ML trading engine.
        
        Args:
            blocking: If True, run in blocking mode
        """
        if not self._initialize():
            logger.error("Failed to initialize, cannot start")
            return
        
        self.running = True
        self.state = MLTradingState.RUNNING
        
        self._print_startup_banner()
        
        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
    
    def _print_startup_banner(self):
        """Print startup information."""
        logger.info("=" * 70)
        logger.info("🤖 ML TRADING ENGINE STARTED")
        logger.info(f"   Mode: {'PAPER' if self.config.paper_mode else 'LIVE'} {'(DRY RUN)' if self.config.dry_run else ''}")
        logger.info(f"   ML Preset: {self.config.ml_preset}")
        logger.info(f"   Symbols: {', '.join(self.config.symbols)}")
        logger.info(f"   Max Daily Loss: ${self.config.max_daily_loss_usd:,.0f}")
        logger.info(f"   Max Positions: {self.config.max_positions}")
        logger.info(f"   ML Confidence Threshold: {self.config.ml_confidence_threshold:.0%}")
        logger.info("=" * 70)
    
    def stop(self):
        """Stop the trading engine."""
        logger.info("Stopping ML Trading Engine...")
        self.running = False
        self.state = MLTradingState.STOPPED
    
    def pause(self):
        """Pause trading (continues monitoring)."""
        self.paused = True
        self.state = MLTradingState.PAUSED
        logger.info("ML Trading Engine PAUSED")
    
    def resume(self):
        """Resume trading."""
        self.paused = False
        self.state = MLTradingState.RUNNING
        logger.info("ML Trading Engine RESUMED")
    
    def _run_loop(self):
        """Main trading loop."""
        last_scan = datetime.min
        last_position_check = datetime.min
        
        try:
            while self.running:
                now = datetime.now(timezone.utc)
                
                # Check if trading is allowed
                allowed, reason = self.safety_manager.is_trading_allowed()
                if not allowed:
                    if self.state != MLTradingState.SAFETY_HALT:
                        logger.warning(f"Trading halted: {reason}")
                        self.state = MLTradingState.SAFETY_HALT
                    time.sleep(10)
                    continue
                elif self.state == MLTradingState.SAFETY_HALT:
                    logger.info("Trading resumed after safety halt")
                    self.state = MLTradingState.RUNNING
                
                # Scan for signals
                if not self.paused and (now - last_scan).seconds >= self.config.scan_interval_seconds:
                    self._scan_for_ml_signals()
                    last_scan = now
                
                # Check positions
                if (now - last_position_check).seconds >= self.config.position_check_interval_seconds:
                    self._check_positions()
                    last_position_check = now
                
                # Update stats
                self._update_stats()
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.state = MLTradingState.ERROR
        finally:
            self._cleanup()
    
    def _scan_for_ml_signals(self):
        """Scan symbols using ML pipeline for trading signals."""
        logger.debug(f"Scanning {len(self.config.symbols)} symbols with ML pipeline...")
        
        for symbol in self.config.symbols:
            try:
                # Skip if already in position
                if symbol in self.positions:
                    continue
                
                # Check max positions
                if len(self.positions) >= self.config.max_positions:
                    logger.debug("Max positions reached")
                    break
                
                # Get ML decision
                ml_decision = self._get_ml_decision(symbol)
                
                if ml_decision is None:
                    continue
                
                self.stats.ml_signals_generated += 1
                
                # Check if signal is strong enough
                if not ml_decision.should_trade:
                    continue
                
                if ml_decision.overall_confidence < self.config.ml_confidence_threshold:
                    logger.debug(
                        f"{symbol}: ML confidence {ml_decision.overall_confidence:.1%} "
                        f"below threshold {self.config.ml_confidence_threshold:.1%}"
                    )
                    continue
                
                # Validate with safety manager
                validation = self._validate_ml_trade(symbol, ml_decision)
                
                if not validation.approved:
                    logger.info(f"{symbol}: Trade blocked - {validation.reason}")
                    self.stats.ml_signals_blocked += 1
                    self.stats.safety_blocks += 1
                    continue
                
                self.stats.ml_signals_approved += 1
                
                # Execute trade
                if not self.paused:
                    self._execute_ml_trade(symbol, ml_decision, validation)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
    
    def _get_ml_decision(self, symbol: str) -> Optional[MLTradeDecision]:
        """Get ML decision for a symbol."""
        try:
            # Build pipeline result (simplified)
            pipeline_result = self._build_pipeline_result(symbol)
            
            if pipeline_result is None:
                return None
            
            # Get market data for regime detection
            market_data = self._get_market_data(symbol)
            
            # Get ML decision
            decision = self.ml_pipeline.process_symbol(
                symbol=symbol,
                pipeline_result=pipeline_result,
                market_data=market_data,
            )
            
            # Track regime changes
            current_regime = self.ml_pipeline.state.current_regime
            if current_regime != self._last_regime:
                self._last_regime = current_regime
                self.stats.regime_changes += 1
                self.stats.current_regime = current_regime.value if current_regime else ""
                logger.info(f"Regime changed to: {self.stats.current_regime}")
            
            # Callback
            if self.on_ml_signal and decision.should_trade:
                self.on_ml_signal(symbol, decision)
            
            return decision
            
        except Exception as e:
            logger.debug(f"Could not get ML decision for {symbol}: {e}")
            return None
    
    def _build_pipeline_result(self, symbol: str) -> Optional[PipelineResult]:
        """Build a pipeline result for a symbol."""
        from schemas.core_schemas import (
            HedgeSnapshot,
            LiquiditySnapshot,
            SentimentSnapshot,
            ElasticitySnapshot,
        )
        
        try:
            spot_price = self._get_spot_price(symbol)
            if spot_price is None or spot_price <= 0:
                return None
            
            now = datetime.now(timezone.utc)
            
            return PipelineResult(
                timestamp=now,
                symbol=symbol,
                hedge_snapshot=HedgeSnapshot(
                    timestamp=now, symbol=symbol,
                    movement_energy=60.0, energy_asymmetry=0.2, hedge_pressure=0.1,
                ),
                liquidity_snapshot=LiquiditySnapshot(
                    timestamp=now, symbol=symbol,
                    bid_depth=100000, ask_depth=100000, spread_bps=5.0, imbalance=0.0,
                ),
                sentiment_snapshot=SentimentSnapshot(
                    timestamp=now, symbol=symbol,
                    composite_score=0.55, news_sentiment=0.5, flow_sentiment=0.6,
                    technical_sentiment=0.55, mtf_score=0.5,
                ),
                elasticity_snapshot=ElasticitySnapshot(
                    timestamp=now, symbol=symbol,
                    elasticity=1.0, volatility=0.25, trend_strength=0.4,
                ),
                consensus={"direction": "long", "confidence": 0.55, "entry_price": spot_price},
            )
        except Exception as e:
            logger.debug(f"Could not build pipeline result for {symbol}: {e}")
            return None
    
    def _get_market_data(self, symbol: str) -> Optional[Any]:
        """Get market data for a symbol."""
        if not self.market_adapter:
            return None
        
        try:
            import pandas as pd
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=30)
            bars = self.market_adapter.get_bars(symbol, start=start, end=end, timeframe="1Day")
            
            if bars:
                return pd.DataFrame([{
                    "timestamp": b.timestamp,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                } for b in bars])
        except Exception:
            pass
        return None
    
    def _get_spot_price(self, symbol: str) -> Optional[float]:
        """Get current spot price."""
        if self.broker:
            try:
                quote = self.broker.get_latest_quote(symbol)
                if quote:
                    bid = quote.get("bid", 0) or 0
                    ask = quote.get("ask", 0) or 0
                    if bid and ask:
                        return (bid + ask) / 2
            except Exception:
                pass
        
        if self.market_adapter:
            try:
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=2)
                bars = self.market_adapter.get_bars(symbol, start=start, end=end, timeframe="1Day")
                if bars:
                    return float(bars[-1].close)
            except Exception:
                pass
        
        # Fallbacks
        fallbacks = {
            "SPY": 600.0, "QQQ": 500.0, "AAPL": 230.0,
            "NVDA": 140.0, "MSFT": 430.0, "GOOGL": 175.0,
        }
        return fallbacks.get(symbol, 100.0)
    
    def _validate_ml_trade(
        self,
        symbol: str,
        decision: MLTradeDecision,
    ) -> TradeValidationResult:
        """Validate ML trade with safety manager."""
        price = decision.signal.forecast_horizons.get(1, 0) or self._get_spot_price(symbol) or 100.0
        
        # Calculate quantity from position size
        position_value = decision.position_size.final_size * self._get_portfolio_value()
        quantity = max(1, int(position_value / price))
        
        side = "buy" if decision.action == "buy" else "sell"
        
        return self.safety_manager.validate_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            metadata={
                "ml_confidence": decision.overall_confidence,
                "regime": decision.signal.regime,
                "liquidity_score": 0.8,  # Would come from pipeline
            },
        )
    
    def _execute_ml_trade(
        self,
        symbol: str,
        decision: MLTradeDecision,
        validation: TradeValidationResult,
    ):
        """Execute an ML-driven trade."""
        try:
            price = self._get_spot_price(symbol) or 100.0
            
            # Apply size adjustments from validation
            position_value = decision.position_size.final_size * self._get_portfolio_value()
            if validation.suggested_size_multiplier < 1.0:
                position_value *= validation.suggested_size_multiplier
            
            quantity = max(1, int(position_value / price))
            
            # Calculate risk levels
            stop_loss = price * (1 - decision.stop_loss) if decision.action == "buy" else price * (1 + decision.stop_loss)
            take_profit = price * (1 + decision.take_profit) if decision.action == "buy" else price * (1 - decision.take_profit)
            
            if self.config.dry_run:
                logger.info(
                    f"[DRY RUN] ML Trade: {decision.action.upper()} {quantity} {symbol} @ ${price:.2f} | "
                    f"Confidence: {decision.overall_confidence:.1%} | Regime: {decision.signal.regime}"
                )
                self._track_position(symbol, decision, quantity, price, "DRY_" + datetime.now().strftime("%H%M%S"))
                self.stats.orders_placed += 1
                return
            
            # Build trade idea
            trade_idea = TradeIdea(
                symbol=symbol,
                direction=DirectionEnum.LONG if decision.action == "buy" else DirectionEnum.SHORT,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=decision.overall_confidence,
                size=position_value,
            )
            
            # Execute through trade agent
            timestamp = datetime.now(timezone.utc)
            results = self.trade_agent.execute_trades([trade_idea], timestamp)
            
            for result in results:
                if result.status == OrderStatus.SUBMITTED:
                    logger.info(
                        f"✅ ML Trade Executed: {decision.action.upper()} {quantity} {symbol} @ ${price:.2f} | "
                        f"Confidence: {decision.overall_confidence:.1%}"
                    )
                    self._track_position(symbol, decision, quantity, price, result.order_id or "")
                    self.stats.orders_placed += 1
                    self.safety_manager.record_order(symbol, decision.action, quantity, price)
                    
                    if self.on_trade_executed:
                        self.on_trade_executed(symbol, decision, result)
                else:
                    logger.warning(f"❌ ML Trade Rejected: {symbol} | {result.message}")
                    self.stats.orders_rejected += 1
                    
        except Exception as e:
            logger.error(f"ML trade execution failed for {symbol}: {e}")
    
    def _track_position(
        self,
        symbol: str,
        decision: MLTradeDecision,
        quantity: int,
        price: float,
        order_id: str,
    ):
        """Track a new position."""
        position = MLPosition(
            symbol=symbol,
            side="long" if decision.action == "buy" else "short",
            quantity=quantity,
            entry_price=price,
            current_price=price,
            entry_time=datetime.now(timezone.utc),
            ml_confidence=decision.overall_confidence,
            ml_regime=decision.signal.regime,
            ml_signal_strength=decision.signal.strength.value,
            stop_loss=price * (1 - decision.stop_loss) if decision.action == "buy" else price * (1 + decision.stop_loss),
            take_profit=price * (1 + decision.take_profit) if decision.action == "buy" else price * (1 - decision.take_profit),
            trailing_stop=price * (1 - decision.trailing_stop) if decision.trailing_stop and decision.action == "buy" else None,
            highest_price=price,
            lowest_price=price,
            order_id=order_id,
        )
        
        self.positions[symbol] = position
        logger.info(f"📊 Tracking ML position: {symbol}")
    
    def _check_positions(self):
        """Check all positions for updates and exits."""
        for symbol, position in list(self.positions.items()):
            try:
                # Update price
                current_price = self._get_spot_price(symbol)
                if current_price:
                    position.current_price = current_price
                    
                    # Update high/low
                    if current_price > position.highest_price:
                        position.highest_price = current_price
                    if current_price < position.lowest_price:
                        position.lowest_price = current_price
                    
                    # Calculate P&L
                    if position.side == "long":
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    else:
                        position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    
                    position.unrealized_pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
                
                # Check exit conditions
                should_exit, exit_reason = self._check_exit_conditions(position)
                
                if should_exit and not position.is_closing:
                    self._close_position(symbol, position, exit_reason)
                
                # Callback
                if self.on_position_update:
                    self.on_position_update(position)
                    
            except Exception as e:
                logger.error(f"Error checking position {symbol}: {e}")
    
    def _check_exit_conditions(self, position: MLPosition) -> Tuple[bool, str]:
        """Check if position should be exited."""
        price = position.current_price
        
        # Stop loss
        if position.side == "long" and price <= position.stop_loss:
            return True, "stop_loss"
        if position.side == "short" and price >= position.stop_loss:
            return True, "stop_loss"
        
        # Take profit
        if position.side == "long" and price >= position.take_profit:
            return True, "take_profit"
        if position.side == "short" and price <= position.take_profit:
            return True, "take_profit"
        
        # Trailing stop
        if position.trailing_stop:
            if position.side == "long":
                # Update trailing stop
                new_trailing = position.highest_price * 0.98  # 2% trail
                if new_trailing > position.trailing_stop:
                    position.trailing_stop = new_trailing
                if price <= position.trailing_stop:
                    return True, "trailing_stop"
        
        return False, ""
    
    def _close_position(self, symbol: str, position: MLPosition, reason: str):
        """Close a position."""
        try:
            position.is_closing = True
            
            if self.config.dry_run:
                logger.info(f"[DRY RUN] Would close {symbol} | reason={reason} | P&L=${position.unrealized_pnl:+,.2f}")
            elif self.broker:
                side = "sell" if position.side == "long" else "buy"
                self.broker.place_order(symbol=symbol, quantity=position.quantity, side=side)
                logger.info(f"✅ Position closed: {symbol} | reason={reason} | P&L=${position.unrealized_pnl:+,.2f}")
            
            # Update stats
            if position.unrealized_pnl > 0:
                self.stats.winning_trades += 1
            else:
                self.stats.losing_trades += 1
            
            self.stats.realized_pnl += position.unrealized_pnl
            
            # Update safety manager
            self.safety_manager.update_pnl(position.unrealized_pnl, self._get_portfolio_value())
            
            # Remove position
            del self.positions[symbol]
            
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            position.is_closing = False
    
    def _handle_safety_event(self, event_type: str, data: Dict[str, Any]):
        """Handle safety events."""
        logger.warning(f"Safety event: {event_type} | {data}")
        
        if event_type == "circuit_breaker_triggered":
            self.stats.circuit_breaker_triggers += 1
            self.state = MLTradingState.SAFETY_HALT
        
        if self.on_safety_event:
            self.on_safety_event(event_type, data)
    
    def _update_stats(self):
        """Update session statistics."""
        if not self.stats:
            return
        
        # Update unrealized P&L
        self.stats.unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        
        # Update equity
        self.stats.current_equity = self._get_portfolio_value()
        
        # Calculate average ML confidence
        if self.stats.ml_signals_approved > 0:
            confidences = [p.ml_confidence for p in self.positions.values()]
            self.stats.avg_ml_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    def _cleanup(self):
        """Cleanup on shutdown."""
        self.running = False
        self.state = MLTradingState.STOPPED
        self._print_session_summary()
    
    def _print_session_summary(self):
        """Print session summary."""
        if not self.stats:
            return
        
        duration = datetime.now(timezone.utc) - self.stats.start_time
        total_trades = self.stats.winning_trades + self.stats.losing_trades
        win_rate = self.stats.winning_trades / total_trades if total_trades > 0 else 0
        
        print("\n" + "=" * 70)
        print("🤖 ML TRADING SESSION SUMMARY")
        print("=" * 70)
        print(f"Duration: {duration}")
        print(f"Mode: {'DRY RUN' if self.config.dry_run else 'PAPER' if self.config.paper_mode else 'LIVE'}")
        print(f"ML Preset: {self.config.ml_preset}")
        print()
        print("📊 ML SIGNALS:")
        print(f"  Generated: {self.stats.ml_signals_generated}")
        print(f"  Approved: {self.stats.ml_signals_approved}")
        print(f"  Blocked: {self.stats.ml_signals_blocked}")
        print(f"  Avg Confidence: {self.stats.avg_ml_confidence:.1%}")
        print()
        print("📈 TRADES:")
        print(f"  Orders Placed: {self.stats.orders_placed}")
        print(f"  Orders Rejected: {self.stats.orders_rejected}")
        print(f"  Winning: {self.stats.winning_trades}")
        print(f"  Losing: {self.stats.losing_trades}")
        print(f"  Win Rate: {win_rate:.1%}")
        print()
        print("💰 P&L:")
        print(f"  Realized: ${self.stats.realized_pnl:+,.2f}")
        print(f"  Unrealized: ${self.stats.unrealized_pnl:+,.2f}")
        print(f"  Total: ${self.stats.realized_pnl + self.stats.unrealized_pnl:+,.2f}")
        print()
        print("🛡️ SAFETY:")
        print(f"  Safety Blocks: {self.stats.safety_blocks}")
        print(f"  Circuit Breakers: {self.stats.circuit_breaker_triggers}")
        print(f"  Regime Changes: {self.stats.regime_changes}")
        print(f"  Final Regime: {self.stats.current_regime}")
        print()
        print(f"Open Positions: {len(self.positions)}")
        for symbol, pos in self.positions.items():
            print(f"  {symbol}: {pos.quantity} @ ${pos.entry_price:.2f} | P&L: ${pos.unrealized_pnl:+,.2f}")
        print("=" * 70 + "\n")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            "state": self.state.value,
            "running": self.running,
            "paused": self.paused,
            "mode": "dry_run" if self.config.dry_run else ("paper" if self.config.paper_mode else "live"),
            "ml_preset": self.config.ml_preset,
            "positions_count": len(self.positions),
            "ml_pipeline_status": self.ml_pipeline.get_status() if self.ml_pipeline else {},
            "safety_status": self.safety_manager.get_status() if self.safety_manager else {},
            "stats": {
                "ml_signals_generated": self.stats.ml_signals_generated if self.stats else 0,
                "ml_signals_approved": self.stats.ml_signals_approved if self.stats else 0,
                "total_pnl": (self.stats.realized_pnl + self.stats.unrealized_pnl) if self.stats else 0,
                "current_regime": self.stats.current_regime if self.stats else "",
            },
        }


# Factory function
def create_ml_trading_engine(
    preset: str = "balanced",
    paper_mode: bool = True,
    dry_run: bool = False,
    symbols: Optional[List[str]] = None,
    max_daily_loss: float = 5000.0,
) -> MLTradingEngine:
    """Create an ML trading engine with preset configuration."""
    config = MLTradingConfig(
        ml_preset=preset,
        paper_mode=paper_mode,
        dry_run=dry_run,
        symbols=symbols or ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"],
        max_daily_loss_usd=max_daily_loss,
    )
    
    return MLTradingEngine(config=config)


__all__ = [
    "MLTradingState",
    "MLTradingConfig",
    "MLPosition",
    "MLTradingStats",
    "MLTradingEngine",
    "create_ml_trading_engine",
]
