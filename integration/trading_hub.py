"""
Trading Hub - Central Integration Layer for All Trading Components

Connects and orchestrates:
- Paper Trading Engine with WebSocket broadcasting
- Anomaly Detector with real-time alerts
- Options Flow Scanner with signal generation
- Notification Bot (Telegram/Discord) with alerts
- Portfolio Analytics with dashboard updates
- Greeks Hedger with automated rebalancing

This is the central nervous system of the trading platform.

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from weakref import WeakSet

from loguru import logger


class HubState(str, Enum):
    """Trading hub states."""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class AlertPriority(str, Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TradingAlert:
    """Universal alert structure."""
    alert_type: str
    priority: AlertPriority
    title: str
    message: str
    symbol: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_type": self.alert_type,
            "priority": self.priority.value,
            "title": self.title,
            "message": self.message,
            "symbol": self.symbol,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }


@dataclass
class HubConfig:
    """Trading Hub configuration."""
    # Component toggles
    enable_paper_trading: bool = True
    enable_websocket: bool = True
    enable_anomaly_detection: bool = True
    enable_flow_scanner: bool = True
    enable_notifications: bool = True
    enable_greeks_hedger: bool = True
    enable_portfolio_analytics: bool = True
    
    # Update intervals (seconds)
    portfolio_broadcast_interval: float = 5.0
    positions_broadcast_interval: float = 5.0
    greeks_broadcast_interval: float = 5.0
    flow_scan_interval: float = 10.0
    anomaly_check_interval: float = 1.0
    
    # Alert thresholds
    pnl_alert_threshold: float = 1000.0  # Alert on $1000+ P&L change
    delta_alert_threshold: float = 100.0  # Alert on large delta changes
    
    # Notification settings
    notify_on_trade: bool = True
    notify_on_signal: bool = True
    notify_on_anomaly: bool = True
    notify_on_flow: bool = True
    
    # Auto-hedging
    auto_hedge_enabled: bool = False
    hedge_delta_threshold: float = 500.0


class TradingHub:
    """
    Central hub connecting all trading components.
    
    Responsibilities:
    - Component lifecycle management
    - Event routing between components
    - WebSocket broadcasting
    - Alert aggregation and distribution
    - State synchronization
    """
    
    def __init__(self, config: Optional[HubConfig] = None):
        """Initialize Trading Hub.
        
        Args:
            config: Hub configuration
        """
        self.config = config or HubConfig()
        self.state = HubState.STOPPED
        
        # Components (lazy loaded)
        self._paper_engine = None
        self._websocket_manager = None
        self._anomaly_detector = None
        self._flow_scanner = None
        self._notification_bot = None
        self._greeks_hedger = None
        self._portfolio_analytics = None
        
        # Alert handlers
        self._alert_handlers: List[Callable[[TradingAlert], None]] = []
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        self._running = False
        
        # State cache for broadcasting
        self._portfolio_state: Dict[str, Any] = {}
        self._positions_state: List[Dict[str, Any]] = []
        self._greeks_state: Dict[str, Any] = {}
        self._flow_alerts: List[Dict[str, Any]] = []
        self._anomaly_alerts: List[Dict[str, Any]] = []
        
        # Metrics
        self._metrics = {
            "alerts_sent": 0,
            "broadcasts_sent": 0,
            "trades_executed": 0,
            "anomalies_detected": 0,
            "flow_signals": 0,
        }
        
        logger.info("TradingHub initialized")
    
    # =========================================================================
    # Component Access (Lazy Loading)
    # =========================================================================
    
    @property
    def paper_engine(self):
        """Get or create paper trading engine."""
        if self._paper_engine is None and self.config.enable_paper_trading:
            try:
                from trade.paper_trading_engine import PaperTradingEngine
                self._paper_engine = PaperTradingEngine(dry_run=True)
                self._wire_paper_engine()
                logger.info("Paper Trading Engine loaded")
            except Exception as e:
                logger.warning(f"Could not load Paper Trading Engine: {e}")
        return self._paper_engine
    
    @property
    def websocket_manager(self):
        """Get or create WebSocket connection manager."""
        if self._websocket_manager is None and self.config.enable_websocket:
            try:
                from routers.websocket_api import manager
                self._websocket_manager = manager
                logger.info("WebSocket Manager loaded")
            except Exception as e:
                logger.warning(f"Could not load WebSocket Manager: {e}")
        return self._websocket_manager
    
    @property
    def anomaly_detector(self):
        """Get or create anomaly detector."""
        if self._anomaly_detector is None and self.config.enable_anomaly_detection:
            try:
                from utils.anomaly_detector import AnomalyDetector
                self._anomaly_detector = AnomalyDetector()
                logger.info("Anomaly Detector loaded")
            except Exception as e:
                logger.warning(f"Could not load Anomaly Detector: {e}")
        return self._anomaly_detector
    
    @property
    def flow_scanner(self):
        """Get or create options flow scanner."""
        if self._flow_scanner is None and self.config.enable_flow_scanner:
            try:
                from scanner.options_flow_scanner import OptionsFlowScanner
                self._flow_scanner = OptionsFlowScanner()
                logger.info("Options Flow Scanner loaded")
            except Exception as e:
                logger.warning(f"Could not load Flow Scanner: {e}")
        return self._flow_scanner
    
    @property
    def notification_bot(self):
        """Get or create notification bot."""
        if self._notification_bot is None and self.config.enable_notifications:
            try:
                from notifications.telegram_bot import TradingBot
                self._notification_bot = TradingBot()
                logger.info("Notification Bot loaded")
            except Exception as e:
                logger.warning(f"Could not load Notification Bot: {e}")
        return self._notification_bot
    
    @property
    def greeks_hedger(self):
        """Get or create Greeks hedger."""
        if self._greeks_hedger is None and self.config.enable_greeks_hedger:
            try:
                from trade.greeks_hedger import GreeksHedger
                self._greeks_hedger = GreeksHedger()
                logger.info("Greeks Hedger loaded")
            except Exception as e:
                logger.warning(f"Could not load Greeks Hedger: {e}")
        return self._greeks_hedger
    
    @property
    def portfolio_analytics(self):
        """Get or create portfolio analytics."""
        if self._portfolio_analytics is None and self.config.enable_portfolio_analytics:
            try:
                from dashboard.portfolio_analytics import PortfolioAnalytics
                self._portfolio_analytics = PortfolioAnalytics()
                logger.info("Portfolio Analytics loaded")
            except Exception as e:
                logger.warning(f"Could not load Portfolio Analytics: {e}")
        return self._portfolio_analytics
    
    # =========================================================================
    # Component Wiring
    # =========================================================================
    
    def _wire_paper_engine(self):
        """Wire paper trading engine callbacks."""
        if self._paper_engine is None:
            return
        
        # On signal generated
        self._paper_engine.on_signal_generated = self._handle_signal_generated
        
        # On order placed
        self._paper_engine.on_order_placed = self._handle_order_placed
        
        # On position update
        self._paper_engine.on_position_update = self._handle_position_update
        
        # On lifecycle action
        self._paper_engine.on_lifecycle_action = self._handle_lifecycle_action
        
        logger.info("Paper Engine callbacks wired")
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    def _handle_signal_generated(self, signal: Dict[str, Any]):
        """Handle trading signal generated by the engine."""
        logger.info(f"Signal generated: {signal.get('symbol')} - {signal.get('direction')}")
        
        # Create alert
        alert = TradingAlert(
            alert_type="signal",
            priority=AlertPriority.MEDIUM,
            title=f"Trading Signal: {signal.get('symbol')}",
            message=f"{signal.get('direction', 'N/A').upper()} signal with {signal.get('confidence', 0):.0%} confidence",
            symbol=signal.get("symbol"),
            data=signal,
            source="trade_agent",
        )
        
        self._dispatch_alert(alert)
        self._metrics["flow_signals"] += 1
        
        # Broadcast via WebSocket (only if event loop is running)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._broadcast_signal(signal))
        except RuntimeError:
            pass  # No event loop running
    
    def _handle_order_placed(self, order: Dict[str, Any]):
        """Handle order placed event."""
        logger.info(f"Order placed: {order.get('symbol')} - {order.get('side')}")
        
        alert = TradingAlert(
            alert_type="order",
            priority=AlertPriority.HIGH,
            title=f"Order Placed: {order.get('symbol')}",
            message=f"{order.get('side', 'N/A').upper()} {order.get('qty', 0)} @ ${order.get('price', 0):.2f}",
            symbol=order.get("symbol"),
            data=order,
            source="execution",
        )
        
        self._dispatch_alert(alert)
        self._metrics["trades_executed"] += 1
        
        # Broadcast order update (only if event loop is running)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._broadcast_order(order))
        except RuntimeError:
            pass  # No event loop running
    
    def _handle_position_update(self, position: Dict[str, Any]):
        """Handle position update event."""
        symbol = position.get("symbol", "UNKNOWN")
        pnl = position.get("unrealized_pnl", 0)
        
        logger.debug(f"Position update: {symbol} P&L: ${pnl:.2f}")
        
        # Check for significant P&L change
        if abs(pnl) > self.config.pnl_alert_threshold:
            alert = TradingAlert(
                alert_type="pnl_alert",
                priority=AlertPriority.HIGH if abs(pnl) > self.config.pnl_alert_threshold * 2 else AlertPriority.MEDIUM,
                title=f"P&L Alert: {symbol}",
                message=f"Position P&L: ${pnl:+,.2f}",
                symbol=symbol,
                data=position,
                source="portfolio",
            )
            self._dispatch_alert(alert)
        
        # Update state cache
        self._update_positions_cache(position)
        
        # Broadcast position update (only if event loop is running)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._broadcast_position(position))
        except RuntimeError:
            pass  # No event loop running
    
    def _handle_lifecycle_action(self, action: Dict[str, Any]):
        """Handle position lifecycle action."""
        logger.info(f"Lifecycle action: {action.get('symbol')} - {action.get('action')}")
        
        alert = TradingAlert(
            alert_type="lifecycle",
            priority=AlertPriority.MEDIUM,
            title=f"Position Action: {action.get('symbol')}",
            message=f"{action.get('action', 'N/A')}: {action.get('reason', 'No reason')}",
            symbol=action.get("symbol"),
            data=action,
            source="lifecycle_manager",
        )
        
        self._dispatch_alert(alert)
    
    # =========================================================================
    # Anomaly Detection Integration
    # =========================================================================
    
    async def process_market_data(self, data_point: Dict[str, Any]):
        """Process market data through anomaly detector.
        
        Args:
            data_point: Market data point with symbol, price, volume, etc.
        """
        if not self.anomaly_detector:
            return
        
        try:
            from utils.anomaly_detector import MarketDataPoint
            
            point = MarketDataPoint(
                symbol=data_point.get("symbol", "UNKNOWN"),
                timestamp=datetime.fromisoformat(data_point.get("timestamp", datetime.utcnow().isoformat())),
                price=float(data_point.get("price", 0)),
                volume=int(data_point.get("volume", 0)),
                bid=float(data_point.get("bid", 0)),
                ask=float(data_point.get("ask", 0)),
            )
            
            alerts = self.anomaly_detector.process_data(point)
            
            for anomaly_alert in alerts:
                self._metrics["anomalies_detected"] += 1
                
                alert = TradingAlert(
                    alert_type="anomaly",
                    priority=AlertPriority.HIGH,
                    title=f"Anomaly Detected: {anomaly_alert.symbol}",
                    message=f"{anomaly_alert.anomaly_type.value}: {anomaly_alert.description}",
                    symbol=anomaly_alert.symbol,
                    data={
                        "anomaly_type": anomaly_alert.anomaly_type.value,
                        "severity": anomaly_alert.severity.value,
                        "expected_value": anomaly_alert.expected_value,
                        "current_value": anomaly_alert.current_value,
                    },
                    source="anomaly_detector",
                )
                
                self._dispatch_alert(alert)
                
                # Broadcast anomaly alert
                await self._broadcast_alert(alert)
                
        except Exception as e:
            logger.error(f"Error processing market data for anomalies: {e}")
    
    # =========================================================================
    # Options Flow Integration
    # =========================================================================
    
    async def process_options_flow(self, trade_data: Dict[str, Any]):
        """Process options trade through flow scanner.
        
        Args:
            trade_data: Options trade data
        """
        if not self.flow_scanner:
            return
        
        try:
            from scanner.options_flow_scanner import OptionTrade
            
            trade = OptionTrade(
                symbol=trade_data.get("symbol", ""),
                underlying=trade_data.get("underlying", ""),
                strike=float(trade_data.get("strike", 0)),
                expiration=datetime.fromisoformat(trade_data.get("expiration", datetime.utcnow().isoformat())),
                option_type=trade_data.get("option_type", "call"),
                price=float(trade_data.get("price", 0)),
                size=int(trade_data.get("size", 0)),
                premium=float(trade_data.get("premium", 0)),
                bid=float(trade_data.get("bid", 0)),
                ask=float(trade_data.get("ask", 0)),
                underlying_price=float(trade_data.get("underlying_price", 0)),
                timestamp=datetime.fromisoformat(trade_data.get("timestamp", datetime.utcnow().isoformat())),
                exchange=trade_data.get("exchange", "UNKNOWN"),
                trade_id=trade_data.get("trade_id", ""),
            )
            
            flow_alert = await self.flow_scanner.process_trade(trade)
            
            if flow_alert:
                self._metrics["flow_signals"] += 1
                
                alert = TradingAlert(
                    alert_type="flow",
                    priority=AlertPriority.HIGH if flow_alert.score >= 80 else AlertPriority.MEDIUM,
                    title=f"Options Flow: {flow_alert.underlying}",
                    message=f"{flow_alert.flow_type.value.upper()}: {flow_alert.total_contracts} contracts, ${flow_alert.total_premium:,.0f} premium",
                    symbol=flow_alert.underlying,
                    data={
                        "flow_type": flow_alert.flow_type.value,
                        "sentiment": flow_alert.sentiment.value,
                        "score": flow_alert.score,
                        "contracts": flow_alert.total_contracts,
                        "premium": flow_alert.total_premium,
                    },
                    source="flow_scanner",
                )
                
                self._dispatch_alert(alert)
                
                # Broadcast flow alert
                await self._broadcast_flow_alert(flow_alert)
                
        except Exception as e:
            logger.error(f"Error processing options flow: {e}")
    
    # =========================================================================
    # Greeks Hedging Integration
    # =========================================================================
    
    async def update_portfolio_greeks(self, greeks: Dict[str, Any]):
        """Update portfolio Greeks and check for hedging needs.
        
        Args:
            greeks: Portfolio Greeks data
        """
        self._greeks_state = greeks
        
        if not self.greeks_hedger:
            return
        
        try:
            from trade.greeks_hedger import GreekExposure
            
            exposure = GreekExposure(
                delta=float(greeks.get("delta", 0)),
                gamma=float(greeks.get("gamma", 0)),
                theta=float(greeks.get("theta", 0)),
                vega=float(greeks.get("vega", 0)),
            )
            
            recommendations = self.greeks_hedger.update_exposure(exposure)
            
            for rec in recommendations:
                alert = TradingAlert(
                    alert_type="hedge_recommendation",
                    priority=AlertPriority.MEDIUM,
                    title=f"Hedge Recommendation: {rec.hedge_type.value}",
                    message=f"Target adjustment: {rec.target_adjustment:.2f}",
                    data={
                        "hedge_type": rec.hedge_type.value,
                        "current_exposure": rec.current_exposure,
                        "target_exposure": rec.target_exposure,
                        "target_adjustment": rec.target_adjustment,
                        "instruments": rec.recommended_instruments,
                    },
                    source="greeks_hedger",
                )
                
                self._dispatch_alert(alert)
                
                # Auto-execute hedge if enabled
                if self.config.auto_hedge_enabled:
                    await self._execute_hedge(rec)
                    
        except Exception as e:
            logger.error(f"Error updating portfolio Greeks: {e}")
    
    async def _execute_hedge(self, recommendation):
        """Execute hedge recommendation if auto-hedge is enabled."""
        logger.info(f"Auto-executing hedge: {recommendation.hedge_type.value}")
        # TODO: Implement auto-hedge execution through broker
        pass
    
    # =========================================================================
    # Alert Distribution
    # =========================================================================
    
    def register_alert_handler(self, handler: Callable[[TradingAlert], None]):
        """Register an alert handler.
        
        Args:
            handler: Callback function to receive alerts
        """
        self._alert_handlers.append(handler)
        logger.info(f"Alert handler registered: {handler.__name__}")
    
    def _dispatch_alert(self, alert: TradingAlert):
        """Dispatch alert to all handlers and notification channels.
        
        Args:
            alert: Alert to dispatch
        """
        self._metrics["alerts_sent"] += 1
        
        # Call registered handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        # Send to notification bot
        if self.config.enable_notifications and self.notification_bot:
            asyncio.create_task(self._send_notification(alert))
    
    async def _send_notification(self, alert: TradingAlert):
        """Send alert via notification bot.
        
        Args:
            alert: Alert to send
        """
        if not self.notification_bot:
            return
        
        try:
            # Only send notifications for configured alert types
            should_notify = (
                (alert.alert_type == "signal" and self.config.notify_on_signal) or
                (alert.alert_type == "order" and self.config.notify_on_trade) or
                (alert.alert_type == "anomaly" and self.config.notify_on_anomaly) or
                (alert.alert_type == "flow" and self.config.notify_on_flow) or
                (alert.priority == AlertPriority.CRITICAL)
            )
            
            if should_notify:
                from notifications.telegram_bot import AlertMessage, AlertPriority as BotPriority
                
                priority_map = {
                    AlertPriority.LOW: BotPriority.LOW,
                    AlertPriority.MEDIUM: BotPriority.MEDIUM,
                    AlertPriority.HIGH: BotPriority.HIGH,
                    AlertPriority.CRITICAL: BotPriority.CRITICAL,
                }
                
                bot_alert = AlertMessage(
                    priority=priority_map.get(alert.priority, BotPriority.MEDIUM),
                    title=alert.title,
                    message=alert.message,
                    symbol=alert.symbol,
                    data=alert.data,
                )
                
                await self.notification_bot.send_alert(bot_alert)
                
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    # =========================================================================
    # WebSocket Broadcasting
    # =========================================================================
    
    async def _broadcast_portfolio(self):
        """Broadcast portfolio update to WebSocket subscribers."""
        if not self.websocket_manager:
            return
        
        try:
            from routers.websocket_api import ChannelType
            
            await self.websocket_manager.broadcast_to_channel(
                ChannelType.PORTFOLIO,
                self._portfolio_state,
            )
            self._metrics["broadcasts_sent"] += 1
            
        except Exception as e:
            logger.error(f"Error broadcasting portfolio: {e}")
    
    async def _broadcast_positions(self):
        """Broadcast positions update to WebSocket subscribers."""
        if not self.websocket_manager:
            return
        
        try:
            from routers.websocket_api import ChannelType
            
            await self.websocket_manager.broadcast_to_channel(
                ChannelType.POSITIONS,
                {"positions": self._positions_state},
            )
            self._metrics["broadcasts_sent"] += 1
            
        except Exception as e:
            logger.error(f"Error broadcasting positions: {e}")
    
    async def _broadcast_greeks(self):
        """Broadcast Greeks update to WebSocket subscribers."""
        if not self.websocket_manager:
            return
        
        try:
            from routers.websocket_api import ChannelType
            
            await self.websocket_manager.broadcast_to_channel(
                ChannelType.GREEKS,
                self._greeks_state,
            )
            self._metrics["broadcasts_sent"] += 1
            
        except Exception as e:
            logger.error(f"Error broadcasting Greeks: {e}")
    
    async def _broadcast_signal(self, signal: Dict[str, Any]):
        """Broadcast trading signal."""
        if not self.websocket_manager:
            return
        
        try:
            from routers.websocket_api import ChannelType
            
            await self.websocket_manager.broadcast_to_channel(
                ChannelType.SIGNALS,
                signal,
            )
            
        except Exception as e:
            logger.error(f"Error broadcasting signal: {e}")
    
    async def _broadcast_order(self, order: Dict[str, Any]):
        """Broadcast order update."""
        if not self.websocket_manager:
            return
        
        try:
            from routers.websocket_api import ChannelType
            
            await self.websocket_manager.broadcast_to_channel(
                ChannelType.ORDERS,
                order,
            )
            
        except Exception as e:
            logger.error(f"Error broadcasting order: {e}")
    
    async def _broadcast_position(self, position: Dict[str, Any]):
        """Broadcast single position update."""
        if not self.websocket_manager:
            return
        
        try:
            from routers.websocket_api import ChannelType
            
            await self.websocket_manager.broadcast_to_channel(
                ChannelType.POSITIONS,
                {"position": position, "type": "update"},
            )
            
        except Exception as e:
            logger.error(f"Error broadcasting position: {e}")
    
    async def _broadcast_alert(self, alert: TradingAlert):
        """Broadcast alert to WebSocket subscribers."""
        if not self.websocket_manager:
            return
        
        try:
            from routers.websocket_api import ChannelType
            
            await self.websocket_manager.broadcast_to_channel(
                ChannelType.ALERTS,
                alert.to_dict(),
            )
            
        except Exception as e:
            logger.error(f"Error broadcasting alert: {e}")
    
    async def _broadcast_flow_alert(self, flow_alert):
        """Broadcast flow scanner alert."""
        if not self.websocket_manager:
            return
        
        try:
            from routers.websocket_api import ChannelType
            
            await self.websocket_manager.broadcast_to_channel(
                ChannelType.SIGNALS,
                {
                    "type": "flow",
                    "underlying": flow_alert.underlying,
                    "flow_type": flow_alert.flow_type.value,
                    "sentiment": flow_alert.sentiment.value,
                    "score": flow_alert.score,
                    "contracts": flow_alert.total_contracts,
                    "premium": flow_alert.total_premium,
                },
            )
            
        except Exception as e:
            logger.error(f"Error broadcasting flow alert: {e}")
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    def update_portfolio_state(self, state: Dict[str, Any]):
        """Update cached portfolio state.
        
        Args:
            state: Portfolio state data
        """
        self._portfolio_state = state
        
        # Update portfolio analytics if enabled
        if self.portfolio_analytics:
            try:
                self.portfolio_analytics.update_portfolio_snapshot(
                    timestamp=datetime.utcnow(),
                    total_value=state.get("total_value", 0),
                    cash=state.get("cash", 0),
                    positions=self._positions_state,
                )
            except Exception as e:
                logger.error(f"Error updating portfolio analytics: {e}")
    
    def _update_positions_cache(self, position: Dict[str, Any]):
        """Update positions cache with new position data.
        
        Args:
            position: Position data
        """
        symbol = position.get("symbol")
        if not symbol:
            return
        
        # Find and update existing position or add new
        for i, p in enumerate(self._positions_state):
            if p.get("symbol") == symbol:
                self._positions_state[i] = position
                return
        
        self._positions_state.append(position)
    
    # =========================================================================
    # Background Tasks
    # =========================================================================
    
    async def _portfolio_broadcast_loop(self):
        """Background task to periodically broadcast portfolio updates."""
        while self._running:
            try:
                await self._broadcast_portfolio()
                await asyncio.sleep(self.config.portfolio_broadcast_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Portfolio broadcast error: {e}")
                await asyncio.sleep(5)
    
    async def _positions_broadcast_loop(self):
        """Background task to periodically broadcast positions updates."""
        while self._running:
            try:
                await self._broadcast_positions()
                await asyncio.sleep(self.config.positions_broadcast_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Positions broadcast error: {e}")
                await asyncio.sleep(5)
    
    async def _greeks_broadcast_loop(self):
        """Background task to periodically broadcast Greeks updates."""
        while self._running:
            try:
                await self._broadcast_greeks()
                await asyncio.sleep(self.config.greeks_broadcast_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Greeks broadcast error: {e}")
                await asyncio.sleep(5)
    
    # =========================================================================
    # Lifecycle Management
    # =========================================================================
    
    async def start(self):
        """Start the trading hub and all components."""
        if self.state == HubState.RUNNING:
            logger.warning("Trading Hub is already running")
            return
        
        self.state = HubState.INITIALIZING
        logger.info("Starting Trading Hub...")
        
        try:
            # Initialize components (lazy loading will trigger)
            _ = self.paper_engine
            _ = self.websocket_manager
            _ = self.anomaly_detector
            _ = self.flow_scanner
            _ = self.notification_bot
            _ = self.greeks_hedger
            _ = self.portfolio_analytics
            
            self._running = True
            
            # Start background tasks
            if self.config.enable_websocket:
                self._tasks.append(asyncio.create_task(self._portfolio_broadcast_loop()))
                self._tasks.append(asyncio.create_task(self._positions_broadcast_loop()))
                self._tasks.append(asyncio.create_task(self._greeks_broadcast_loop()))
            
            self.state = HubState.RUNNING
            logger.info("Trading Hub started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Trading Hub: {e}")
            self.state = HubState.ERROR
            raise
    
    async def stop(self):
        """Stop the trading hub and all components."""
        logger.info("Stopping Trading Hub...")
        
        self._running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        
        self.state = HubState.STOPPED
        logger.info("Trading Hub stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get hub status and metrics.
        
        Returns:
            Status dictionary
        """
        return {
            "state": self.state.value,
            "components": {
                "paper_engine": self._paper_engine is not None,
                "websocket_manager": self._websocket_manager is not None,
                "anomaly_detector": self._anomaly_detector is not None,
                "flow_scanner": self._flow_scanner is not None,
                "notification_bot": self._notification_bot is not None,
                "greeks_hedger": self._greeks_hedger is not None,
                "portfolio_analytics": self._portfolio_analytics is not None,
            },
            "metrics": self._metrics,
            "config": {
                "enable_paper_trading": self.config.enable_paper_trading,
                "enable_websocket": self.config.enable_websocket,
                "enable_notifications": self.config.enable_notifications,
                "auto_hedge_enabled": self.config.auto_hedge_enabled,
            },
        }


# =============================================================================
# Singleton Instance
# =============================================================================

# Global trading hub instance
_trading_hub: Optional[TradingHub] = None


def get_trading_hub(config: Optional[HubConfig] = None) -> TradingHub:
    """Get or create the global trading hub instance.
    
    Args:
        config: Optional configuration (only used on first call)
        
    Returns:
        TradingHub instance
    """
    global _trading_hub
    
    if _trading_hub is None:
        _trading_hub = TradingHub(config)
    
    return _trading_hub


async def start_trading_hub(config: Optional[HubConfig] = None) -> TradingHub:
    """Start the global trading hub.
    
    Args:
        config: Optional configuration
        
    Returns:
        Started TradingHub instance
    """
    hub = get_trading_hub(config)
    await hub.start()
    return hub


async def stop_trading_hub():
    """Stop the global trading hub."""
    global _trading_hub
    
    if _trading_hub is not None:
        await _trading_hub.stop()
