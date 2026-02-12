"""
Telegram/Discord Bot for Trading Alerts

Chat-based interface for:
- Real-time trading alerts
- Portfolio status queries
- Position management
- Order execution
- Market data queries

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from abc import ABC, abstractmethod

from loguru import logger


class Platform(str, Enum):
    """Supported messaging platforms."""
    TELEGRAM = "telegram"
    DISCORD = "discord"


class AlertPriority(str, Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CommandType(str, Enum):
    """Bot command types."""
    PORTFOLIO = "portfolio"
    POSITIONS = "positions"
    GREEKS = "greeks"
    ORDERS = "orders"
    ALERTS = "alerts"
    QUOTE = "quote"
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    HELP = "help"
    STATUS = "status"
    WATCHLIST = "watchlist"
    FLOW = "flow"
    EARNINGS = "earnings"


@dataclass
class BotUser:
    """Bot user information."""
    user_id: str
    platform: Platform
    username: Optional[str] = None
    is_authorized: bool = False
    is_admin: bool = False
    subscriptions: List[str] = field(default_factory=list)
    last_activity: Optional[datetime] = None


@dataclass
class BotMessage:
    """Bot message structure."""
    platform: Platform
    user_id: str
    chat_id: str
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    is_command: bool = False
    command: Optional[CommandType] = None
    args: List[str] = field(default_factory=list)


@dataclass
class BotResponse:
    """Bot response structure."""
    text: str
    parse_mode: str = "Markdown"  # or "HTML"
    disable_preview: bool = True
    reply_markup: Optional[Dict] = None  # Keyboard markup


@dataclass
class AlertMessage:
    """Alert to be sent."""
    priority: AlertPriority
    title: str
    message: str
    symbol: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def format_telegram(self) -> str:
        """Format for Telegram."""
        emoji = {
            AlertPriority.LOW: "ℹ️",
            AlertPriority.MEDIUM: "⚠️",
            AlertPriority.HIGH: "🔔",
            AlertPriority.CRITICAL: "🚨",
        }
        
        msg = f"{emoji[self.priority]} *{self.title}*\n\n"
        msg += f"{self.message}\n"
        
        if self.symbol:
            msg += f"\n📊 Symbol: `{self.symbol}`"
        
        msg += f"\n\n🕐 {self.timestamp.strftime('%H:%M:%S')}"
        
        return msg
    
    def format_discord(self) -> Dict[str, Any]:
        """Format for Discord embed."""
        colors = {
            AlertPriority.LOW: 0x3498db,    # Blue
            AlertPriority.MEDIUM: 0xf39c12,  # Yellow
            AlertPriority.HIGH: 0xe74c3c,    # Red
            AlertPriority.CRITICAL: 0x9b59b6, # Purple
        }
        
        return {
            "title": self.title,
            "description": self.message,
            "color": colors[self.priority],
            "fields": [
                {"name": "Symbol", "value": self.symbol or "N/A", "inline": True},
                {"name": "Priority", "value": self.priority.value, "inline": True},
            ],
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BotConfig:
    """Bot configuration."""
    # Telegram
    telegram_token: Optional[str] = None
    telegram_webhook_url: Optional[str] = None
    
    # Discord
    discord_token: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    
    # Security
    authorized_users: List[str] = field(default_factory=list)
    admin_users: List[str] = field(default_factory=list)
    
    # Features
    allow_trading: bool = False
    max_order_value: float = 10000
    require_confirmation: bool = True
    
    # Rate limiting
    max_messages_per_minute: int = 30
    alert_cooldown_seconds: int = 60


class CommandHandler(ABC):
    """Base class for command handlers."""
    
    @abstractmethod
    async def handle(self, message: BotMessage, bot: "TradingBot") -> BotResponse:
        """Handle command."""
        pass


class PortfolioCommandHandler(CommandHandler):
    """Handler for /portfolio command."""
    
    async def handle(self, message: BotMessage, bot: "TradingBot") -> BotResponse:
        """Get portfolio summary."""
        data = bot.get_portfolio_data()
        
        text = "📊 *Portfolio Summary*\n\n"
        text += f"💰 Total Value: `${data.get('total_value', 0):,.2f}`\n"
        text += f"💵 Cash: `${data.get('cash', 0):,.2f}`\n"
        text += f"📈 Buying Power: `${data.get('buying_power', 0):,.2f}`\n\n"
        text += f"📅 Day P&L: `${data.get('day_pnl', 0):,.2f}` ({data.get('day_pnl_pct', 0):.2f}%)\n"
        text += f"📊 Total P&L: `${data.get('total_pnl', 0):,.2f}`\n\n"
        text += f"📋 Positions: {data.get('positions_count', 0)}"
        
        return BotResponse(text=text)


class PositionsCommandHandler(CommandHandler):
    """Handler for /positions command."""
    
    async def handle(self, message: BotMessage, bot: "TradingBot") -> BotResponse:
        """Get positions list."""
        positions = bot.get_positions_data()
        
        if not positions:
            return BotResponse(text="📋 No open positions")
        
        text = "📋 *Open Positions*\n\n"
        
        for pos in positions[:10]:  # Limit to 10
            symbol = pos.get('symbol', 'N/A')
            qty = pos.get('quantity', 0)
            pnl = pos.get('unrealized_pnl', 0)
            pnl_pct = pos.get('unrealized_pnl_pct', 0)
            
            emoji = "🟢" if pnl >= 0 else "🔴"
            text += f"{emoji} `{symbol}`: {qty} @ ${pnl:,.2f} ({pnl_pct:.1f}%)\n"
        
        if len(positions) > 10:
            text += f"\n... and {len(positions) - 10} more"
        
        return BotResponse(text=text)


class GreeksCommandHandler(CommandHandler):
    """Handler for /greeks command."""
    
    async def handle(self, message: BotMessage, bot: "TradingBot") -> BotResponse:
        """Get Greeks exposure."""
        greeks = bot.get_greeks_data()
        
        text = "🔢 *Greeks Exposure*\n\n"
        text += f"Δ Delta: `{greeks.get('delta', 0):.2f}`\n"
        text += f"Γ Gamma: `{greeks.get('gamma', 0):.4f}`\n"
        text += f"Θ Theta: `{greeks.get('theta', 0):.2f}`\n"
        text += f"ν Vega: `{greeks.get('vega', 0):.2f}`\n"
        text += f"ρ Rho: `{greeks.get('rho', 0):.2f}`\n\n"
        text += f"β-Δ Beta-Weighted: `{greeks.get('beta_weighted_delta', 0):.2f}`"
        
        return BotResponse(text=text)


class QuoteCommandHandler(CommandHandler):
    """Handler for /quote command."""
    
    async def handle(self, message: BotMessage, bot: "TradingBot") -> BotResponse:
        """Get stock quote."""
        if not message.args:
            return BotResponse(text="Usage: /quote SYMBOL")
        
        symbol = message.args[0].upper()
        quote = bot.get_quote(symbol)
        
        if not quote:
            return BotResponse(text=f"❌ Could not get quote for {symbol}")
        
        change = quote.get('change', 0)
        change_pct = quote.get('change_pct', 0)
        emoji = "🟢" if change >= 0 else "🔴"
        
        text = f"📈 *{symbol}*\n\n"
        text += f"💵 Price: `${quote.get('price', 0):.2f}`\n"
        text += f"{emoji} Change: `${change:.2f}` ({change_pct:.2f}%)\n\n"
        text += f"📊 Bid: `${quote.get('bid', 0):.2f}` | Ask: `${quote.get('ask', 0):.2f}`\n"
        text += f"📊 Volume: `{quote.get('volume', 0):,}`"
        
        return BotResponse(text=text)


class HelpCommandHandler(CommandHandler):
    """Handler for /help command."""
    
    async def handle(self, message: BotMessage, bot: "TradingBot") -> BotResponse:
        """Show help message."""
        text = "🤖 *GNOSIS Trading Bot*\n\n"
        text += "*Available Commands:*\n\n"
        text += "📊 /portfolio - Portfolio summary\n"
        text += "📋 /positions - Open positions\n"
        text += "🔢 /greeks - Greeks exposure\n"
        text += "📈 /quote SYMBOL - Get stock quote\n"
        text += "👀 /watchlist - Your watchlist\n"
        text += "🔔 /alerts - Recent alerts\n"
        text += "📅 /earnings SYMBOL - Earnings info\n"
        text += "💹 /flow - Options flow scanner\n"
        text += "📊 /status - System status\n"
        text += "❓ /help - This message\n"
        
        if bot.config.allow_trading:
            text += "\n*Trading Commands:*\n"
            text += "🟢 /buy SYMBOL QTY - Buy stock\n"
            text += "🔴 /sell SYMBOL QTY - Sell stock\n"
            text += "❌ /close SYMBOL - Close position\n"
        
        return BotResponse(text=text)


class StatusCommandHandler(CommandHandler):
    """Handler for /status command."""
    
    async def handle(self, message: BotMessage, bot: "TradingBot") -> BotResponse:
        """Get system status."""
        status = bot.get_system_status()
        
        text = "🖥️ *System Status*\n\n"
        text += f"✅ Status: `{status.get('status', 'Unknown')}`\n"
        text += f"⏱️ Uptime: `{status.get('uptime', 'N/A')}`\n"
        text += f"📡 Market: `{status.get('market_status', 'Unknown')}`\n"
        text += f"🔌 Connections: `{status.get('connections', 0)}`\n"
        text += f"⚡ Latency: `{status.get('latency_ms', 0):.0f}ms`"
        
        return BotResponse(text=text)


class TradingBot:
    """
    Multi-platform trading bot.
    
    Features:
    - Telegram and Discord support
    - Real-time alerts
    - Portfolio queries
    - Order execution
    - Command handling
    """
    
    def __init__(self, config: Optional[BotConfig] = None):
        """Initialize trading bot."""
        self.config = config or BotConfig()
        
        # Users
        self._users: Dict[str, BotUser] = {}
        
        # Command handlers
        self._handlers: Dict[CommandType, CommandHandler] = {
            CommandType.PORTFOLIO: PortfolioCommandHandler(),
            CommandType.POSITIONS: PositionsCommandHandler(),
            CommandType.GREEKS: GreeksCommandHandler(),
            CommandType.QUOTE: QuoteCommandHandler(),
            CommandType.HELP: HelpCommandHandler(),
            CommandType.STATUS: StatusCommandHandler(),
        }
        
        # Alert subscribers
        self._alert_subscribers: Dict[str, List[str]] = {}  # category -> [user_ids]
        
        # Rate limiting
        self._message_counts: Dict[str, List[datetime]] = {}
        
        # Data callbacks
        self._portfolio_callback: Optional[Callable] = None
        self._positions_callback: Optional[Callable] = None
        self._greeks_callback: Optional[Callable] = None
        self._quote_callback: Optional[Callable] = None
        self._order_callback: Optional[Callable] = None
        
        # Alert queue
        self._alert_queue: asyncio.Queue = asyncio.Queue()
        
        logger.info("TradingBot initialized")
    
    def set_data_callbacks(
        self,
        portfolio: Optional[Callable] = None,
        positions: Optional[Callable] = None,
        greeks: Optional[Callable] = None,
        quote: Optional[Callable] = None,
        order: Optional[Callable] = None,
    ) -> None:
        """Set data retrieval callbacks."""
        self._portfolio_callback = portfolio
        self._positions_callback = positions
        self._greeks_callback = greeks
        self._quote_callback = quote
        self._order_callback = order
    
    def get_portfolio_data(self) -> Dict[str, Any]:
        """Get portfolio data."""
        if self._portfolio_callback:
            return self._portfolio_callback()
        return {
            "total_value": 100000,
            "cash": 50000,
            "buying_power": 100000,
            "day_pnl": 500,
            "day_pnl_pct": 0.5,
            "total_pnl": 5000,
            "positions_count": 5,
        }
    
    def get_positions_data(self) -> List[Dict[str, Any]]:
        """Get positions data."""
        if self._positions_callback:
            return self._positions_callback()
        return [
            {"symbol": "AAPL", "quantity": 100, "unrealized_pnl": 500, "unrealized_pnl_pct": 2.5},
            {"symbol": "MSFT", "quantity": 50, "unrealized_pnl": -200, "unrealized_pnl_pct": -1.2},
        ]
    
    def get_greeks_data(self) -> Dict[str, float]:
        """Get Greeks data."""
        if self._greeks_callback:
            return self._greeks_callback()
        return {
            "delta": 150.5,
            "gamma": 2.3,
            "theta": -45.2,
            "vega": 320.1,
            "rho": 12.5,
            "beta_weighted_delta": 180.3,
        }
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get stock quote."""
        if self._quote_callback:
            return self._quote_callback(symbol)
        return {
            "symbol": symbol,
            "price": 150.00,
            "bid": 149.95,
            "ask": 150.05,
            "change": 2.50,
            "change_pct": 1.69,
            "volume": 50000000,
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "status": "Operational",
            "uptime": "5d 12h 30m",
            "market_status": "Open",
            "connections": len(self._users),
            "latency_ms": 25,
        }
    
    def parse_message(self, text: str, platform: Platform, user_id: str, chat_id: str) -> BotMessage:
        """Parse incoming message."""
        message = BotMessage(
            platform=platform,
            user_id=user_id,
            chat_id=chat_id,
            text=text,
        )
        
        # Check if command
        if text.startswith('/'):
            message.is_command = True
            parts = text[1:].split()
            
            if parts:
                cmd = parts[0].lower()
                message.args = parts[1:]
                
                # Map to command type
                cmd_map = {
                    'portfolio': CommandType.PORTFOLIO,
                    'p': CommandType.PORTFOLIO,
                    'positions': CommandType.POSITIONS,
                    'pos': CommandType.POSITIONS,
                    'greeks': CommandType.GREEKS,
                    'g': CommandType.GREEKS,
                    'quote': CommandType.QUOTE,
                    'q': CommandType.QUOTE,
                    'orders': CommandType.ORDERS,
                    'alerts': CommandType.ALERTS,
                    'buy': CommandType.BUY,
                    'sell': CommandType.SELL,
                    'close': CommandType.CLOSE,
                    'help': CommandType.HELP,
                    'h': CommandType.HELP,
                    'status': CommandType.STATUS,
                    'watchlist': CommandType.WATCHLIST,
                    'w': CommandType.WATCHLIST,
                    'flow': CommandType.FLOW,
                    'earnings': CommandType.EARNINGS,
                }
                
                message.command = cmd_map.get(cmd)
        
        return message
    
    async def handle_message(self, message: BotMessage) -> Optional[BotResponse]:
        """Handle incoming message."""
        # Check rate limit
        if not self._check_rate_limit(message.user_id):
            return BotResponse(text="⚠️ Rate limit exceeded. Please wait.")
        
        # Check authorization
        user = self._get_or_create_user(message)
        if not user.is_authorized:
            if message.user_id not in self.config.authorized_users:
                return BotResponse(text="⛔ You are not authorized to use this bot.")
            user.is_authorized = True
        
        # Handle command
        if message.is_command and message.command:
            handler = self._handlers.get(message.command)
            if handler:
                return await handler.handle(message, self)
            return BotResponse(text=f"❓ Unknown command. Use /help for available commands.")
        
        return None
    
    def _get_or_create_user(self, message: BotMessage) -> BotUser:
        """Get or create user."""
        key = f"{message.platform.value}:{message.user_id}"
        
        if key not in self._users:
            self._users[key] = BotUser(
                user_id=message.user_id,
                platform=message.platform,
                is_authorized=message.user_id in self.config.authorized_users,
                is_admin=message.user_id in self.config.admin_users,
            )
        
        user = self._users[key]
        user.last_activity = datetime.now()
        return user
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is rate limited."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        if user_id not in self._message_counts:
            self._message_counts[user_id] = []
        
        # Clean old messages
        self._message_counts[user_id] = [
            t for t in self._message_counts[user_id] if t > cutoff
        ]
        
        # Check limit
        if len(self._message_counts[user_id]) >= self.config.max_messages_per_minute:
            return False
        
        self._message_counts[user_id].append(now)
        return True
    
    async def send_alert(self, alert: AlertMessage, user_ids: Optional[List[str]] = None) -> None:
        """Send alert to users."""
        await self._alert_queue.put((alert, user_ids))
    
    async def process_alert_queue(self) -> None:
        """Process alert queue (background task)."""
        while True:
            try:
                alert, user_ids = await self._alert_queue.get()
                
                # Send to all subscribers or specific users
                targets = user_ids or list(self._users.keys())
                
                for target in targets:
                    user = self._users.get(target)
                    if not user:
                        continue
                    
                    # Format message based on platform
                    if user.platform == Platform.TELEGRAM:
                        text = alert.format_telegram()
                        # Would send via Telegram API
                        logger.info(f"Would send Telegram alert to {user.user_id}: {alert.title}")
                    elif user.platform == Platform.DISCORD:
                        embed = alert.format_discord()
                        # Would send via Discord API
                        logger.info(f"Would send Discord alert to {user.user_id}: {alert.title}")
                
                self._alert_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    def subscribe_alerts(self, user_id: str, categories: List[str]) -> None:
        """Subscribe user to alert categories."""
        for category in categories:
            if category not in self._alert_subscribers:
                self._alert_subscribers[category] = []
            if user_id not in self._alert_subscribers[category]:
                self._alert_subscribers[category].append(user_id)
    
    def unsubscribe_alerts(self, user_id: str, categories: Optional[List[str]] = None) -> None:
        """Unsubscribe user from alert categories."""
        if categories is None:
            categories = list(self._alert_subscribers.keys())
        
        for category in categories:
            if category in self._alert_subscribers:
                self._alert_subscribers[category] = [
                    u for u in self._alert_subscribers[category] if u != user_id
                ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bot statistics."""
        return {
            "total_users": len(self._users),
            "authorized_users": sum(1 for u in self._users.values() if u.is_authorized),
            "alert_subscribers": sum(len(s) for s in self._alert_subscribers.values()),
            "alert_queue_size": self._alert_queue.qsize(),
        }


# Telegram-specific webhook handler
class TelegramWebhookHandler:
    """Handle Telegram webhook requests."""
    
    def __init__(self, bot: TradingBot, token: str):
        """Initialize handler."""
        self.bot = bot
        self.token = token
    
    async def handle_update(self, update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle Telegram update."""
        if "message" not in update:
            return None
        
        msg = update["message"]
        text = msg.get("text", "")
        user_id = str(msg["from"]["id"])
        chat_id = str(msg["chat"]["id"])
        
        message = self.bot.parse_message(text, Platform.TELEGRAM, user_id, chat_id)
        response = await self.bot.handle_message(message)
        
        if response:
            return {
                "method": "sendMessage",
                "chat_id": chat_id,
                "text": response.text,
                "parse_mode": response.parse_mode,
                "disable_web_page_preview": response.disable_preview,
            }
        
        return None


# Discord-specific handler
class DiscordHandler:
    """Handle Discord interactions."""
    
    def __init__(self, bot: TradingBot, token: str):
        """Initialize handler."""
        self.bot = bot
        self.token = token
    
    async def handle_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle Discord message."""
        content = data.get("content", "")
        user_id = data.get("author", {}).get("id", "")
        channel_id = data.get("channel_id", "")
        
        message = self.bot.parse_message(content, Platform.DISCORD, user_id, channel_id)
        response = await self.bot.handle_message(message)
        
        if response:
            return {
                "content": response.text,
                "channel_id": channel_id,
            }
        
        return None


# Convenience functions
def create_trading_bot(config: Optional[BotConfig] = None) -> TradingBot:
    """Create trading bot."""
    return TradingBot(config)


def create_alert(
    priority: AlertPriority,
    title: str,
    message: str,
    symbol: Optional[str] = None,
) -> AlertMessage:
    """Create alert message."""
    return AlertMessage(
        priority=priority,
        title=title,
        message=message,
        symbol=symbol,
    )
