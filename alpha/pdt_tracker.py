"""
PDT (Pattern Day Trader) Tracker

Tracks day trades to ensure compliance with FINRA PDT rules:
- Accounts under $25,000 are limited to 3 day trades per 5 rolling business days
- A day trade = buying and selling the same security on the same day
- PDT violations can result in account restrictions

This module helps Robinhood/Webull users avoid PDT violations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DayTrade:
    """Record of a single day trade."""
    symbol: str
    trade_date: date
    buy_time: datetime
    sell_time: datetime
    buy_price: float
    sell_price: float
    quantity: int
    profit_loss: float
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "trade_date": self.trade_date.isoformat(),
            "buy_time": self.buy_time.isoformat(),
            "sell_time": self.sell_time.isoformat(),
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
            "quantity": self.quantity,
            "profit_loss": self.profit_loss
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DayTrade":
        return cls(
            symbol=data["symbol"],
            trade_date=date.fromisoformat(data["trade_date"]),
            buy_time=datetime.fromisoformat(data["buy_time"]),
            sell_time=datetime.fromisoformat(data["sell_time"]),
            buy_price=data["buy_price"],
            sell_price=data["sell_price"],
            quantity=data["quantity"],
            profit_loss=data["profit_loss"]
        )


@dataclass
class PDTTracker:
    """
    Pattern Day Trader compliance tracker.
    
    Tracks day trades over a rolling 5-day window and prevents
    violations of PDT rules for accounts under $25,000.
    """
    
    max_day_trades: int = 3
    lookback_days: int = 5
    account_value: float = 0.0  # Current account value
    pdt_threshold: float = 25000.0  # PDT threshold
    day_trades: List[DayTrade] = field(default_factory=list)
    persistence_path: Optional[Path] = None
    
    def __post_init__(self):
        """Load persisted data if available."""
        if self.persistence_path:
            self._load()
    
    @property
    def is_pdt_restricted(self) -> bool:
        """Check if account is under PDT threshold."""
        return self.account_value < self.pdt_threshold
    
    @property
    def trades_in_window(self) -> List[DayTrade]:
        """Get day trades within the rolling lookback window."""
        cutoff = date.today() - timedelta(days=self.lookback_days)
        return [t for t in self.day_trades if t.trade_date >= cutoff]
    
    @property
    def day_trades_used(self) -> int:
        """Number of day trades used in current window."""
        return len(self.trades_in_window)
    
    @property
    def day_trades_remaining(self) -> int:
        """Number of day trades remaining in current window."""
        if not self.is_pdt_restricted:
            return 999  # Unlimited for accounts over threshold
        return max(0, self.max_day_trades - self.day_trades_used)
    
    def can_day_trade(self) -> bool:
        """Check if a day trade is currently allowed."""
        if not self.is_pdt_restricted:
            return True
        return self.day_trades_remaining > 0
    
    def can_open_position(self, symbol: str, allow_day_trade: bool = False) -> tuple[bool, str]:
        """
        Check if we can open a position on a symbol.
        
        Args:
            symbol: Stock symbol to trade
            allow_day_trade: If True, allow opening even if it might become a day trade
            
        Returns:
            (can_open, reason) tuple
        """
        if not self.is_pdt_restricted:
            return True, "Account above PDT threshold"
        
        if allow_day_trade:
            if self.can_day_trade():
                return True, f"Day trades remaining: {self.day_trades_remaining}"
            else:
                return False, "No day trades remaining in 5-day window"
        
        # If not allowing day trades, we can always open
        # (we just won't close same day)
        return True, "Position opened for swing trade (no same-day exit)"
    
    def can_close_position(self, symbol: str, open_date: date) -> tuple[bool, str]:
        """
        Check if we can close a position without violating PDT.
        
        Args:
            symbol: Stock symbol
            open_date: Date the position was opened
            
        Returns:
            (can_close, reason) tuple
        """
        if not self.is_pdt_restricted:
            return True, "Account above PDT threshold"
        
        today = date.today()
        
        # If opened on a different day, not a day trade
        if open_date < today:
            return True, "Not a day trade (opened on previous day)"
        
        # Would be a day trade
        if self.can_day_trade():
            return True, f"Day trade allowed ({self.day_trades_remaining} remaining)"
        
        return False, "Would exceed PDT limit - hold until tomorrow"
    
    def record_day_trade(
        self,
        symbol: str,
        buy_time: datetime,
        sell_time: datetime,
        buy_price: float,
        sell_price: float,
        quantity: int
    ) -> DayTrade:
        """Record a completed day trade."""
        trade = DayTrade(
            symbol=symbol,
            trade_date=date.today(),
            buy_time=buy_time,
            sell_time=sell_time,
            buy_price=buy_price,
            sell_price=sell_price,
            quantity=quantity,
            profit_loss=(sell_price - buy_price) * quantity
        )
        
        self.day_trades.append(trade)
        logger.info(
            f"Day trade recorded: {symbol} | P/L: ${trade.profit_loss:.2f} | "
            f"Day trades used: {self.day_trades_used}/{self.max_day_trades}"
        )
        
        # Persist if configured
        if self.persistence_path:
            self._save()
        
        return trade
    
    def update_account_value(self, value: float) -> None:
        """Update account value for PDT threshold check."""
        old_restricted = self.is_pdt_restricted
        self.account_value = value
        new_restricted = self.is_pdt_restricted
        
        if old_restricted and not new_restricted:
            logger.info(f"Account now above PDT threshold (${value:,.2f})")
        elif not old_restricted and new_restricted:
            logger.warning(f"Account now below PDT threshold (${value:,.2f})")
    
    def get_status(self) -> dict:
        """Get current PDT status summary."""
        return {
            "account_value": self.account_value,
            "pdt_threshold": self.pdt_threshold,
            "is_pdt_restricted": self.is_pdt_restricted,
            "day_trades_used": self.day_trades_used,
            "day_trades_remaining": self.day_trades_remaining,
            "max_day_trades": self.max_day_trades,
            "lookback_days": self.lookback_days,
            "can_day_trade": self.can_day_trade(),
            "recent_trades": [t.to_dict() for t in self.trades_in_window]
        }
    
    def get_next_available_date(self) -> Optional[date]:
        """Get the date when a day trade will become available again."""
        if self.can_day_trade():
            return date.today()
        
        if not self.trades_in_window:
            return date.today()
        
        # Find oldest trade in window - it will expire first
        oldest = min(self.trades_in_window, key=lambda t: t.trade_date)
        return oldest.trade_date + timedelta(days=self.lookback_days + 1)
    
    def cleanup_old_trades(self) -> int:
        """Remove day trades older than lookback window."""
        cutoff = date.today() - timedelta(days=self.lookback_days * 2)
        original_count = len(self.day_trades)
        self.day_trades = [t for t in self.day_trades if t.trade_date >= cutoff]
        removed = original_count - len(self.day_trades)
        
        if removed > 0:
            logger.debug(f"Cleaned up {removed} old day trade records")
            if self.persistence_path:
                self._save()
        
        return removed
    
    def _save(self) -> None:
        """Save day trades to disk."""
        if not self.persistence_path:
            return
        
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "account_value": self.account_value,
            "day_trades": [t.to_dict() for t in self.day_trades]
        }
        self.persistence_path.write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        """Load day trades from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        
        try:
            data = json.loads(self.persistence_path.read_text())
            self.account_value = data.get("account_value", 0.0)
            self.day_trades = [
                DayTrade.from_dict(t) for t in data.get("day_trades", [])
            ]
            logger.info(f"Loaded {len(self.day_trades)} day trade records")
        except Exception as e:
            logger.warning(f"Failed to load PDT data: {e}")
