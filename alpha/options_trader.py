"""
Gnosis Alpha - Options Trading Execution

Simple options trading for Robinhood-style retail traders.
Executes single-leg options orders via Alpaca.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
    )
    from alpaca.trading.enums import (
        OrderSide,
        TimeInForce,
        AssetClass,
        OrderType,
    )
    ALPACA_AVAILABLE = True
except ImportError:
    logger.warning("Alpaca trading SDK not available")
    ALPACA_AVAILABLE = False

from alpha.alpha_config import AlphaConfig
from alpha.options_signal import OptionsSignal, OptionStrategy, OptionContract


@dataclass
class OptionsOrder:
    """An options order result."""
    order_id: str
    symbol: str
    contract_symbol: str
    side: str
    quantity: int
    order_type: str
    status: str
    submitted_at: datetime
    strategy: str
    filled_price: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "contract_symbol": self.contract_symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "status": self.status,
            "strategy": self.strategy,
            "submitted_at": self.submitted_at.isoformat(),
            "filled_price": self.filled_price,
        }


@dataclass 
class OptionsPosition:
    """An options position."""
    symbol: str
    contract_symbol: str
    quantity: int
    side: str  # "long" or "short"
    avg_entry_price: float
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    market_value: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "contract_symbol": self.contract_symbol,
            "quantity": self.quantity,
            "side": self.side,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "market_value": self.market_value,
        }


class OptionsTrader:
    """
    Options trading execution for Gnosis Alpha.
    
    Handles:
    - Long calls and puts (buy to open)
    - Closing positions (sell to close)
    - Position tracking
    
    Does NOT handle multi-leg strategies (spreads, etc.)
    """
    
    def __init__(
        self,
        config: Optional[AlphaConfig] = None,
        paper: bool = True,
    ):
        self.config = config or AlphaConfig.from_env()
        self.paper = paper
        self.trading_client: Optional[TradingClient] = None
        
        if ALPACA_AVAILABLE:
            self._init_client()
    
    def _init_client(self) -> None:
        """Initialize Alpaca trading client."""
        try:
            self.trading_client = TradingClient(
                api_key=self.config.alpaca_api_key,
                secret_key=self.config.alpaca_secret_key,
                paper=self.paper,
            )
            logger.info("Alpaca options trading client initialized")
        except Exception as e:
            logger.error(f"Failed to init Alpaca client: {e}")
    
    def execute_signal(
        self,
        signal: OptionsSignal,
        contracts: int = 1,
        use_limit: bool = False,
    ) -> Optional[OptionsOrder]:
        """
        Execute an options signal.
        
        Args:
            signal: The options signal to execute
            contracts: Number of contracts (overrides signal suggestion)
            use_limit: Use limit order at mid price vs market
            
        Returns:
            OptionsOrder if successful
        """
        if not self.trading_client:
            logger.error("Trading client not available")
            return None
        
        if not signal.contracts:
            logger.error("No contract in signal")
            return None
        
        contract = signal.contracts[0]
        
        # Determine order side based on strategy
        if signal.strategy in [OptionStrategy.LONG_CALL, OptionStrategy.LONG_PUT]:
            side = OrderSide.BUY
        elif signal.strategy in [OptionStrategy.COVERED_CALL, OptionStrategy.CASH_SECURED_PUT]:
            side = OrderSide.SELL
        else:
            logger.error(f"Unknown strategy: {signal.strategy}")
            return None
        
        try:
            # Create order request
            if use_limit and contract.mid_price:
                order_request = LimitOrderRequest(
                    symbol=contract.contract_symbol,
                    qty=contracts,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(contract.mid_price, 2),
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=contract.contract_symbol,
                    qty=contracts,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            
            # Submit order
            order = self.trading_client.submit_order(order_request)
            
            return OptionsOrder(
                order_id=str(order.id),
                symbol=signal.symbol,
                contract_symbol=contract.contract_symbol,
                side=side.value,
                quantity=contracts,
                order_type="limit" if use_limit else "market",
                status=str(order.status),
                strategy=signal.strategy.value,
                submitted_at=datetime.now(timezone.utc),
            )
            
        except Exception as e:
            logger.error(f"Failed to execute options order: {e}")
            return None
    
    def buy_call(
        self,
        contract_symbol: str,
        quantity: int = 1,
        limit_price: Optional[float] = None,
    ) -> Optional[OptionsOrder]:
        """
        Buy a call option (buy to open).
        
        Args:
            contract_symbol: The options contract symbol
            quantity: Number of contracts
            limit_price: Optional limit price
            
        Returns:
            OptionsOrder if successful
        """
        return self._place_order(contract_symbol, OrderSide.BUY, quantity, limit_price)
    
    def buy_put(
        self,
        contract_symbol: str,
        quantity: int = 1,
        limit_price: Optional[float] = None,
    ) -> Optional[OptionsOrder]:
        """
        Buy a put option (buy to open).
        """
        return self._place_order(contract_symbol, OrderSide.BUY, quantity, limit_price)
    
    def sell_to_close(
        self,
        contract_symbol: str,
        quantity: int = 1,
        limit_price: Optional[float] = None,
    ) -> Optional[OptionsOrder]:
        """
        Sell an option to close a long position.
        """
        return self._place_order(contract_symbol, OrderSide.SELL, quantity, limit_price)
    
    def _place_order(
        self,
        contract_symbol: str,
        side: OrderSide,
        quantity: int,
        limit_price: Optional[float] = None,
    ) -> Optional[OptionsOrder]:
        """Place a single-leg options order."""
        if not self.trading_client:
            logger.error("Trading client not available")
            return None
        
        try:
            if limit_price:
                order_request = LimitOrderRequest(
                    symbol=contract_symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(limit_price, 2),
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=contract_symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            
            order = self.trading_client.submit_order(order_request)
            
            # Extract underlying symbol from contract
            # Format: AAPL240119C00190000
            underlying = contract_symbol[:4].rstrip('0123456789')
            if not underlying:
                underlying = contract_symbol.split('2')[0]  # Try another approach
            
            return OptionsOrder(
                order_id=str(order.id),
                symbol=underlying,
                contract_symbol=contract_symbol,
                side=side.value,
                quantity=quantity,
                order_type="limit" if limit_price else "market",
                status=str(order.status),
                strategy="manual",
                submitted_at=datetime.now(timezone.utc),
            )
            
        except Exception as e:
            logger.error(f"Failed to place options order: {e}")
            return None
    
    def get_options_positions(self) -> List[OptionsPosition]:
        """Get all open options positions."""
        if not self.trading_client:
            return []
        
        positions = []
        
        try:
            all_positions = self.trading_client.get_all_positions()
            
            for pos in all_positions:
                # Filter for options (asset_class = us_option)
                if hasattr(pos, 'asset_class') and str(pos.asset_class) == 'us_option':
                    positions.append(OptionsPosition(
                        symbol=pos.symbol[:4].rstrip('0123456789'),  # Extract underlying
                        contract_symbol=pos.symbol,
                        quantity=abs(int(pos.qty)),
                        side="long" if int(pos.qty) > 0 else "short",
                        avg_entry_price=float(pos.avg_entry_price),
                        current_price=float(pos.current_price) if pos.current_price else None,
                        unrealized_pnl=float(pos.unrealized_pl) if pos.unrealized_pl else 0.0,
                        market_value=float(pos.market_value) if pos.market_value else 0.0,
                    ))
                    
        except Exception as e:
            logger.error(f"Failed to get options positions: {e}")
        
        return positions
    
    def close_position(self, contract_symbol: str) -> Optional[OptionsOrder]:
        """Close an entire options position."""
        positions = self.get_options_positions()
        
        for pos in positions:
            if pos.contract_symbol == contract_symbol:
                if pos.side == "long":
                    return self.sell_to_close(contract_symbol, pos.quantity)
                else:
                    return self.buy_call(contract_symbol, pos.quantity)  # Buy to close short
        
        logger.warning(f"No position found for {contract_symbol}")
        return None
    
    def get_account_buying_power(self) -> float:
        """Get available buying power for options."""
        if not self.trading_client:
            return 0.0
        
        try:
            account = self.trading_client.get_account()
            return float(account.buying_power)
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return 0.0
