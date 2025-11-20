"""Order Executor - Converts trade ideas into actual broker orders."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from schemas.core_schemas import DirectionEnum, OrderResult, TradeIdea


class OrderExecutor:
    """
    Executes trade ideas by placing orders through the broker adapter.
    
    Includes safety checks:
    - Confidence threshold filtering
    - Position size limits
    - Risk management
    """
    
    def __init__(
        self, 
        broker_adapter: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize order executor.
        
        Args:
            broker_adapter: Broker adapter (e.g., AlpacaBrokerAdapter)
            config: Configuration dict with safety parameters
        """
        self.broker = broker_adapter
        self.config = config or {}
        
        # Safety thresholds
        self.min_confidence = self.config.get("min_confidence", 0.7)
        self.max_position_size = self.config.get("max_position_size", 500.0)  # Start small
        self.max_portfolio_exposure = self.config.get("max_portfolio_exposure", 0.1)  # 10% per trade
        
        logger.info(f"OrderExecutor initialized (min_confidence={self.min_confidence}, max_size=${self.max_position_size})")
    
    def execute_ideas(
        self, 
        trade_ideas: List[TradeIdea],
        current_positions: Optional[List] = None
    ) -> List[OrderResult]:
        """
        Execute a list of trade ideas.
        
        Args:
            trade_ideas: List of trade ideas to execute
            current_positions: Current broker positions (for risk checks)
            
        Returns:
            List of order results
        """
        if not trade_ideas:
            return []
        
        results = []
        
        for idea in trade_ideas:
            try:
                # Safety check: confidence threshold
                if idea.confidence < self.min_confidence:
                    logger.debug(
                        f"Skipping {idea.symbol}: confidence {idea.confidence:.2f} < {self.min_confidence}"
                    )
                    continue
                
                # Safety check: neutral direction
                if idea.direction == DirectionEnum.NEUTRAL:
                    logger.debug(f"Skipping {idea.symbol}: neutral direction")
                    continue
                
                # Convert trade idea to order
                order_result = self._execute_single_idea(idea, current_positions)
                
                if order_result:
                    results.append(order_result)
                    logger.info(
                        f"✅ Order executed: {idea.symbol} {order_result.side} "
                        f"{order_result.quantity} @ {order_result.fill_price or 'market'}"
                    )
            
            except Exception as e:
                logger.error(f"Error executing trade idea for {idea.symbol}: {e}")
                # Create failed order result
                results.append(OrderResult(
                    timestamp=datetime.now(),
                    symbol=idea.symbol,
                    side="",
                    quantity=0,
                    order_type="",
                    status="failed",
                    error_message=str(e)
                ))
        
        return results
    
    def _execute_single_idea(
        self, 
        idea: TradeIdea,
        current_positions: Optional[List]
    ) -> Optional[OrderResult]:
        """
        Execute a single trade idea.
        
        Args:
            idea: Trade idea to execute
            current_positions: Current positions
            
        Returns:
            OrderResult or None if skipped
        """
        # Determine order side
        if idea.direction == DirectionEnum.LONG:
            side = "buy"
        elif idea.direction == DirectionEnum.SHORT:
            side = "sell"
        else:
            return None
        
        # Calculate quantity
        quantity = self._calculate_quantity(idea)
        
        if quantity <= 0:
            logger.debug(f"Skipping {idea.symbol}: quantity {quantity} <= 0")
            return None
        
        # Check if we already have a position (avoid doubling up)
        if self._has_position(idea.symbol, current_positions):
            logger.debug(f"Skipping {idea.symbol}: already have position")
            return None
        
        # Place order through broker
        try:
            broker_order = self.broker.place_order(
                symbol=idea.symbol,
                qty=quantity,
                side=side,
                order_type="market",  # Start with market orders for simplicity
                time_in_force="day"
            )
            
            # Convert broker response to OrderResult
            order_result = OrderResult(
                timestamp=datetime.now(),
                symbol=idea.symbol,
                side=side,
                quantity=quantity,
                order_type="market",
                status=broker_order.get("status", "submitted"),
                order_id=broker_order.get("id"),
                fill_price=broker_order.get("filled_avg_price"),
                commission=broker_order.get("commission"),
            )
            
            return order_result
        
        except Exception as e:
            logger.error(f"Broker error placing order for {idea.symbol}: {e}")
            raise
    
    def _calculate_quantity(self, idea: TradeIdea) -> int:
        """
        Calculate order quantity with safety checks.
        
        Args:
            idea: Trade idea
            
        Returns:
            Quantity (number of shares)
        """
        # Start with idea's suggested size
        dollar_size = min(idea.size, self.max_position_size)
        
        # Get current price (simplified - in production would get real quote)
        # For now, assume $100/share as rough estimate
        estimated_price = 100.0
        
        # Calculate shares
        quantity = int(dollar_size / estimated_price)
        
        # Ensure at least 1 share
        quantity = max(1, quantity)
        
        # Cap at reasonable max (avoid huge orders)
        max_shares = int(self.max_position_size / 10)  # Max 50 shares if max_position_size=$500
        quantity = min(quantity, max_shares)
        
        return quantity
    
    def _has_position(self, symbol: str, current_positions: Optional[List]) -> bool:
        """
        Check if we already have a position in this symbol.
        
        Args:
            symbol: Symbol to check
            current_positions: Current positions from broker
            
        Returns:
            True if position exists
        """
        if not current_positions:
            return False
        
        for pos in current_positions:
            if hasattr(pos, 'symbol') and pos.symbol == symbol:
                return True
        
        return False
    
    def check_daily_loss_limit(self, account) -> bool:
        """
        Check if daily loss limit has been hit.
        
        Args:
            account: Broker account object
            
        Returns:
            True if trading should continue, False if limit hit
        """
        daily_loss_limit = self.config.get("daily_loss_limit", 100.0)
        
        # Check if account has meaningful loss
        if hasattr(account, 'equity') and hasattr(account, 'last_equity'):
            daily_pnl = float(account.equity) - float(account.last_equity)
            
            if daily_pnl < -daily_loss_limit:
                logger.error(f"🛑 Daily loss limit hit: ${daily_pnl:.2f} < -${daily_loss_limit}")
                return False
        
        return True
