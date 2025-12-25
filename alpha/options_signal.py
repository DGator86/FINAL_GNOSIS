"""
Gnosis Alpha - Options Signal Generator

Simple directional options strategies for retail traders.
Supports Robinhood-style options trading:
- Long Calls (bullish)
- Long Puts (bearish)
- Covered Calls (income on existing shares)
- Cash-Secured Puts (buy shares at discount)

Focus: Simple, defined-risk strategies with clear max loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)

# Try to import Alpaca options
try:
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import (
        OptionChainRequest,
        OptionSnapshotRequest,
        OptionLatestQuoteRequest,
    )
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
    )
    from alpaca.trading.enums import (
        OrderSide,
        TimeInForce,
        AssetClass,
    )
    ALPACA_OPTIONS_AVAILABLE = True
except ImportError:
    logger.warning("Alpaca options SDK not available")
    ALPACA_OPTIONS_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class OptionType(str, Enum):
    """Option type."""
    CALL = "call"
    PUT = "put"


class OptionStrategy(str, Enum):
    """Simple options strategies for retail traders."""
    LONG_CALL = "long_call"           # Bullish: Buy call
    LONG_PUT = "long_put"             # Bearish: Buy put
    COVERED_CALL = "covered_call"     # Income: Own shares + sell call
    CASH_SECURED_PUT = "cash_secured_put"  # Buy shares at discount


class OptionSignalDirection(str, Enum):
    """Directional bias for options."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class OptionContract:
    """Represents a single option contract."""
    symbol: str              # Underlying symbol (e.g., "AAPL")
    contract_symbol: str     # Full contract symbol (e.g., "AAPL240119C00190000")
    option_type: OptionType
    strike: float
    expiration: date
    
    # Pricing
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_price: Optional[float] = None
    mid_price: Optional[float] = None
    
    # Greeks (if available)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    iv: Optional[float] = None  # Implied volatility
    
    # Volume/OI
    volume: int = 0
    open_interest: int = 0
    avg_volume: int = 0  # Average daily volume for comparison
    
    @property
    def unusual_volume(self) -> bool:
        """Check if volume is unusual (2x+ average)."""
        if self.avg_volume > 0:
            return self.volume >= self.avg_volume * 2
        return self.volume > 1000  # Fallback threshold
    
    @property
    def volume_ratio(self) -> float:
        """Ratio of current volume to average."""
        if self.avg_volume > 0:
            return self.volume / self.avg_volume
        return 0.0
    
    @property
    def days_to_expiration(self) -> int:
        """Days until expiration."""
        return (self.expiration - date.today()).days
    
    @property
    def is_itm(self) -> bool:
        """Check if in-the-money (requires underlying price)."""
        return False  # Override in signal
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "contract_symbol": self.contract_symbol,
            "option_type": self.option_type.value,
            "strike": self.strike,
            "expiration": self.expiration.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "mid_price": self.mid_price,
            "delta": self.delta,
            "theta": self.theta,
            "iv": self.iv,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "avg_volume": self.avg_volume,
            "unusual_volume": self.unusual_volume,
            "volume_ratio": round(self.volume_ratio, 2),
            "days_to_expiration": self.days_to_expiration,
        }


@dataclass
class OptionsSignal:
    """
    A complete options trading signal.
    
    Includes the recommended strategy, contract details,
    risk metrics, and entry/exit guidance.
    """
    # Basic info
    symbol: str
    strategy: OptionStrategy
    direction: OptionSignalDirection
    confidence: float
    
    # Contracts involved
    contracts: List[OptionContract] = field(default_factory=list)
    
    # Underlying price info
    underlying_price: Optional[float] = None
    
    # Entry/Exit
    entry_price: Optional[float] = None  # Per contract cost
    max_loss: Optional[float] = None     # Maximum possible loss
    max_profit: Optional[float] = None   # Maximum possible profit (None = unlimited)
    break_even: Optional[float] = None   # Break-even price at expiration
    
    # Position sizing
    suggested_contracts: int = 1
    total_cost: Optional[float] = None   # Total capital required
    
    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: Optional[datetime] = None
    
    # Reasoning
    reasoning: str = ""
    risk_factors: List[str] = field(default_factory=list)
    
    # Volume indicators
    unusual_volume: bool = False
    volume_ratio: float = 0.0  # Current vs average volume
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy.value,
            "direction": self.direction.value,
            "confidence": round(self.confidence, 3),
            "underlying_price": self.underlying_price,
            "entry_price": self.entry_price,
            "max_loss": self.max_loss,
            "max_profit": self.max_profit,
            "break_even": self.break_even,
            "suggested_contracts": self.suggested_contracts,
            "total_cost": self.total_cost,
            "timestamp": self.timestamp.isoformat(),
            "reasoning": self.reasoning,
            "risk_factors": self.risk_factors,
            "unusual_volume": self.unusual_volume,
            "volume_ratio": round(self.volume_ratio, 2),
            "contracts": [c.to_dict() for c in self.contracts],
        }
    
    def to_robinhood_format(self) -> str:
        """Format signal for Robinhood-style display."""
        direction_emoji = {
            "BULLISH": "🟢",
            "BEARISH": "🔴", 
            "NEUTRAL": "⚪"
        }[self.direction.value]
        
        strategy_name = {
            "long_call": "Buy Call",
            "long_put": "Buy Put",
            "covered_call": "Covered Call",
            "cash_secured_put": "Cash-Secured Put",
        }[self.strategy.value]
        
        lines = [
            f"{direction_emoji} {self.symbol}: {strategy_name}",
            f"Confidence: {self.confidence * 100:.0f}%",
        ]
        
        if self.contracts:
            contract = self.contracts[0]
            lines.append(f"Strike: ${contract.strike:.2f} {contract.option_type.value.upper()}")
            lines.append(f"Expires: {contract.expiration} ({contract.days_to_expiration}d)")
            if contract.mid_price:
                lines.append(f"Price: ${contract.mid_price:.2f}/contract")
        
        if self.underlying_price:
            lines.append(f"Stock: ${self.underlying_price:.2f}")
        
        if self.max_loss:
            lines.append(f"Max Loss: ${self.max_loss:.2f}")
        
        if self.break_even:
            lines.append(f"Break-Even: ${self.break_even:.2f}")
        
        if self.risk_factors:
            lines.append(f"⚠️ {', '.join(self.risk_factors[:2])}")
        
        return "\n".join(lines)


class OptionsChainFetcher:
    """
    Fetches options chain data from Alpaca or yfinance.
    """
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self._option_client = None
        self._trading_client = None
        
        if ALPACA_OPTIONS_AVAILABLE:
            try:
                self._option_client = OptionHistoricalDataClient(
                    api_key=api_key,
                    secret_key=secret_key,
                )
                self._trading_client = TradingClient(
                    api_key=api_key,
                    secret_key=secret_key,
                    paper=True,
                )
            except Exception as e:
                logger.warning(f"Failed to init Alpaca options client: {e}")
    
    def get_chain(
        self,
        symbol: str,
        min_dte: int = 7,
        max_dte: int = 45,
        strike_count: int = 10,
    ) -> List[OptionContract]:
        """
        Get options chain for a symbol.
        
        Args:
            symbol: Underlying symbol
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration  
            strike_count: Number of strikes around ATM
            
        Returns:
            List of OptionContract objects
        """
        contracts = []
        
        # Try yfinance first (more reliable for retail)
        if YFINANCE_AVAILABLE:
            try:
                contracts = self._get_chain_yfinance(symbol, min_dte, max_dte, strike_count)
                if contracts:
                    return contracts
            except Exception as e:
                logger.debug(f"yfinance chain failed: {e}")
        
        # Fall back to Alpaca
        if self._option_client:
            try:
                contracts = self._get_chain_alpaca(symbol, min_dte, max_dte, strike_count)
            except Exception as e:
                logger.debug(f"Alpaca chain failed: {e}")
        
        return contracts
    
    def _get_chain_yfinance(
        self,
        symbol: str,
        min_dte: int,
        max_dte: int,
        strike_count: int,
    ) -> List[OptionContract]:
        """Get chain from yfinance."""
        contracts = []
        
        ticker = yf.Ticker(symbol)
        
        # Get current price
        hist = ticker.history(period="1d")
        if hist.empty:
            return contracts
        current_price = float(hist['Close'].iloc[-1])
        
        # Get expiration dates
        try:
            expirations = ticker.options
        except Exception:
            return contracts
        
        if not expirations:
            return contracts
        
        # Filter expirations by DTE
        today = date.today()
        valid_expirations = []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                valid_expirations.append(exp_str)
        
        if not valid_expirations:
            # Use first available if none in range
            valid_expirations = expirations[:2]
        
        # Get option chain for each expiration
        for exp_str in valid_expirations[:2]:  # Limit to 2 expirations
            try:
                chain = ticker.option_chain(exp_str)
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                
                # Process calls
                calls_df = chain.calls
                if not calls_df.empty:
                    # Filter strikes around current price
                    calls_df = calls_df[
                        (calls_df['strike'] >= current_price * 0.9) &
                        (calls_df['strike'] <= current_price * 1.1)
                    ].head(strike_count)
                    
                    for _, row in calls_df.iterrows():
                        contract = OptionContract(
                            symbol=symbol,
                            contract_symbol=row.get('contractSymbol', ''),
                            option_type=OptionType.CALL,
                            strike=float(row['strike']),
                            expiration=exp_date,
                            bid=float(row.get('bid', 0)) if row.get('bid') else None,
                            ask=float(row.get('ask', 0)) if row.get('ask') else None,
                            last_price=float(row.get('lastPrice', 0)) if row.get('lastPrice') else None,
                            volume=int(row.get('volume', 0)) if row.get('volume') else 0,
                            open_interest=int(row.get('openInterest', 0)) if row.get('openInterest') else 0,
                            iv=float(row.get('impliedVolatility', 0)) if row.get('impliedVolatility') else None,
                        )
                        if contract.bid and contract.ask:
                            contract.mid_price = (contract.bid + contract.ask) / 2
                        contracts.append(contract)
                
                # Process puts
                puts_df = chain.puts
                if not puts_df.empty:
                    puts_df = puts_df[
                        (puts_df['strike'] >= current_price * 0.9) &
                        (puts_df['strike'] <= current_price * 1.1)
                    ].head(strike_count)
                    
                    for _, row in puts_df.iterrows():
                        contract = OptionContract(
                            symbol=symbol,
                            contract_symbol=row.get('contractSymbol', ''),
                            option_type=OptionType.PUT,
                            strike=float(row['strike']),
                            expiration=exp_date,
                            bid=float(row.get('bid', 0)) if row.get('bid') else None,
                            ask=float(row.get('ask', 0)) if row.get('ask') else None,
                            last_price=float(row.get('lastPrice', 0)) if row.get('lastPrice') else None,
                            volume=int(row.get('volume', 0)) if row.get('volume') else 0,
                            open_interest=int(row.get('openInterest', 0)) if row.get('openInterest') else 0,
                            iv=float(row.get('impliedVolatility', 0)) if row.get('impliedVolatility') else None,
                        )
                        if contract.bid and contract.ask:
                            contract.mid_price = (contract.bid + contract.ask) / 2
                        contracts.append(contract)
                        
            except Exception as e:
                logger.debug(f"Error processing expiration {exp_str}: {e}")
        
        return contracts
    
    def _get_chain_alpaca(
        self,
        symbol: str,
        min_dte: int,
        max_dte: int,
        strike_count: int,
    ) -> List[OptionContract]:
        """Get chain from Alpaca."""
        contracts = []
        
        # Alpaca options chain implementation
        # Note: Alpaca options API may require specific formatting
        
        return contracts


class OptionsSignalGenerator:
    """
    Generates simple options trading signals.
    
    Focuses on strategies suitable for Robinhood/retail traders:
    - Long calls for bullish bets
    - Long puts for bearish bets  
    - Covered calls for income
    - Cash-secured puts for accumulation
    """
    
    # Default parameters - Short-term focus for Alpha
    DEFAULT_MIN_DTE = 1         # Minimum 1 day (next day expiration)
    DEFAULT_MAX_DTE = 14        # Maximum 14 days (2 weeks)
    DEFAULT_DELTA_TARGET = 0.35  # Slightly higher delta for short-term trades
    
    # Unusual volume thresholds
    UNUSUAL_VOLUME_MULTIPLIER = 2.0  # Volume > 2x average = unusual
    HIGH_VOLUME_MULTIPLIER = 3.0     # Volume > 3x average = very unusual
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        min_confidence: float = 0.6,
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.min_confidence = min_confidence
        
        self.chain_fetcher = OptionsChainFetcher(api_key, secret_key)
        
        # Import technical analyzer
        from alpha.technical_analyzer import TechnicalAnalyzer
        self.tech_analyzer = TechnicalAnalyzer()
    
    def generate_signal(
        self,
        symbol: str,
        strategy: Optional[OptionStrategy] = None,
        account_size: float = 10000,
        max_risk_pct: float = 0.05,
    ) -> Optional[OptionsSignal]:
        """
        Generate an options signal for a symbol.
        
        Args:
            symbol: Stock symbol
            strategy: Specific strategy (or auto-select based on direction)
            account_size: Account size for position sizing
            max_risk_pct: Maximum percentage of account to risk
            
        Returns:
            OptionsSignal or None if no good setup found
        """
        # Get technical analysis for direction
        tech_signals = self.tech_analyzer.analyze(symbol)
        
        if not tech_signals.current_price:
            logger.warning(f"Could not get price for {symbol}")
            return None
        
        underlying_price = tech_signals.current_price
        
        # Determine direction and confidence
        if tech_signals.overall_signal == "bullish":
            direction = OptionSignalDirection.BULLISH
            confidence = tech_signals.confidence
        elif tech_signals.overall_signal == "bearish":
            direction = OptionSignalDirection.BEARISH
            confidence = tech_signals.confidence
        else:
            direction = OptionSignalDirection.NEUTRAL
            confidence = 0.5
        
        # Skip low confidence
        if confidence < self.min_confidence:
            logger.info(f"{symbol}: Confidence {confidence:.0%} below threshold")
            return None
        
        # Auto-select strategy if not specified
        if strategy is None:
            if direction == OptionSignalDirection.BULLISH:
                strategy = OptionStrategy.LONG_CALL
            elif direction == OptionSignalDirection.BEARISH:
                strategy = OptionStrategy.LONG_PUT
            else:
                return None  # No neutral strategies for now
        
        # Get options chain
        contracts = self.chain_fetcher.get_chain(
            symbol,
            min_dte=self.DEFAULT_MIN_DTE,
            max_dte=self.DEFAULT_MAX_DTE,
        )
        
        if not contracts:
            logger.warning(f"No options chain available for {symbol}")
            return None
        
        # Select best contract based on strategy
        selected_contract = self._select_contract(
            contracts, 
            strategy, 
            underlying_price,
            confidence,
        )
        
        if not selected_contract:
            logger.warning(f"No suitable contract found for {symbol}")
            return None
        
        # Calculate risk metrics
        max_loss, max_profit, break_even = self._calculate_risk_metrics(
            strategy,
            selected_contract,
            underlying_price,
        )
        
        # Position sizing
        max_risk = account_size * max_risk_pct
        if selected_contract.mid_price:
            contract_cost = selected_contract.mid_price * 100  # 100 shares per contract
            suggested_contracts = max(1, int(max_risk / contract_cost))
            total_cost = contract_cost * suggested_contracts
        else:
            suggested_contracts = 1
            total_cost = None
        
        # Build reasoning
        reasoning = self._build_reasoning(
            symbol,
            strategy,
            tech_signals,
            selected_contract,
        )
        
        # Risk factors
        risk_factors = []
        if selected_contract.days_to_expiration <= 3:
            risk_factors.append("Very short DTE - rapid theta decay")
        elif selected_contract.days_to_expiration < 7:
            risk_factors.append("Short DTE - elevated theta decay")
        if selected_contract.iv and selected_contract.iv > 0.5:
            risk_factors.append(f"High IV ({selected_contract.iv:.0%})")
        if selected_contract.open_interest < 100:
            risk_factors.append("Low liquidity")
        if confidence < 0.7:
            risk_factors.append("Moderate confidence")
        
        # Check for unusual volume
        has_unusual_volume = selected_contract.unusual_volume
        vol_ratio = selected_contract.volume_ratio
        
        if has_unusual_volume:
            if vol_ratio >= self.HIGH_VOLUME_MULTIPLIER:
                risk_factors.insert(0, f"🔥 Very High Volume ({vol_ratio:.1f}x avg)")
            else:
                risk_factors.insert(0, f"📈 Unusual Volume ({vol_ratio:.1f}x avg)")
        
        return OptionsSignal(
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            confidence=confidence,
            contracts=[selected_contract],
            underlying_price=underlying_price,
            entry_price=selected_contract.mid_price,
            max_loss=max_loss,
            max_profit=max_profit,
            break_even=break_even,
            suggested_contracts=suggested_contracts,
            total_cost=total_cost,
            reasoning=reasoning,
            risk_factors=risk_factors,
            unusual_volume=has_unusual_volume,
            volume_ratio=vol_ratio,
            valid_until=datetime.now(timezone.utc) + timedelta(hours=4),
        )
    
    def _select_contract(
        self,
        contracts: List[OptionContract],
        strategy: OptionStrategy,
        underlying_price: float,
        confidence: float,
    ) -> Optional[OptionContract]:
        """Select the best contract for the strategy."""
        
        # Filter by option type
        if strategy in [OptionStrategy.LONG_CALL, OptionStrategy.COVERED_CALL]:
            filtered = [c for c in contracts if c.option_type == OptionType.CALL]
        else:
            filtered = [c for c in contracts if c.option_type == OptionType.PUT]
        
        if not filtered:
            return None
        
        # Filter by liquidity (need bid/ask)
        filtered = [c for c in filtered if c.bid and c.ask and c.bid > 0]
        
        if not filtered:
            return None
        
        # Select based on strategy
        if strategy == OptionStrategy.LONG_CALL:
            # For long calls: slightly OTM, good liquidity
            # Higher confidence = closer to ATM
            if confidence > 0.75:
                # ATM or slightly ITM
                target_strike = underlying_price * 0.98
            else:
                # Slightly OTM
                target_strike = underlying_price * 1.02
            
            # Find closest strike
            best = min(filtered, key=lambda c: abs(c.strike - target_strike))
            
        elif strategy == OptionStrategy.LONG_PUT:
            # For long puts: slightly OTM
            if confidence > 0.75:
                target_strike = underlying_price * 1.02
            else:
                target_strike = underlying_price * 0.98
            
            best = min(filtered, key=lambda c: abs(c.strike - target_strike))
            
        elif strategy == OptionStrategy.COVERED_CALL:
            # For covered calls: OTM (want to keep shares)
            target_strike = underlying_price * 1.05  # 5% OTM
            otm_calls = [c for c in filtered if c.strike > underlying_price]
            if otm_calls:
                best = min(otm_calls, key=lambda c: abs(c.strike - target_strike))
            else:
                best = filtered[0]
                
        elif strategy == OptionStrategy.CASH_SECURED_PUT:
            # For CSP: slightly OTM (willing to buy at discount)
            target_strike = underlying_price * 0.95  # 5% below
            otm_puts = [c for c in filtered if c.strike < underlying_price]
            if otm_puts:
                best = min(otm_puts, key=lambda c: abs(c.strike - target_strike))
            else:
                best = filtered[0]
        else:
            best = filtered[0]
        
        return best
    
    def _calculate_risk_metrics(
        self,
        strategy: OptionStrategy,
        contract: OptionContract,
        underlying_price: float,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate max loss, max profit, and break-even.
        
        Returns: (max_loss, max_profit, break_even)
        """
        premium = (contract.mid_price or 0) * 100  # Per contract
        
        if strategy == OptionStrategy.LONG_CALL:
            # Max loss = premium paid
            # Max profit = unlimited
            # Break-even = strike + premium
            max_loss = premium
            max_profit = None  # Unlimited
            break_even = contract.strike + (contract.mid_price or 0)
            
        elif strategy == OptionStrategy.LONG_PUT:
            # Max loss = premium paid
            # Max profit = strike - premium (if stock goes to 0)
            # Break-even = strike - premium
            max_loss = premium
            max_profit = (contract.strike - (contract.mid_price or 0)) * 100
            break_even = contract.strike - (contract.mid_price or 0)
            
        elif strategy == OptionStrategy.COVERED_CALL:
            # Max loss = stock price - premium (if stock goes to 0)
            # Max profit = (strike - current price) + premium
            # Break-even = current price - premium
            max_loss = (underlying_price - (contract.mid_price or 0)) * 100
            max_profit = (contract.strike - underlying_price + (contract.mid_price or 0)) * 100
            break_even = underlying_price - (contract.mid_price or 0)
            
        elif strategy == OptionStrategy.CASH_SECURED_PUT:
            # Max loss = strike - premium (if stock goes to 0)
            # Max profit = premium received
            # Break-even = strike - premium
            max_loss = (contract.strike - (contract.mid_price or 0)) * 100
            max_profit = premium
            break_even = contract.strike - (contract.mid_price or 0)
            
        else:
            max_loss = premium
            max_profit = None
            break_even = None
        
        return max_loss, max_profit, break_even
    
    def _build_reasoning(
        self,
        symbol: str,
        strategy: OptionStrategy,
        tech_signals: Any,
        contract: OptionContract,
    ) -> str:
        """Build reasoning string for the signal."""
        parts = []
        
        # Direction reasoning
        if tech_signals.trend_signal != "neutral":
            parts.append(f"Trend: {tech_signals.trend_signal}")
        
        if tech_signals.rsi_14:
            if tech_signals.rsi_14 < 30:
                parts.append(f"RSI oversold ({tech_signals.rsi_14:.0f})")
            elif tech_signals.rsi_14 > 70:
                parts.append(f"RSI overbought ({tech_signals.rsi_14:.0f})")
        
        if tech_signals.macd and tech_signals.macd_signal:
            macd_cross = "bullish" if tech_signals.macd > tech_signals.macd_signal else "bearish"
            parts.append(f"MACD {macd_cross}")
        
        # Contract info
        parts.append(f"${contract.strike} {contract.option_type.value} @ ${contract.mid_price:.2f}")
        parts.append(f"{contract.days_to_expiration}d to exp")
        
        return " | ".join(parts)
    
    def scan_for_options(
        self,
        symbols: List[str],
        strategies: Optional[List[OptionStrategy]] = None,
    ) -> List[OptionsSignal]:
        """
        Scan multiple symbols for options opportunities.
        
        Args:
            symbols: List of stock symbols
            strategies: Filter by specific strategies (or all)
            
        Returns:
            List of OptionsSignal sorted by confidence
        """
        signals = []
        
        if strategies is None:
            strategies = [OptionStrategy.LONG_CALL, OptionStrategy.LONG_PUT]
        
        for symbol in symbols:
            for strategy in strategies:
                try:
                    signal = self.generate_signal(symbol, strategy=strategy)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.debug(f"Error generating {strategy.value} for {symbol}: {e}")
        
        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)
        
        return signals


def get_recommended_strategy(direction: str, has_shares: bool = False) -> OptionStrategy:
    """
    Get recommended strategy based on direction and share ownership.
    
    Args:
        direction: "bullish", "bearish", or "neutral"
        has_shares: Whether the user already owns shares
        
    Returns:
        Recommended OptionStrategy
    """
    if direction.lower() == "bullish":
        if has_shares:
            return OptionStrategy.COVERED_CALL  # Generate income
        return OptionStrategy.LONG_CALL  # Leverage upside
        
    elif direction.lower() == "bearish":
        return OptionStrategy.LONG_PUT  # Profit from decline
        
    else:  # Neutral
        if has_shares:
            return OptionStrategy.COVERED_CALL
        return OptionStrategy.CASH_SECURED_PUT  # Get paid to wait
