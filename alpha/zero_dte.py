"""
Gnosis Alpha - 0DTE (Zero Days to Expiration) Options

High-risk, high-reward same-day options trading.
Popular with retail traders for quick directional bets.

WARNING: 0DTE options are extremely risky:
- 100% loss of premium is common
- Rapid time decay (theta)
- High sensitivity to price moves (gamma)
- Wide bid-ask spreads near expiration
- NOT suitable for beginners

Use only with money you can afford to lose entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta, timezone
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from alpha.options_signal import (
    OptionContract,
    OptionType,
    OptionStrategy,
    OptionSignalDirection,
    OptionsSignal,
    OptionsChainFetcher,
)
from alpha.technical_analyzer import TechnicalAnalyzer


class ZeroDTERisk(str, Enum):
    """Risk level for 0DTE trades."""
    EXTREME = "EXTREME"      # Very high gamma, near expiration
    HIGH = "HIGH"            # Standard 0DTE risk
    ELEVATED = "ELEVATED"    # Slightly better odds


class ZeroDTEStrategy(str, Enum):
    """0DTE-specific strategies."""
    SCALP_CALL = "scalp_call"          # Quick bullish scalp
    SCALP_PUT = "scalp_put"            # Quick bearish scalp
    MOMENTUM_CALL = "momentum_call"    # Ride momentum up
    MOMENTUM_PUT = "momentum_put"      # Ride momentum down
    LOTTO_CALL = "lotto_call"          # Deep OTM lottery ticket
    LOTTO_PUT = "lotto_put"            # Deep OTM lottery ticket


@dataclass
class ZeroDTESignal:
    """
    A 0DTE options trading signal.
    
    Includes extra risk warnings and timing guidance
    specific to same-day expiration trades.
    """
    # Basic info
    symbol: str
    strategy: ZeroDTEStrategy
    direction: OptionSignalDirection
    confidence: float
    risk_level: ZeroDTERisk
    
    # Contract
    contract: Optional[OptionContract] = None
    
    # Pricing
    underlying_price: Optional[float] = None
    entry_price: Optional[float] = None  # Per contract
    
    # Risk metrics (0DTE specific)
    max_loss: Optional[float] = None        # Premium paid (100% loss likely)
    potential_gain_pct: Optional[float] = None  # If move happens
    break_even: Optional[float] = None
    distance_to_strike_pct: Optional[float] = None  # How far OTM
    
    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    market_close: Optional[datetime] = None  # When options expire worthless
    time_remaining: Optional[str] = None     # Human readable
    
    # Position sizing
    suggested_contracts: int = 1
    max_position_dollars: float = 100.0  # Default small size for 0DTE
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    reasoning: str = ""
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy.value,
            "direction": self.direction.value,
            "confidence": round(self.confidence, 3),
            "risk_level": self.risk_level.value,
            "underlying_price": self.underlying_price,
            "entry_price": self.entry_price,
            "max_loss": self.max_loss,
            "potential_gain_pct": self.potential_gain_pct,
            "break_even": self.break_even,
            "distance_to_strike_pct": self.distance_to_strike_pct,
            "time_remaining": self.time_remaining,
            "suggested_contracts": self.suggested_contracts,
            "max_position_dollars": self.max_position_dollars,
            "warnings": self.warnings,
            "reasoning": self.reasoning,
            "contract": self.contract.to_dict() if self.contract else None,
        }
    
    def to_robinhood_format(self) -> str:
        """Format for Robinhood-style display with 0DTE warnings."""
        direction_emoji = {
            "BULLISH": "🟢",
            "BEARISH": "🔴",
            "NEUTRAL": "⚪"
        }[self.direction.value]
        
        risk_emoji = {
            "EXTREME": "🔥🔥🔥",
            "HIGH": "🔥🔥",
            "ELEVATED": "🔥"
        }[self.risk_level.value]
        
        strategy_name = {
            "scalp_call": "0DTE Scalp Call",
            "scalp_put": "0DTE Scalp Put",
            "momentum_call": "0DTE Momentum Call",
            "momentum_put": "0DTE Momentum Put",
            "lotto_call": "0DTE Lotto Call 🎰",
            "lotto_put": "0DTE Lotto Put 🎰",
        }[self.strategy.value]
        
        lines = [
            f"⚡ 0DTE OPTIONS ⚡",
            f"{direction_emoji} {self.symbol}: {strategy_name}",
            f"Risk: {self.risk_level.value} {risk_emoji}",
        ]
        
        if self.contract:
            lines.append(f"Strike: ${self.contract.strike:.2f} {self.contract.option_type.value.upper()}")
            if self.contract.mid_price:
                lines.append(f"Price: ${self.contract.mid_price:.2f}/contract")
        
        if self.underlying_price:
            lines.append(f"Stock: ${self.underlying_price:.2f}")
        
        if self.distance_to_strike_pct is not None:
            otm_itm = "OTM" if self.distance_to_strike_pct > 0 else "ITM"
            lines.append(f"Distance: {abs(self.distance_to_strike_pct):.1f}% {otm_itm}")
        
        if self.max_loss:
            lines.append(f"Max Loss: ${self.max_loss:.2f} (100% likely)")
        
        if self.break_even:
            lines.append(f"Break-Even: ${self.break_even:.2f}")
        
        if self.time_remaining:
            lines.append(f"⏰ Expires: {self.time_remaining}")
        
        # Always show warnings for 0DTE
        lines.append("")
        lines.append("⚠️ 0DTE WARNINGS:")
        for warning in self.warnings[:3]:
            lines.append(f"  • {warning}")
        
        return "\n".join(lines)


class ZeroDTEGenerator:
    """
    Generates 0DTE options signals.
    
    Focuses on same-day expiration options with:
    - Extra risk warnings
    - Smaller position sizes
    - Time-of-day awareness
    - Scalping vs momentum strategies
    """
    
    # 0DTE specific parameters
    MAX_POSITION_DOLLARS = 200  # Default max for 0DTE
    LOTTO_MAX_PRICE = 0.50      # Max price for lotto tickets
    SCALP_TARGET_DELTA = 0.40   # Near ATM for scalps
    MOMENTUM_TARGET_DELTA = 0.30  # Slightly OTM for momentum
    LOTTO_TARGET_DELTA = 0.10    # Deep OTM for lottos
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        max_position_dollars: float = 200,
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.max_position_dollars = max_position_dollars
        
        self.chain_fetcher = OptionsChainFetcher(api_key, secret_key)
        self.tech_analyzer = TechnicalAnalyzer()
    
    def is_0dte_available(self, symbol: str) -> bool:
        """
        Check if 0DTE options are available for a symbol today.
        
        Major indices (SPY, QQQ, IWM) have daily expirations.
        Most stocks only have weekly/monthly.
        """
        # Symbols with daily 0DTE options
        daily_0dte_symbols = {
            "SPY", "QQQ", "IWM",  # Major ETFs
            "SPX", "NDX",         # Index options
            "AAPL", "TSLA", "AMZN", "NVDA", "META", "GOOGL", "MSFT",  # Select stocks
        }
        
        return symbol.upper() in daily_0dte_symbols
    
    def get_time_to_close(self) -> Tuple[timedelta, str]:
        """Get time remaining until market close (4 PM ET)."""
        now = datetime.now(timezone.utc)
        
        # Market close is 4 PM ET (9 PM UTC during EST, 8 PM UTC during EDT)
        # Simplified: assume 9 PM UTC
        today_close = now.replace(hour=21, minute=0, second=0, microsecond=0)
        
        if now > today_close:
            return timedelta(0), "Market Closed"
        
        remaining = today_close - now
        hours = remaining.seconds // 3600
        minutes = (remaining.seconds % 3600) // 60
        
        if hours > 0:
            time_str = f"{hours}h {minutes}m to close"
        else:
            time_str = f"{minutes}m to close"
        
        return remaining, time_str
    
    def get_0dte_chain(self, symbol: str) -> List[OptionContract]:
        """Get options expiring today."""
        if not YFINANCE_AVAILABLE:
            return []
        
        contracts = []
        today = date.today()
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current price
            hist = ticker.history(period="1d")
            if hist.empty:
                return contracts
            current_price = float(hist['Close'].iloc[-1])
            
            # Get expiration dates
            expirations = ticker.options
            
            # Find today's expiration
            today_str = today.strftime("%Y-%m-%d")
            
            if today_str not in expirations:
                # Try to find closest expiration (might be labeled differently)
                logger.info(f"No 0DTE found for {symbol} on {today_str}")
                return contracts
            
            # Get chain for today
            chain = ticker.option_chain(today_str)
            
            # Helper to safely convert to int (handles NaN)
            import math
            def safe_int(val, default=0):
                if val is None:
                    return default
                try:
                    if math.isnan(val):
                        return default
                    return int(val)
                except (ValueError, TypeError):
                    return default
            
            def safe_float(val, default=None):
                if val is None:
                    return default
                try:
                    fval = float(val)
                    if math.isnan(fval):
                        return default
                    return fval
                except (ValueError, TypeError):
                    return default
            
            # Process calls
            for _, row in chain.calls.iterrows():
                bid = safe_float(row.get('bid'), 0)
                ask = safe_float(row.get('ask'), 0)
                if bid and bid > 0 and ask and ask > 0:
                    contract = OptionContract(
                        symbol=symbol,
                        contract_symbol=row.get('contractSymbol', ''),
                        option_type=OptionType.CALL,
                        strike=float(row['strike']),
                        expiration=today,
                        bid=bid,
                        ask=ask,
                        last_price=safe_float(row.get('lastPrice'), 0),
                        volume=safe_int(row.get('volume'), 0),
                        open_interest=safe_int(row.get('openInterest'), 0),
                        iv=safe_float(row.get('impliedVolatility')),
                    )
                    contract.mid_price = (contract.bid + contract.ask) / 2
                    contracts.append(contract)
            
            # Process puts
            for _, row in chain.puts.iterrows():
                bid = safe_float(row.get('bid'), 0)
                ask = safe_float(row.get('ask'), 0)
                if bid and bid > 0 and ask and ask > 0:
                    contract = OptionContract(
                        symbol=symbol,
                        contract_symbol=row.get('contractSymbol', ''),
                        option_type=OptionType.PUT,
                        strike=float(row['strike']),
                        expiration=today,
                        bid=bid,
                        ask=ask,
                        last_price=safe_float(row.get('lastPrice'), 0),
                        volume=safe_int(row.get('volume'), 0),
                        open_interest=safe_int(row.get('openInterest'), 0),
                        iv=safe_float(row.get('impliedVolatility')),
                    )
                    contract.mid_price = (contract.bid + contract.ask) / 2
                    contracts.append(contract)
                    
        except Exception as e:
            logger.error(f"Error getting 0DTE chain for {symbol}: {e}")
        
        return contracts
    
    def generate_signal(
        self,
        symbol: str,
        strategy: Optional[ZeroDTEStrategy] = None,
        max_dollars: Optional[float] = None,
    ) -> Optional[ZeroDTESignal]:
        """
        Generate a 0DTE options signal.
        
        Args:
            symbol: Stock symbol
            strategy: Specific 0DTE strategy (or auto-select)
            max_dollars: Maximum position size in dollars
            
        Returns:
            ZeroDTESignal or None
        """
        max_pos = max_dollars or self.max_position_dollars
        
        # Check if 0DTE is available
        if not self.is_0dte_available(symbol):
            logger.warning(f"0DTE not available for {symbol}")
            return None
        
        # Get time remaining
        time_remaining, time_str = self.get_time_to_close()
        
        if time_remaining.total_seconds() < 900:  # Less than 15 minutes
            logger.warning("Less than 15 minutes to close - 0DTE too risky")
            return None
        
        # Get technical analysis
        tech = self.tech_analyzer.analyze(symbol)
        
        if not tech.current_price:
            return None
        
        underlying_price = tech.current_price
        
        # Determine direction
        if tech.overall_signal == "bullish":
            direction = OptionSignalDirection.BULLISH
        elif tech.overall_signal == "bearish":
            direction = OptionSignalDirection.BEARISH
        else:
            direction = OptionSignalDirection.NEUTRAL
        
        confidence = tech.confidence
        
        # Auto-select strategy if not specified
        if strategy is None:
            strategy = self._select_strategy(tech, time_remaining)
        
        # Get 0DTE chain
        contracts = self.get_0dte_chain(symbol)
        
        if not contracts:
            logger.warning(f"No 0DTE options available for {symbol}")
            return None
        
        # Select contract based on strategy
        contract = self._select_contract(
            contracts,
            strategy,
            underlying_price,
            max_pos,
        )
        
        if not contract:
            return None
        
        # Calculate risk metrics
        distance_pct = ((contract.strike - underlying_price) / underlying_price) * 100
        if contract.option_type == OptionType.PUT:
            distance_pct = -distance_pct
        
        max_loss = (contract.mid_price or 0) * 100
        break_even = contract.strike + (contract.mid_price or 0) if contract.option_type == OptionType.CALL \
                     else contract.strike - (contract.mid_price or 0)
        
        # Position sizing
        if contract.mid_price and contract.mid_price > 0:
            contract_cost = contract.mid_price * 100
            suggested_contracts = max(1, int(max_pos / contract_cost))
        else:
            suggested_contracts = 1
        
        # Determine risk level
        risk_level = self._assess_risk(contract, time_remaining, distance_pct)
        
        # Build warnings
        warnings = self._build_warnings(contract, time_remaining, distance_pct, strategy)
        
        # Build reasoning
        reasoning = self._build_reasoning(tech, contract, strategy, time_str)
        
        return ZeroDTESignal(
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            confidence=confidence,
            risk_level=risk_level,
            contract=contract,
            underlying_price=underlying_price,
            entry_price=contract.mid_price,
            max_loss=max_loss * suggested_contracts,
            break_even=break_even,
            distance_to_strike_pct=distance_pct,
            time_remaining=time_str,
            suggested_contracts=suggested_contracts,
            max_position_dollars=max_pos,
            warnings=warnings,
            reasoning=reasoning,
        )
    
    def _select_strategy(
        self,
        tech: Any,
        time_remaining: timedelta,
    ) -> ZeroDTEStrategy:
        """Auto-select strategy based on technicals and time."""
        hours_left = time_remaining.total_seconds() / 3600
        
        # Lotto plays for low confidence or long time remaining
        if tech.confidence < 0.5 or hours_left > 5:
            if tech.overall_signal == "bullish":
                return ZeroDTEStrategy.LOTTO_CALL
            elif tech.overall_signal == "bearish":
                return ZeroDTEStrategy.LOTTO_PUT
            return ZeroDTEStrategy.LOTTO_CALL
        
        # Momentum plays for strong signals
        if tech.confidence > 0.65:
            if tech.overall_signal == "bullish":
                return ZeroDTEStrategy.MOMENTUM_CALL
            elif tech.overall_signal == "bearish":
                return ZeroDTEStrategy.MOMENTUM_PUT
        
        # Default to scalps
        if tech.overall_signal == "bullish":
            return ZeroDTEStrategy.SCALP_CALL
        elif tech.overall_signal == "bearish":
            return ZeroDTEStrategy.SCALP_PUT
        
        return ZeroDTEStrategy.SCALP_CALL
    
    def _select_contract(
        self,
        contracts: List[OptionContract],
        strategy: ZeroDTEStrategy,
        underlying_price: float,
        max_dollars: float,
    ) -> Optional[OptionContract]:
        """Select the best contract for the strategy."""
        
        # Filter by option type
        if strategy in [ZeroDTEStrategy.SCALP_CALL, ZeroDTEStrategy.MOMENTUM_CALL, ZeroDTEStrategy.LOTTO_CALL]:
            filtered = [c for c in contracts if c.option_type == OptionType.CALL]
        else:
            filtered = [c for c in contracts if c.option_type == OptionType.PUT]
        
        if not filtered:
            return None
        
        # Filter by price (must be affordable)
        max_price = max_dollars / 100  # Per share price
        filtered = [c for c in filtered if c.mid_price and c.mid_price <= max_price]
        
        if not filtered:
            return None
        
        # Select based on strategy
        if strategy in [ZeroDTEStrategy.LOTTO_CALL, ZeroDTEStrategy.LOTTO_PUT]:
            # Lotto: Cheapest OTM options
            if strategy == ZeroDTEStrategy.LOTTO_CALL:
                otm = [c for c in filtered if c.strike > underlying_price]
            else:
                otm = [c for c in filtered if c.strike < underlying_price]
            
            if otm:
                # Get cheapest that's not too far OTM
                otm.sort(key=lambda c: c.mid_price or 999)
                return otm[0]
        
        elif strategy in [ZeroDTEStrategy.SCALP_CALL, ZeroDTEStrategy.SCALP_PUT]:
            # Scalp: Near ATM
            return min(filtered, key=lambda c: abs(c.strike - underlying_price))
        
        elif strategy in [ZeroDTEStrategy.MOMENTUM_CALL, ZeroDTEStrategy.MOMENTUM_PUT]:
            # Momentum: Slightly OTM
            if strategy == ZeroDTEStrategy.MOMENTUM_CALL:
                target = underlying_price * 1.005  # 0.5% OTM
                otm = [c for c in filtered if c.strike >= underlying_price]
            else:
                target = underlying_price * 0.995
                otm = [c for c in filtered if c.strike <= underlying_price]
            
            if otm:
                return min(otm, key=lambda c: abs(c.strike - target))
        
        # Fallback: nearest strike
        return min(filtered, key=lambda c: abs(c.strike - underlying_price))
    
    def _assess_risk(
        self,
        contract: OptionContract,
        time_remaining: timedelta,
        distance_pct: float,
    ) -> ZeroDTERisk:
        """Assess the risk level of the 0DTE trade."""
        hours_left = time_remaining.total_seconds() / 3600
        
        # Extreme risk conditions
        if hours_left < 1:
            return ZeroDTERisk.EXTREME
        if abs(distance_pct) > 3:  # More than 3% OTM
            return ZeroDTERisk.EXTREME
        if contract.mid_price and contract.mid_price < 0.10:
            return ZeroDTERisk.EXTREME
        
        # High risk conditions
        if hours_left < 3:
            return ZeroDTERisk.HIGH
        if abs(distance_pct) > 1.5:
            return ZeroDTERisk.HIGH
        
        return ZeroDTERisk.ELEVATED
    
    def _build_warnings(
        self,
        contract: OptionContract,
        time_remaining: timedelta,
        distance_pct: float,
        strategy: ZeroDTEStrategy,
    ) -> List[str]:
        """Build 0DTE-specific warnings."""
        warnings = [
            "0DTE: 100% loss is common",
            "Expires TODAY - no time to recover",
        ]
        
        hours_left = time_remaining.total_seconds() / 3600
        
        if hours_left < 2:
            warnings.append(f"Only {hours_left:.1f}h until expiration")
        
        if abs(distance_pct) > 2:
            warnings.append(f"Far OTM ({abs(distance_pct):.1f}%) - unlikely to profit")
        
        if contract.mid_price and contract.mid_price < 0.20:
            warnings.append("Very cheap = very unlikely to pay off")
        
        if strategy in [ZeroDTEStrategy.LOTTO_CALL, ZeroDTEStrategy.LOTTO_PUT]:
            warnings.append("LOTTO play: Treat as entertainment, not investment")
        
        # Bid-ask spread warning
        if contract.bid and contract.ask:
            spread_pct = (contract.ask - contract.bid) / contract.mid_price * 100
            if spread_pct > 20:
                warnings.append(f"Wide spread ({spread_pct:.0f}%) - hard to exit")
        
        return warnings
    
    def _build_reasoning(
        self,
        tech: Any,
        contract: OptionContract,
        strategy: ZeroDTEStrategy,
        time_str: str,
    ) -> str:
        """Build reasoning string."""
        parts = []
        
        if tech.trend_signal != "neutral":
            parts.append(f"Trend: {tech.trend_signal}")
        
        if tech.rsi_14:
            if tech.rsi_14 < 30:
                parts.append("RSI oversold")
            elif tech.rsi_14 > 70:
                parts.append("RSI overbought")
        
        parts.append(f"${contract.strike} {contract.option_type.value}")
        parts.append(f"${contract.mid_price:.2f}")
        parts.append(time_str)
        
        return " | ".join(parts)
    
    def scan_0dte(
        self,
        symbols: Optional[List[str]] = None,
        max_dollars: float = 200,
    ) -> List[ZeroDTESignal]:
        """
        Scan for 0DTE opportunities.
        
        Args:
            symbols: Symbols to scan (default: major 0DTE symbols)
            max_dollars: Max position size
            
        Returns:
            List of ZeroDTESignal sorted by confidence
        """
        if symbols is None:
            # Default to symbols with daily 0DTE
            symbols = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
        
        signals = []
        
        for symbol in symbols:
            try:
                signal = self.generate_signal(symbol, max_dollars=max_dollars)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.debug(f"Error scanning {symbol} for 0DTE: {e}")
        
        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)
        
        return signals


def print_0dte_disclaimer():
    """Print 0DTE risk disclaimer."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    ⚠️  0DTE OPTIONS WARNING ⚠️                    ║
╠══════════════════════════════════════════════════════════════════╣
║  0DTE (Zero Days to Expiration) options are EXTREMELY RISKY:     ║
║                                                                  ║
║  • 100% loss of premium is COMMON                                ║
║  • Options expire WORTHLESS at market close                      ║
║  • Rapid time decay destroys value quickly                       ║
║  • High gamma = extreme price sensitivity                        ║
║  • Wide bid-ask spreads make exits difficult                     ║
║                                                                  ║
║  ONLY trade 0DTE with money you can afford to LOSE ENTIRELY.     ║
║  This is closer to GAMBLING than investing.                      ║
╚══════════════════════════════════════════════════════════════════╝
""")
