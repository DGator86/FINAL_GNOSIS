"""
Gnosis Alpha - Backtesting Framework

Walk-forward backtesting for realistic strategy evaluation.
Prevents look-ahead bias through proper train/test splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class Trade:
    """Represents a single backtest trade."""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    direction: str  # "BUY" or "SELL"
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    confidence: float
    expected_return: float
    
    # Results (filled after exit)
    actual_return: float = 0.0
    pnl: float = 0.0
    hold_days: int = 0
    hit_target: bool = False
    hit_stop: bool = False
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date.isoformat(),
            "exit_date": self.exit_date.isoformat() if self.exit_date else None,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "confidence": self.confidence,
            "expected_return": self.expected_return,
            "actual_return": self.actual_return,
            "pnl": self.pnl,
            "hold_days": self.hold_days,
        }


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Basic info
    start_date: datetime
    end_date: datetime
    symbols: List[str]
    initial_capital: float
    
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_hold_days: float = 0.0
    
    # Model performance
    direction_accuracy: float = 0.0
    prediction_correlation: float = 0.0
    
    # Equity curve
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    
    # All trades
    trades: List[Trade] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "symbols": self.symbols,
            "initial_capital": self.initial_capital,
            "performance": {
                "total_return": self.total_return,
                "annualized_return": self.annualized_return,
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "max_drawdown": self.max_drawdown,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
            },
            "trades": {
                "total": self.total_trades,
                "winning": self.winning_trades,
                "losing": self.losing_trades,
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
                "avg_hold_days": self.avg_hold_days,
            },
            "model": {
                "direction_accuracy": self.direction_accuracy,
                "prediction_correlation": self.prediction_correlation,
            },
        }
    
    def print_summary(self) -> None:
        """Print formatted backtest summary."""
        print("\n" + "="*60)
        print("  GNOSIS ALPHA - Backtest Results")
        print("="*60)
        print(f"\nPeriod: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Symbols: {', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        
        print("\n📊 Performance:")
        print(f"  Total Return: {self.total_return*100:+.2f}%")
        print(f"  Annualized Return: {self.annualized_return*100:+.2f}%")
        print(f"  Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {self.sortino_ratio:.2f}")
        print(f"  Max Drawdown: {self.max_drawdown*100:.2f}%")
        
        print("\n📈 Trades:")
        print(f"  Total Trades: {self.total_trades}")
        print(f"  Win Rate: {self.win_rate*100:.1f}%")
        print(f"  Profit Factor: {self.profit_factor:.2f}")
        print(f"  Avg Win: {self.avg_win*100:+.2f}%")
        print(f"  Avg Loss: {self.avg_loss*100:.2f}%")
        print(f"  Avg Hold: {self.avg_hold_days:.1f} days")
        
        print("\n🤖 Model Accuracy:")
        print(f"  Direction Accuracy: {self.direction_accuracy*100:.1f}%")
        print(f"  Prediction Correlation: {self.prediction_correlation:.3f}")
        print("="*60 + "\n")


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000
    position_size_pct: float = 0.10  # 10% of portfolio per trade
    max_positions: int = 5
    
    # Trade management
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit
    max_hold_days: int = 7
    
    # Signal filtering
    min_confidence: float = 0.6
    
    # Costs
    commission_per_trade: float = 0.0  # Robinhood style
    slippage_pct: float = 0.001  # 0.1% slippage
    
    # Walk-forward settings
    train_days: int = 180  # 6 months training
    test_days: int = 30   # 1 month test
    retrain_interval: int = 30  # Retrain every 30 days


class AlphaBacktester:
    """
    Walk-forward backtester for Alpha signals.
    
    Simulates realistic trading with:
    - Walk-forward validation (retrain periodically)
    - Position sizing based on confidence
    - Stop loss and take profit orders
    - Transaction costs and slippage
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.results: Optional[BacktestResult] = None
    
    def run(
        self,
        model,  # AlphaModel
        feature_engine,  # AlphaFeatureEngine  
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> BacktestResult:
        """
        Run walk-forward backtest.
        
        Args:
            model: Trained AlphaModel or model class to train
            feature_engine: Feature extraction engine
            symbols: List of symbols to trade
            start_date: Backtest start date
            end_date: Backtest end date
            price_data: Optional pre-loaded price data
            
        Returns:
            BacktestResult with all metrics and trades
        """
        if not PANDAS_AVAILABLE:
            raise RuntimeError("pandas required for backtesting")
        
        # Initialize result
        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            initial_capital=self.config.initial_capital,
        )
        
        # Load price data if not provided
        if price_data is None:
            price_data = self._load_price_data(symbols, start_date, end_date)
        
        # Initialize portfolio state
        cash = self.config.initial_capital
        positions: Dict[str, Trade] = {}
        equity_history = []
        all_trades = []
        predictions = []  # For model evaluation
        actuals = []
        
        # Generate trading days
        trading_days = self._get_trading_days(start_date, end_date, price_data)
        
        logger.info(f"Running backtest over {len(trading_days)} trading days")
        
        # Walk-forward simulation
        last_train_date = None
        
        for current_date in trading_days:
            # Check if we need to retrain
            if self._should_retrain(current_date, last_train_date):
                # Retrain model on historical data
                train_start = current_date - timedelta(days=self.config.train_days)
                model = self._retrain_model(
                    model, feature_engine, symbols, train_start, current_date
                )
                last_train_date = current_date
            
            # Calculate current equity
            equity = cash + self._calculate_position_value(positions, price_data, current_date)
            equity_history.append((current_date, equity))
            
            # Check and close positions (stop loss, take profit, max hold)
            closed_trades = self._check_exits(
                positions, price_data, current_date
            )
            for trade in closed_trades:
                cash += trade.exit_price * trade.quantity * (1 if trade.direction == "BUY" else -1)
                all_trades.append(trade)
                del positions[trade.symbol]
            
            # Generate signals and open new positions
            if len(positions) < self.config.max_positions:
                for symbol in symbols:
                    if symbol in positions:
                        continue
                    
                    # Extract features
                    features = feature_engine.extract(symbol, as_of=current_date)
                    
                    if not features.has_sufficient_data:
                        continue
                    
                    # Get prediction
                    try:
                        prediction = model.predict(features.to_array())
                    except Exception as e:
                        logger.debug(f"Prediction error for {symbol}: {e}")
                        continue
                    
                    # Record for model evaluation
                    predictions.append(prediction.expected_return)
                    
                    # Get actual future return for evaluation
                    actual_return = self._get_future_return(
                        symbol, current_date, price_data, 
                        days=self.config.max_hold_days
                    )
                    actuals.append(actual_return)
                    
                    # Check signal quality
                    if prediction.direction == "HOLD":
                        continue
                    if prediction.confidence < self.config.min_confidence:
                        continue
                    
                    # Calculate position size
                    position_value = equity * self.config.position_size_pct
                    position_value *= prediction.confidence  # Scale by confidence
                    
                    # Get current price
                    price = self._get_price(symbol, current_date, price_data)
                    if price is None:
                        continue
                    
                    # Apply slippage
                    if prediction.direction == "BUY":
                        entry_price = price * (1 + self.config.slippage_pct)
                    else:
                        entry_price = price * (1 - self.config.slippage_pct)
                    
                    # Calculate quantity
                    quantity = int(position_value / entry_price)
                    if quantity <= 0:
                        continue
                    
                    # Check if we have enough cash
                    cost = entry_price * quantity + self.config.commission_per_trade
                    if cost > cash:
                        continue
                    
                    # Open position
                    trade = Trade(
                        symbol=symbol,
                        entry_date=current_date,
                        exit_date=None,
                        direction=prediction.direction,
                        entry_price=entry_price,
                        exit_price=None,
                        quantity=quantity,
                        confidence=prediction.confidence,
                        expected_return=prediction.expected_return,
                    )
                    
                    positions[symbol] = trade
                    cash -= cost
                    
                    if len(positions) >= self.config.max_positions:
                        break
        
        # Close any remaining positions at end
        for symbol, trade in list(positions.items()):
            exit_price = self._get_price(symbol, end_date, price_data)
            if exit_price:
                trade.exit_date = end_date
                trade.exit_price = exit_price
                trade.hold_days = (end_date - trade.entry_date).days
                trade.actual_return = (
                    (exit_price - trade.entry_price) / trade.entry_price
                    * (1 if trade.direction == "BUY" else -1)
                )
                trade.pnl = (exit_price - trade.entry_price) * trade.quantity * (1 if trade.direction == "BUY" else -1)
                all_trades.append(trade)
        
        # Calculate final metrics
        result.trades = all_trades
        result.equity_curve = equity_history
        result = self._calculate_metrics(result, predictions, actuals)
        
        self.results = result
        return result
    
    def _load_price_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """Load price data for all symbols."""
        import yfinance as yf
        
        price_data = {}
        
        # Add buffer for calculations
        buffer_start = start_date - timedelta(days=200)
        buffer_end = end_date + timedelta(days=30)
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=buffer_start, end=buffer_end)
                if not df.empty:
                    price_data[symbol] = df
            except Exception as e:
                logger.warning(f"Could not load data for {symbol}: {e}")
        
        return price_data
    
    def _get_trading_days(
        self,
        start_date: datetime,
        end_date: datetime,
        price_data: Dict[str, pd.DataFrame],
    ) -> List[datetime]:
        """Get list of trading days."""
        # Use SPY or first symbol's trading days
        ref_symbol = "SPY" if "SPY" in price_data else list(price_data.keys())[0]
        df = price_data[ref_symbol]
        
        days = [
            d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d
            for d in df.index
            if start_date <= (d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d) <= end_date
        ]
        
        return days
    
    def _should_retrain(
        self,
        current_date: datetime,
        last_train_date: Optional[datetime],
    ) -> bool:
        """Check if model should be retrained."""
        if last_train_date is None:
            return True
        
        days_since_train = (current_date - last_train_date).days
        return days_since_train >= self.config.retrain_interval
    
    def _retrain_model(
        self,
        model,
        feature_engine,
        symbols: List[str],
        train_start: datetime,
        train_end: datetime,
    ):
        """Retrain model on historical data."""
        from alpha.ml.models import DirectionalClassifier, ModelConfig
        
        logger.info(f"Retraining model from {train_start.date()} to {train_end.date()}")
        
        # Build training dataset
        all_features = []
        all_labels = []
        
        for symbol in symbols[:10]:  # Limit symbols for speed
            X, y, _ = feature_engine.build_training_dataset(
                symbol,
                train_start,
                train_end,
                forward_days=5,
            )
            
            if len(X) > 0:
                all_features.append(X)
                all_labels.append(y)
        
        if not all_features:
            logger.warning("No training data available")
            return model
        
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        # Create fresh model and train
        if hasattr(model, 'config'):
            new_model = DirectionalClassifier(model.config)
        else:
            new_model = DirectionalClassifier()
        
        try:
            new_model.fit(X, y, feature_names=feature_engine.FEATURE_NAMES)
            return new_model
        except Exception as e:
            logger.warning(f"Training failed: {e}")
            return model
    
    def _get_price(
        self,
        symbol: str,
        date: datetime,
        price_data: Dict[str, pd.DataFrame],
    ) -> Optional[float]:
        """Get price for symbol on date."""
        if symbol not in price_data:
            return None
        
        df = price_data[symbol]
        
        # Find closest date
        for d in [date, date - timedelta(days=1), date - timedelta(days=2)]:
            try:
                idx = df.index.get_indexer([d], method='nearest')[0]
                if idx >= 0:
                    return float(df['Close'].iloc[idx])
            except Exception:
                continue
        
        return None
    
    def _calculate_position_value(
        self,
        positions: Dict[str, Trade],
        price_data: Dict[str, pd.DataFrame],
        date: datetime,
    ) -> float:
        """Calculate total position value."""
        total = 0.0
        
        for symbol, trade in positions.items():
            price = self._get_price(symbol, date, price_data)
            if price:
                if trade.direction == "BUY":
                    total += price * trade.quantity
                else:
                    # Short position
                    total += (2 * trade.entry_price - price) * trade.quantity
        
        return total
    
    def _check_exits(
        self,
        positions: Dict[str, Trade],
        price_data: Dict[str, pd.DataFrame],
        current_date: datetime,
    ) -> List[Trade]:
        """Check positions for exit conditions."""
        closed = []
        
        for symbol, trade in list(positions.items()):
            price = self._get_price(symbol, current_date, price_data)
            if price is None:
                continue
            
            # Calculate current return
            if trade.direction == "BUY":
                current_return = (price - trade.entry_price) / trade.entry_price
            else:
                current_return = (trade.entry_price - price) / trade.entry_price
            
            hold_days = (current_date - trade.entry_date).days
            should_exit = False
            
            # Stop loss
            if current_return <= -self.config.stop_loss_pct:
                should_exit = True
                trade.hit_stop = True
            
            # Take profit
            if current_return >= self.config.take_profit_pct:
                should_exit = True
                trade.hit_target = True
            
            # Max hold time
            if hold_days >= self.config.max_hold_days:
                should_exit = True
            
            if should_exit:
                # Apply slippage on exit
                if trade.direction == "BUY":
                    exit_price = price * (1 - self.config.slippage_pct)
                else:
                    exit_price = price * (1 + self.config.slippage_pct)
                
                trade.exit_date = current_date
                trade.exit_price = exit_price
                trade.hold_days = hold_days
                trade.actual_return = current_return
                trade.pnl = (exit_price - trade.entry_price) * trade.quantity * (1 if trade.direction == "BUY" else -1)
                
                closed.append(trade)
        
        return closed
    
    def _get_future_return(
        self,
        symbol: str,
        date: datetime,
        price_data: Dict[str, pd.DataFrame],
        days: int,
    ) -> float:
        """Get actual return over next N days."""
        current_price = self._get_price(symbol, date, price_data)
        future_price = self._get_price(symbol, date + timedelta(days=days), price_data)
        
        if current_price and future_price:
            return (future_price - current_price) / current_price
        return 0.0
    
    def _calculate_metrics(
        self,
        result: BacktestResult,
        predictions: List[float],
        actuals: List[float],
    ) -> BacktestResult:
        """Calculate all performance metrics."""
        trades = result.trades
        equity_curve = result.equity_curve
        
        if not trades:
            return result
        
        # Basic trade stats
        result.total_trades = len(trades)
        
        winning = [t for t in trades if t.actual_return > 0]
        losing = [t for t in trades if t.actual_return <= 0]
        
        result.winning_trades = len(winning)
        result.losing_trades = len(losing)
        result.win_rate = len(winning) / len(trades) if trades else 0
        
        result.avg_win = np.mean([t.actual_return for t in winning]) if winning else 0
        result.avg_loss = np.mean([t.actual_return for t in losing]) if losing else 0
        result.avg_hold_days = np.mean([t.hold_days for t in trades])
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Portfolio metrics
        if equity_curve:
            equities = [e[1] for e in equity_curve]
            returns = np.diff(equities) / equities[:-1]
            
            result.total_return = (equities[-1] - equities[0]) / equities[0]
            
            # Annualized return
            days = (result.end_date - result.start_date).days
            result.annualized_return = (1 + result.total_return) ** (365 / days) - 1 if days > 0 else 0
            
            # Sharpe ratio (assuming 0% risk-free rate)
            if len(returns) > 1 and np.std(returns) > 0:
                result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 1:
                downside_std = np.std(downside_returns)
                result.sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
            
            # Max drawdown
            peak = equities[0]
            max_dd = 0
            for equity in equities:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
            result.max_drawdown = max_dd
        
        # Model evaluation
        if predictions and actuals and len(predictions) == len(actuals):
            # Direction accuracy
            pred_directions = [1 if p > 0 else -1 for p in predictions]
            actual_directions = [1 if a > 0 else -1 for a in actuals]
            correct = sum(1 for p, a in zip(pred_directions, actual_directions) if p == a)
            result.direction_accuracy = correct / len(predictions)
            
            # Prediction correlation
            if np.std(predictions) > 0 and np.std(actuals) > 0:
                result.prediction_correlation = np.corrcoef(predictions, actuals)[0, 1]
        
        return result
