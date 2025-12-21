"""
Unit Tests for Elite Backtest Engine

Tests cover:
- HistoricalSnapshotGenerator (snapshot generation from price data)
- EliteBacktestEngine (core backtesting functionality)
- Metrics calculations (Sharpe, Sortino, drawdown, etc.)
- Position management (open/close positions)
- Monte Carlo simulation

Run with: pytest tests/test_elite_backtest_engine.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.elite_backtest_engine import (
    EliteBacktestConfig,
    EliteBacktestResults,
    EliteBacktestEngine,
    HistoricalSnapshotGenerator,
    SimulatedTrade,
    AssetType,
    print_elite_results,
    run_elite_backtest,
)
from schemas.core_schemas import (
    DirectionEnum,
    HedgeSnapshot,
    LiquiditySnapshot,
    SentimentSnapshot,
    ElasticitySnapshot,
    PipelineResult,
    TradeIdea,
    StrategyType,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_history():
    """Create sample historical price data."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.015, 100)
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 - np.random.uniform(0, 0.01, 100)),
        'high': prices * (1 + np.random.uniform(0, 0.02, 100)),
        'low': prices * (1 - np.random.uniform(0, 0.02, 100)),
        'close': prices,
        'volume': np.random.randint(1_000_000, 10_000_000, 100),
    })
    return df


@pytest.fixture
def sample_bar(sample_history):
    """Create a sample bar from history."""
    return sample_history.iloc[-1].to_dict()


@pytest.fixture
def snapshot_generator():
    """Create a HistoricalSnapshotGenerator instance."""
    return HistoricalSnapshotGenerator(lookback=50)


@pytest.fixture
def backtest_config():
    """Create a sample backtest configuration."""
    return EliteBacktestConfig(
        symbols=["TEST"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_capital=100_000.0,
        max_positions=3,
        max_position_pct=0.04,
        min_confidence=0.30,
        use_agent_signals=False,  # Use simple consensus for testing
        disable_event_risk=True,
        monte_carlo_runs=100,  # Fewer runs for faster tests
    )


@pytest.fixture
def mock_alpaca_data(sample_history):
    """Mock Alpaca data adapter."""
    with patch('backtesting.elite_backtest_engine.EliteBacktestEngine.fetch_historical_data') as mock:
        mock.return_value = sample_history.copy()
        yield mock


# =============================================================================
# HISTORICAL SNAPSHOT GENERATOR TESTS
# =============================================================================

class TestHistoricalSnapshotGenerator:
    """Tests for HistoricalSnapshotGenerator."""
    
    def test_init(self):
        """Test generator initialization."""
        gen = HistoricalSnapshotGenerator(lookback=30)
        assert gen.lookback == 30
    
    def test_compute_hedge_snapshot_basic(self, snapshot_generator, sample_history, sample_bar):
        """Test hedge snapshot computation."""
        timestamp = datetime.now(timezone.utc)
        
        snapshot = snapshot_generator.compute_hedge_snapshot(
            bar=sample_bar,
            history=sample_history,
            timestamp=timestamp,
            symbol="TEST"
        )
        
        assert isinstance(snapshot, HedgeSnapshot)
        assert snapshot.symbol == "TEST"
        assert snapshot.timestamp == timestamp
        assert 0.0 <= snapshot.elasticity <= 1.0
        assert -1.0 <= snapshot.energy_asymmetry <= 1.0
        assert 0.0 <= snapshot.pressure_up <= 1.0
        assert 0.0 <= snapshot.pressure_down <= 1.0
        assert snapshot.regime in ["trending", "volatile", "compressed", "neutral"]
    
    def test_compute_hedge_snapshot_insufficient_data(self, snapshot_generator, sample_bar):
        """Test hedge snapshot with insufficient history."""
        short_history = pd.DataFrame({
            'close': [100, 101, 102],
            'open': [99, 100, 101],
            'high': [101, 102, 103],
            'low': [98, 99, 100],
            'volume': [1000000, 1000000, 1000000],
        })
        timestamp = datetime.now(timezone.utc)
        
        snapshot = snapshot_generator.compute_hedge_snapshot(
            bar=sample_bar,
            history=short_history,
            timestamp=timestamp,
            symbol="TEST"
        )
        
        # Should return default values
        assert snapshot.symbol == "TEST"
        assert snapshot.elasticity == 0.0
    
    def test_compute_liquidity_snapshot(self, snapshot_generator, sample_history, sample_bar):
        """Test liquidity snapshot computation."""
        timestamp = datetime.now(timezone.utc)
        
        snapshot = snapshot_generator.compute_liquidity_snapshot(
            bar=sample_bar,
            history=sample_history,
            timestamp=timestamp,
            symbol="TEST"
        )
        
        assert isinstance(snapshot, LiquiditySnapshot)
        assert snapshot.symbol == "TEST"
        assert 0.0 <= snapshot.liquidity_score <= 1.0
        assert snapshot.bid_ask_spread >= 0
        assert snapshot.volume > 0
        assert snapshot.impact_cost >= 0
    
    def test_compute_sentiment_snapshot(self, snapshot_generator, sample_history, sample_bar):
        """Test sentiment snapshot computation."""
        timestamp = datetime.now(timezone.utc)
        
        snapshot = snapshot_generator.compute_sentiment_snapshot(
            bar=sample_bar,
            history=sample_history,
            timestamp=timestamp,
            symbol="TEST"
        )
        
        assert isinstance(snapshot, SentimentSnapshot)
        assert snapshot.symbol == "TEST"
        assert -1.0 <= snapshot.sentiment_score <= 1.0
        assert -1.0 <= snapshot.technical_sentiment <= 1.0
        assert -1.0 <= snapshot.flow_sentiment <= 1.0
        assert 0.0 <= snapshot.confidence <= 1.0
    
    def test_compute_elasticity_snapshot(self, snapshot_generator, sample_history, sample_bar):
        """Test elasticity snapshot computation."""
        timestamp = datetime.now(timezone.utc)
        
        snapshot = snapshot_generator.compute_elasticity_snapshot(
            bar=sample_bar,
            history=sample_history,
            timestamp=timestamp,
            symbol="TEST"
        )
        
        assert isinstance(snapshot, ElasticitySnapshot)
        assert snapshot.symbol == "TEST"
        assert snapshot.volatility >= 0
        assert 0.0 <= snapshot.trend_strength <= 1.0
        assert snapshot.volatility_regime in ["high_volatility", "moderate", "low_volatility", "compressed"]
    
    def test_compute_atr(self, snapshot_generator, sample_history):
        """Test ATR computation."""
        atr = snapshot_generator.compute_atr(sample_history, period=14)
        
        assert atr > 0
        # ATR should be reasonable relative to price
        avg_price = sample_history['close'].mean()
        assert atr < avg_price * 0.1  # ATR should be less than 10% of price
    
    def test_compute_atr_insufficient_data(self, snapshot_generator):
        """Test ATR with insufficient data."""
        short_history = pd.DataFrame({
            'close': [100, 101],
            'high': [101, 102],
            'low': [99, 100],
        })
        
        atr = snapshot_generator.compute_atr(short_history, period=14)
        assert atr == 0.0


# =============================================================================
# ELITE BACKTEST CONFIG TESTS
# =============================================================================

class TestEliteBacktestConfig:
    """Tests for EliteBacktestConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = EliteBacktestConfig()
        
        assert config.symbols == ["SPY"]
        assert config.initial_capital == 100_000.0
        assert config.max_positions == 5
        assert config.max_position_pct == 0.04
        assert config.min_confidence == 0.30
        assert config.monte_carlo_runs == 1000
        assert config.disable_event_risk == True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = EliteBacktestConfig(
            symbols=["AAPL", "GOOGL"],
            initial_capital=50_000.0,
            max_positions=10,
            min_confidence=0.50,
        )
        
        assert config.symbols == ["AAPL", "GOOGL"]
        assert config.initial_capital == 50_000.0
        assert config.max_positions == 10
        assert config.min_confidence == 0.50


# =============================================================================
# SIMULATED TRADE TESTS
# =============================================================================

class TestSimulatedTrade:
    """Tests for SimulatedTrade dataclass."""
    
    def test_default_values(self):
        """Test default trade values."""
        trade = SimulatedTrade()
        
        assert trade.trade_id == ""
        assert trade.symbol == ""
        assert trade.asset_type == AssetType.EQUITY
        assert trade.direction == "long"
        assert trade.is_winner == False
        assert trade.r_multiple == 0.0
    
    def test_custom_trade(self):
        """Test custom trade values."""
        entry_date = datetime.now(timezone.utc)
        exit_date = entry_date + timedelta(days=5)
        
        trade = SimulatedTrade(
            trade_id="T00001",
            symbol="AAPL",
            entry_date=entry_date,
            exit_date=exit_date,
            direction="long",
            entry_price=150.0,
            exit_price=160.0,
            position_size=10.0,
            net_pnl=100.0,
            is_winner=True,
            r_multiple=1.5,
        )
        
        assert trade.trade_id == "T00001"
        assert trade.symbol == "AAPL"
        assert trade.entry_price == 150.0
        assert trade.exit_price == 160.0
        assert trade.is_winner == True
        assert trade.r_multiple == 1.5


# =============================================================================
# ELITE BACKTEST ENGINE TESTS
# =============================================================================

class TestEliteBacktestEngine:
    """Tests for EliteBacktestEngine."""
    
    def test_init(self, backtest_config):
        """Test engine initialization."""
        engine = EliteBacktestEngine(backtest_config)
        
        assert engine.capital == backtest_config.initial_capital
        assert engine.positions == {}
        assert engine.trades == []
        assert engine.equity_curve == []
        assert engine.trade_counter == 0
    
    def test_build_pipeline_result(self, backtest_config, sample_history, sample_bar):
        """Test pipeline result building."""
        engine = EliteBacktestEngine(backtest_config)
        timestamp = datetime.now(timezone.utc)
        
        pipeline = engine._build_pipeline_result(
            symbol="TEST",
            bar=sample_bar,
            history=sample_history,
            timestamp=timestamp,
        )
        
        assert isinstance(pipeline, PipelineResult)
        assert pipeline.symbol == "TEST"
        assert pipeline.hedge_snapshot is not None
        assert pipeline.liquidity_snapshot is not None
        assert pipeline.sentiment_snapshot is not None
        assert pipeline.elasticity_snapshot is not None
        assert 'direction' in pipeline.consensus
        assert 'confidence' in pipeline.consensus
    
    def test_check_position_exit_stop_loss(self, backtest_config):
        """Test position exit on stop loss."""
        engine = EliteBacktestEngine(backtest_config)
        
        position = SimulatedTrade(
            direction="long",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            atr=2.0,
        )
        
        # Price below stop loss
        should_exit, reason = engine._check_position_exit(
            position=position,
            current_price=94.0,
            timestamp=datetime.now(timezone.utc),
            consensus={"direction": "bullish", "confidence": 0.5},
        )
        
        assert should_exit == True
        assert reason == "stop_loss"
    
    def test_check_position_exit_take_profit(self, backtest_config):
        """Test position exit on take profit."""
        engine = EliteBacktestEngine(backtest_config)
        
        position = SimulatedTrade(
            direction="long",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            atr=2.0,
        )
        
        # Price above take profit
        should_exit, reason = engine._check_position_exit(
            position=position,
            current_price=112.0,
            timestamp=datetime.now(timezone.utc),
            consensus={"direction": "bullish", "confidence": 0.5},
        )
        
        assert should_exit == True
        assert reason == "take_profit"
    
    def test_check_position_exit_signal_reversal(self, backtest_config):
        """Test position exit on signal reversal."""
        engine = EliteBacktestEngine(backtest_config)
        
        position = SimulatedTrade(
            direction="long",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            atr=2.0,
        )
        
        # Strong bearish signal (reversal)
        should_exit, reason = engine._check_position_exit(
            position=position,
            current_price=102.0,
            timestamp=datetime.now(timezone.utc),
            consensus={"direction": "bearish", "confidence": 0.7},
        )
        
        assert should_exit == True
        assert reason == "signal_reversal"
    
    def test_check_position_exit_no_exit(self, backtest_config):
        """Test position stays open when no exit conditions met."""
        engine = EliteBacktestEngine(backtest_config)
        
        position = SimulatedTrade(
            direction="long",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            atr=2.0,
        )
        
        # Price in middle, no reversal
        should_exit, reason = engine._check_position_exit(
            position=position,
            current_price=103.0,
            timestamp=datetime.now(timezone.utc),
            consensus={"direction": "bullish", "confidence": 0.5},
        )
        
        assert should_exit == False
        assert reason == ""
    
    def test_open_position(self, backtest_config, sample_history, sample_bar):
        """Test opening a position."""
        engine = EliteBacktestEngine(backtest_config)
        timestamp = datetime.now(timezone.utc)
        
        # Build pipeline result
        pipeline = engine._build_pipeline_result(
            symbol="TEST",
            bar=sample_bar,
            history=sample_history,
            timestamp=timestamp,
        )
        
        # Create trade idea
        trade_idea = TradeIdea(
            symbol="TEST",
            direction=DirectionEnum.LONG,
            strategy_type=StrategyType.DIRECTIONAL,
            confidence=0.6,
            timestamp=timestamp,
        )
        
        # Open position
        initial_capital = engine.capital
        trade = engine._open_position(
            symbol="TEST",
            trade_idea=trade_idea,
            current_price=sample_bar['close'],
            timestamp=timestamp,
            pipeline_result=pipeline,
        )
        
        assert trade is not None
        assert trade.symbol == "TEST"
        assert trade.direction == "long"
        assert trade.entry_price > 0
        assert trade.stop_loss > 0
        assert trade.take_profit > 0
        assert "TEST" in engine.positions
        assert engine.capital < initial_capital  # Capital deducted
    
    def test_open_position_max_positions_reached(self, backtest_config, sample_history, sample_bar):
        """Test that no position opens when max positions reached."""
        backtest_config.max_positions = 1
        engine = EliteBacktestEngine(backtest_config)
        timestamp = datetime.now(timezone.utc)
        
        # Add existing position
        engine.positions["EXISTING"] = SimulatedTrade(symbol="EXISTING")
        
        pipeline = engine._build_pipeline_result(
            symbol="TEST",
            bar=sample_bar,
            history=sample_history,
            timestamp=timestamp,
        )
        
        trade_idea = TradeIdea(
            symbol="TEST",
            direction=DirectionEnum.LONG,
            strategy_type=StrategyType.DIRECTIONAL,
            confidence=0.6,
            timestamp=timestamp,
        )
        
        trade = engine._open_position(
            symbol="TEST",
            trade_idea=trade_idea,
            current_price=sample_bar['close'],
            timestamp=timestamp,
            pipeline_result=pipeline,
        )
        
        assert trade is None
        assert "TEST" not in engine.positions
    
    def test_close_position(self, backtest_config, sample_history, sample_bar):
        """Test closing a position."""
        engine = EliteBacktestEngine(backtest_config)
        timestamp = datetime.now(timezone.utc)
        
        # Add a position
        entry_price = 100.0
        position = SimulatedTrade(
            trade_id="T00001",
            symbol="TEST",
            entry_date=timestamp - timedelta(days=5),
            direction="long",
            entry_price=entry_price,
            position_size=40.0,  # $4000 / $100
            stop_loss=95.0,
            take_profit=110.0,
            atr=2.0,
            entry_cost=2.0,
        )
        engine.positions["TEST"] = position
        engine.capital = 96_000.0  # $4000 allocated
        
        # Close with profit
        exit_price = 105.0
        closed_trade = engine._close_position(
            symbol="TEST",
            current_price=exit_price,
            timestamp=timestamp,
            exit_reason="take_profit",
        )
        
        assert closed_trade is not None
        assert closed_trade.exit_date == timestamp
        assert closed_trade.exit_reason == "take_profit"
        assert closed_trade.net_pnl > 0
        assert closed_trade.is_winner == True
        assert "TEST" not in engine.positions
        assert len(engine.trades) == 1
        assert engine.capital > 96_000.0  # Capital returned with profit
    
    def test_record_equity(self, backtest_config, sample_bar):
        """Test equity curve recording."""
        engine = EliteBacktestEngine(backtest_config)
        timestamp = datetime.now(timezone.utc)
        
        # Record equity
        engine._record_equity(timestamp, {"TEST": sample_bar})
        
        assert len(engine.equity_curve) == 1
        assert 'timestamp' in engine.equity_curve[0]
        assert 'equity' in engine.equity_curve[0]
        assert 'capital' in engine.equity_curve[0]
        assert engine.equity_curve[0]['equity'] == backtest_config.initial_capital


# =============================================================================
# METRICS CALCULATION TESTS
# =============================================================================

class TestMetricsCalculation:
    """Tests for metrics calculations in results."""
    
    def test_calculate_results_empty(self, backtest_config):
        """Test results calculation with no trades."""
        engine = EliteBacktestEngine(backtest_config)
        
        # Add minimal equity curve
        engine.equity_curve = [
            {'timestamp': '2023-01-01', 'equity': 100000},
            {'timestamp': '2023-01-02', 'equity': 100100},
        ]
        
        results = engine._calculate_results()
        
        assert results.total_trades == 0
        assert results.win_rate == 0.0
        assert results.final_capital == backtest_config.initial_capital
    
    def test_calculate_results_with_trades(self, backtest_config):
        """Test results calculation with trades."""
        engine = EliteBacktestEngine(backtest_config)
        
        # Add trades
        engine.trades = [
            SimulatedTrade(net_pnl=100, is_winner=True, hold_time_hours=24, strategy="directional", symbol="TEST", r_multiple=1.0, entry_cost=2, exit_cost=2),
            SimulatedTrade(net_pnl=-50, is_winner=False, hold_time_hours=12, strategy="directional", symbol="TEST", r_multiple=-0.5, entry_cost=2, exit_cost=2),
            SimulatedTrade(net_pnl=75, is_winner=True, hold_time_hours=48, strategy="directional", symbol="TEST", r_multiple=0.8, entry_cost=2, exit_cost=2),
        ]
        
        # Add equity curve
        engine.equity_curve = [
            {'timestamp': '2023-01-01', 'equity': 100000},
            {'timestamp': '2023-01-02', 'equity': 100100},
            {'timestamp': '2023-01-03', 'equity': 100050},
            {'timestamp': '2023-01-04', 'equity': 100125},
        ]
        
        engine.capital = 100125
        
        results = engine._calculate_results()
        
        assert results.total_trades == 3
        assert results.winning_trades == 2
        assert results.losing_trades == 1
        assert results.win_rate == 2/3
        assert results.profit_factor == 175/50  # 175 gains / 50 losses
        assert results.avg_win == 87.5  # (100 + 75) / 2
        assert results.avg_loss == 50.0
    
    def test_calculate_sharpe_ratio(self, backtest_config):
        """Test Sharpe ratio calculation."""
        engine = EliteBacktestEngine(backtest_config)
        
        # Create equity curve with positive returns
        base = 100000
        returns = [0.01, 0.02, -0.005, 0.015, 0.01]  # ~3% gain
        equity = [base]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        
        engine.equity_curve = [
            {'timestamp': f'2023-01-{i+1:02d}', 'equity': e}
            for i, e in enumerate(equity)
        ]
        engine.capital = equity[-1]
        
        results = engine._calculate_results()
        
        assert results.sharpe_ratio > 0  # Should be positive with positive returns
        assert results.volatility > 0
    
    def test_calculate_max_drawdown(self, backtest_config):
        """Test max drawdown calculation."""
        engine = EliteBacktestEngine(backtest_config)
        
        # Create equity curve with drawdown
        engine.equity_curve = [
            {'timestamp': '2023-01-01', 'equity': 100000},
            {'timestamp': '2023-01-02', 'equity': 105000},  # Peak
            {'timestamp': '2023-01-03', 'equity': 100000},  # -4.76%
            {'timestamp': '2023-01-04', 'equity': 95000},   # -9.52% from peak
            {'timestamp': '2023-01-05', 'equity': 98000},
        ]
        engine.capital = 98000
        
        results = engine._calculate_results()
        
        # Max drawdown should be ~9.52% (105000 -> 95000)
        assert results.max_drawdown == 10000  # Dollar drawdown
        assert abs(results.max_drawdown_pct - 0.0952) < 0.01  # Percentage
    
    def test_calculate_streaks(self, backtest_config):
        """Test win/loss streak calculation."""
        engine = EliteBacktestEngine(backtest_config)
        
        # Create trades with streaks: W, W, W, L, L, W
        engine.trades = [
            SimulatedTrade(is_winner=True, net_pnl=10, strategy="directional", symbol="TEST", r_multiple=0.5, entry_cost=1, exit_cost=1, hold_time_hours=24),
            SimulatedTrade(is_winner=True, net_pnl=10, strategy="directional", symbol="TEST", r_multiple=0.5, entry_cost=1, exit_cost=1, hold_time_hours=24),
            SimulatedTrade(is_winner=True, net_pnl=10, strategy="directional", symbol="TEST", r_multiple=0.5, entry_cost=1, exit_cost=1, hold_time_hours=24),
            SimulatedTrade(is_winner=False, net_pnl=-5, strategy="directional", symbol="TEST", r_multiple=-0.25, entry_cost=1, exit_cost=1, hold_time_hours=24),
            SimulatedTrade(is_winner=False, net_pnl=-5, strategy="directional", symbol="TEST", r_multiple=-0.25, entry_cost=1, exit_cost=1, hold_time_hours=24),
            SimulatedTrade(is_winner=True, net_pnl=10, strategy="directional", symbol="TEST", r_multiple=0.5, entry_cost=1, exit_cost=1, hold_time_hours=24),
        ]
        
        engine.equity_curve = [{'timestamp': '2023-01-01', 'equity': 100000}]
        engine.capital = 100020
        
        results = engine._calculate_results()
        
        assert results.max_consecutive_wins == 3
        assert results.max_consecutive_losses == 2


# =============================================================================
# MONTE CARLO TESTS
# =============================================================================

class TestMonteCarlo:
    """Tests for Monte Carlo simulation."""
    
    def test_monte_carlo_runs(self, backtest_config):
        """Test Monte Carlo simulation runs."""
        backtest_config.monte_carlo_runs = 100
        engine = EliteBacktestEngine(backtest_config)
        
        # Create results with returns
        results = EliteBacktestResults()
        results.daily_returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.005] * 10  # 60 days
        
        engine._run_monte_carlo(results)
        
        assert results.mc_median_return != 0
        assert results.mc_var_5 < results.mc_median_return < results.mc_var_95
        assert 0 <= results.mc_prob_profit <= 1
    
    def test_monte_carlo_empty_returns(self, backtest_config):
        """Test Monte Carlo with empty returns."""
        engine = EliteBacktestEngine(backtest_config)
        
        results = EliteBacktestResults()
        results.daily_returns = []
        
        # Should not crash
        engine._run_monte_carlo(results)
        
        # Values should remain at default (0)
        assert results.mc_median_return == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestBacktestIntegration:
    """Integration tests for full backtest runs."""
    
    def test_full_backtest_with_mock_data(self, backtest_config, sample_history):
        """Test full backtest run with mocked data."""
        engine = EliteBacktestEngine(backtest_config)
        
        # Mock fetch_historical_data
        with patch.object(engine, 'fetch_historical_data', return_value=sample_history):
            results = engine.run_backtest()
        
        assert isinstance(results, EliteBacktestResults)
        assert results.total_bars > 0
        assert results.initial_capital == backtest_config.initial_capital
        assert len(results.equity_curve) > 0
    
    def test_backtest_config_integration(self):
        """Test configuration is properly applied."""
        config = EliteBacktestConfig(
            symbols=["TEST"],
            initial_capital=50_000.0,
            max_positions=2,
            min_confidence=0.50,
            monte_carlo_runs=50,
        )
        
        engine = EliteBacktestEngine(config)
        
        assert engine.capital == 50_000.0
        assert engine.config.max_positions == 2
        assert engine.config.min_confidence == 0.50


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_print_elite_results(self, capsys):
        """Test results printing (just check it doesn't crash)."""
        results = EliteBacktestResults(
            config=EliteBacktestConfig(symbols=["TEST"]),
            start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2023, 12, 31, tzinfo=timezone.utc),
            total_bars=250,
            initial_capital=100_000,
            final_capital=110_000,
            total_return=10_000,
            total_return_pct=0.10,
            cagr=0.10,
            total_trades=50,
            win_rate=0.55,
            profit_factor=1.5,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown_pct=0.08,
            strategy_returns={"directional": 10000},
            symbol_returns={"TEST": 10000},
        )
        
        print_elite_results(results)
        
        captured = capsys.readouterr()
        assert "ELITE BACKTEST RESULTS" in captured.out
        assert "TEST" in captured.out
        assert "10,000" in captured.out


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_short_position_handling(self, backtest_config):
        """Test short position stop loss and take profit."""
        engine = EliteBacktestEngine(backtest_config)
        
        position = SimulatedTrade(
            direction="short",
            entry_price=100.0,
            stop_loss=105.0,  # Stop loss above for short
            take_profit=90.0,  # Take profit below for short
            atr=2.0,
        )
        
        # Price hits stop loss (goes up)
        should_exit, reason = engine._check_position_exit(
            position=position,
            current_price=106.0,
            timestamp=datetime.now(timezone.utc),
            consensus={"direction": "bearish", "confidence": 0.5},
        )
        
        assert should_exit == True
        assert reason == "stop_loss"
        
        # Reset and test take profit
        should_exit, reason = engine._check_position_exit(
            position=position,
            current_price=89.0,
            timestamp=datetime.now(timezone.utc),
            consensus={"direction": "bearish", "confidence": 0.5},
        )
        
        assert should_exit == True
        assert reason == "take_profit"
    
    def test_zero_volume_handling(self, snapshot_generator, sample_bar):
        """Test handling of zero volume data."""
        history = pd.DataFrame({
            'close': [100.0] * 50,
            'open': [99.0] * 50,
            'high': [101.0] * 50,
            'low': [98.0] * 50,
            'volume': [0] * 50,  # Zero volume
        })
        
        timestamp = datetime.now(timezone.utc)
        
        # Should not crash
        snapshot = snapshot_generator.compute_hedge_snapshot(
            bar=sample_bar,
            history=history,
            timestamp=timestamp,
            symbol="TEST"
        )
        
        assert snapshot is not None
    
    def test_negative_pnl_tracking(self, backtest_config):
        """Test that losing trades are properly tracked."""
        engine = EliteBacktestEngine(backtest_config)
        
        # Add losing trades
        engine.trades = [
            SimulatedTrade(net_pnl=-100, is_winner=False, strategy="directional", symbol="TEST", r_multiple=-1.0, entry_cost=2, exit_cost=2, hold_time_hours=24),
            SimulatedTrade(net_pnl=-200, is_winner=False, strategy="directional", symbol="TEST", r_multiple=-2.0, entry_cost=2, exit_cost=2, hold_time_hours=24),
        ]
        
        engine.equity_curve = [
            {'timestamp': '2023-01-01', 'equity': 100000},
            {'timestamp': '2023-01-02', 'equity': 99700},
        ]
        engine.capital = 99700
        
        results = engine._calculate_results()
        
        assert results.total_return < 0
        assert results.win_rate == 0.0
        assert results.avg_loss == 150.0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
