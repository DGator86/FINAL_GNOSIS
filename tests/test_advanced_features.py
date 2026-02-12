"""
Tests for Advanced Features

Covers:
- Options Flow Scanner
- Earnings Calendar
- Volatility Surface
- Greeks Hedging
- RL Agent
- Transformer Predictor
- Anomaly Detection
- Trading Dashboard
- Telegram Bot
- Portfolio Analytics

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

import pytest
import sys
import importlib.util
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio


# Helper to load modules without triggering dashboard.py import conflict
def load_module_direct(module_name, file_path):
    """Load module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Register in sys.modules first
    spec.loader.exec_module(module)
    return module

# Pre-load dashboard modules to avoid conflicts
_trading_dashboard = load_module_direct(
    "dashboard.trading_dashboard",
    "/home/root/webapp/dashboard/trading_dashboard.py"
)
_portfolio_analytics = load_module_direct(
    "dashboard.portfolio_analytics",
    "/home/root/webapp/dashboard/portfolio_analytics.py"
)


# =============================================================================
# Options Flow Scanner Tests
# =============================================================================

class TestOptionsFlowScanner:
    """Test Options Flow Scanner."""
    
    def test_option_trade_creation(self):
        """Test OptionTrade dataclass."""
        from scanner.options_flow_scanner import OptionTrade, TradeAggressor
        
        trade = OptionTrade(
            symbol="AAPL230120C00180000",
            underlying="AAPL",
            strike=180.0,
            expiration=datetime(2023, 1, 20),
            option_type="call",
            price=5.50,
            size=200,
            premium=110000,  # 200 * 5.50 * 100
            bid=5.40,
            ask=5.60,
            underlying_price=175.0,
            timestamp=datetime.now(),
            exchange="CBOE",
            trade_id="trade123",
        )
        
        assert trade.symbol == "AAPL230120C00180000"
        assert trade.is_block is True  # 200 contracts or $110k premium
        assert trade.moneyness == "OTM"  # 175 < 180
        
    def test_trade_aggressor_detection(self):
        """Test aggressor detection."""
        from scanner.options_flow_scanner import OptionTrade, TradeAggressor
        
        # Buyer aggressor (at ask)
        buy_trade = OptionTrade(
            symbol="SPY_CALL",
            underlying="SPY",
            strike=500,
            expiration=datetime(2024, 1, 19),
            option_type="call",
            price=5.00,
            size=100,
            premium=50000,
            bid=4.90,
            ask=5.00,
            underlying_price=495.0,
            timestamp=datetime.now(),
            exchange="ISE",
            trade_id="t1",
        )
        
        assert buy_trade.aggressor == TradeAggressor.BUYER
        
    def test_flow_scanner_initialization(self):
        """Test scanner initialization."""
        from scanner.options_flow_scanner import OptionsFlowScanner, FlowScannerConfig
        
        config = FlowScannerConfig(
            block_size_threshold=50,
            block_premium_threshold=50000,
        )
        
        scanner = OptionsFlowScanner(config)
        
        assert scanner.config.block_size_threshold == 50
        
    @pytest.mark.asyncio
    async def test_process_block_trade(self):
        """Test block trade processing."""
        from scanner.options_flow_scanner import (
            OptionsFlowScanner, OptionTrade, FlowType
        )
        
        scanner = OptionsFlowScanner()
        
        trade = OptionTrade(
            symbol="MSFT_CALL",
            underlying="MSFT",
            strike=400,
            expiration=datetime(2024, 3, 15),
            option_type="call",
            price=10.0,
            size=500,  # Block trade
            premium=500000,
            bid=9.90,
            ask=10.10,
            underlying_price=390.0,
            timestamp=datetime.now(),
            exchange="AMEX",
            trade_id="block1",
        )
        
        alert = await scanner.process_trade(trade)
        
        assert alert is not None
        assert alert.flow_type == FlowType.BLOCK
        assert alert.total_contracts == 500
        
    def test_flow_alert_scoring(self):
        """Test alert scoring logic."""
        from scanner.options_flow_scanner import (
            OptionsFlowScanner, FlowType, TradeAggressor
        )
        
        scanner = OptionsFlowScanner()
        
        # High premium, clear direction, short DTE
        score = scanner._calculate_alert_score(
            premium=500000,
            contracts=500,
            flow_type=FlowType.SWEEP,
            aggressor=TradeAggressor.BUYER,
            moneyness="OTM",
            days_to_expiry=5,
        )
        
        assert score >= 70  # Should be high score
        
    def test_flow_summary(self):
        """Test flow summary generation."""
        from scanner.options_flow_scanner import OptionsFlowScanner
        
        scanner = OptionsFlowScanner()
        summary = scanner.get_flow_summary()
        
        assert "total_alerts" in summary
        assert "total_premium" in summary
        assert "sentiment_breakdown" in summary


# =============================================================================
# Earnings Calendar Tests
# =============================================================================

class TestEarningsCalendar:
    """Test Earnings Calendar."""
    
    def test_earnings_event_creation(self):
        """Test EarningsEvent dataclass."""
        from data.earnings_calendar import EarningsEvent, EarningsTime
        
        event = EarningsEvent(
            symbol="AAPL",
            company_name="Apple Inc.",
            earnings_date=date.today() + timedelta(days=7),
            earnings_time=EarningsTime.AMC,
            eps_estimate=1.50,
            revenue_estimate=100000,
        )
        
        assert event.symbol == "AAPL"
        assert event.days_until == 7
        assert event.eps_surprise is None  # No actual yet
        
    def test_earnings_surprise_calculation(self):
        """Test EPS surprise calculation."""
        from data.earnings_calendar import EarningsEvent, EarningsTime, EarningsSurprise
        
        event = EarningsEvent(
            symbol="MSFT",
            company_name="Microsoft",
            earnings_date=date.today() - timedelta(days=1),
            earnings_time=EarningsTime.AMC,
            eps_estimate=2.00,
            eps_actual=2.20,  # Beat
        )
        
        assert event.eps_surprise == EarningsSurprise.BEAT
        assert event.eps_surprise_percent == pytest.approx(10.0, rel=0.01)
        
    def test_calendar_upcoming_earnings(self):
        """Test getting upcoming earnings."""
        from data.earnings_calendar import EarningsCalendar, EarningsEvent, EarningsTime
        
        calendar = EarningsCalendar()
        
        # Add events
        for i in range(5):
            event = EarningsEvent(
                symbol=f"SYM{i}",
                company_name=f"Company {i}",
                earnings_date=date.today() + timedelta(days=i+1),
                earnings_time=EarningsTime.BMO,
            )
            calendar.add_earnings_event(event)
        
        upcoming = calendar.get_upcoming_earnings(days_ahead=3)
        
        assert len(upcoming) == 3
        
    def test_expected_move_calculation(self):
        """Test expected move calculation."""
        from data.earnings_calendar import EarningsCalendar
        
        calendar = EarningsCalendar()
        
        result = calendar.calculate_expected_move(
            symbol="AAPL",
            atm_iv=0.50,  # 50% IV
            days_to_earnings=1,
            stock_price=180.0,
        )
        
        assert "expected_move_percent" in result
        assert "upper_bound" in result
        assert "lower_bound" in result
        assert result["upper_bound"] > result["lower_bound"]
        
    def test_iv_crush_prediction(self):
        """Test IV crush prediction."""
        from data.earnings_calendar import EarningsCalendar
        
        calendar = EarningsCalendar()
        
        result = calendar.predict_iv_crush(
            symbol="TSLA",
            current_iv=0.80,
        )
        
        assert "estimated_post_iv" in result
        assert "estimated_crush_percent" in result
        assert result["estimated_post_iv"] < 0.80  # Should be lower


# =============================================================================
# Volatility Surface Tests
# =============================================================================

class TestVolatilitySurface:
    """Test Volatility Surface modeling."""
    
    def test_iv_point_creation(self):
        """Test IVPoint dataclass."""
        from models.volatility_surface import IVPoint
        
        point = IVPoint(
            strike=500.0,
            expiration=date.today() + timedelta(days=30),
            iv=0.25,
            option_type="call",
            underlying_price=495.0,
        )
        
        assert point.moneyness == pytest.approx(500.0/495.0, rel=0.01)
        assert point.days_to_expiry == 30
        
    def test_volatility_smile(self):
        """Test VolatilitySmile."""
        from models.volatility_surface import VolatilitySmile
        
        smile = VolatilitySmile(
            underlying="SPY",
            expiration=date.today() + timedelta(days=30),
            underlying_price=500.0,
            strikes=[480, 490, 500, 510, 520],
            ivs=[0.28, 0.25, 0.23, 0.24, 0.26],
            atm_strike=500.0,
            atm_iv=0.23,
        )
        
        # Test skew (puts more expensive)
        assert smile.skew > 0
        
        # Test IV interpolation
        iv_505 = smile.get_iv(505.0)
        assert 0.23 < iv_505 < 0.24
        
    def test_term_structure(self):
        """Test TermStructure."""
        from models.volatility_surface import TermStructure
        
        today = date.today()
        
        term = TermStructure(
            underlying="QQQ",
            as_of=datetime.now(),
            expirations=[
                today + timedelta(days=7),
                today + timedelta(days=30),
                today + timedelta(days=60),
            ],
            atm_ivs=[0.28, 0.25, 0.24],  # Backwardation
        )
        
        assert term.is_backwardation is True
        assert term.is_contango is False
        assert term.slope < 0  # Negative slope
        
    def test_surface_builder(self):
        """Test VolatilitySurfaceBuilder."""
        from models.volatility_surface import VolatilitySurfaceBuilder, IVPoint
        
        builder = VolatilitySurfaceBuilder()
        
        # Create IV points
        today = date.today()
        points = []
        
        for days in [7, 30, 60]:
            exp = today + timedelta(days=days)
            for strike in [480, 490, 500, 510, 520]:
                points.append(IVPoint(
                    strike=strike,
                    expiration=exp,
                    iv=0.25 + (abs(strike - 500) * 0.001),
                    option_type="call",
                ))
        
        surface = builder.build_surface("SPY", 500.0, points)
        
        assert surface.underlying == "SPY"
        assert len(surface.smiles) == 3  # 3 expirations
        assert surface.term_structure is not None
        
    def test_surface_interpolation(self):
        """Test surface IV interpolation."""
        from models.volatility_surface import VolatilitySurfaceBuilder, IVPoint
        
        builder = VolatilitySurfaceBuilder()
        
        today = date.today()
        exp = today + timedelta(days=30)
        
        points = [
            IVPoint(strike=490, expiration=exp, iv=0.26, option_type="call"),
            IVPoint(strike=500, expiration=exp, iv=0.24, option_type="call"),
            IVPoint(strike=510, expiration=exp, iv=0.25, option_type="call"),
        ]
        
        surface = builder.build_surface("SPY", 500.0, points)
        
        # Get interpolated IV
        iv = surface.get_iv(505.0, exp)
        assert 0.24 < iv < 0.25


# =============================================================================
# Greeks Hedger Tests
# =============================================================================

class TestGreeksHedger:
    """Test Greeks Hedging system."""
    
    def test_greek_exposure_creation(self):
        """Test GreekExposure dataclass."""
        from trade.greeks_hedger import GreekExposure
        
        exposure = GreekExposure(
            delta=100.0,
            gamma=5.0,
            theta=-50.0,
            vega=200.0,
        )
        
        assert exposure.delta == 100.0
        
        # Test addition
        exposure2 = GreekExposure(delta=50.0, gamma=2.0)
        combined = exposure + exposure2
        
        assert combined.delta == 150.0
        assert combined.gamma == 7.0
        
    def test_hedge_limits(self):
        """Test HedgeLimits breach detection."""
        from trade.greeks_hedger import HedgeLimits, GreekExposure
        
        limits = HedgeLimits(max_delta=200.0, max_gamma=20.0)
        
        # Within limits
        exposure1 = GreekExposure(delta=100.0, gamma=10.0)
        breaches1 = limits.check_breach(exposure1)
        assert len(breaches1) == 0
        
        # Breach delta
        exposure2 = GreekExposure(delta=300.0, gamma=10.0)
        breaches2 = limits.check_breach(exposure2)
        assert len(breaches2) == 1
        assert breaches2[0]["greek"] == "delta"
        
    def test_hedger_initialization(self):
        """Test GreeksHedger initialization."""
        from trade.greeks_hedger import GreeksHedger, HedgerConfig
        
        config = HedgerConfig(
            hedge_delta=True,
            hedge_gamma=False,
            auto_hedge=False,
        )
        
        hedger = GreeksHedger(config)
        
        assert hedger.config.hedge_delta is True
        assert hedger.config.hedge_gamma is False
        
    def test_hedge_recommendation_generation(self):
        """Test hedge recommendation generation."""
        from trade.greeks_hedger import GreeksHedger, GreekExposure, HedgeType
        
        hedger = GreeksHedger()
        
        # Create exposure that exceeds limits
        exposure = GreekExposure(delta=600.0, gamma=5.0)  # Over default 500 limit
        
        recommendations = hedger.update_exposure(exposure)
        
        assert len(recommendations) >= 1
        assert recommendations[0].hedge_type == HedgeType.DELTA
        
    def test_hedge_stats(self):
        """Test hedger statistics."""
        from trade.greeks_hedger import GreeksHedger
        
        hedger = GreeksHedger()
        stats = hedger.get_hedge_stats()
        
        assert "total_hedges" in stats
        assert "pending_recommendations" in stats


# =============================================================================
# RL Agent Tests
# =============================================================================

class TestRLAgent:
    """Test Reinforcement Learning Agent."""
    
    def test_market_state_creation(self):
        """Test MarketState dataclass."""
        from models.rl_agent import MarketState
        
        state = MarketState(
            price=100.0,
            price_change_1d=0.01,
            price_change_5d=0.03,
            price_change_20d=0.10,
            realized_vol=0.20,
            implied_vol=0.25,
            vol_ratio=1.25,
            rsi=55.0,
            macd=0.5,
            macd_signal=0.3,
            volume_ratio=1.2,
            position_size=0.0,
            position_pnl=0.0,
            position_duration=0,
            portfolio_heat=0.0,
            cash_ratio=1.0,
            day_of_week=1,
            hour_of_day=10,
        )
        
        array = state.to_array()
        assert len(array) == state.state_dim
        
    def test_dqn_agent_creation(self):
        """Test DQN agent creation."""
        from models.rl_agent import DQNAgent, RLAgentConfig
        
        config = RLAgentConfig(
            state_dim=17,
            action_dim=8,
            hidden_dims=[64, 32],
        )
        
        agent = DQNAgent(config)
        
        assert agent.config.state_dim == 17
        assert agent.epsilon == config.epsilon_start
        
    def test_dqn_action_selection(self):
        """Test DQN action selection."""
        from models.rl_agent import DQNAgent, MarketState, TradingAction
        
        agent = DQNAgent()
        agent.epsilon = 0.0  # Greedy mode
        
        state = MarketState(
            price=100.0, price_change_1d=0.01, price_change_5d=0.03,
            price_change_20d=0.10, realized_vol=0.20, implied_vol=0.25,
            vol_ratio=1.25, rsi=55.0, macd=0.5, macd_signal=0.3,
            volume_ratio=1.2, position_size=0.0, position_pnl=0.0,
            position_duration=0, portfolio_heat=0.0, cash_ratio=1.0,
            day_of_week=1, hour_of_day=10,
        )
        
        action = agent.select_action(state, training=False)
        
        assert isinstance(action, TradingAction)
        
    def test_trading_environment(self):
        """Test trading environment."""
        from models.rl_agent import TradingRLEnvironment, TradingAction
        
        env = TradingRLEnvironment(initial_capital=100000)
        
        # Generate some price history
        prices = [100 + i * 0.1 for i in range(100)]
        
        state = env.reset(prices)
        
        assert state.cash_ratio == 1.0
        
        # Take action
        next_state, reward, done = env.step(TradingAction.HOLD)
        
        assert not done
        
    def test_replay_buffer(self):
        """Test experience replay buffer."""
        from models.rl_agent import ReplayBuffer, Experience, MarketState, TradingAction
        
        buffer = ReplayBuffer(capacity=100)
        
        state = MarketState(
            price=100.0, price_change_1d=0.01, price_change_5d=0.03,
            price_change_20d=0.10, realized_vol=0.20, implied_vol=0.25,
            vol_ratio=1.25, rsi=55.0, macd=0.5, macd_signal=0.3,
            volume_ratio=1.2, position_size=0.0, position_pnl=0.0,
            position_duration=0, portfolio_heat=0.0, cash_ratio=1.0,
            day_of_week=1, hour_of_day=10,
        )
        
        exp = Experience(
            state=state,
            action=TradingAction.BUY_SMALL,
            reward=0.01,
            next_state=state,
            done=False,
        )
        
        buffer.push(exp)
        
        assert len(buffer) == 1


# =============================================================================
# Transformer Predictor Tests
# =============================================================================

class TestTransformerPredictor:
    """Test Transformer Price Predictor."""
    
    def test_price_features_creation(self):
        """Test PriceFeatures dataclass."""
        from models.transformer_predictor import PriceFeatures
        
        features = PriceFeatures(
            open_prices=[100, 101, 102],
            high_prices=[102, 103, 104],
            low_prices=[99, 100, 101],
            close_prices=[101, 102, 103],
            volumes=[1000000, 1100000, 1200000],
        )
        
        assert len(features.close_prices) == 3
        
    def test_transformer_config(self):
        """Test TransformerConfig."""
        from models.transformer_predictor import TransformerConfig
        
        config = TransformerConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
        )
        
        assert config.d_model == 64
        assert config.d_model % config.n_heads == 0
        
    def test_positional_encoding(self):
        """Test positional encoding."""
        from models.transformer_predictor import PositionalEncoding
        
        pe = PositionalEncoding(d_model=64)
        encodings = pe.encode(seq_len=10)
        
        assert len(encodings) == 10
        assert len(encodings[0]) == 64
        
    def test_transformer_predictor_creation(self):
        """Test TransformerPredictor creation."""
        from models.transformer_predictor import TransformerPredictor, TransformerConfig
        
        config = TransformerConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            sequence_length=20,
        )
        
        predictor = TransformerPredictor(config)
        
        assert predictor.config.d_model == 32
        
    def test_transformer_prediction(self):
        """Test transformer prediction."""
        from models.transformer_predictor import (
            TransformerPredictor, PriceFeatures, PredictionHorizon
        )
        
        predictor = TransformerPredictor()
        
        # Create features with enough data
        n = predictor.config.sequence_length + 10
        features = PriceFeatures(
            open_prices=[100 + i * 0.1 for i in range(n)],
            high_prices=[101 + i * 0.1 for i in range(n)],
            low_prices=[99 + i * 0.1 for i in range(n)],
            close_prices=[100 + i * 0.1 for i in range(n)],
            volumes=[1000000] * n,
        )
        
        prediction = predictor.predict("SPY", features, PredictionHorizon.HOUR_1)
        
        assert prediction.symbol == "SPY"
        assert prediction.predicted_price > 0
        assert prediction.confidence > 0


# =============================================================================
# Anomaly Detection Tests
# =============================================================================

class TestAnomalyDetector:
    """Test Anomaly Detection system."""
    
    def test_market_data_point(self):
        """Test MarketDataPoint."""
        from utils.anomaly_detector import MarketDataPoint
        
        point = MarketDataPoint(
            symbol="AAPL",
            timestamp=datetime.now(),
            price=180.0,
            volume=50000000,
            bid=179.95,
            ask=180.05,
        )
        
        assert point.symbol == "AAPL"
        
    def test_anomaly_detector_creation(self):
        """Test AnomalyDetector creation."""
        from utils.anomaly_detector import AnomalyDetector, AnomalyDetectorConfig
        
        config = AnomalyDetectorConfig(
            price_spike_zscore=3.0,
            volume_spike_multiplier=5.0,
        )
        
        detector = AnomalyDetector(config)
        
        assert detector.config.price_spike_zscore == 3.0
        
    def test_price_spike_detection(self):
        """Test price spike detection."""
        from utils.anomaly_detector import AnomalyDetector, MarketDataPoint, AnomalyType
        
        detector = AnomalyDetector()
        
        # Add normal data points
        for i in range(50):
            point = MarketDataPoint(
                symbol="TEST",
                timestamp=datetime.now() - timedelta(minutes=50-i),
                price=100.0 + (i % 3) * 0.1,  # Small variance
                volume=1000000,
                bid=99.95,
                ask=100.05,
            )
            detector.process_data(point)
        
        # Add spike
        spike = MarketDataPoint(
            symbol="TEST",
            timestamp=datetime.now(),
            price=120.0,  # Big spike
            volume=1000000,
            bid=119.95,
            ask=120.05,
        )
        
        alerts = detector.process_data(spike)
        
        # Should detect price spike
        price_alerts = [a for a in alerts if a.anomaly_type == AnomalyType.PRICE_SPIKE]
        assert len(price_alerts) >= 1
        
    def test_volume_spike_detection(self):
        """Test volume spike detection."""
        from utils.anomaly_detector import AnomalyDetector, MarketDataPoint, AnomalyType
        
        detector = AnomalyDetector()
        
        # Add normal volume data
        for i in range(50):
            point = MarketDataPoint(
                symbol="VOL_TEST",
                timestamp=datetime.now() - timedelta(minutes=50-i),
                price=100.0,
                volume=1000000,
                bid=99.95,
                ask=100.05,
            )
            detector.process_data(point)
        
        # Add volume spike
        spike = MarketDataPoint(
            symbol="VOL_TEST",
            timestamp=datetime.now(),
            price=100.0,
            volume=10000000,  # 10x normal
            bid=99.95,
            ask=100.05,
        )
        
        alerts = detector.process_data(spike)
        
        vol_alerts = [a for a in alerts if a.anomaly_type == AnomalyType.VOLUME_SPIKE]
        assert len(vol_alerts) >= 1
        
    def test_detector_stats(self):
        """Test detector statistics."""
        from utils.anomaly_detector import AnomalyDetector
        
        detector = AnomalyDetector()
        stats = detector.get_stats()
        
        assert "total_alerts" in stats
        assert "symbols_tracked" in stats


# =============================================================================
# Trading Dashboard Tests
# =============================================================================

class TestTradingDashboard:
    """Test Trading Dashboard."""
    
    def test_widget_config(self):
        """Test WidgetConfig."""
        WidgetConfig = _trading_dashboard.WidgetConfig
        WidgetType = _trading_dashboard.WidgetType
        
        config = WidgetConfig(
            widget_type=WidgetType.PORTFOLIO_SUMMARY,
            title="Portfolio",
            position={"x": 0, "y": 0, "width": 4, "height": 2},
        )
        
        assert config.widget_type == WidgetType.PORTFOLIO_SUMMARY
        
    def test_dashboard_layout(self):
        """Test DashboardLayout."""
        DashboardLayout = _trading_dashboard.DashboardLayout
        DashboardTheme = _trading_dashboard.DashboardTheme
        
        layout = DashboardLayout(
            name="Test Layout",
            theme=DashboardTheme.DARK,
        )
        
        assert layout.name == "Test Layout"
        assert layout.theme == DashboardTheme.DARK
        
    def test_dashboard_data_provider(self):
        """Test DashboardDataProvider."""
        DashboardDataProvider = _trading_dashboard.DashboardDataProvider
        
        provider = DashboardDataProvider()
        
        provider.update_portfolio({
            "total_value": 100000,
            "cash": 50000,
            "buying_power": 100000,
            "day_pnl": 500,
        })
        
        summary = provider.get_portfolio_summary()
        
        assert summary["total_value"] == 100000
        
    def test_dashboard_creation(self):
        """Test TradingDashboard creation."""
        TradingDashboard = _trading_dashboard.TradingDashboard
        
        dashboard = TradingDashboard()
        
        assert len(dashboard.layout.widgets) > 0
        
    def test_dashboard_html_generation(self):
        """Test HTML generation."""
        TradingDashboard = _trading_dashboard.TradingDashboard
        
        dashboard = TradingDashboard()
        html = dashboard.generate_html()
        
        assert "GNOSIS Trading Dashboard" in html
        assert "<html" in html


# =============================================================================
# Telegram Bot Tests
# =============================================================================

class TestTelegramBot:
    """Test Telegram/Discord Bot."""
    
    def test_bot_user(self):
        """Test BotUser dataclass."""
        from notifications.telegram_bot import BotUser, Platform
        
        user = BotUser(
            user_id="123456",
            platform=Platform.TELEGRAM,
            username="trader",
        )
        
        assert user.user_id == "123456"
        assert user.is_authorized is False
        
    def test_bot_message_parsing(self):
        """Test message parsing."""
        from notifications.telegram_bot import TradingBot, Platform, CommandType
        
        bot = TradingBot()
        
        message = bot.parse_message(
            "/portfolio",
            Platform.TELEGRAM,
            "user123",
            "chat123",
        )
        
        assert message.is_command is True
        assert message.command == CommandType.PORTFOLIO
        
    def test_bot_message_with_args(self):
        """Test message with arguments."""
        from notifications.telegram_bot import TradingBot, Platform, CommandType
        
        bot = TradingBot()
        
        message = bot.parse_message(
            "/quote AAPL",
            Platform.TELEGRAM,
            "user123",
            "chat123",
        )
        
        assert message.command == CommandType.QUOTE
        assert message.args == ["AAPL"]
        
    def test_alert_message_formatting(self):
        """Test alert message formatting."""
        from notifications.telegram_bot import AlertMessage, AlertPriority
        
        alert = AlertMessage(
            priority=AlertPriority.HIGH,
            title="Price Alert",
            message="AAPL crossed $180",
            symbol="AAPL",
        )
        
        telegram_msg = alert.format_telegram()
        discord_embed = alert.format_discord()
        
        assert "Price Alert" in telegram_msg
        assert "🔔" in telegram_msg  # High priority emoji
        assert discord_embed["title"] == "Price Alert"
        
    @pytest.mark.asyncio
    async def test_bot_handle_message(self):
        """Test bot message handling."""
        from notifications.telegram_bot import TradingBot, BotConfig, Platform
        
        config = BotConfig(
            authorized_users=["user123"],
        )
        bot = TradingBot(config)
        
        message = bot.parse_message(
            "/help",
            Platform.TELEGRAM,
            "user123",
            "chat123",
        )
        
        response = await bot.handle_message(message)
        
        assert response is not None
        assert "Available Commands" in response.text


# =============================================================================
# Portfolio Analytics Tests
# =============================================================================

class TestPortfolioAnalytics:
    """Test Portfolio Analytics."""
    
    def test_performance_metrics(self):
        """Test PerformanceMetrics."""
        PerformanceMetrics = _portfolio_analytics.PerformanceMetrics
        
        metrics = PerformanceMetrics(
            total_return=5000,
            total_return_pct=5.0,
            daily_return_avg=0.02,
            daily_return_std=0.01,
            sharpe_ratio=2.0,
            sortino_ratio=2.5,
            calmar_ratio=1.5,
            max_drawdown=5.0,
            max_drawdown_duration_days=10,
            current_drawdown=2.0,
            win_rate=60.0,
            avg_win=1.5,
            avg_loss=-0.8,
            profit_factor=1.8,
            best_day=3.0,
            worst_day=-2.0,
            positive_days=12,
            negative_days=8,
        )
        
        data = metrics.to_dict()
        
        assert data["sharpe_ratio"] == 2.0
        
    def test_risk_metrics(self):
        """Test RiskMetrics."""
        RiskMetrics = _portfolio_analytics.RiskMetrics
        
        risk = RiskMetrics(
            portfolio_delta=100.0,
            portfolio_gamma=5.0,
            portfolio_theta=-50.0,
            portfolio_vega=200.0,
            portfolio_beta=1.2,
            beta_weighted_delta=120.0,
            var_95=5000,
            var_99=7500,
            cvar_95=6000,
            gross_exposure=100000,
            net_exposure=50000,
            long_exposure=75000,
            short_exposure=25000,
            largest_position_pct=15.0,
            top_5_concentration=60.0,
            hhi_index=1500,
        )
        
        data = risk.to_dict()
        
        assert data["greeks"]["delta"] == 100.0
        assert data["var"]["var_95"] == 5000
        
    def test_analytics_portfolio_update(self):
        """Test portfolio snapshot update."""
        PortfolioAnalytics = _portfolio_analytics.PortfolioAnalytics
        
        analytics = PortfolioAnalytics()
        
        analytics.update_portfolio_snapshot(
            timestamp=datetime.now(),
            total_value=100000,
            cash=50000,
            positions=[
                {"symbol": "AAPL", "market_value": 25000},
                {"symbol": "MSFT", "market_value": 25000},
            ],
        )
        
        assert len(analytics._portfolio_history) == 1
        
    def test_performance_calculation(self):
        """Test performance calculation."""
        PortfolioAnalytics = _portfolio_analytics.PortfolioAnalytics
        TimeFrame = _portfolio_analytics.TimeFrame
        
        analytics = PortfolioAnalytics()
        
        # Add history
        for i in range(30):
            analytics.update_portfolio_snapshot(
                timestamp=datetime.now() - timedelta(days=29-i),
                total_value=100000 + i * 100,  # Steady growth
                cash=50000,
                positions=[],
            )
        
        perf = analytics.calculate_performance(TimeFrame.MONTH)
        
        assert perf.total_return > 0
        assert perf.positive_days > 0
        
    def test_sector_allocation(self):
        """Test sector allocation calculation."""
        PortfolioAnalytics = _portfolio_analytics.PortfolioAnalytics
        
        analytics = PortfolioAnalytics()
        
        analytics._positions = [
            {"symbol": "AAPL", "sector": "Technology", "market_value": 30000},
            {"symbol": "MSFT", "sector": "Technology", "market_value": 20000},
            {"symbol": "JPM", "sector": "Financials", "market_value": 25000},
            {"symbol": "XOM", "sector": "Energy", "market_value": 25000},
        ]
        
        allocations = analytics.get_sector_allocation()
        
        assert len(allocations) == 3  # Technology, Financials, Energy
        
        # Technology should be largest
        tech = next(a for a in allocations if a.sector == "Technology")
        assert tech.market_value == 50000
        
    def test_chart_generation(self):
        """Test chart data generation."""
        PortfolioAnalytics = _portfolio_analytics.PortfolioAnalytics
        ChartType = _portfolio_analytics.ChartType
        
        analytics = PortfolioAnalytics()
        
        # Add some history
        for i in range(30):
            analytics.update_portfolio_snapshot(
                timestamp=datetime.now() - timedelta(days=29-i),
                total_value=100000 + i * 50,
                cash=50000,
                positions=[],
            )
        
        chart = analytics.generate_pnl_chart()
        
        assert chart.chart_type == ChartType.LINE
        assert len(chart.labels) > 0
        
    def test_full_report_generation(self):
        """Test full report generation."""
        PortfolioAnalytics = _portfolio_analytics.PortfolioAnalytics
        
        analytics = PortfolioAnalytics()
        
        # Add minimal data
        analytics.update_portfolio_snapshot(
            timestamp=datetime.now() - timedelta(days=1),
            total_value=100000,
            cash=50000,
            positions=[{"symbol": "SPY", "market_value": 50000}],
        )
        analytics.update_portfolio_snapshot(
            timestamp=datetime.now(),
            total_value=101000,
            cash=51000,
            positions=[{"symbol": "SPY", "market_value": 50000}],
        )
        
        report = analytics.generate_full_report()
        
        assert "performance" in report
        assert "risk" in report
        assert "charts" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
