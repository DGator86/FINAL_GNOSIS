"""Tests for MultiTimeframeScanner integration.

Tests cover:
- Scanner initialization
- Priority symbol management
- Scan execution
- Result handling
- Factory creation

Author: Super Gnosis Elite Trading System
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from gnosis.scanner import MultiTimeframeScanner, ScannerResult, create_scanner
from engines.scanner import Opportunity


class TestScannerResult:
    """Tests for ScannerResult dataclass."""

    def test_empty_result(self):
        """Test empty scanner result."""
        result = ScannerResult(scan_time=datetime.now())
        
        assert result.opportunities == []
        assert result.symbols_scanned == 0
        assert result.top_opportunity is None
        assert result.long_opportunities == []
        assert result.short_opportunities == []

    def test_result_with_opportunities(self):
        """Test result with opportunities."""
        opp1 = Opportunity(
            rank=1,
            symbol="AAPL",
            score=0.85,
            opportunity_type="breakout",
            direction="long",
            confidence=0.8,
            energy_asymmetry=0.5,
            movement_energy=0.7,
            liquidity_score=0.9,
            reasoning="Test opportunity",
        )
        opp2 = Opportunity(
            rank=2,
            symbol="SPY",
            score=0.75,
            opportunity_type="swing",
            direction="short",
            confidence=0.7,
            energy_asymmetry=-0.3,
            movement_energy=0.5,
            liquidity_score=0.8,
            reasoning="Test opportunity 2",
        )
        
        result = ScannerResult(
            scan_time=datetime.now(),
            opportunities=[opp1, opp2],
            symbols_scanned=10,
            scan_duration_seconds=1.5,
        )
        
        assert len(result.opportunities) == 2
        assert result.top_opportunity == opp1
        assert len(result.long_opportunities) == 1
        assert len(result.short_opportunities) == 1
        assert result.long_opportunities[0].symbol == "AAPL"
        assert result.short_opportunities[0].symbol == "SPY"


class TestMultiTimeframeScannerInit:
    """Tests for MultiTimeframeScanner initialization."""

    def test_init_without_engines(self):
        """Test initialization without engines."""
        scanner = MultiTimeframeScanner(config={"auto_create_engines": False})
        
        assert scanner.scanner is None
        assert len(scanner._priority_symbols) > 0

    def test_init_with_config(self):
        """Test initialization with config."""
        config = {
            "auto_create_engines": False,
            "priority_symbols": ["AAPL", "GOOGL", "MSFT"],
        }
        scanner = MultiTimeframeScanner(config=config)
        
        assert scanner._priority_symbols == ["AAPL", "GOOGL", "MSFT"]

    def test_default_priority_symbols(self):
        """Test default priority symbols are set."""
        scanner = MultiTimeframeScanner(config={"auto_create_engines": False})
        
        assert "SPY" in scanner._priority_symbols
        assert "QQQ" in scanner._priority_symbols


class TestPrioritySymbolManagement:
    """Tests for priority symbol management."""

    @pytest.fixture
    def scanner(self):
        """Create scanner for testing."""
        return MultiTimeframeScanner(config={"auto_create_engines": False})

    def test_get_priority_symbols(self, scanner):
        """Test getting priority symbols."""
        symbols = scanner.get_priority_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0

    def test_set_priority_symbols(self, scanner):
        """Test setting priority symbols."""
        new_symbols = ["TSLA", "AMD", "NVDA"]
        scanner.set_priority_symbols(new_symbols)
        
        assert scanner.get_priority_symbols() == new_symbols

    def test_add_priority_symbol(self, scanner):
        """Test adding a priority symbol."""
        original_count = len(scanner.get_priority_symbols())
        scanner.add_priority_symbol("UNIQUE_SYMBOL")
        
        assert "UNIQUE_SYMBOL" in scanner.get_priority_symbols()
        assert len(scanner.get_priority_symbols()) == original_count + 1

    def test_add_duplicate_priority_symbol(self, scanner):
        """Test adding duplicate symbol doesn't duplicate."""
        scanner.add_priority_symbol("SPY")  # Already in default list
        
        count = scanner.get_priority_symbols().count("SPY")
        assert count == 1

    def test_remove_priority_symbol(self, scanner):
        """Test removing a priority symbol."""
        scanner.remove_priority_symbol("SPY")
        
        assert "SPY" not in scanner.get_priority_symbols()

    def test_remove_nonexistent_symbol(self, scanner):
        """Test removing nonexistent symbol doesn't error."""
        scanner.remove_priority_symbol("NONEXISTENT")
        # Should not raise


class TestScanExecution:
    """Tests for scan execution."""

    @pytest.fixture
    def mock_opportunity_scanner(self):
        """Create mock OpportunityScanner."""
        mock = Mock()
        mock.scan.return_value = Mock(
            scan_timestamp=datetime.now(),
            symbols_scanned=10,
            scan_duration_seconds=1.0,
            opportunities=[
                Opportunity(
                    rank=1,
                    symbol="AAPL",
                    score=0.9,
                    opportunity_type="breakout",
                    direction="long",
                    confidence=0.85,
                    energy_asymmetry=0.6,
                    movement_energy=0.8,
                    liquidity_score=0.95,
                    reasoning="Strong breakout",
                )
            ],
        )
        return mock

    def test_run_with_no_scanner(self):
        """Test run returns empty result when no scanner."""
        scanner = MultiTimeframeScanner(config={"auto_create_engines": False})
        result = scanner.run(symbols=["AAPL"])
        
        assert isinstance(result, ScannerResult)
        assert result.opportunities == []

    def test_run_with_scanner(self, mock_opportunity_scanner):
        """Test run with working scanner."""
        scanner = MultiTimeframeScanner(config={"auto_create_engines": False})
        scanner.scanner = mock_opportunity_scanner
        
        result = scanner.run(symbols=["AAPL", "MSFT"], top_n=5)
        
        assert isinstance(result, ScannerResult)
        assert len(result.opportunities) == 1
        mock_opportunity_scanner.scan.assert_called_once()

    def test_scan_priority(self, mock_opportunity_scanner):
        """Test priority symbol scanning."""
        scanner = MultiTimeframeScanner(config={"auto_create_engines": False})
        scanner.scanner = mock_opportunity_scanner
        
        result = scanner.scan_priority(top_n=5)
        
        assert isinstance(result, ScannerResult)
        mock_opportunity_scanner.scan.assert_called_once()

    def test_scan_handles_exception(self, mock_opportunity_scanner):
        """Test scan handles exceptions gracefully."""
        mock_opportunity_scanner.scan.side_effect = Exception("Scan error")
        
        scanner = MultiTimeframeScanner(config={"auto_create_engines": False})
        scanner.scanner = mock_opportunity_scanner
        
        result = scanner.run(symbols=["AAPL"])
        
        assert isinstance(result, ScannerResult)
        assert result.opportunities == []


class TestAsyncScanning:
    """Tests for async scanning."""

    @pytest.fixture
    def mock_opportunity_scanner(self):
        """Create mock OpportunityScanner."""
        mock = Mock()
        mock.scan.return_value = Mock(
            scan_timestamp=datetime.now(),
            symbols_scanned=5,
            scan_duration_seconds=0.5,
            opportunities=[],
        )
        return mock

    @pytest.mark.asyncio
    async def test_scan_all_priority_only(self, mock_opportunity_scanner):
        """Test async scan with priority only."""
        scanner = MultiTimeframeScanner(config={"auto_create_engines": False})
        scanner.scanner = mock_opportunity_scanner
        
        result = await scanner.scan_all(priority_only=True, top_n=5)
        
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_scan_all_full_universe(self, mock_opportunity_scanner):
        """Test async scan of full universe."""
        scanner = MultiTimeframeScanner(config={"auto_create_engines": False})
        scanner.scanner = mock_opportunity_scanner
        
        with patch('gnosis.scanner.get_dynamic_universe', return_value=["SPY", "QQQ"]):
            result = await scanner.scan_all(priority_only=False, top_n=10)
        
        assert isinstance(result, list)


class TestFactoryFunction:
    """Tests for create_scanner factory function."""

    def test_create_scanner_default(self):
        """Test creating scanner with defaults."""
        with patch.object(MultiTimeframeScanner, '_create_scanner_from_factory'):
            scanner = create_scanner()
            
            assert isinstance(scanner, MultiTimeframeScanner)

    def test_create_scanner_with_config(self):
        """Test creating scanner with config."""
        config = {
            "auto_create_engines": False,
            "priority_symbols": ["TEST"],
        }
        
        scanner = create_scanner(config=config)
        
        assert scanner._priority_symbols == ["TEST"]


class TestScannerWithEngines:
    """Tests for scanner with actual engine kwargs."""

    def test_init_with_engine_kwargs(self):
        """Test initialization with engine kwargs."""
        mock_hedge = Mock()
        mock_liquidity = Mock()
        mock_sentiment = Mock()
        mock_elasticity = Mock()
        mock_options = Mock()
        mock_market = Mock()
        
        with patch('gnosis.scanner.OpportunityScanner') as MockScanner:
            scanner = MultiTimeframeScanner(
                config={"auto_create_engines": False},
                hedge_engine=mock_hedge,
                liquidity_engine=mock_liquidity,
                sentiment_engine=mock_sentiment,
                elasticity_engine=mock_elasticity,
                options_adapter=mock_options,
                market_adapter=mock_market,
            )
            
            MockScanner.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
