"""
Tests for Infrastructure Components

Tests cover:
- WebSocket API (connection management, subscriptions, broadcasts)
- Prometheus Metrics (counters, gauges, histograms, export)
- Redis Cache (get/set, TTL, namespaces, fallback to memory)

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

import pytest
import asyncio
import json
import time
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# =============================================================================
# WebSocket API Tests
# =============================================================================

# Import WebSocket module directly to avoid routers/__init__.py loading ml_trades
import sys
import importlib.util

def load_websocket_api():
    """Load websocket_api module directly without going through routers/__init__.py."""
    spec = importlib.util.spec_from_file_location(
        "websocket_api",
        "/home/root/webapp/routers/websocket_api.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["websocket_api"] = module
    spec.loader.exec_module(module)
    return module

# Load the module once at module level
websocket_api = load_websocket_api()


class TestWebSocketModels:
    """Test WebSocket data models."""
    
    def test_channel_type_enum(self):
        """Test ChannelType enum values."""
        ChannelType = websocket_api.ChannelType
        
        assert ChannelType.PORTFOLIO.value == "portfolio"
        assert ChannelType.POSITIONS.value == "positions"
        assert ChannelType.ORDERS.value == "orders"
        assert ChannelType.GREEKS.value == "greeks"
        
    def test_message_type_enum(self):
        """Test MessageType enum values."""
        MessageType = websocket_api.MessageType
        
        assert MessageType.SUBSCRIBE.value == "subscribe"
        assert MessageType.UPDATE.value == "update"
        assert MessageType.HEARTBEAT.value == "heartbeat"
        
    def test_websocket_message_to_json(self):
        """Test WebSocketMessage serialization."""
        WebSocketMessage = websocket_api.WebSocketMessage
        MessageType = websocket_api.MessageType
        ChannelType = websocket_api.ChannelType
        
        msg = WebSocketMessage(
            type=MessageType.UPDATE,
            channel=ChannelType.PORTFOLIO,
            data={"value": 100000},
            sequence=1,
        )
        
        json_str = msg.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["type"] == "update"
        assert parsed["channel"] == "portfolio"
        assert parsed["data"]["value"] == 100000
        assert parsed["sequence"] == 1


class TestConnectionManager:
    """Test WebSocket ConnectionManager."""
    
    @pytest.fixture
    def manager(self):
        """Create connection manager."""
        ConnectionManager = websocket_api.ConnectionManager
        return ConnectionManager()
    
    def test_manager_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager.get_connection_count() == 0
        
    @pytest.mark.asyncio
    async def test_generate_client_id(self, manager):
        """Test client ID generation."""
        client_id = manager._generate_client_id()
        assert len(client_id) == 8
        
    @pytest.mark.asyncio
    async def test_get_connection_count(self, manager):
        """Test connection counting."""
        assert manager.get_connection_count() == 0


# =============================================================================
# Prometheus Metrics Tests
# =============================================================================

class TestMetricsRegistry:
    """Test MetricsRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Create fresh metrics registry."""
        from utils.metrics import MetricsRegistry
        return MetricsRegistry(prefix="test")
    
    def test_registry_initialization(self, registry):
        """Test registry initializes with default metrics."""
        assert registry.prefix == "test"
        assert len(registry._metrics) > 0
        
    def test_inc_counter(self, registry):
        """Test counter increment."""
        registry.inc_counter("orders_total", 1, {"side": "buy"})
        registry.inc_counter("orders_total", 2, {"side": "buy"})
        
        # Value should be accumulated
        key = registry._get_label_key({"side": "buy"})
        assert registry._counters[f"test_orders_total"][key] == 3
        
    def test_set_gauge(self, registry):
        """Test gauge setting."""
        registry.set_gauge("portfolio_value_dollars", 100000)
        registry.set_gauge("portfolio_value_dollars", 105000)
        
        # Gauge should be overwritten
        assert registry._gauges["test_portfolio_value_dollars"][""] == 105000
        
    def test_observe_histogram(self, registry):
        """Test histogram observation."""
        registry.observe_histogram("order_latency_seconds", 0.05)
        registry.observe_histogram("order_latency_seconds", 0.10)
        registry.observe_histogram("order_latency_seconds", 0.15)
        
        observations = registry._histograms["test_order_latency_seconds"][""]
        assert len(observations) == 3
        assert observations[0] == 0.05
        
    def test_prometheus_output(self, registry):
        """Test Prometheus format output."""
        registry.set_gauge("portfolio_value_dollars", 100000)
        registry.inc_counter("orders_total", 5)
        
        output = registry.get_prometheus_output()
        
        assert "test_portfolio_value_dollars 100000" in output
        assert "test_orders_total 5" in output
        
    def test_prometheus_output_with_labels(self, registry):
        """Test Prometheus output with labels."""
        registry.inc_counter("orders_total", 3, {"side": "buy", "type": "market"})
        
        output = registry.get_prometheus_output()
        
        assert 'side="buy"' in output
        assert 'type="market"' in output
        
    def test_get_label_key(self, registry):
        """Test label key generation."""
        key = registry._get_label_key({"b": "2", "a": "1"})
        
        # Should be sorted
        assert key == 'a="1",b="2"'
        
    def test_get_label_key_empty(self, registry):
        """Test empty labels."""
        key = registry._get_label_key({})
        assert key == ""
        
    def test_reset(self, registry):
        """Test metric reset."""
        registry.set_gauge("test_metric", 100)
        registry.reset()
        
        assert len(registry._gauges) == 0


class TestMetricHelpers:
    """Test metric helper functions."""
    
    def test_update_portfolio_metrics(self):
        """Test update_portfolio_metrics helper."""
        from utils.metrics import update_portfolio_metrics, metrics
        
        metrics.reset()
        update_portfolio_metrics(
            total_value=100000,
            daily_pnl=500,
            unrealized_pnl=300,
            realized_pnl=200,
        )
        
        assert metrics._gauges["gnosis_portfolio_value_dollars"][""] == 100000
        assert metrics._gauges["gnosis_daily_pnl_dollars"][""] == 500
        
    def test_update_greeks_metrics(self):
        """Test update_greeks_metrics helper."""
        from utils.metrics import update_greeks_metrics, metrics
        
        metrics.reset()
        update_greeks_metrics(
            delta=100.0,
            gamma=5.0,
            theta=-20.0,
            vega=50.0,
        )
        
        assert metrics._gauges["gnosis_portfolio_delta"][""] == 100.0
        assert metrics._gauges["gnosis_portfolio_theta"][""] == -20.0
        
    def test_record_order(self):
        """Test record_order helper."""
        from utils.metrics import record_order, metrics
        
        metrics.reset()
        record_order(
            side="buy",
            order_type="market",
            status="filled",
            latency=0.05,
            slippage=0.10,
        )
        
        key = metrics._get_label_key({"side": "buy", "type": "market", "status": "filled"})
        assert metrics._counters["gnosis_orders_total"][key] == 1


class TestMetricDecorators:
    """Test metric decorators."""
    
    def test_track_latency_sync(self):
        """Test track_latency decorator on sync function."""
        from utils.metrics import track_latency, metrics
        
        metrics.reset()
        
        @track_latency("test_latency")
        def slow_function():
            time.sleep(0.01)
            return "done"
        
        result = slow_function()
        
        assert result == "done"
        observations = metrics._histograms["gnosis_test_latency"][""]
        assert len(observations) == 1
        assert observations[0] >= 0.01
        
    @pytest.mark.asyncio
    async def test_track_latency_async(self):
        """Test track_latency decorator on async function."""
        from utils.metrics import track_latency, metrics
        
        metrics.reset()
        
        @track_latency("test_async_latency")
        async def slow_async_function():
            await asyncio.sleep(0.01)
            return "async done"
        
        result = await slow_async_function()
        
        assert result == "async done"
        observations = metrics._histograms["gnosis_test_async_latency"][""]
        assert len(observations) == 1


# =============================================================================
# Redis Cache Tests
# =============================================================================

class TestCacheNamespace:
    """Test CacheNamespace enum."""
    
    def test_namespace_values(self):
        """Test namespace enum values."""
        from utils.redis_cache import CacheNamespace
        
        assert CacheNamespace.MARKET_DATA.value == "market"
        assert CacheNamespace.GREEKS.value == "greeks"
        assert CacheNamespace.PORTFOLIO.value == "portfolio"


class TestCacheConfig:
    """Test CacheConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        from utils.redis_cache import CacheConfig
        
        config = CacheConfig()
        
        assert config.redis_url == "redis://localhost:6379/0"
        assert config.default_ttl == 300
        assert config.quote_ttl == 5
        assert config.greeks_ttl == 10


class TestMemoryCacheBackend:
    """Test in-memory cache backend."""
    
    @pytest.fixture
    def cache(self):
        """Create memory cache."""
        from utils.redis_cache import MemoryCacheBackend
        return MemoryCacheBackend(max_size=100, default_ttl=60)
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        """Test basic set and get."""
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        
        assert result == "value1"
        
    @pytest.mark.asyncio
    async def test_get_nonexistent(self, cache):
        """Test getting nonexistent key."""
        result = await cache.get("nonexistent")
        assert result is None
        
    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Test key deletion."""
        await cache.set("key1", "value1")
        await cache.delete("key1")
        
        result = await cache.get("key1")
        assert result is None
        
    @pytest.mark.asyncio
    async def test_exists(self, cache):
        """Test key existence check."""
        await cache.set("key1", "value1")
        
        assert await cache.exists("key1") is True
        assert await cache.exists("nonexistent") is False
        
    @pytest.mark.asyncio
    async def test_ttl_expiration(self, cache):
        """Test TTL-based expiration."""
        await cache.set("short_key", "value", ttl=1)
        
        # Should exist immediately
        assert await cache.get("short_key") == "value"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        assert await cache.get("short_key") is None
        
    @pytest.mark.asyncio
    async def test_clear_namespace(self, cache):
        """Test namespace clearing."""
        await cache.set("ns1:key1", "value1")
        await cache.set("ns1:key2", "value2")
        await cache.set("ns2:key1", "value3")
        
        deleted = await cache.clear_namespace("ns1")
        
        assert deleted == 2
        assert await cache.get("ns1:key1") is None
        assert await cache.get("ns2:key1") == "value3"
        
    @pytest.mark.asyncio
    async def test_get_many(self, cache):
        """Test getting multiple values."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        result = await cache.get_many(["key1", "key2", "key3"])
        
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        assert "key3" not in result
        
    @pytest.mark.asyncio
    async def test_set_many(self, cache):
        """Test setting multiple values."""
        await cache.set_many({"key1": "value1", "key2": "value2"})
        
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"
        
    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        from utils.redis_cache import MemoryCacheBackend
        
        cache = MemoryCacheBackend(max_size=3, default_ttl=60)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        await cache.get("key1")
        
        # Add new key, should evict key2 (oldest)
        await cache.set("key4", "value4")
        
        assert await cache.get("key1") is not None  # Recently used
        assert await cache.get("key4") is not None  # Just added
        
    def test_get_stats(self, cache):
        """Test cache statistics."""
        stats = cache.get_stats()
        
        assert "size" in stats
        assert "max_size" in stats
        assert "utilization" in stats


class TestCacheManager:
    """Test CacheManager."""
    
    @pytest.fixture
    def manager(self):
        """Create cache manager."""
        from utils.redis_cache import CacheManager, CacheConfig
        
        config = CacheConfig()
        return CacheManager(config)
    
    def test_build_key(self, manager):
        """Test key building with namespace."""
        from utils.redis_cache import CacheNamespace
        
        key = manager._build_key(CacheNamespace.GREEKS, "AAPL", "call", "100")
        
        assert key == "greeks:AAPL:call:100"
        
    def test_get_ttl(self, manager):
        """Test TTL retrieval for namespaces."""
        from utils.redis_cache import CacheNamespace
        
        quote_ttl = manager._get_ttl(CacheNamespace.QUOTES)
        greeks_ttl = manager._get_ttl(CacheNamespace.GREEKS)
        
        assert quote_ttl == 5  # Fast refresh for quotes
        assert greeks_ttl == 10
        
    @pytest.mark.asyncio
    async def test_get_set_with_namespace(self, manager):
        """Test get/set with namespace."""
        from utils.redis_cache import CacheNamespace
        
        await manager.set(
            CacheNamespace.GREEKS,
            "AAPL_CALL",
            value={"delta": 0.5, "gamma": 0.02},
        )
        
        result = await manager.get(CacheNamespace.GREEKS, "AAPL_CALL")
        
        assert result["delta"] == 0.5
        
    @pytest.mark.asyncio
    async def test_specialized_methods(self, manager):
        """Test specialized cache methods."""
        # Quote
        await manager.set_quote("AAPL", {"bid": 150.0, "ask": 150.10})
        quote = await manager.get_quote("AAPL")
        assert quote["bid"] == 150.0
        
        # Greeks
        await manager.set_greeks("AAPL_CALL", {"delta": 0.55})
        greeks = await manager.get_greeks("AAPL_CALL")
        assert greeks["delta"] == 0.55
        
        # Portfolio
        await manager.set_portfolio({"value": 100000})
        portfolio = await manager.get_portfolio()
        assert portfolio["value"] == 100000
        
    def test_get_stats(self, manager):
        """Test cache statistics."""
        stats = manager.get_stats()
        
        assert "backend" in stats
        assert "config" in stats


class TestCacheIntegration:
    """Integration tests for cache system."""
    
    @pytest.mark.asyncio
    async def test_full_cache_workflow(self):
        """Test complete cache workflow."""
        from utils.redis_cache import CacheManager, CacheConfig, CacheNamespace
        
        config = CacheConfig()
        manager = CacheManager(config)
        
        # Initialize (will use memory backend as Redis not available)
        await manager.initialize()
        
        # Set various data types
        await manager.set_quote("SPY", {"bid": 500.0, "ask": 500.05, "last": 500.02})
        await manager.set_greeks("SPY_CALL", {"delta": 0.50, "gamma": 0.015})
        await manager.set_portfolio({"total_value": 100000, "positions": []})
        
        # Retrieve data
        spy_quote = await manager.get_quote("SPY")
        spy_greeks = await manager.get_greeks("SPY_CALL")
        portfolio = await manager.get_portfolio()
        
        assert spy_quote["last"] == 500.02
        assert spy_greeks["delta"] == 0.50
        assert portfolio["total_value"] == 100000
        
        # Clear namespace
        from utils.redis_cache import CacheNamespace
        deleted = await manager.clear_namespace(CacheNamespace.QUOTES)
        assert deleted >= 1
        
        # Quote should be gone
        spy_quote = await manager.get_quote("SPY")
        assert spy_quote is None
        
        # Cleanup
        await manager.close()


# =============================================================================
# Integration Tests
# =============================================================================

class TestInfrastructureIntegration:
    """Integration tests across infrastructure components."""
    
    @pytest.mark.asyncio
    async def test_metrics_with_cache(self):
        """Test metrics recording with cache operations."""
        from utils.metrics import metrics, track_latency
        from utils.redis_cache import CacheManager, CacheConfig
        
        metrics.reset()
        
        config = CacheConfig()
        cache = CacheManager(config)
        await cache.initialize()
        
        # Cache operation with latency tracking
        @track_latency("cache_operation")
        async def cached_operation():
            await cache.set_quote("TEST", {"price": 100})
            return await cache.get_quote("TEST")
        
        result = await cached_operation()
        
        assert result["price"] == 100
        assert len(metrics._histograms["gnosis_cache_operation"][""]) == 1
        
        await cache.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
