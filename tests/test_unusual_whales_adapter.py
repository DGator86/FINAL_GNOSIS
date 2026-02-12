from datetime import datetime

import pytest

pytest.importorskip("httpx")
pytest.importorskip("loguru")

import httpx

from engines.inputs.unusual_whales_adapter import UnusualWhalesOptionsAdapter


class FakeUWClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def get(self, url, params=None):
        response = self.responses[self.calls]
        self.calls += 1
        return response


def response_with_contracts():
    data = {
        "data": [
            {
                "symbol": "TEST241220C00200000",
                "nbbo_bid": 1.0,
                "nbbo_ask": 1.2,
                "last_price": 1.1,
                "volume": 500,
                "open_interest": 1000,
                "implied_volatility": 0.3,
                "delta": 0.4,
            }
        ]
    }
    return httpx.Response(status_code=200, json=data, request=httpx.Request("GET", "http://test"))


def response_404():
    return httpx.Response(status_code=404, json={}, request=httpx.Request("GET", "http://test"))


def response_401():
    return httpx.Response(status_code=401, json={"detail": "unauthorized"}, request=httpx.Request("GET", "http://test"))


def response_500():
    return httpx.Response(status_code=500, json={}, request=httpx.Request("GET", "http://test"))


def test_successful_chain_parsing():
    adapter = UnusualWhalesOptionsAdapter(token="x", client=FakeUWClient([response_with_contracts()]))
    contracts = adapter.get_chain("AAPL", datetime.now())
    assert len(contracts) == 1
    assert contracts[0].strike == 200


def test_404_does_not_disable_future_calls():
    """Test that 404 raises RuntimeError but doesn't disable the adapter.
    
    The adapter raises RuntimeError on 404 (no data for symbol), but this
    shouldn't permanently disable the adapter for future calls.
    """
    client = FakeUWClient([response_404(), response_with_contracts()])
    adapter = UnusualWhalesOptionsAdapter(token="x", client=client)

    # First call should raise RuntimeError due to 404
    with pytest.raises(RuntimeError, match="has no data for MSFT"):
        adapter.get_chain("MSFT", datetime.now())

    # Second call should succeed (404 didn't permanently disable the adapter)
    second = adapter.get_chain("MSFT", datetime.now())
    assert len(second) == 1  # 404 did not permanently disable real calls
    assert adapter.use_stub is False


def test_401_raises_runtime_error():
    """Test that 401 (authentication error) raises RuntimeError.
    
    The adapter raises RuntimeError on authentication failures since
    real data is required for the backtest/live trading use case.
    """
    client = FakeUWClient([response_401()])
    adapter = UnusualWhalesOptionsAdapter(token="x", client=client)

    # 401 should raise RuntimeError indicating auth failure
    with pytest.raises(RuntimeError, match="auth/subscription error"):
        adapter.get_chain("TSLA", datetime.now())


def test_generic_http_error_raises_runtime_error():
    """Test that server errors (500) raise RuntimeError.
    
    The adapter raises RuntimeError on transient server errors since
    the caller should handle retries or fallback logic at a higher level.
    """
    client = FakeUWClient([response_500()])
    adapter = UnusualWhalesOptionsAdapter(token="x", client=client)

    # 500 should raise RuntimeError indicating server error
    with pytest.raises(RuntimeError, match="transient/unavailable"):
        adapter.get_chain("NVDA", datetime.now())
