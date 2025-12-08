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
        "contracts": [
            {
                "symbol": "TEST200",
                "strike": 200,
                "expiration_date": "2024-12-20",
                "type": "call",
                "bid": 1.0,
                "ask": 1.2,
                "last": 1.1,
                "volume": 500,
                "open_interest": 1000,
                "implied_volatility": 0.3,
                "greeks": {"delta": 0.4},
            }
        ]
    }
    return httpx.Response(status_code=200, json=data, request=httpx.Request("GET", "http://test"))


def response_404():
    return httpx.Response(status_code=404, json={}, request=httpx.Request("GET", "http://test"))


def test_successful_chain_parsing():
    adapter = UnusualWhalesOptionsAdapter(token="x", client=FakeUWClient([response_with_contracts()]))
    contracts = adapter.get_chain("AAPL", datetime.now())
    assert len(contracts) == 1
    assert contracts[0].strike == 200


def test_404_does_not_disable_future_calls():
    client = FakeUWClient([response_404(), response_with_contracts()])
    adapter = UnusualWhalesOptionsAdapter(token="x", client=client)

    first = adapter.get_chain("MSFT", datetime.now())
    second = adapter.get_chain("MSFT", datetime.now())

    assert first  # falls back to stub but still returns list
    assert len(second) == 1  # 404 did not permanently force stub-only
    assert adapter.use_stub is False
