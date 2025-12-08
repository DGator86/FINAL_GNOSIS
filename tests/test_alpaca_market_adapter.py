from datetime import datetime

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("loguru")

from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter


class FakeBar:
    def __init__(self, ts, open, high, low, close, volume):
        self.timestamp = ts
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeClient:
    def __init__(self, response):
        self.response = response

    def get_stock_bars(self, request):
        return self.response


class EmptyClient(FakeClient):
    def __init__(self):
        super().__init__(FakeResponse({}))


class RichClient(FakeClient):
    def __init__(self):
        bars = [
            FakeBar(datetime(2024, 1, 1, 10, 0), 10, 11, 9, 10.5, 1000),
            FakeBar(datetime(2024, 1, 1, 10, 1), 10.5, 11.5, 10, 11, 1200),
        ]
        super().__init__(FakeResponse({"AAPL": bars}))


def test_extracts_bars_from_dict_response(monkeypatch):
    adapter = AlpacaMarketDataAdapter(client=RichClient(), data_feed="IEX")
    bars = adapter.get_bars(
        symbol="AAPL",
        start=datetime(2024, 1, 1, 9, 30),
        end=datetime(2024, 1, 1, 16, 0),
        timeframe="1Min",
    )

    assert len(bars) == 2
    assert bars[0].open == 10


def test_empty_response_logs_and_returns_empty():
    adapter = AlpacaMarketDataAdapter(client=EmptyClient(), data_feed="IEX")

    bars = adapter.get_bars(
        symbol="MSFT",
        start=datetime(2024, 1, 1, 9, 30),
        end=datetime(2024, 1, 1, 16, 0),
        timeframe="1Min",
    )

    assert bars == []
