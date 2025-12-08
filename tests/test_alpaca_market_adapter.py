import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("ALPACA_API_KEY", "test_key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test_secret")

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


class ItemResponse:
    """Simulate Alpaca SDK object that supports __getitem__ instead of .data dict."""

    def __init__(self, symbol, bars):
        self._symbol = symbol
        self._bars = bars

    def __getitem__(self, key):
        if key == self._symbol:
            return self._bars
        raise KeyError(key)


class MultiBarsResponse:
    """Simulate MultiStockBars where .data is dict mapping symbols to bar lists."""

    def __init__(self, data):
        self.data = data


class ListDataResponse:
    """Simulate response where .data is a list of bar objects carrying symbol attribute."""

    def __init__(self, bars):
        self.data = bars


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


def test_extracts_bars_from_getitem_response():
    bars = [
        FakeBar(datetime(2024, 2, 1, 10, 0), 100, 101, 99, 100.5, 5000),
    ]
    adapter = AlpacaMarketDataAdapter(client=FakeClient(ItemResponse("SPY", bars)), data_feed="IEX")

    result = adapter.get_bars(
        symbol="SPY",
        start=datetime(2024, 2, 1, 9, 30),
        end=datetime(2024, 2, 1, 16, 0),
        timeframe="1Day",
    )

    assert len(result) == 1
    assert result[0].close == 100.5


def test_naive_datetime_converted_to_utc(monkeypatch):
    captured_request = {}

    class InspectClient(FakeClient):
        def get_stock_bars(self, request):
            captured_request["start"] = request.start
            captured_request["end"] = request.end
            return super().get_stock_bars(request)

    adapter = AlpacaMarketDataAdapter(client=InspectClient(RichClient().response), data_feed="IEX")

    start = datetime(2024, 3, 1, 9, 30)  # naive
    end = start + timedelta(hours=6)

    adapter.get_bars(symbol="AAPL", start=start, end=end, timeframe="1Day")

    assert captured_request["start"].tzinfo == timezone.utc
    assert captured_request["end"].tzinfo == timezone.utc


def test_multi_symbol_response_extracts_only_requested():
    bars_a = [FakeBar(datetime(2024, 1, 1, 10, 0), 1, 2, 1, 1.5, 100)]
    bars_b = [FakeBar(datetime(2024, 1, 1, 10, 0), 5, 6, 5, 5.5, 200)]
    client = FakeClient(MultiBarsResponse({"AAA": bars_a, "BBB": bars_b}))

    adapter = AlpacaMarketDataAdapter(client=client, data_feed="IEX")
    result = adapter.get_bars(
        symbol="BBB",
        start=datetime(2024, 1, 1, 9, 30),
        end=datetime(2024, 1, 1, 16, 0),
        timeframe="5Min",
    )

    assert len(result) == 1
    assert result[0].close == 5.5


def test_list_data_response_filters_by_symbol_attribute():
    class BarWithSymbol(FakeBar):
        def __init__(self, symbol, *args):
            super().__init__(*args)
            self.symbol = symbol

    bars = [
        BarWithSymbol("AAA", datetime(2024, 1, 1, 10, 0), 1, 2, 1, 1.5, 100),
        BarWithSymbol("BBB", datetime(2024, 1, 1, 10, 1), 5, 6, 5, 5.5, 200),
    ]

    adapter = AlpacaMarketDataAdapter(client=FakeClient(ListDataResponse(bars)), data_feed="IEX")
    result = adapter.get_bars(
        symbol="AAA",
        start=datetime(2024, 1, 1, 9, 30),
        end=datetime(2024, 1, 1, 16, 0),
        timeframe="1Min",
    )

    assert len(result) == 1
    assert result[0].close == 1.5
