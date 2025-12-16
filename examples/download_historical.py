"""Utility script to cache historical Massive.com data for backtests."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import polars as pl
import typer
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

try:
    from massive import RESTClient
except Exception as exc:  # pragma: no cover - library may be absent in CI
    RESTClient = None
    logger.warning("Massive client not available: %s", exc)

app = typer.Typer(help="Download and cache historical data for Super Gnosis backtests")

DEFAULT_SYMBOLS: List[str] = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMZN",
    "META",
    "GOOGL",
    "GOOG",
    "AMD",
    "NFLX",
    "AVGO",
    "COST",
    "JPM",
    "BAC",
    "WMT",
    "UNH",
    "HD",
    "LLY",
    "V",
    "MA",
    "CRM",
    "BABA",
    "ADBE",
    "PYPL",
]

TIMEFRAMES = {"1min": (1, "minute"), "5min": (5, "minute"), "15min": (15, "minute"), "1hour": (1, "hour"), "1day": (1, "day")}


def _client() -> RESTClient:
    api_key = os.getenv("MASSIVE_API_KEY")
    if not RESTClient:
        raise RuntimeError("massive package not installed; run `pip install massive`.")
    if not api_key:
        raise RuntimeError("MASSIVE_API_KEY missing in environment")
    return RESTClient(api_key=api_key)


def _bars_path(cache_dir: Path, symbol: str, timeframe: str) -> Path:
    return cache_dir / symbol / f"bars_{timeframe}.parquet"


def _options_path(cache_dir: Path, symbol: str, suffix: str) -> Path:
    return cache_dir / symbol / "options" / f"{suffix}.parquet"


def _save_parquet(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(list(records))
    if df.is_empty():
        logger.warning("No records to write for %s", path)
        return
    if path.exists():
        try:
            existing = pl.read_parquet(path)
            df = (
                pl.concat([existing, df])
                .unique(subset=df.columns, keep="last")
                .sort(df.columns[0])
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not merge existing cache %s: %s", path, exc)
    df.write_parquet(path)
    logger.info("Cached %s rows to %s", df.height, path)


def download_bars(symbol: str, start: datetime, end: datetime, cache_dir: Path) -> None:
    client = _client()
    for tf, (mult, span) in TIMEFRAMES.items():
        logger.info("Fetching %s bars for %s", tf, symbol)
        aggs = client.get_aggs(
            ticker=symbol,
            multiplier=mult,
            timespan=span,
            from_=start.strftime("%Y-%m-%d"),
            to=end.strftime("%Y-%m-%d"),
            adjusted=True,
            sort="asc",
        )
        if not aggs:
            logger.warning("No bars returned for %s (%s)", symbol, tf)
            continue
        rows = [
            {
                "timestamp": datetime.fromtimestamp(a.timestamp / 1000),
                "open": float(a.open),
                "high": float(a.high),
                "low": float(a.low),
                "close": float(a.close),
                "volume": float(a.volume),
            }
            for a in aggs
        ]
        _save_parquet(_bars_path(cache_dir, symbol, tf), rows)


def download_options(symbol: str, cache_dir: Path) -> None:
    client = _client()
    logger.info("Fetching options contracts for %s", symbol)
    contracts = list(client.list_options_contracts(underlying_ticker=symbol, limit=1000))
    contract_rows = [
        {
            "symbol": getattr(c, "ticker", ""),
            "strike": float(getattr(c, "strike_price", 0)),
            "expiration": getattr(c, "expiration_date", ""),
            "option_type": getattr(c, "contract_type", "").lower(),
        }
        for c in contracts
    ]
    _save_parquet(_options_path(cache_dir, symbol, "contracts"), contract_rows)

    logger.info("Fetching daily options snapshots for %s", symbol)
    today = datetime.utcnow().date()
    for days in range(30):
        day = today - timedelta(days=days)
        snapshot = client.get_options_snapshot_v5(underlying_asset=symbol, date=day.strftime("%Y-%m-%d"))
        if not snapshot:
            continue
        rows = [
            {
                "as_of": day,
                "symbol": getattr(s, "ticker", ""),
                "last": float(getattr(s.last_quote, "P", 0) or 0),
                "bid": float(getattr(s.last_quote, "bP", 0) or 0),
                "ask": float(getattr(s.last_quote, "aP", 0) or 0),
                "open_interest": float(getattr(s, "open_interest", 0) or 0),
                "implied_volatility": float(getattr(s, "implied_volatility", 0) or 0),
            }
            for s in snapshot
        ]
        _save_parquet(_options_path(cache_dir, symbol, f"snapshot_{day}"), rows)


@app.command("download-data")
def cli_download(
    symbols: str = typer.Option(
        ",".join(DEFAULT_SYMBOLS), help="Comma separated symbols (default top 25)"
    ),
    start: str = typer.Option("2020-01-01", help="Start date YYYY-MM-DD"),
    end: str = typer.Option(datetime.utcnow().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD"),
    cache_dir: Path = typer.Option(Path("data/historical"), help="Cache root directory"),
) -> None:
    """Download multi-timeframe bars and options data to local cache."""

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    cache_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbol_list:
        download_bars(symbol, start_dt, end_dt, cache_dir)
        download_options(symbol, cache_dir)


if __name__ == "__main__":
    app()
