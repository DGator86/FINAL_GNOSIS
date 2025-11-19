"""Stub adapters for testing without external dependencies."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import List

from engines.inputs.market_data_adapter import OHLCV, MarketDataAdapter, Quote
from engines.inputs.news_adapter import NewsAdapter, NewsArticle
from engines.inputs.options_chain_adapter import OptionContract, OptionsChainAdapter


class StaticOptionsAdapter:
    """Static options chain adapter for testing."""
    
    def get_chain(self, symbol: str, timestamp: datetime) -> List[OptionContract]:
        """Generate synthetic options chain."""
        base_price = 450.0  # Assume SPY-like price
        contracts = []
        
        for days in [7, 14, 30, 45, 60]:
            expiration = timestamp + timedelta(days=days)
            
            for strike_offset in range(-10, 11, 2):
                strike = base_price + (strike_offset * 5)
                
                # Calls
                contracts.append(OptionContract(
                    symbol=f"{symbol}_{expiration.strftime('%Y%m%d')}C{strike}",
                    strike=strike,
                    expiration=expiration,
                    option_type="call",
                    bid=max(0.1, random.uniform(1, 20)),
                    ask=max(0.2, random.uniform(1.1, 21)),
                    last=max(0.15, random.uniform(1, 20.5)),
                    volume=random.uniform(0, 10000),
                    open_interest=random.uniform(0, 50000),
                    implied_volatility=random.uniform(0.15, 0.40),
                    delta=random.uniform(0.1, 0.9),
                    gamma=random.uniform(0.001, 0.05),
                    theta=-random.uniform(0.01, 0.1),
                    vega=random.uniform(0.05, 0.3),
                    rho=random.uniform(0.01, 0.1),
                ))
                
                # Puts
                contracts.append(OptionContract(
                    symbol=f"{symbol}_{expiration.strftime('%Y%m%d')}P{strike}",
                    strike=strike,
                    expiration=expiration,
                    option_type="put",
                    bid=max(0.1, random.uniform(1, 20)),
                    ask=max(0.2, random.uniform(1.1, 21)),
                    last=max(0.15, random.uniform(1, 20.5)),
                    volume=random.uniform(0, 10000),
                    open_interest=random.uniform(0, 50000),
                    implied_volatility=random.uniform(0.15, 0.40),
                    delta=-random.uniform(0.1, 0.9),
                    gamma=random.uniform(0.001, 0.05),
                    theta=-random.uniform(0.01, 0.1),
                    vega=random.uniform(0.05, 0.3),
                    rho=-random.uniform(0.01, 0.1),
                ))
        
        return contracts


class StaticMarketDataAdapter:
    """Static market data adapter for testing."""
    
    def get_bars(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime,
        timeframe: str = "1Day"
    ) -> List[OHLCV]:
        """Generate synthetic OHLCV bars."""
        bars = []
        current = start
        base_price = 450.0
        
        while current <= end:
            daily_change = random.uniform(-0.02, 0.02)
            open_price = base_price
            close_price = base_price * (1 + daily_change)
            high_price = max(open_price, close_price) * (1 + abs(random.uniform(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(random.uniform(0, 0.01)))
            
            bars.append(OHLCV(
                timestamp=current,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=random.uniform(50_000_000, 150_000_000),
            ))
            
            base_price = close_price
            current += timedelta(days=1)
        
        return bars
    
    def get_quote(self, symbol: str) -> Quote:
        """Generate synthetic quote."""
        price = 450.0 + random.uniform(-5, 5)
        spread = price * 0.0001
        
        return Quote(
            timestamp=datetime.now(),
            symbol=symbol,
            bid=price - spread / 2,
            ask=price + spread / 2,
            bid_size=random.uniform(100, 1000),
            ask_size=random.uniform(100, 1000),
            last=price,
            last_size=random.uniform(10, 500),
        )


class StaticNewsAdapter:
    """Static news adapter for testing."""
    
    def get_news(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime
    ) -> List[NewsArticle]:
        """Generate synthetic news articles."""
        articles = []
        current = start
        
        sentiments = ["positive", "negative", "neutral"]
        headlines = [
            f"{symbol} Shows Strong Performance",
            f"Analysts Upgrade {symbol} Target",
            f"{symbol} Faces Headwinds",
            f"Market Volatility Affects {symbol}",
            f"{symbol} Announces Strategic Initiative",
        ]
        
        # Generate 1-3 articles per day
        while current <= end:
            for _ in range(random.randint(1, 3)):
                sentiment_type = random.choice(sentiments)
                sentiment_value = {
                    "positive": random.uniform(0.3, 1.0),
                    "negative": random.uniform(-1.0, -0.3),
                    "neutral": random.uniform(-0.2, 0.2),
                }[sentiment_type]
                
                articles.append(NewsArticle(
                    timestamp=current + timedelta(hours=random.randint(0, 23)),
                    headline=random.choice(headlines),
                    summary=f"Article about {symbol} with {sentiment_type} sentiment.",
                    source="Test News Source",
                    url=f"https://example.com/news/{symbol}",
                    sentiment=sentiment_value,
                ))
            
            current += timedelta(days=1)
        
        return articles
