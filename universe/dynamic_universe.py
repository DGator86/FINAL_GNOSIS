#!/usr/bin/env python3
"""
Dynamic Universe Manager

Manages a dynamic stock universe of 25 stocks with real-time ranking
to identify and closely watch the top 10 performers.

Universe Composition:
- Magnificent 7 (Core Tech Giants)
- Growth Leaders (High momentum tech/growth)
- Value Stalwarts (Established large caps)
- Sector Leaders (Best in class by sector)

Ranking Criteria:
- Momentum (price performance)
- Volume surge (unusual activity)
- Volatility regime
- Options flow signals
- Technical strength

Author: GNOSIS Trading System
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# UNIVERSE CONFIGURATION
# =============================================================================

@dataclass
class StockInfo:
    """Information about a stock in the universe."""
    symbol: str
    name: str
    sector: str
    category: str  # magnificent_7, growth, value, sector_leader
    market_cap: str  # mega, large, mid
    weight: float = 1.0  # Base weight in universe
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "sector": self.sector,
            "category": self.category,
            "market_cap": self.market_cap,
            "weight": self.weight,
        }


# The Dynamic Universe of 25 Stocks
DYNAMIC_UNIVERSE = [
    # === MAGNIFICENT 7 (Core Holdings) ===
    StockInfo("AAPL", "Apple Inc.", "Technology", "magnificent_7", "mega", 1.5),
    StockInfo("MSFT", "Microsoft Corp.", "Technology", "magnificent_7", "mega", 1.5),
    StockInfo("GOOGL", "Alphabet Inc.", "Technology", "magnificent_7", "mega", 1.5),
    StockInfo("AMZN", "Amazon.com Inc.", "Consumer Cyclical", "magnificent_7", "mega", 1.5),
    StockInfo("NVDA", "NVIDIA Corp.", "Technology", "magnificent_7", "mega", 1.5),
    StockInfo("META", "Meta Platforms", "Technology", "magnificent_7", "mega", 1.5),
    StockInfo("TSLA", "Tesla Inc.", "Consumer Cyclical", "magnificent_7", "mega", 1.5),
    
    # === GROWTH LEADERS ===
    StockInfo("AMD", "Advanced Micro Devices", "Technology", "growth", "large", 1.2),
    StockInfo("CRM", "Salesforce Inc.", "Technology", "growth", "large", 1.0),
    StockInfo("NFLX", "Netflix Inc.", "Communication", "growth", "large", 1.0),
    StockInfo("AVGO", "Broadcom Inc.", "Technology", "growth", "mega", 1.2),
    StockInfo("ADBE", "Adobe Inc.", "Technology", "growth", "large", 1.0),
    
    # === VALUE STALWARTS ===
    StockInfo("JPM", "JPMorgan Chase", "Financial", "value", "mega", 1.0),
    StockInfo("V", "Visa Inc.", "Financial", "value", "mega", 1.0),
    StockInfo("JNJ", "Johnson & Johnson", "Healthcare", "value", "mega", 0.8),
    StockInfo("UNH", "UnitedHealth Group", "Healthcare", "value", "mega", 1.0),
    StockInfo("HD", "Home Depot", "Consumer Cyclical", "value", "large", 0.9),
    
    # === SECTOR LEADERS ===
    StockInfo("XOM", "Exxon Mobil", "Energy", "sector_leader", "mega", 0.8),
    StockInfo("LLY", "Eli Lilly", "Healthcare", "sector_leader", "mega", 1.2),
    StockInfo("MA", "Mastercard", "Financial", "sector_leader", "large", 1.0),
    StockInfo("COST", "Costco Wholesale", "Consumer Defensive", "sector_leader", "large", 0.9),
    StockInfo("PEP", "PepsiCo Inc.", "Consumer Defensive", "sector_leader", "large", 0.7),
    StockInfo("ORCL", "Oracle Corp.", "Technology", "sector_leader", "large", 1.0),
    StockInfo("MRK", "Merck & Co.", "Healthcare", "sector_leader", "large", 0.8),
    StockInfo("BA", "Boeing Co.", "Industrials", "sector_leader", "large", 0.9),
]


@dataclass
class StockRanking:
    """Ranking data for a stock."""
    symbol: str
    rank: int
    score: float
    momentum_score: float
    volume_score: float
    volatility_score: float
    flow_score: float
    technical_score: float
    signals: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "rank": self.rank,
            "score": self.score,
            "momentum_score": self.momentum_score,
            "volume_score": self.volume_score,
            "volatility_score": self.volatility_score,
            "flow_score": self.flow_score,
            "technical_score": self.technical_score,
            "signals": self.signals,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class DynamicUniverseManager:
    """
    Manages the dynamic 25-stock universe and ranks the top 10.
    
    Features:
    - Real-time stock ranking based on multiple factors
    - Dynamic watchlist for top 10 performers
    - Automatic rebalancing signals
    - Sector and category exposure tracking
    """
    
    def __init__(self):
        self.universe = {s.symbol: s for s in DYNAMIC_UNIVERSE}
        self.rankings: Dict[str, StockRanking] = {}
        self.top_10: List[str] = []
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.history_cache: Dict[str, Any] = {}
        
        # Load environment
        self._load_env()
        
        print(f"DynamicUniverseManager initialized with {len(self.universe)} stocks")
    
    def _load_env(self):
        """Load environment variables."""
        env_path = Path(__file__).parent.parent / ".env"
        try:
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    os.environ[key] = value
        except FileNotFoundError:
            pass
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols in the universe."""
        return list(self.universe.keys())
    
    def get_magnificent_7(self) -> List[str]:
        """Get Magnificent 7 symbols."""
        return [s for s, info in self.universe.items() if info.category == "magnificent_7"]
    
    def get_by_category(self, category: str) -> List[str]:
        """Get symbols by category."""
        return [s for s, info in self.universe.items() if info.category == category]
    
    def get_by_sector(self, sector: str) -> List[str]:
        """Get symbols by sector."""
        return [s for s, info in self.universe.items() if info.sector == sector]
    
    def fetch_current_prices(self) -> Dict[str, Dict[str, Any]]:
        """Fetch current prices for all stocks."""
        import urllib.request
        import urllib.error
        
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not api_secret:
            print("Warning: Alpaca API credentials not found")
            return {}
        
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        
        symbols = ",".join(self.get_all_symbols())
        url = f"https://data.alpaca.markets/v2/stocks/quotes/latest?symbols={symbols}"
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                quotes = data.get("quotes", {})
                
                for symbol, quote in quotes.items():
                    bid = float(quote.get("bp", 0))
                    ask = float(quote.get("ap", 0))
                    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                    
                    self.price_cache[symbol] = {
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "spread": ask - bid,
                        "timestamp": datetime.now(),
                    }
                
                return self.price_cache
        except Exception as e:
            print(f"Error fetching prices: {e}")
            return self.price_cache
    
    def fetch_historical_data(self, days: int = 30) -> Dict[str, Any]:
        """Fetch historical data for ranking calculations."""
        import urllib.request
        
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not api_secret:
            return {}
        
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        
        end = datetime.now()
        start = end - timedelta(days=days)
        
        for symbol in self.get_all_symbols():
            try:
                url = (
                    f"https://data.alpaca.markets/v2/stocks/{symbol}/bars?"
                    f"timeframe=1Day&start={start.strftime('%Y-%m-%d')}&end={end.strftime('%Y-%m-%d')}"
                )
                
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    bars = data.get("bars", [])
                    
                    if bars:
                        closes = [b["c"] for b in bars]
                        volumes = [b["v"] for b in bars]
                        highs = [b["h"] for b in bars]
                        lows = [b["l"] for b in bars]
                        
                        self.history_cache[symbol] = {
                            "closes": closes,
                            "volumes": volumes,
                            "highs": highs,
                            "lows": lows,
                            "bars": len(bars),
                            "latest_close": closes[-1] if closes else 0,
                            "avg_volume": np.mean(volumes) if volumes else 0,
                        }
            except Exception as e:
                print(f"Error fetching history for {symbol}: {e}")
        
        return self.history_cache
    
    def calculate_momentum_score(self, symbol: str) -> float:
        """Calculate momentum score (0-100)."""
        history = self.history_cache.get(symbol, {})
        closes = history.get("closes", [])
        
        if len(closes) < 5:
            return 50.0
        
        # Short-term momentum (5-day)
        short_return = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 0
        
        # Medium-term momentum (20-day)
        if len(closes) >= 20:
            med_return = (closes[-1] - closes[-20]) / closes[-20] if closes[-20] > 0 else 0
        else:
            med_return = short_return
        
        # Score: combine short and medium momentum
        momentum = (short_return * 0.6 + med_return * 0.4)
        
        # Normalize to 0-100 scale
        score = 50 + (momentum * 500)  # 10% move = 100 points
        return max(0, min(100, score))
    
    def calculate_volume_score(self, symbol: str) -> float:
        """Calculate volume surge score (0-100)."""
        history = self.history_cache.get(symbol, {})
        volumes = history.get("volumes", [])
        
        if len(volumes) < 10:
            return 50.0
        
        recent_vol = np.mean(volumes[-3:])
        avg_vol = np.mean(volumes[:-3]) if len(volumes) > 3 else recent_vol
        
        if avg_vol > 0:
            vol_ratio = recent_vol / avg_vol
            score = 50 + (vol_ratio - 1) * 50  # 2x volume = 100
        else:
            score = 50.0
        
        return max(0, min(100, score))
    
    def calculate_volatility_score(self, symbol: str) -> float:
        """Calculate volatility regime score (0-100)."""
        history = self.history_cache.get(symbol, {})
        closes = history.get("closes", [])
        
        if len(closes) < 10:
            return 50.0
        
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized %
        
        # Sweet spot: 20-40% volatility
        if 20 <= volatility <= 40:
            score = 80 + (10 - abs(volatility - 30))
        elif volatility < 20:
            score = 50 + volatility * 1.5
        else:
            score = max(30, 100 - (volatility - 40) * 2)
        
        return max(0, min(100, score))
    
    def calculate_flow_score(self, symbol: str) -> float:
        """Calculate options flow score (0-100)."""
        # Placeholder - would integrate with Unusual Whales
        # For now, use a weighted random based on category
        info = self.universe.get(symbol)
        if not info:
            return 50.0
        
        base_score = 50.0
        
        # Magnificent 7 gets a boost
        if info.category == "magnificent_7":
            base_score += 10
        
        # Growth stocks tend to have more options activity
        if info.category == "growth":
            base_score += 5
        
        # Add some variance based on recent momentum
        momentum = self.calculate_momentum_score(symbol)
        flow_adjustment = (momentum - 50) * 0.2
        
        return max(0, min(100, base_score + flow_adjustment))
    
    def calculate_technical_score(self, symbol: str) -> float:
        """Calculate technical strength score (0-100)."""
        history = self.history_cache.get(symbol, {})
        closes = history.get("closes", [])
        highs = history.get("highs", [])
        lows = history.get("lows", [])
        
        if len(closes) < 20:
            return 50.0
        
        score = 50.0
        
        # Above 20-day SMA
        sma_20 = np.mean(closes[-20:])
        if closes[-1] > sma_20:
            score += 15
        
        # Above 50-day SMA (if available)
        if len(closes) >= 50:
            sma_50 = np.mean(closes[-50:])
            if closes[-1] > sma_50:
                score += 10
        
        # Near 52-week high (proxy with available data)
        period_high = max(highs)
        if closes[-1] >= period_high * 0.95:
            score += 15
        elif closes[-1] >= period_high * 0.90:
            score += 10
        
        # RSI approximation
        gains = []
        losses = []
        for i in range(1, min(14, len(closes))):
            change = closes[-i] - closes[-(i+1)]
            if change > 0:
                gains.append(change)
            else:
                losses.append(abs(change))
        
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0.0001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # RSI between 40-70 is ideal
        if 40 <= rsi <= 70:
            score += 10
        elif rsi > 70:
            score -= 5  # Overbought
        elif rsi < 30:
            score -= 5  # Oversold (might bounce but risky)
        
        return max(0, min(100, score))
    
    def rank_stocks(self) -> List[StockRanking]:
        """Rank all stocks and return sorted list."""
        
        # Fetch fresh data
        self.fetch_current_prices()
        self.fetch_historical_data(days=60)
        
        rankings = []
        
        for symbol, info in self.universe.items():
            momentum = self.calculate_momentum_score(symbol)
            volume = self.calculate_volume_score(symbol)
            volatility = self.calculate_volatility_score(symbol)
            flow = self.calculate_flow_score(symbol)
            technical = self.calculate_technical_score(symbol)
            
            # Weighted composite score
            weights = {
                "momentum": 0.30,
                "volume": 0.15,
                "volatility": 0.15,
                "flow": 0.20,
                "technical": 0.20,
            }
            
            composite = (
                momentum * weights["momentum"] +
                volume * weights["volume"] +
                volatility * weights["volatility"] +
                flow * weights["flow"] +
                technical * weights["technical"]
            )
            
            # Apply category weight
            composite *= info.weight
            
            # Signals
            signals = {
                "trend": "bullish" if momentum > 60 else ("bearish" if momentum < 40 else "neutral"),
                "volume_surge": volume > 70,
                "technically_strong": technical > 65,
                "high_activity": flow > 60,
            }
            
            rankings.append(StockRanking(
                symbol=symbol,
                rank=0,  # Will be set after sorting
                score=composite,
                momentum_score=momentum,
                volume_score=volume,
                volatility_score=volatility,
                flow_score=flow,
                technical_score=technical,
                signals=signals,
                last_updated=datetime.now(),
            ))
        
        # Sort by score descending
        rankings.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for i, r in enumerate(rankings):
            r.rank = i + 1
            self.rankings[r.symbol] = r
        
        # Update top 10
        self.top_10 = [r.symbol for r in rankings[:10]]
        
        return rankings
    
    def get_top_10(self) -> List[StockRanking]:
        """Get the top 10 ranked stocks."""
        if not self.rankings:
            self.rank_stocks()
        
        return [self.rankings[s] for s in self.top_10 if s in self.rankings]
    
    def get_watchlist(self) -> Dict[str, Any]:
        """Get the current watchlist with detailed info."""
        top_10 = self.get_top_10()
        
        watchlist = {
            "timestamp": datetime.now().isoformat(),
            "top_10": [],
            "summary": {
                "avg_score": 0,
                "bullish_count": 0,
                "volume_surge_count": 0,
                "sectors": {},
                "categories": {},
            },
        }
        
        for ranking in top_10:
            info = self.universe.get(ranking.symbol, StockInfo(ranking.symbol, "", "", "", ""))
            price_data = self.price_cache.get(ranking.symbol, {})
            
            stock_data = {
                "rank": ranking.rank,
                "symbol": ranking.symbol,
                "name": info.name,
                "sector": info.sector,
                "category": info.category,
                "price": price_data.get("mid", 0),
                "score": ranking.score,
                "scores": {
                    "momentum": ranking.momentum_score,
                    "volume": ranking.volume_score,
                    "volatility": ranking.volatility_score,
                    "flow": ranking.flow_score,
                    "technical": ranking.technical_score,
                },
                "signals": ranking.signals,
            }
            
            watchlist["top_10"].append(stock_data)
            
            # Update summary
            watchlist["summary"]["avg_score"] += ranking.score
            if ranking.signals.get("trend") == "bullish":
                watchlist["summary"]["bullish_count"] += 1
            if ranking.signals.get("volume_surge"):
                watchlist["summary"]["volume_surge_count"] += 1
            
            # Track sectors
            if info.sector not in watchlist["summary"]["sectors"]:
                watchlist["summary"]["sectors"][info.sector] = 0
            watchlist["summary"]["sectors"][info.sector] += 1
            
            # Track categories
            if info.category not in watchlist["summary"]["categories"]:
                watchlist["summary"]["categories"][info.category] = 0
            watchlist["summary"]["categories"][info.category] += 1
        
        watchlist["summary"]["avg_score"] /= len(top_10) if top_10 else 1
        
        return watchlist
    
    def print_rankings(self):
        """Print formatted rankings."""
        rankings = self.rank_stocks()
        
        print("\n" + "="*90)
        print("  DYNAMIC UNIVERSE RANKINGS - TOP 25 STOCKS")
        print("="*90)
        print(f"  Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*90)
        
        print(f"\n  {'Rank':<6} {'Symbol':<8} {'Name':<20} {'Score':>8} {'Mom':>6} {'Vol':>6} {'Tech':>6} {'Trend':<10}")
        print("  " + "-"*82)
        
        for r in rankings:
            info = self.universe.get(r.symbol, StockInfo(r.symbol, "", "", "", ""))
            trend = r.signals.get("trend", "neutral")
            
            # Color coding
            if r.rank <= 10:
                color = "\033[92m"  # Green for top 10
            elif r.rank <= 15:
                color = "\033[93m"  # Yellow
            else:
                color = "\033[0m"  # Normal
            reset = "\033[0m"
            
            marker = "*" if r.rank <= 10 else " "
            
            print(f"  {color}{marker}{r.rank:<5} {r.symbol:<8} {info.name[:18]:<20} {r.score:>7.1f} "
                  f"{r.momentum_score:>5.1f} {r.volume_score:>5.1f} {r.technical_score:>5.1f} {trend:<10}{reset}")
        
        # Print top 10 summary
        print("\n" + "="*90)
        print("  TOP 10 WATCHLIST")
        print("="*90)
        
        top_10 = self.get_top_10()
        mag7_in_top10 = sum(1 for r in top_10 if self.universe[r.symbol].category == "magnificent_7")
        
        print(f"\n  Magnificent 7 in Top 10: {mag7_in_top10}/7")
        print(f"  Average Score: {np.mean([r.score for r in top_10]):.1f}")
        print(f"  Bullish Signals: {sum(1 for r in top_10 if r.signals.get('trend') == 'bullish')}/10")
        print(f"  Volume Surges: {sum(1 for r in top_10 if r.signals.get('volume_surge'))}/10")
        
        print("\n  Sector Distribution:")
        sectors = {}
        for r in top_10:
            sector = self.universe[r.symbol].sector
            sectors[sector] = sectors.get(sector, 0) + 1
        for sector, count in sorted(sectors.items(), key=lambda x: -x[1]):
            print(f"    {sector}: {count}")
        
        print("\n" + "="*90)
    
    def export_config(self, filepath: str = None):
        """Export top 10 as trading configuration."""
        top_10 = self.get_top_10()
        symbols = [r.symbol for r in top_10]
        
        config = {
            "TRADING_SYMBOLS": ",".join(symbols),
            "UNIVERSE_SIZE": len(self.universe),
            "WATCHLIST_SIZE": 10,
            "LAST_UPDATED": datetime.now().isoformat(),
            "TOP_10": [r.to_dict() for r in top_10],
        }
        
        if filepath:
            with open(filepath, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Config exported to {filepath}")
        
        return config


def main():
    """Run the dynamic universe manager."""
    manager = DynamicUniverseManager()
    manager.print_rankings()
    
    # Export config
    watchlist = manager.get_watchlist()
    print(f"\nTop 10 Symbols: {', '.join(manager.top_10)}")


if __name__ == "__main__":
    main()
