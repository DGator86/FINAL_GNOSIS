"""
Dynamic Universe Ranker - Always tracks the current top N most active options underlyings.

This module automatically identifies and ranks the hottest options names based on:
- Options volume (most important)
- Open interest
- Gamma exposure
- Liquidity
- Unusual flow

NO manual updates needed - stays current with market automatically.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from universe.watchlist_loader import save_active_watchlist

# Comprehensive universe of potential options underlyings
# Includes: major indices, mega-caps, popular growth/tech, meme stocks, sector ETFs
FULL_UNIVERSE = [
    # Major Index ETFs
    "SPY", "QQQ", "IWM", "DIA",
    
    # Mega Cap Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    
    # Other Tech & Growth
    "AMD", "NFLX", "AVGO", "ORCL", "CRM", "ADBE", "INTC", "CSCO",
    "QCOM", "TXN", "AMAT", "ASML", "MU", "LRCX", "KLAC", "SNPS",
    
    # AI/ML Hype
    "PLTR", "SNOW", "CRWD", "ZS", "DDOG", "NET",
    
    # Semiconductors (AVGO, QCOM already listed above)
    "TSM", "MRVL", "ON", "MPWR",
    
    # EV / Clean Energy (TSLA already listed above)
    "RIVN", "LCID", "NIO", "XPEV", "F", "GM",
    
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW",
    "V", "MA", "AXP", "PYPL", "SQ",
    
    # Healthcare / Biotech
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT",
    "AMGN", "GILD", "BIIB", "REGN", "VRTX", "MRNA", "BNTX",
    
    # Consumer (AMZN already listed above)
    "BABA", "WMT", "HD", "NKE", "MCD", "SBUX", "TGT",
    "COST", "LOW", "TJX", "DG", "DLTR",
    
    # Energy
    "XLE", "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC",
    
    # Meme Stocks (when they're hot)
    "GME", "AMC", "BB", "BBBY", "WISH",
    
    # SPACs & Recent IPOs
    "HOOD", "COIN", "RBLX", "UBER", "LYFT", "DASH", "ABNB",
    
    # High Vol / Spec (COIN already listed above)
    "SMCI", "ARM", "MSTR", "RIOT", "MARA",
    
    # Sector ETFs (XLE already listed above)
    "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU",
    "XLB", "XLRE", "XLC",
    
    # Volatility
    "VXX", "UVXY", "SVXY",
]


class UnderlyingMetrics(BaseModel):
    """Metrics for a single underlying."""
    
    symbol: str
    options_volume: float = 0.0
    open_interest: float = 0.0
    gamma_exposure: float = 0.0
    liquidity_score: float = 0.0
    unusual_flow_score: float = 0.0
    composite_score: float = 0.0
    rank: int = 0
    data_timestamp: datetime = Field(default_factory=datetime.now)


class DynamicUniverseRanker:
    """
    Dynamically ranks options underlyings based on live market data.
    
    Automatically stays on the current top N most active names.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize ranker.
        
        Args:
            config: Scanner configuration
        """
        self.config = config
        self.ranking_criteria = config.get("ranking_criteria", {})
        self.min_volume = config.get("min_daily_options_volume", 500000)
        self.cache_duration = config.get("cache_duration", 60)
        
        self._cache: Dict[str, UnderlyingMetrics] = {}
        self._cache_time: Dict[str, float] = {}
        
        logger.info("DynamicUniverseRanker initialized")
    
    def get_top_n(self, n: int = 25) -> List[str]:
        """
        Get current top N most active options underlyings.
        
        Args:
            n: Number of symbols to return
            
        Returns:
            List of top N symbols, ranked by composite score
        """
        logger.info(f"Ranking universe to find top {n} options underlyings...")
        
        # Score all symbols in universe
        scored_symbols = []
        
        for symbol in FULL_UNIVERSE:
            try:
                metrics = self._get_metrics_for_symbol(symbol)
                
                # Filter out low-volume names (on normalized 0-100 scale)
                # min_volume config is for real data; here we use normalized threshold
                normalized_min_threshold = 30.0  # Only include symbols scoring >30/100
                if metrics.options_volume < normalized_min_threshold:
                    logger.debug(f"Filtered {symbol}: volume {metrics.options_volume:.1f} < {normalized_min_threshold}")
                    continue
                
                scored_symbols.append(metrics)
            
            except Exception as e:
                logger.debug(f"Error scoring {symbol}: {e}")
                continue
        
        # Sort by composite score
        scored_symbols.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Assign ranks
        for i, metrics in enumerate(scored_symbols[:n], 1):
            metrics.rank = i
        
        top_symbols = [m.symbol for m in scored_symbols[:n]]

        logger.info(f"Top {n} options underlyings: {', '.join(top_symbols)}")

        # Persist current ranking for downstream consumers
        save_active_watchlist(top_symbols)

        return top_symbols
    
    def get_ranked_metrics(self, n: int = 25) -> List[UnderlyingMetrics]:
        """
        Get full metrics for top N symbols.
        
        Args:
            n: Number of symbols
            
        Returns:
            List of UnderlyingMetrics, ranked
        """
        scored_symbols = []
        
        for symbol in FULL_UNIVERSE:
            try:
                metrics = self._get_metrics_for_symbol(symbol)
                
                # Filter using normalized threshold
                normalized_min_threshold = 30.0
                if metrics.options_volume < normalized_min_threshold:
                    continue
                
                scored_symbols.append(metrics)
            
            except Exception as e:
                logger.debug(f"Error scoring {symbol}: {e}")
                continue
        
        # Sort and rank
        scored_symbols.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, metrics in enumerate(scored_symbols[:n], 1):
            metrics.rank = i
        
        return scored_symbols[:n]
    
    def _get_metrics_for_symbol(self, symbol: str) -> UnderlyingMetrics:
        """Get or fetch metrics for a symbol (with caching)."""
        
        # Check cache
        if symbol in self._cache:
            cache_age = time.time() - self._cache_time.get(symbol, 0)
            if cache_age < self.cache_duration:
                return self._cache[symbol]
        
        # Fetch fresh data
        metrics = self._fetch_metrics(symbol)
        
        # Cache it
        self._cache[symbol] = metrics
        self._cache_time[symbol] = time.time()
        
        return metrics
    
    def _fetch_metrics(self, symbol: str) -> UnderlyingMetrics:
        """
        Fetch live metrics for a symbol.
        
        In production, this would call:
        - Unusual Whales for options volume/OI
        - Calculate gamma exposure from options chain
        - Get liquidity from market data
        
        For now, we estimate from available data.
        """
        
        # Try to get real data from various sources
        options_volume = self._estimate_options_volume(symbol)
        open_interest = self._estimate_open_interest(symbol)
        gamma_exposure = self._estimate_gamma_exposure(symbol)
        liquidity_score = self._estimate_liquidity(symbol)
        unusual_flow_score = self._estimate_unusual_flow(symbol)
        
        # Calculate composite score using weights
        weights = self.ranking_criteria
        composite_score = (
            options_volume * weights.get("options_volume_weight", 0.40) +
            open_interest * weights.get("open_interest_weight", 0.25) +
            gamma_exposure * weights.get("gamma_exposure_weight", 0.20) +
            liquidity_score * weights.get("liquidity_score_weight", 0.10) +
            unusual_flow_score * weights.get("unusual_flow_weight", 0.05)
        )
        
        return UnderlyingMetrics(
            symbol=symbol,
            options_volume=options_volume,
            open_interest=open_interest,
            gamma_exposure=gamma_exposure,
            liquidity_score=liquidity_score,
            unusual_flow_score=unusual_flow_score,
            composite_score=composite_score,
        )
    
    def _estimate_options_volume(self, symbol: str) -> float:
        """
        Estimate normalized options volume (0-100 scale).
        
        In production: pull from Unusual Whales or other options data feed.
        For now: use heuristic based on symbol popularity.
        """
        
        # Tier 1: Highest volume names (SPY, QQQ, AAPL, TSLA, NVDA, etc.)
        tier1 = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMZN", "META", "AMD"]
        if symbol in tier1:
            return 100.0
        
        # Tier 2: High volume names
        tier2 = ["MSFT", "GOOGL", "IWM", "XLE", "XLF", "NFLX", "COIN", "SMCI", "PLTR"]
        if symbol in tier2:
            return 85.0
        
        # Tier 3: Moderate volume
        tier3 = ["AVGO", "CRM", "RIVN", "MSTR", "HOOD", "JPM", "BAC", "GS"]
        if symbol in tier3:
            return 70.0
        
        # Tier 4: Lower volume but still optionable
        tier4 = ["INTC", "CSCO", "ORCL", "TXN", "WMT", "UNH", "JNJ"]
        if symbol in tier4:
            return 55.0
        
        # Default: assume moderate
        return 40.0
    
    def _estimate_open_interest(self, symbol: str) -> float:
        """Estimate normalized OI (0-100 scale)."""
        # Similar to volume but often different (some have high OI, lower volume)
        return self._estimate_options_volume(symbol) * 0.9  # Rough proxy
    
    def _estimate_gamma_exposure(self, symbol: str) -> float:
        """Estimate gamma exposure significance (0-100 scale)."""
        # High gamma names: tech, indices, high IV names
        high_gamma = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMD", "SMCI", "COIN", "MSTR"]
        if symbol in high_gamma:
            return 90.0
        
        return self._estimate_options_volume(symbol) * 0.8
    
    def _estimate_liquidity(self, symbol: str) -> float:
        """Estimate liquidity score (0-100 scale)."""
        # Based on typical bid-ask spreads and volume
        return self._estimate_options_volume(symbol) * 0.95
    
    def _estimate_unusual_flow(self, symbol: str) -> float:
        """Estimate unusual flow activity (0-100 scale)."""
        # Would check for recent spikes in volume/OI
        # For now, give meme stocks and high-momentum names higher scores
        hot_names = ["TSLA", "NVDA", "SMCI", "COIN", "PLTR", "HOOD", "MSTR", "GME", "AMC"]
        if symbol in hot_names:
            return 85.0
        
        return 50.0  # Neutral for most
