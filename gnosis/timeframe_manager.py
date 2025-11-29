"""Multi-Timeframe Data Management.

Aggregates 1-minute bars into higher timeframes (5m, 15m, 30m, 1h, 4h, 1d)
for multi-timeframe analysis and confidence building.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class OHLCV:
    """OHLCV bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @classmethod
    def aggregate(cls, bars: List['OHLCV']) -> 'OHLCV':
        """Aggregate multiple bars into one."""
        if not bars:
            raise ValueError("Cannot aggregate empty bar list")
        
        return cls(
            timestamp=bars[-1].timestamp,  # Use last bar's timestamp
            open=bars[0].open,
            high=max(b.high for b in bars),
            low=min(b.low for b in bars),
            close=bars[-1].close,
            volume=sum(b.volume for b in bars)
        )


class TimeframeManager:
    """Manages multi-timeframe bar collection and aggregation.
    
    Maintains bars for 7 timeframes:
    - 1Min (base)
    - 5Min
    - 15Min  
    - 30Min
    - 1Hour
    - 4Hour
    - 1Day
    
    Automatically aggregates 1-minute bars into higher timeframes.
    """
    
    TIMEFRAMES = {
        '1Min': 1,
        '5Min': 5,
        '15Min': 15,
        '30Min': 30,
        '1Hour': 60,
        '4Hour': 240,
        '1Day': 1440  # minutes in a day
    }
    
    def __init__(self, max_bars: int = 200):
        """Initialize TimeframeManager.
        
        Args:
            max_bars: Maximum number of bars to keep per timeframe
        """
        self.max_bars = max_bars
        
        # Storage for each timeframe (deque for efficient append/pop)
        self.bars: Dict[str, deque] = {
            tf: deque(maxlen=max_bars) for tf in self.TIMEFRAMES
        }
        
        # Track last aggregation time for each timeframe
        self.last_aggregation: Dict[str, Optional[datetime]] = {
            tf: None for tf in self.TIMEFRAMES
        }
        
        # Buffer for incomplete higher timeframe bars
        self.aggregation_buffer: Dict[str, List[OHLCV]] = {
            tf: [] for tf in self.TIMEFRAMES if tf != '1Min'
        }
        
        logger.info(f"TimeframeManager initialized | max_bars={max_bars}")
    
    def add_1min_bar(self, bar_dict: Dict) -> None:
        """Add 1-minute bar and trigger aggregations.
        
        Args:
            bar_dict: Dictionary with keys: timestamp, open, high, low, close, volume
        """
        # Convert to OHLCV object
        bar = OHLCV(
            timestamp=bar_dict['t_event'],
            open=bar_dict['open'],
            high=bar_dict['high'],
            low=bar_dict['low'],
            close=bar_dict['close'],
            volume=bar_dict['volume']
        )
        
        # Add to 1Min storage
        self.bars['1Min'].append(bar)
        
        # Trigger aggregations for higher timeframes
        self._aggregate_5min(bar)
        self._aggregate_15min(bar)
        self._aggregate_30min(bar)
        self._aggregate_1hour(bar)
        self._aggregate_4hour(bar)
        self._aggregate_1day(bar)
        
        logger.debug(f"Added 1min bar | bars: {self.get_bar_counts()}")
    
    def _aggregate_5min(self, bar: OHLCV) -> None:
        """Aggregate to 5-minute bars."""
        self.aggregation_buffer['5Min'].append(bar)
        
        # Check if we have 5 minutes worth
        if len(self.aggregation_buffer['5Min']) >= 5:
            agg_bar = OHLCV.aggregate(self.aggregation_buffer['5Min'][:5])
            self.bars['5Min'].append(agg_bar)
            self.aggregation_buffer['5Min'] = self.aggregation_buffer['5Min'][5:]
            logger.debug(f"Aggregated 5Min bar | {agg_bar.timestamp}")
    
    def _aggregate_15min(self, bar: OHLCV) -> None:
        """Aggregate to 15-minute bars."""
        self.aggregation_buffer['15Min'].append(bar)
        
        if len(self.aggregation_buffer['15Min']) >= 15:
            agg_bar = OHLCV.aggregate(self.aggregation_buffer['15Min'][:15])
            self.bars['15Min'].append(agg_bar)
            self.aggregation_buffer['15Min'] = self.aggregation_buffer['15Min'][15:]
            logger.debug(f"Aggregated 15Min bar | {agg_bar.timestamp}")
    
    def _aggregate_30min(self, bar: OHLCV) -> None:
        """Aggregate to 30-minute bars."""
        self.aggregation_buffer['30Min'].append(bar)
        
        if len(self.aggregation_buffer['30Min']) >= 30:
            agg_bar = OHLCV.aggregate(self.aggregation_buffer['30Min'][:30])
            self.bars['30Min'].append(agg_bar)
            self.aggregation_buffer['30Min'] = self.aggregation_buffer['30Min'][30:]
            logger.debug(f"Aggregated 30Min bar | {agg_bar.timestamp}")
    
    def _aggregate_1hour(self, bar: OHLCV) -> None:
        """Aggregate to 1-hour bars."""
        self.aggregation_buffer['1Hour'].append(bar)
        
        if len(self.aggregation_buffer['1Hour']) >= 60:
            agg_bar = OHLCV.aggregate(self.aggregation_buffer['1Hour'][:60])
            self.bars['1Hour'].append(agg_bar)
            self.aggregation_buffer['1Hour'] = self.aggregation_buffer['1Hour'][60:]
            logger.debug(f"Aggregated 1Hour bar | {agg_bar.timestamp}")
    
    def _aggregate_4hour(self, bar: OHLCV) -> None:
        """Aggregate to 4-hour bars."""
        self.aggregation_buffer['4Hour'].append(bar)
        
        if len(self.aggregation_buffer['4Hour']) >= 240:
            agg_bar = OHLCV.aggregate(self.aggregation_buffer['4Hour'][:240])
            self.bars['4Hour'].append(agg_bar)
            self.aggregation_buffer['4Hour'] = self.aggregation_buffer['4Hour'][240:]
            logger.debug(f"Aggregated 4Hour bar | {agg_bar.timestamp}")
    
    def _aggregate_1day(self, bar: OHLCV) -> None:
        """Aggregate to 1-day bars."""
        self.aggregation_buffer['1Day'].append(bar)
        
        # For daily bars, also check if it's a new day
        if len(self.aggregation_buffer['1Day']) >= 390:  # Market hours: 6.5h * 60min = 390min
            agg_bar = OHLCV.aggregate(self.aggregation_buffer['1Day'])
            self.bars['1Day'].append(agg_bar)
            self.aggregation_buffer['1Day'] = []
            logger.debug(f"Aggregated 1Day bar | {agg_bar.timestamp}")
    
    def get_bars(self, timeframe: str, count: Optional[int] = None) -> List[OHLCV]:
        """Get bars for a specific timeframe.
        
        Args:
            timeframe: '1Min', '5Min', '15Min', '30Min', '1Hour', '4Hour', '1Day'
            count: Number of recent bars to return (None = all)
        
        Returns:
            List of OHLCV bars
        """
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        bars = list(self.bars[timeframe])
        if count is not None:
            bars = bars[-count:]
        return bars
    
    def get_latest_bar(self, timeframe: str) -> Optional[OHLCV]:
        """Get most recent bar for a timeframe.
        
        Returns:
            Latest OHLCV bar or None if no bars available
        """
        bars = self.bars.get(timeframe, [])
        return bars[-1] if bars else None
    
    def get_bar_counts(self) -> Dict[str, int]:
        """Get number of bars available for each timeframe.
        
        Returns:
            Dictionary mapping timeframe to bar count
        """
        return {tf: len(self.bars[tf]) for tf in self.TIMEFRAMES}
    
    def get_all_timeframes_data(self) -> Dict[str, List[OHLCV]]:
        """Get all bars for all timeframes.
        
        Returns:
            Dictionary mapping timeframe to list of bars
        """
        return {tf: list(self.bars[tf]) for tf in self.TIMEFRAMES}
    
    def has_sufficient_data(self, min_bars_per_timeframe: int = 50) -> bool:
        """Check if all timeframes have sufficient data.
        
        Args:
            min_bars_per_timeframe: Minimum bars required per timeframe
        
        Returns:
            True if all timeframes have enough bars
        """
        counts = self.get_bar_counts()
        return all(count >= min_bars_per_timeframe for count in counts.values())
    
    def clear(self) -> None:
        """Clear all stored bars."""
        for tf in self.TIMEFRAMES:
            self.bars[tf].clear()
            if tf != '1Min':
                self.aggregation_buffer[tf].clear()
        logger.info("TimeframeManager cleared")


__all__ = ['TimeframeManager', 'OHLCV']
