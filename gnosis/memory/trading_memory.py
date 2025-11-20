"""Lightweight in-memory store for trading memories.

This is a simplified drop-in replacement for the documented memory system so
examples and demos can run without extra infrastructure.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MemoryItem:
    kind: str
    content: str
    score: float
    symbol: str
    metadata: Dict[str, object] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class TradingMemoryStore:
    def __init__(self, decay_half_life: float = 3600.0):
        self.decay_half_life = decay_half_life
        self._items: List[MemoryItem] = []

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()

    def add(
        self,
        kind: str,
        content: str,
        score: float,
        symbol: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> MemoryItem:
        item = MemoryItem(kind, content, score, symbol, metadata or {})
        self._items.append(item)
        return item

    def _decay_factor(self, item: MemoryItem) -> float:
        age = max(time.time() - item.timestamp, 0)
        return 0.5 ** (age / self.decay_half_life)

    def _score(self, item: MemoryItem) -> float:
        return item.score * self._decay_factor(item)

    def search(
        self,
        query: str | None = None,
        symbol: Optional[str] = None,
        kind: Optional[str] = None,
        topk: int = 5,
    ) -> List[MemoryItem]:
        results = []
        for item in self._items:
            if symbol and item.symbol != symbol:
                continue
            if kind and item.kind != kind:
                continue
            if query and query.lower() not in item.content.lower():
                continue
            results.append((self._score(item), item))

        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:topk]]

    def get_recent(self, symbol: Optional[str] = None, n: int = 5) -> List[MemoryItem]:
        items = [i for i in self._items if not symbol or i.symbol == symbol]
        items.sort(key=lambda i: i.timestamp, reverse=True)
        return items[:n]

    def summarize(self, symbol: str, lookback_hours: int = 24) -> Dict[str, object]:
        cutoff = time.time() - lookback_hours * 3600
        recent = [i for i in self._items if i.symbol == symbol and i.timestamp >= cutoff]
        by_kind: Dict[str, int] = {}
        for item in recent:
            by_kind[item.kind] = by_kind.get(item.kind, 0) + 1

        avg_score = sum(i.score for i in recent) / len(recent) if recent else 0.0

        return {
            "total_memories": len(self._items),
            "recent_count": len(recent),
            "avg_score": avg_score,
            "by_kind": by_kind,
        }


__all__ = ["TradingMemoryStore", "MemoryItem"]
