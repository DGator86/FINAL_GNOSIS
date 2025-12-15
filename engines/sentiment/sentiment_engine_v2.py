"""Sentiment Engine v2 - transformer-powered, flow-aware scoring.

# NEW: Adds Hugging Face transformers, Unusual Whales flow sentiment, and
# multi-timeframe decay with source weighting.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

try:  # pragma: no cover - optional heavy dependency
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None

from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter
from schemas.core_schemas import SentimentSnapshot


class SentimentEngineV2:
    """Advanced NLP sentiment engine with options flow overlay."""

    def __init__(
        self,
        unusual_whales_adapter: Optional[UnusualWhalesAdapter] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.unusual_whales_adapter = unusual_whales_adapter
        self.config = config or {}
        self.prev_score: float = 0.0
        self._nlp = pipeline("sentiment-analysis") if pipeline else None
        logger.info("SentimentEngineV2 initialized with transformer pipeline=%s", bool(self._nlp))

    def run(self, text: str, symbol: str, timestamp: datetime) -> SentimentSnapshot:
        """Run sentiment on free text plus options flow overlay."""

        base_score = self._analyze_text(text)
        flow_score = self._flow_sentiment(symbol)
        combined_flow = self._combine_flow_scores(base_score, flow_score)
        mtf_score = self._mtf_decay(combined_flow)

        relevance = self._keyword_relevance(text)
        intensity = abs(mtf_score)
        confidence = min(1.0, 0.5 + 0.5 * intensity)

        snapshot = SentimentSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            sentiment_score=combined_flow,
            news_sentiment=base_score,
            flow_sentiment=flow_score,
            technical_sentiment=0.0,
            confidence=confidence,
            intensity=intensity,
            relevance=relevance,
            mtf_score=mtf_score,
        )
        # Test: assert snapshot.intensity >= 0
        self.prev_score = snapshot.sentiment_score
        return snapshot

    def _analyze_text(self, text: str) -> float:
        if not text:
            return 0.0
        if not self._nlp:
            logger.debug("Transformers pipeline unavailable; returning neutral score")
            return 0.0
        result = self._nlp(text[:512])[0]
        label = result.get("label", "NEUTRAL").upper()
        score = result.get("score", 0.0)
        signed_score = score if label == "POSITIVE" else -score
        return signed_score

    def _flow_sentiment(self, symbol: str) -> float:
        if not self.unusual_whales_adapter:
            return 0.0
        try:
            flow = self.unusual_whales_adapter.get_unusual_activity(symbol) or []
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Flow sentiment unavailable: {exc}")
            return 0.0

        premiums = Counter()
        for trade in flow:
            premium = trade.get("premium", 0.0)
            direction = trade.get("type", "").lower()
            if "call" in direction or "bull" in direction:
                premiums["bullish"] += premium
            elif "put" in direction or "bear" in direction:
                premiums["bearish"] += premium

        total = premiums["bullish"] + premiums["bearish"]
        if total == 0:
            return 0.0
        return (premiums["bullish"] - premiums["bearish"]) / total

    def _combine_flow_scores(self, text_score: float, flow_score: float) -> float:
        flow_weight = self.config.get("flow_weight", 0.4)
        text_weight = self.config.get("news_weight", 0.6)
        raw = (text_score * text_weight) + (flow_score * flow_weight)
        return float(np.clip(raw, -1.0, 1.0))

    def _mtf_decay(self, score: float) -> float:
        decayed = (score * 0.8) + (self.prev_score * 0.2)
        return float(decayed)

    def _keyword_relevance(self, text: str) -> float:
        keywords = self.config.get("keywords", ["earnings", "guidance", "flow"])
        if not text:
            return 0.0
        count = sum(text.lower().count(k.lower()) for k in keywords)
        return count / max(len(text.split()), 1)


__all__ = ["SentimentEngineV2"]
