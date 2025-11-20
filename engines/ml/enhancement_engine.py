"""Composite ML enhancement engine that wires forecasting, similarity, anomalies, and curriculum RL."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from loguru import logger

from engines.inputs.market_data_adapter import MarketDataAdapter
from engines.ml.anomaly import AnomalyDetector
from engines.ml.curriculum import CurriculumRLEvaluator
from engines.ml.forecasting import KatsForecasterAdapter
from engines.ml.similarity import FaissRegimeRetriever
from schemas.core_schemas import (
    MLEnhancementSnapshot,
    PolicyRecommendation,
    RegimeSimilaritySnapshot,
    ForecastSnapshot,
    AnomalySnapshot,
    PipelineResult,
)


class MLEnhancementEngine:
    """Implements cross-cutting ML upgrades referenced in the suggestion doc."""

    def __init__(
        self,
        market_adapter: MarketDataAdapter,
        forecaster: KatsForecasterAdapter,
        similarity_retriever: FaissRegimeRetriever,
        anomaly_detector: AnomalyDetector,
        rl_evaluator: CurriculumRLEvaluator,
    ) -> None:
        self.market_adapter = market_adapter
        self.forecaster = forecaster
        self.similarity_retriever = similarity_retriever
        self.anomaly_detector = anomaly_detector
        self.rl_evaluator = rl_evaluator

    def enhance(self, pipeline_result: PipelineResult, timestamp: datetime) -> MLEnhancementSnapshot:
        """Generate ML enhancements using outputs from the base engines."""

        symbol = pipeline_result.symbol
        logger.info(f"Running ML enhancement engine for {symbol}")
        horizon = self.forecaster.forecast_horizon
        latest_bar = self._get_latest_bar(symbol, timestamp)
        forecast_payload = self.forecaster.forecast(symbol, timestamp, horizon=horizon)
        forecast_snapshot = ForecastSnapshot(**forecast_payload)

        feature_vector = self._build_feature_vector(pipeline_result, latest_bar)
        similarity_payload = self.similarity_retriever.query(feature_vector)
        similarity_snapshot = RegimeSimilaritySnapshot(
            similarity_score=float(similarity_payload.get("similarity", 0.0)),
            neighbors=similarity_payload.get("neighbors", []),
            feature_vector=feature_vector,
        )
        anomaly_payload = self.anomaly_detector.score(feature_vector)
        anomaly_snapshot = AnomalySnapshot(
            score=float(anomaly_payload.get("score", 0.0)),
            flagged=bool(anomaly_payload.get("flagged", False)),
            feature_vector=feature_vector,
            metadata={"vector_length": len(feature_vector)},
        )

        # update similarity history with the latest feature vector and metadata
        self.similarity_retriever.add_reference(
            feature_vector,
            label=pipeline_result.hedge_snapshot.regime
            if pipeline_result.hedge_snapshot
            else "unknown",
            metadata={"timestamp": timestamp.isoformat()},
        )

        sentiment_score = (
            pipeline_result.sentiment_snapshot.sentiment_score
            if pipeline_result.sentiment_snapshot
            else 0.0
        )
        policy_signal = self.rl_evaluator.recommend(
            forecast_payload, similarity_payload, anomaly_payload, sentiment_score
        )
        policy_recommendation = PolicyRecommendation(
            action=policy_signal.action,
            risk_multiplier=policy_signal.risk_multiplier,
            rationale=policy_signal.rationale,
            curriculum_stage=policy_signal.curriculum_stage,
        )

        return MLEnhancementSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            forecast=forecast_snapshot,
            regime_similarity=similarity_snapshot,
            anomaly=anomaly_snapshot,
            policy_recommendation=policy_recommendation,
        )

    def _get_latest_bar(
        self, symbol: str, timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        start = timestamp - timedelta(days=5)
        bars = self.market_adapter.get_bars(symbol, start, timestamp, timeframe="1Day")
        if not bars:
            return None
        bar = bars[-1].model_dump()
        return bar

    def _build_feature_vector(
        self, pipeline_result: PipelineResult, latest_bar: Optional[Dict[str, Any]]
    ) -> List[float]:
        vector: List[float] = []
        hedge = pipeline_result.hedge_snapshot
        liquidity = pipeline_result.liquidity_snapshot
        sentiment = pipeline_result.sentiment_snapshot

        if hedge:
            vector.extend(
                [
                    hedge.elasticity,
                    hedge.movement_energy,
                    hedge.energy_asymmetry,
                    hedge.gamma_pressure,
                    hedge.vanna_pressure,
                    hedge.charm_pressure,
                    hedge.dealer_gamma_sign,
                ]
            )
        if liquidity:
            vector.extend(
                [
                    liquidity.liquidity_score,
                    liquidity.bid_ask_spread,
                    liquidity.volume,
                    liquidity.depth,
                    liquidity.impact_cost,
                ]
            )
        if sentiment:
            vector.extend(
                [
                    sentiment.sentiment_score,
                    sentiment.news_sentiment,
                    sentiment.flow_sentiment,
                    sentiment.technical_sentiment,
                ]
            )
        if latest_bar:
            vector.append(float(latest_bar.get("close", 0.0)))

        if not vector:
            logger.warning("Empty feature vector for ML enhancements; defaulting to zeros")
            vector = [0.0]

        return vector
