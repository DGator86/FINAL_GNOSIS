"""
Greeks Hedging Automation

Automated hedging system for portfolio Greeks:
- Delta hedging
- Gamma scalping
- Vega neutralization
- Portfolio balancing
- Dynamic hedge adjustment

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import defaultdict
import math

from loguru import logger


class HedgeType(str, Enum):
    """Types of hedges."""
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    THETA = "theta"
    RHO = "rho"
    PORTFOLIO = "portfolio"  # Multi-greek hedge


class HedgeInstrument(str, Enum):
    """Instruments for hedging."""
    STOCK = "stock"
    FUTURES = "futures"
    OPTIONS = "options"
    INDEX_OPTIONS = "index_options"
    VIX_OPTIONS = "vix_options"


class HedgeUrgency(str, Enum):
    """Hedge urgency level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GreekExposure:
    """Current Greek exposures."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    
    # Dollar values
    delta_dollars: float = 0.0
    gamma_dollars: float = 0.0
    theta_dollars: float = 0.0
    vega_dollars: float = 0.0
    
    # Metadata
    as_of: datetime = field(default_factory=datetime.now)
    portfolio_value: float = 0.0
    
    def __add__(self, other: "GreekExposure") -> "GreekExposure":
        """Add exposures."""
        return GreekExposure(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho,
            delta_dollars=self.delta_dollars + other.delta_dollars,
            gamma_dollars=self.gamma_dollars + other.gamma_dollars,
            theta_dollars=self.theta_dollars + other.theta_dollars,
            vega_dollars=self.vega_dollars + other.vega_dollars,
        )
    
    def __neg__(self) -> "GreekExposure":
        """Negate exposures."""
        return GreekExposure(
            delta=-self.delta,
            gamma=-self.gamma,
            theta=-self.theta,
            vega=-self.vega,
            rho=-self.rho,
            delta_dollars=-self.delta_dollars,
            gamma_dollars=-self.gamma_dollars,
            theta_dollars=-self.theta_dollars,
            vega_dollars=-self.vega_dollars,
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            "delta_dollars": self.delta_dollars,
            "gamma_dollars": self.gamma_dollars,
            "theta_dollars": self.theta_dollars,
            "vega_dollars": self.vega_dollars,
        }


@dataclass
class HedgeLimits:
    """Risk limits for hedging."""
    # Maximum exposures
    max_delta: float = 500.0  # Share equivalent
    max_gamma: float = 50.0
    max_vega: float = 1000.0
    max_theta: float = -500.0  # Negative limit
    
    # Dollar limits
    max_delta_dollars: float = 50000.0
    max_gamma_dollars: float = 5000.0
    max_vega_dollars: float = 10000.0
    
    # Hedge triggers (% of max)
    trigger_threshold: float = 0.7  # 70% of limit
    rehedge_threshold: float = 0.5  # 50% of limit
    
    # Frequency limits
    min_hedge_interval_seconds: int = 60
    max_hedges_per_hour: int = 20
    
    def check_breach(self, exposure: GreekExposure) -> List[Dict[str, Any]]:
        """Check for limit breaches."""
        breaches = []
        
        if abs(exposure.delta) > self.max_delta:
            breaches.append({
                "greek": "delta",
                "value": exposure.delta,
                "limit": self.max_delta,
                "severity": "critical" if abs(exposure.delta) > self.max_delta * 1.5 else "high",
            })
        
        if abs(exposure.gamma) > self.max_gamma:
            breaches.append({
                "greek": "gamma",
                "value": exposure.gamma,
                "limit": self.max_gamma,
                "severity": "high",
            })
        
        if abs(exposure.vega) > self.max_vega:
            breaches.append({
                "greek": "vega",
                "value": exposure.vega,
                "limit": self.max_vega,
                "severity": "high",
            })
        
        if exposure.theta < self.max_theta:
            breaches.append({
                "greek": "theta",
                "value": exposure.theta,
                "limit": self.max_theta,
                "severity": "medium",
            })
        
        return breaches


@dataclass
class HedgeRecommendation:
    """Recommended hedge action."""
    hedge_id: str
    hedge_type: HedgeType
    urgency: HedgeUrgency
    
    # What to trade
    instrument: HedgeInstrument
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    
    # Expected impact
    delta_impact: float = 0.0
    gamma_impact: float = 0.0
    vega_impact: float = 0.0
    theta_impact: float = 0.0
    
    # Cost estimate
    estimated_cost: float = 0.0
    estimated_slippage: float = 0.0
    
    # Metadata
    rationale: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hedge_id": self.hedge_id,
            "hedge_type": self.hedge_type.value,
            "urgency": self.urgency.value,
            "instrument": self.instrument.value,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "delta_impact": self.delta_impact,
            "gamma_impact": self.gamma_impact,
            "vega_impact": self.vega_impact,
            "theta_impact": self.theta_impact,
            "estimated_cost": self.estimated_cost,
            "rationale": self.rationale,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class HedgeExecution:
    """Executed hedge record."""
    execution_id: str
    recommendation: HedgeRecommendation
    
    # Execution details
    executed_at: datetime
    fill_price: float
    fill_quantity: int
    commission: float
    slippage: float
    
    # Actual impact
    actual_delta_impact: float = 0.0
    actual_gamma_impact: float = 0.0
    actual_vega_impact: float = 0.0
    
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class HedgerConfig:
    """Hedger configuration."""
    # Enabled hedges
    hedge_delta: bool = True
    hedge_gamma: bool = True
    hedge_vega: bool = True
    hedge_theta: bool = False  # Usually not hedged directly
    
    # Automation
    auto_hedge: bool = False  # Require approval by default
    auto_hedge_delta_only: bool = True  # Auto-hedge delta only
    
    # Instruments
    delta_hedge_instrument: HedgeInstrument = HedgeInstrument.STOCK
    gamma_hedge_instrument: HedgeInstrument = HedgeInstrument.OPTIONS
    vega_hedge_instrument: HedgeInstrument = HedgeInstrument.OPTIONS
    
    # Cost management
    max_hedge_cost_percent: float = 0.01  # Max 1% of portfolio
    min_hedge_size: int = 10  # Minimum shares/contracts
    
    # Timing
    check_interval_seconds: int = 30
    hedge_delay_seconds: int = 5  # Delay before executing


class GreeksHedger:
    """
    Automated Greeks hedging system.
    
    Features:
    - Continuous exposure monitoring
    - Automatic hedge recommendations
    - Multi-greek optimization
    - Cost-efficient execution
    - Risk limit enforcement
    """
    
    def __init__(
        self,
        config: Optional[HedgerConfig] = None,
        limits: Optional[HedgeLimits] = None,
    ):
        """Initialize hedger."""
        self.config = config or HedgerConfig()
        self.limits = limits or HedgeLimits()
        
        # State
        self._current_exposure: Optional[GreekExposure] = None
        self._pending_recommendations: List[HedgeRecommendation] = []
        self._execution_history: List[HedgeExecution] = []
        self._last_hedge_time: Dict[str, datetime] = {}
        self._hedge_count_this_hour: int = 0
        
        # ID counter
        self._recommendation_counter = 0
        
        # Callbacks
        self._on_recommendation: Optional[Callable] = None
        self._on_execution: Optional[Callable] = None
        self._order_executor: Optional[Callable] = None
        
        # Background task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("GreeksHedger initialized")
    
    def set_order_executor(self, executor: Callable) -> None:
        """Set order execution callback."""
        self._order_executor = executor
    
    def set_recommendation_callback(self, callback: Callable) -> None:
        """Set callback for new recommendations."""
        self._on_recommendation = callback
    
    def update_exposure(self, exposure: GreekExposure) -> List[HedgeRecommendation]:
        """
        Update current exposure and generate recommendations.
        
        Returns list of hedge recommendations if needed.
        """
        self._current_exposure = exposure
        
        # Check limits
        breaches = self.limits.check_breach(exposure)
        
        recommendations = []
        
        for breach in breaches:
            rec = self._generate_recommendation(breach, exposure)
            if rec:
                recommendations.append(rec)
                self._pending_recommendations.append(rec)
        
        # Check if we need to rehedge (even without breach)
        rehedge_recs = self._check_rehedge_needs(exposure)
        recommendations.extend(rehedge_recs)
        
        # Notify
        if recommendations and self._on_recommendation:
            for rec in recommendations:
                try:
                    self._on_recommendation(rec)
                except Exception as e:
                    logger.error(f"Recommendation callback error: {e}")
        
        return recommendations
    
    def _generate_recommendation(
        self,
        breach: Dict[str, Any],
        exposure: GreekExposure,
    ) -> Optional[HedgeRecommendation]:
        """Generate hedge recommendation for breach."""
        greek = breach["greek"]
        value = breach["value"]
        limit = breach["limit"]
        severity = breach["severity"]
        
        self._recommendation_counter += 1
        hedge_id = f"HEDGE_{self._recommendation_counter}"
        
        # Determine urgency
        urgency = HedgeUrgency.CRITICAL if severity == "critical" else HedgeUrgency.HIGH
        
        if greek == "delta":
            return self._generate_delta_hedge(hedge_id, value, urgency)
        elif greek == "gamma":
            return self._generate_gamma_hedge(hedge_id, value, urgency)
        elif greek == "vega":
            return self._generate_vega_hedge(hedge_id, value, urgency)
        elif greek == "theta":
            return self._generate_theta_hedge(hedge_id, value, urgency)
        
        return None
    
    def _generate_delta_hedge(
        self,
        hedge_id: str,
        current_delta: float,
        urgency: HedgeUrgency,
    ) -> HedgeRecommendation:
        """Generate delta hedge recommendation."""
        # Simple delta hedge: trade underlying to neutralize
        target_delta = 0  # Neutral target
        hedge_delta = target_delta - current_delta
        
        # Round to tradeable quantity
        quantity = abs(int(hedge_delta))
        if quantity < self.config.min_hedge_size:
            quantity = 0
        
        side = "buy" if hedge_delta > 0 else "sell"
        
        return HedgeRecommendation(
            hedge_id=hedge_id,
            hedge_type=HedgeType.DELTA,
            urgency=urgency,
            instrument=self.config.delta_hedge_instrument,
            symbol="SPY",  # Default to SPY, would be configurable
            side=side,
            quantity=quantity,
            delta_impact=-current_delta if quantity > 0 else 0,
            rationale=f"Delta hedge: current delta {current_delta:.1f}, trading {quantity} shares to neutralize",
        )
    
    def _generate_gamma_hedge(
        self,
        hedge_id: str,
        current_gamma: float,
        urgency: HedgeUrgency,
    ) -> HedgeRecommendation:
        """Generate gamma hedge recommendation."""
        # Gamma hedge typically requires options
        # Buy ATM options for positive gamma, sell for negative
        
        side = "buy" if current_gamma < 0 else "sell"
        quantity = max(self.config.min_hedge_size, abs(int(current_gamma / 0.05)))  # Assume 0.05 gamma per option
        
        return HedgeRecommendation(
            hedge_id=hedge_id,
            hedge_type=HedgeType.GAMMA,
            urgency=urgency,
            instrument=HedgeInstrument.OPTIONS,
            symbol="SPY_ATM_STRADDLE",
            side=side,
            quantity=quantity,
            gamma_impact=-current_gamma * 0.5,  # Estimate
            delta_impact=0,  # ATM straddle is delta neutral
            rationale=f"Gamma hedge: current gamma {current_gamma:.2f}, {side} {quantity} ATM straddles",
        )
    
    def _generate_vega_hedge(
        self,
        hedge_id: str,
        current_vega: float,
        urgency: HedgeUrgency,
    ) -> HedgeRecommendation:
        """Generate vega hedge recommendation."""
        # Vega hedge with VIX options or long-dated options
        side = "buy" if current_vega < 0 else "sell"
        quantity = max(self.config.min_hedge_size, abs(int(current_vega / 10)))  # Assume 10 vega per contract
        
        return HedgeRecommendation(
            hedge_id=hedge_id,
            hedge_type=HedgeType.VEGA,
            urgency=urgency,
            instrument=HedgeInstrument.VIX_OPTIONS,
            symbol="VIX_CALL",
            side=side,
            quantity=quantity,
            vega_impact=-current_vega * 0.5,
            rationale=f"Vega hedge: current vega {current_vega:.1f}, {side} {quantity} VIX calls",
        )
    
    def _generate_theta_hedge(
        self,
        hedge_id: str,
        current_theta: float,
        urgency: HedgeUrgency,
    ) -> HedgeRecommendation:
        """Generate theta hedge recommendation."""
        # Theta management typically involves rolling or closing positions
        return HedgeRecommendation(
            hedge_id=hedge_id,
            hedge_type=HedgeType.THETA,
            urgency=urgency,
            instrument=HedgeInstrument.OPTIONS,
            symbol="REVIEW_POSITIONS",
            side="review",
            quantity=0,
            theta_impact=0,
            rationale=f"Theta exposure {current_theta:.1f}/day - review and roll positions",
        )
    
    def _check_rehedge_needs(self, exposure: GreekExposure) -> List[HedgeRecommendation]:
        """Check if rehedging is needed even without breach."""
        recommendations = []
        
        # Check delta drift
        if self.config.hedge_delta:
            delta_threshold = self.limits.max_delta * self.limits.rehedge_threshold
            if abs(exposure.delta) > delta_threshold:
                # Check if we can hedge (rate limit)
                if self._can_hedge("delta"):
                    self._recommendation_counter += 1
                    rec = self._generate_delta_hedge(
                        f"HEDGE_{self._recommendation_counter}",
                        exposure.delta,
                        HedgeUrgency.MEDIUM,
                    )
                    if rec.quantity > 0:
                        recommendations.append(rec)
                        self._pending_recommendations.append(rec)
        
        return recommendations
    
    def _can_hedge(self, hedge_type: str) -> bool:
        """Check if we can execute a hedge (rate limiting)."""
        now = datetime.now()
        
        # Check interval
        last_time = self._last_hedge_time.get(hedge_type)
        if last_time:
            elapsed = (now - last_time).total_seconds()
            if elapsed < self.limits.min_hedge_interval_seconds:
                return False
        
        # Check hourly count
        if self._hedge_count_this_hour >= self.limits.max_hedges_per_hour:
            return False
        
        return True
    
    async def execute_recommendation(
        self,
        recommendation: HedgeRecommendation,
    ) -> HedgeExecution:
        """Execute a hedge recommendation."""
        execution_id = f"EXEC_{recommendation.hedge_id}"
        
        # Check if we have an executor
        if not self._order_executor:
            logger.warning("No order executor configured")
            return HedgeExecution(
                execution_id=execution_id,
                recommendation=recommendation,
                executed_at=datetime.now(),
                fill_price=0,
                fill_quantity=0,
                commission=0,
                slippage=0,
                success=False,
                error_message="No order executor configured",
            )
        
        try:
            # Execute order
            result = await self._order_executor(
                symbol=recommendation.symbol,
                side=recommendation.side,
                quantity=recommendation.quantity,
                order_type="market",
            )
            
            execution = HedgeExecution(
                execution_id=execution_id,
                recommendation=recommendation,
                executed_at=datetime.now(),
                fill_price=result.get("fill_price", 0),
                fill_quantity=result.get("fill_quantity", recommendation.quantity),
                commission=result.get("commission", 0),
                slippage=result.get("slippage", 0),
                actual_delta_impact=recommendation.delta_impact,
                actual_gamma_impact=recommendation.gamma_impact,
                actual_vega_impact=recommendation.vega_impact,
                success=True,
            )
            
            # Update tracking
            self._execution_history.append(execution)
            self._last_hedge_time[recommendation.hedge_type.value] = datetime.now()
            self._hedge_count_this_hour += 1
            
            # Remove from pending
            self._pending_recommendations = [
                r for r in self._pending_recommendations
                if r.hedge_id != recommendation.hedge_id
            ]
            
            # Callback
            if self._on_execution:
                self._on_execution(execution)
            
            logger.info(f"Executed hedge {execution_id}: {recommendation.side} {recommendation.quantity} {recommendation.symbol}")
            return execution
            
        except Exception as e:
            logger.error(f"Hedge execution failed: {e}")
            return HedgeExecution(
                execution_id=execution_id,
                recommendation=recommendation,
                executed_at=datetime.now(),
                fill_price=0,
                fill_quantity=0,
                commission=0,
                slippage=0,
                success=False,
                error_message=str(e),
            )
    
    def get_optimal_hedge(
        self,
        exposure: GreekExposure,
        available_instruments: List[Dict[str, Any]],
    ) -> List[HedgeRecommendation]:
        """
        Find optimal hedge using available instruments.
        
        Multi-objective optimization considering:
        - Greek neutralization
        - Cost minimization
        - Execution feasibility
        """
        recommendations = []
        
        # Score each instrument for hedging
        scored_instruments = []
        
        for inst in available_instruments:
            score = self._score_hedge_instrument(inst, exposure)
            scored_instruments.append((score, inst))
        
        # Sort by score
        scored_instruments.sort(reverse=True, key=lambda x: x[0])
        
        # Build hedge from best instruments
        remaining_exposure = GreekExposure(
            delta=exposure.delta,
            gamma=exposure.gamma,
            vega=exposure.vega,
        )
        
        for score, inst in scored_instruments[:5]:  # Top 5
            if abs(remaining_exposure.delta) > self.limits.max_delta * 0.1:
                rec = self._create_instrument_recommendation(inst, remaining_exposure)
                if rec:
                    recommendations.append(rec)
                    # Update remaining
                    remaining_exposure.delta += rec.delta_impact
                    remaining_exposure.gamma += rec.gamma_impact
                    remaining_exposure.vega += rec.vega_impact
        
        return recommendations
    
    def _score_hedge_instrument(
        self,
        instrument: Dict[str, Any],
        exposure: GreekExposure,
    ) -> float:
        """Score an instrument for hedging effectiveness."""
        score = 0.0
        
        inst_delta = instrument.get("delta", 0)
        inst_gamma = instrument.get("gamma", 0)
        inst_vega = instrument.get("vega", 0)
        inst_cost = instrument.get("price", 0) * instrument.get("multiplier", 100)
        
        # Delta effectiveness
        if exposure.delta != 0 and inst_delta != 0:
            if (exposure.delta > 0 and inst_delta < 0) or (exposure.delta < 0 and inst_delta > 0):
                score += 40 * min(1, abs(inst_delta / exposure.delta))
        
        # Gamma effectiveness
        if exposure.gamma != 0 and inst_gamma != 0:
            if (exposure.gamma > 0 and inst_gamma < 0) or (exposure.gamma < 0 and inst_gamma > 0):
                score += 30 * min(1, abs(inst_gamma / exposure.gamma))
        
        # Cost efficiency (lower is better)
        if inst_cost > 0:
            efficiency = abs(inst_delta * 100) / inst_cost
            score += 20 * min(1, efficiency / 10)
        
        # Liquidity
        volume = instrument.get("volume", 0)
        oi = instrument.get("open_interest", 0)
        if volume > 100:
            score += 10
        elif volume > 50:
            score += 5
        
        return score
    
    def _create_instrument_recommendation(
        self,
        instrument: Dict[str, Any],
        exposure: GreekExposure,
    ) -> Optional[HedgeRecommendation]:
        """Create recommendation for specific instrument."""
        inst_delta = instrument.get("delta", 0)
        if inst_delta == 0:
            return None
        
        # Calculate quantity needed
        quantity = abs(int(exposure.delta / inst_delta))
        if quantity < 1:
            return None
        
        self._recommendation_counter += 1
        
        side = "buy" if (exposure.delta > 0 and inst_delta < 0) or (exposure.delta < 0 and inst_delta > 0) else "sell"
        if exposure.delta > 0:
            side = "sell" if inst_delta > 0 else "buy"
        else:
            side = "buy" if inst_delta > 0 else "sell"
        
        return HedgeRecommendation(
            hedge_id=f"HEDGE_{self._recommendation_counter}",
            hedge_type=HedgeType.PORTFOLIO,
            urgency=HedgeUrgency.MEDIUM,
            instrument=HedgeInstrument.OPTIONS,
            symbol=instrument.get("symbol", "UNKNOWN"),
            side=side,
            quantity=quantity,
            delta_impact=-inst_delta * quantity if side == "buy" else inst_delta * quantity,
            gamma_impact=instrument.get("gamma", 0) * quantity,
            vega_impact=instrument.get("vega", 0) * quantity,
            estimated_cost=instrument.get("price", 0) * quantity * 100,
            rationale=f"Multi-greek hedge using {instrument.get('symbol')}",
        )
    
    def get_pending_recommendations(self) -> List[HedgeRecommendation]:
        """Get pending hedge recommendations."""
        # Filter expired
        now = datetime.now()
        self._pending_recommendations = [
            r for r in self._pending_recommendations
            if not r.expires_at or r.expires_at > now
        ]
        return self._pending_recommendations.copy()
    
    def get_execution_history(self, limit: int = 100) -> List[HedgeExecution]:
        """Get recent execution history."""
        return self._execution_history[-limit:]
    
    def get_hedge_stats(self) -> Dict[str, Any]:
        """Get hedging statistics."""
        successful = [e for e in self._execution_history if e.success]
        failed = [e for e in self._execution_history if not e.success]
        
        total_cost = sum(e.commission + abs(e.slippage * e.fill_quantity) for e in successful)
        
        return {
            "total_hedges": len(self._execution_history),
            "successful": len(successful),
            "failed": len(failed),
            "pending_recommendations": len(self._pending_recommendations),
            "total_cost": total_cost,
            "hedges_this_hour": self._hedge_count_this_hour,
            "current_exposure": self._current_exposure.to_dict() if self._current_exposure else None,
        }
    
    async def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started hedge monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped hedge monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                # Auto-execute critical hedges if enabled
                if self.config.auto_hedge:
                    for rec in self._pending_recommendations:
                        if rec.urgency == HedgeUrgency.CRITICAL:
                            if self.config.auto_hedge_delta_only and rec.hedge_type != HedgeType.DELTA:
                                continue
                            await self.execute_recommendation(rec)
                
                # Reset hourly counter
                # In production, use proper time tracking
                
                await asyncio.sleep(self.config.check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)


# Singleton instance
greeks_hedger = GreeksHedger()


# Convenience functions
def update_exposure(exposure: GreekExposure) -> List[HedgeRecommendation]:
    """Update exposure and get recommendations."""
    return greeks_hedger.update_exposure(exposure)


def get_pending_hedges() -> List[HedgeRecommendation]:
    """Get pending hedge recommendations."""
    return greeks_hedger.get_pending_recommendations()


async def execute_hedge(recommendation: HedgeRecommendation) -> HedgeExecution:
    """Execute a hedge."""
    return await greeks_hedger.execute_recommendation(recommendation)
