"""Hedge Engine v3.0 - Dealer flow and elasticity analysis."""

from __future__ import annotations

from datetime import datetime
from math import exp
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import statsmodels.api as sm

from loguru import logger

from engines.inputs.options_chain_adapter import OptionsChainAdapter
from engines.hedge.regime_models import MultiDimensionalRegimeDetector
from schemas.core_schemas import HedgeSnapshot


class HedgeEngineV3:
    """
    Hedge Engine v3.0 with elasticity theory and movement energy.
    
    Analyzes dealer hedge positioning, Greek pressure fields, and market elasticity.
    """
    
    def __init__(self, options_adapter: OptionsChainAdapter, config: Dict[str, Any]):
        """
        Initialize Hedge Engine.
        
        Args:
            options_adapter: Options chain data provider
            config: Engine configuration
        """
        self.options_adapter = options_adapter
        self.config = config
        self.regime_detector = MultiDimensionalRegimeDetector(
            n_components=config.get("regime_components", 3),
            history=config.get("regime_history", 256),
            min_samples=config.get("regime_min_samples", 32),
        )
        self.feature_ema: Dict[str, float] = {}
        self.weight_state = {
            "gamma": config.get("gamma_weight", 0.6),
            "vanna": config.get("vanna_weight", 0.4),
        }
        self.ledger_history: Iterable[Dict[str, Any]] | None = config.get(
            "ledger_history", None
        )
        logger.info("HedgeEngineV3 initialized with adaptive regime detection")
    
    def run(self, symbol: str, timestamp: datetime) -> HedgeSnapshot:
        """
        Run hedge analysis for a symbol.
        
        Args:
            symbol: Trading symbol
            timestamp: Analysis timestamp
            
        Returns:
            HedgeSnapshot with elasticity and pressure metrics
        """
        logger.debug(f"Running HedgeEngineV3 for {symbol} at {timestamp}")
        
        # Get options chain
        chain = self.options_adapter.get_chain(symbol, timestamp)
        
        if not chain:
            logger.warning(f"No options chain data for {symbol}")
            return HedgeSnapshot(
                timestamp=timestamp,
                symbol=symbol,
            )
        
        dealer_gamma_sign = self._dealer_gamma_sign(chain)
        gamma_pressure, vanna_pressure, charm_pressure = self._greek_pressures(chain)
        vanna_pressure = self._apply_vanna_shock_absorber(chain, vanna_pressure)
        jump_intensity = self._estimate_jump_risk(chain)
        liquidity_friction = self._estimate_liquidity_friction(chain)

        pressure_up, pressure_down, pressure_net = self._directional_pressures(chain)
        elasticity, directional_elasticity = self._compute_elasticity(chain, gamma_pressure, vanna_pressure)
        movement_energy = self._movement_energy(pressure_net, elasticity)
        energy_asymmetry = self._energy_asymmetry(pressure_up, pressure_down)

        regime_features = self._collect_features(
            dealer_gamma_sign,
            gamma_pressure,
            vanna_pressure,
            charm_pressure,
            movement_energy,
            energy_asymmetry,
            jump_intensity,
            liquidity_friction,
        )
        self.regime_detector.update(regime_features)
        regime, regime_probabilities = self.regime_detector.infer(regime_features)

        # Confidence based on data quality + probabilistic clarity
        confidence = min(1.0, (len(chain) / 100.0) * (1 + max(regime_probabilities.values(), default=0.0)))

        return HedgeSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            elasticity=elasticity,
            movement_energy=movement_energy,
            energy_asymmetry=energy_asymmetry,
            pressure_up=pressure_up,
            pressure_down=pressure_down,
            pressure_net=pressure_net,
            gamma_pressure=gamma_pressure,
            vanna_pressure=vanna_pressure,
            charm_pressure=charm_pressure,
            dealer_gamma_sign=dealer_gamma_sign,
            regime=regime,
            regime_features={
                "dealer_gamma_sign": dealer_gamma_sign,
                "gamma_pressure": gamma_pressure,
                "vanna_pressure": vanna_pressure,
                "charm_pressure": charm_pressure,
                "movement_energy": movement_energy,
                "energy_asymmetry": energy_asymmetry,
                "jump_intensity": jump_intensity,
                "liquidity_friction": liquidity_friction,
            },
            regime_probabilities=regime_probabilities,
            jump_intensity=jump_intensity,
            liquidity_friction=liquidity_friction,
            adaptive_weights=self.weight_state,
            confidence=confidence,
            directional_elasticity=directional_elasticity,
        )

    def _dealer_gamma_sign(self, chain) -> float:
        total_call_oi = sum(c.open_interest for c in chain if c.option_type == "call")
        total_put_oi = sum(c.open_interest for c in chain if c.option_type == "put")
        if total_call_oi + total_put_oi == 0:
            return 0.0
        return (total_call_oi - total_put_oi) / (total_call_oi + total_put_oi)

    def _greek_pressures(self, chain) -> Tuple[float, float, float]:
        gamma_pressure = sum(abs(c.gamma) * c.open_interest for c in chain) / len(chain) if chain else 0.0
        vanna_pressure = sum(abs(c.vega * c.delta) * c.open_interest for c in chain) / len(chain) if chain else 0.0
        charm_pressure = sum(abs(c.theta * c.delta) * c.open_interest for c in chain) / len(chain) if chain else 0.0
        return gamma_pressure, vanna_pressure, charm_pressure

    def _apply_vanna_shock_absorber(self, chain, vanna_pressure: float) -> float:
        """Dampen vanna influence during volatility spikes."""
        ivs = [getattr(c, "implied_volatility", 0.0) or 0.0 for c in chain]
        if not ivs:
            return vanna_pressure

        iv_mean = sum(ivs) / len(ivs)
        iv_median = sorted(ivs)[len(ivs) // 2]
        vol_spike = max(0.0, iv_mean - iv_median)
        decay_rate = self.config.get("vanna_shock_decay", 1.2)
        shock_factor = exp(-decay_rate * vol_spike)
        smoothed = self._adaptive_smoothing("vanna_pressure", vanna_pressure)
        return smoothed * shock_factor

    def _directional_pressures(self, chain) -> Tuple[float, float, float]:
        pressure_up = sum(
            c.gamma * c.open_interest
            for c in chain
            if c.option_type == "call" and getattr(c, "delta", 0.0) > 0.3
        )
        pressure_down = sum(
            abs(c.gamma * c.open_interest)
            for c in chain
            if c.option_type == "put" and getattr(c, "delta", 0.0) < -0.3
        )
        return pressure_up, pressure_down, pressure_up - pressure_down

    def _compute_elasticity(
        self, chain, gamma_pressure: float, vanna_pressure: float
    ) -> Tuple[float, Dict[str, float]]:
        """Enhanced elasticity using IMH + flow regression and OI weighting."""

        self._update_weights(gamma_pressure, vanna_pressure)
        base_elasticity = max(
            0.1,
            gamma_pressure * self.weight_state["gamma"] + vanna_pressure * self.weight_state["vanna"],
        )

        # Flow multiplier via OLS on ledger data (Inelastic Markets Hypothesis)
        ledger_flows = self._load_flow_history()
        flow_multiplier = 0.0
        if len(ledger_flows) >= 5:
            try:
                volumes = np.array([item.get("flow", 0.0) for item in ledger_flows])
                prices = np.array([item.get("price", 0.0) for item in ledger_flows])
                X = sm.add_constant(volumes)
                model = sm.OLS(prices, X).fit()
                flow_multiplier = float(model.params[-1])
            except Exception as exc:  # pragma: no cover
                logger.debug(f"Flow regression failed: {exc}")

        elasticity = base_elasticity * (1 + flow_multiplier)

        # Open interest distribution weighting
        total_oi = sum(getattr(c, "open_interest", 0.0) for c in chain) or 1.0
        weighted_elasticity = 0.0
        for contract in chain:
            oi_weight = getattr(contract, "open_interest", 0.0) / total_oi
            greek_pressure = abs(
                getattr(contract, "gamma", 0.0) * getattr(contract, "delta", 0.0)
            )
            weighted_elasticity += oi_weight * greek_pressure
        elasticity += weighted_elasticity

        # Directional splits for calls vs puts
        up_elasticity = self._directional_elasticity(chain, option_type="call")
        down_elasticity = self._directional_elasticity(chain, option_type="put")
        directional = {"up": up_elasticity, "down": down_elasticity}

        smoothed = self._adaptive_smoothing("elasticity", elasticity)
        return smoothed, directional

    def _directional_elasticity(self, chain, option_type: str) -> float:
        filtered = [c for c in chain if getattr(c, "option_type", "") == option_type]
        if not filtered:
            return 0.0
        total_oi = sum(getattr(c, "open_interest", 0.0) for c in filtered) or 1.0
        pressures = [abs(getattr(c, "gamma", 0.0) * getattr(c, "delta", 0.0)) for c in filtered]
        weights = [getattr(c, "open_interest", 0.0) / total_oi for c in filtered]
        return float(np.dot(weights, pressures))

    def _load_flow_history(self) -> List[Dict[str, Any]]:
        """Load recent ledger-derived flows for regression multiplier.

        Supports either a pre-loaded iterable on config or a callable that returns
        a sequence of flow dictionaries. Only the most recent 30 are retained to
        avoid stale effects.
        """

        source = self.ledger_history or self.config.get("ledger_flows", [])
        try:
            flows = list(source() if callable(source) else source)
            return flows[-30:]
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Failed to load ledger flow history: {exc}")
            return []

    def _movement_energy(self, pressure_net: float, elasticity: float) -> float:
        return abs(pressure_net) / elasticity if elasticity > 0 else 0.0

    def _energy_asymmetry(self, pressure_up: float, pressure_down: float) -> float:
        if pressure_up + pressure_down == 0:
            return 0.0
        return (pressure_up - pressure_down) / (pressure_up + pressure_down)

    def _estimate_jump_risk(self, chain) -> float:
        deep_tail = [c for c in chain if abs(getattr(c, "delta", 0.0)) < 0.05]
        if not chain:
            return 0.0
        return len(deep_tail) / len(chain)

    def _estimate_liquidity_friction(self, chain) -> float:
        total_oi = sum(getattr(c, "open_interest", 0.0) for c in chain)
        if total_oi == 0:
            return 0.0
        top_oi = sorted((getattr(c, "open_interest", 0.0) for c in chain), reverse=True)
        concentration = sum(top_oi[: max(1, len(top_oi) // 10)]) / total_oi
        return self._adaptive_smoothing("liquidity_friction", concentration)

    def _adaptive_smoothing(self, key: str, value: float) -> float:
        alpha = self.config.get("smoothing_alpha", 0.2)
        prev = self.feature_ema.get(key, value)
        smoothed = alpha * value + (1 - alpha) * prev
        self.feature_ema[key] = smoothed
        return smoothed

    def _update_weights(self, gamma_pressure: float, vanna_pressure: float) -> None:
        """Small adaptive step to avoid static elasticity weights."""
        target_gamma = 0.6 + 0.2 * (gamma_pressure > vanna_pressure)
        target_vanna = 1 - target_gamma
        step = self.config.get("weight_step", 0.02)
        self.weight_state["gamma"] = self.weight_state["gamma"] + step * (target_gamma - self.weight_state["gamma"])
        self.weight_state["vanna"] = self.weight_state["vanna"] + step * (target_vanna - self.weight_state["vanna"])

    def _collect_features(
        self,
        dealer_gamma_sign: float,
        gamma_pressure: float,
        vanna_pressure: float,
        charm_pressure: float,
        movement_energy: float,
        energy_asymmetry: float,
        jump_intensity: float,
        liquidity_friction: float,
    ) -> List[float]:
        """Prepare multi-dimensional feature vector for regime detection."""
        return [
            dealer_gamma_sign,
            gamma_pressure,
            vanna_pressure,
            charm_pressure,
            movement_energy,
            energy_asymmetry,
            jump_intensity,
            liquidity_friction,
        ]
