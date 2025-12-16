"""Hedge Engine v3.0 - Dealer flow and elasticity analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from loguru import logger

from adapters.options_chain_adapter import OptionsChainAdapter
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
        logger.info("HedgeEngineV3 initialized")
    
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
        
        # Calculate dealer positioning
        total_call_oi = sum(c.open_interest for c in chain if c.option_type == "call")
        total_put_oi = sum(c.open_interest for c in chain if c.option_type == "put")
        
        # Net dealer gamma sign (positive = long gamma, negative = short gamma)
        if total_call_oi + total_put_oi > 0:
            dealer_gamma_sign = (total_call_oi - total_put_oi) / (total_call_oi + total_put_oi)
        else:
            dealer_gamma_sign = 0.0
        
        # Calculate Greek pressures
        gamma_pressure = sum(abs(c.gamma) * c.open_interest for c in chain) / len(chain) if chain else 0.0
        vanna_pressure = sum(abs(c.vega * c.delta) * c.open_interest for c in chain) / len(chain) if chain else 0.0
        charm_pressure = sum(abs(c.theta * c.delta) * c.open_interest for c in chain) / len(chain) if chain else 0.0
        
        # Calculate directional pressures
        pressure_up = sum(
            c.gamma * c.open_interest 
            for c in chain 
            if c.option_type == "call" and c.delta > 0.3
        )
        pressure_down = sum(
            abs(c.gamma * c.open_interest)
            for c in chain 
            if c.option_type == "put" and c.delta < -0.3
        )
        pressure_net = pressure_up - pressure_down
        
        # Calculate market elasticity (resistance to price movement)
        # Higher gamma and vanna = stiffer market
        elasticity = max(0.1, gamma_pressure * 0.6 + vanna_pressure * 0.4)
        
        # Calculate movement energy (energy required to move price)
        # Energy = Pressure / Elasticity
        if elasticity > 0:
            movement_energy = abs(pressure_net) / elasticity
        else:
            movement_energy = 0.0
        
        # Calculate energy asymmetry (directional bias)
        if pressure_up + pressure_down > 0:
            energy_asymmetry = (pressure_up - pressure_down) / (pressure_up + pressure_down)
        else:
            energy_asymmetry = 0.0
        
        # Determine regime
        regime = self._classify_regime(dealer_gamma_sign, gamma_pressure)
        
        # Confidence based on data quality
        confidence = min(1.0, len(chain) / 100.0)
        
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
            confidence=confidence,
        )
    
    def _classify_regime(self, dealer_sign: float, gamma_pressure: float) -> str:
        """Classify market regime based on dealer positioning."""
        if dealer_sign > 0.3 and gamma_pressure > 0.5:
            return "short_squeeze"
        elif dealer_sign < -0.3 and gamma_pressure > 0.5:
            return "long_compression"
        elif gamma_pressure < 0.2:
            return "low_expansion"
        else:
            return "neutral"
