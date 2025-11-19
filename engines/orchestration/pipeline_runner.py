"""Pipeline orchestration runner."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from loguru import logger

from schemas.core_schemas import PipelineResult


class PipelineRunner:
    """Orchestrates the full DHPE pipeline."""
    
    def __init__(
        self,
        symbol: str,
        engines: Dict[str, Any],
        primary_agents: Dict[str, Any],
        composer: Any,
        trade_agent: Any,
        ledger_store: Any,
        config: Dict[str, Any],
    ):
        """
        Initialize Pipeline Runner.
        
        Args:
            symbol: Trading symbol
            engines: Dictionary of engines
            primary_agents: Dictionary of primary agents
            composer: Composer agent
            trade_agent: Trade agent
            ledger_store: Ledger store
            config: Pipeline configuration
        """
        self.symbol = symbol
        self.engines = engines
        self.primary_agents = primary_agents
        self.composer = composer
        self.trade_agent = trade_agent
        self.ledger_store = ledger_store
        self.config = config
        logger.info(f"PipelineRunner initialized for {symbol}")
    
    def run_once(self, timestamp: datetime) -> PipelineResult:
        """
        Run a single pipeline iteration.
        
        Args:
            timestamp: Execution timestamp
            
        Returns:
            PipelineResult with all outputs
        """
        logger.info(f"Running pipeline for {self.symbol} at {timestamp}")
        
        result = PipelineResult(
            timestamp=timestamp,
            symbol=self.symbol,
        )
        
        try:
            # Run engines
            if "hedge" in self.engines:
                result.hedge_snapshot = self.engines["hedge"].run(self.symbol, timestamp)
            
            if "liquidity" in self.engines:
                result.liquidity_snapshot = self.engines["liquidity"].run(self.symbol, timestamp)
            
            if "sentiment" in self.engines:
                result.sentiment_snapshot = self.engines["sentiment"].run(self.symbol, timestamp)
            
            if "elasticity" in self.engines:
                result.elasticity_snapshot = self.engines["elasticity"].run(self.symbol, timestamp)
            
            # Run primary agents
            for agent_name, agent in self.primary_agents.items():
                try:
                    suggestion = agent.suggest(result, timestamp)
                    if suggestion:
                        result.suggestions.append(suggestion)
                except Exception as e:
                    logger.error(f"Error in agent {agent_name}: {e}")
            
            # Run composer for consensus
            if self.composer and result.suggestions:
                try:
                    result.consensus = self.composer.compose(result.suggestions, timestamp)
                except Exception as e:
                    logger.error(f"Error in composer: {e}")
            
            # Generate trade ideas
            if self.trade_agent:
                try:
                    trade_ideas = self.trade_agent.generate_ideas(result, timestamp)
                    result.trade_ideas = trade_ideas if trade_ideas else []
                except Exception as e:
                    logger.error(f"Error in trade agent: {e}")
            
            # Store in ledger
            if self.ledger_store:
                try:
                    self.ledger_store.append(result)
                except Exception as e:
                    logger.error(f"Error storing to ledger: {e}")
            
            logger.info(f"Pipeline complete for {self.symbol}")
            
        except Exception as e:
            logger.error(f"Pipeline error for {self.symbol}: {e}")
        
        return result
