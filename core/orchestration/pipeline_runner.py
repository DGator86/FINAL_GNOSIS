"""Pipeline orchestration runner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Optional, Set

from loguru import logger

from schemas.core_schemas import PipelineResult, WatchlistEntry
from watchlist import AdaptiveWatchlist


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
        watchlist: Optional[AdaptiveWatchlist] = None,
        active_positions: Optional[Set[str]] = None,
        tracking_agent: Optional[Any] = None,
        adaptation_agent: Optional[Any] = None,
        auto_execute: bool = False,
        ml_engine: Optional[Any] = None,
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
            watchlist: Adaptive watchlist for ranking/trade gating
            active_positions: Symbols currently held (to avoid scaling beyond plan)
        """
        self.symbol = symbol
        self.engines = engines
        self.primary_agents = primary_agents
        self.composer = composer
        self.trade_agent = trade_agent
        self.ledger_store = ledger_store
        self.config = config
        self.watchlist = watchlist
        self.active_positions = active_positions or set()
        self.tracking_agent = tracking_agent
        self.adaptation_agent = adaptation_agent
        self.auto_execute = auto_execute
        self.ml_engine = ml_engine
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
            engine_fns = {
                "hedge": lambda: self.engines["hedge"].run(self.symbol, timestamp),
                "liquidity": lambda: self.engines["liquidity"].run(self.symbol, timestamp),
                "sentiment": lambda: self.engines["sentiment"].run(self.symbol, timestamp),
                "elasticity": lambda: self.engines["elasticity"].run(self.symbol, timestamp),
            }

            with ThreadPoolExecutor(max_workers=len(self.engines)) as executor:
                futures = {
                    executor.submit(fn): name
                    for name, fn in engine_fns.items()
                    if name in self.engines
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        snapshot = future.result()
                        if name == "hedge":
                            result.hedge_snapshot = snapshot
                        elif name == "liquidity":
                            result.liquidity_snapshot = snapshot
                        elif name == "sentiment":
                            result.sentiment_snapshot = snapshot
                        elif name == "elasticity":
                            result.elasticity_snapshot = snapshot
                    except Exception as exc:
                        logger.error(f"Error running {name} engine: {exc}")

            # Run ML enhancement engine (e.g., LSTM lookahead predictions)
            # Can be MLEnhancementEngine (composite) or LSTMPredictionEngine (specialized)
            if self.ml_engine:
                try:
                    result.ml_snapshot = self.ml_engine.enhance(result, timestamp)
                    logger.debug(f"ML enhancement completed for {self.symbol}")
                except Exception as e:
                    logger.error(f"Error in ML enhancement engine: {e}")
            
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

            # Update adaptive watchlist and enforce gating
            if self.watchlist:
                try:
                    entry: WatchlistEntry = self.watchlist.update_from_pipeline(
                        result,
                        active_positions=self.active_positions,
                    )
                    result.watchlist_entry = entry
                    result.watchlist_snapshot = self.watchlist.get_active_watchlist()
                except Exception as e:
                    logger.error(f"Error updating adaptive watchlist: {e}")

            # Generate trade ideas (watchlist gating temporarily disabled for testing)
            if self.trade_agent:
                try:
                    # TEMPORARY: Skip watchlist gating to test execution
                    # TODO: Fix HedgeSnapshot.data attribute error and re-enable
                    # if self.watchlist and not self.watchlist.is_symbol_active(self.symbol):
                    #     logger.info(
                    #         f"Skipping trade idea generation for {self.symbol} — not on adaptive watchlist"
                    #     )
                    #     result.trade_ideas = []
                    # else:
                    trade_ideas = self.trade_agent.generate_ideas(result, timestamp)
                    result.trade_ideas = trade_ideas if trade_ideas else []
                except Exception as e:
                    logger.error(f"Error in trade agent: {e}")

            # Execute trades when enabled
            if self.auto_execute and result.trade_ideas:
                try:
                    result.order_results = self.trade_agent.execute_trades(
                        result.trade_ideas, timestamp
                    )
                except Exception as e:
                    logger.error(f"Error executing trades: {e}")

            # Capture broker position tracking
            if self.tracking_agent:
                try:
                    result.tracking_snapshot = self.tracking_agent.snapshot_positions()
                except Exception as e:
                    logger.error(f"Error collecting tracking snapshot: {e}")

            # Run adaptation feedback loop
            if self.adaptation_agent:
                try:
                    update = self.adaptation_agent.update_from_feedback(
                        result, result.tracking_snapshot
                    )
                    result.adaptation_update = update
                    if update and self.trade_agent and "risk_per_trade" in update.changes:
                        self.trade_agent.update_risk_per_trade(
                            update.changes["risk_per_trade"]
                        )
                except Exception as e:
                    logger.error(f"Error in adaptation agent: {e}")
            
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
