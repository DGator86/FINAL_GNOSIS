"""Pipeline orchestration runner.

Provides efficient async/sync execution of the DHPE pipeline with:
- Concurrent engine execution
- Persistent event loop management
- Thread pool for sync adapters
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, Optional, Set

from loguru import logger

from schemas.core_schemas import PipelineResult, WatchlistEntry
from watchlist import AdaptiveWatchlist


# Global event loop management for efficient reuse
_loop_lock = threading.Lock()
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_thread_pool: Optional[ThreadPoolExecutor] = None


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop for sync contexts.
    
    This avoids the overhead of creating a new event loop for each
    pipeline run when called from synchronous code.
    """
    global _event_loop
    
    with _loop_lock:
        if _event_loop is None or _event_loop.is_closed():
            _event_loop = asyncio.new_event_loop()
            # Start loop in background thread
            thread = threading.Thread(
                target=_event_loop.run_forever,
                daemon=True,
                name="pipeline-event-loop"
            )
            thread.start()
            logger.debug("Created persistent event loop for pipeline")
        return _event_loop


def get_thread_pool() -> ThreadPoolExecutor:
    """Get or create a shared thread pool for sync adapter execution."""
    global _thread_pool
    
    with _loop_lock:
        if _thread_pool is None:
            _thread_pool = ThreadPoolExecutor(
                max_workers=8,
                thread_name_prefix="pipeline-worker"
            )
            logger.debug("Created shared thread pool for pipeline")
        return _thread_pool


class PipelineRunner:
    """Orchestrates the full DHPE pipeline.
    
    Supports both sync and async execution with efficient resource reuse.
    """
    
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
            tracking_agent: Position tracking agent
            adaptation_agent: Adaptive feedback agent
            auto_execute: Whether to execute trades automatically
            ml_engine: ML enhancement engine
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
        self._thread_pool = get_thread_pool()
        logger.info(f"PipelineRunner initialized for {symbol}")
    
    async def _run_engine(self, name: str, engine, timestamp: datetime):
        """Run a single engine, handling both async and sync implementations.
        
        Uses shared thread pool for sync engines to avoid blocking.
        """
        try:
            if hasattr(engine, "run_async") and asyncio.iscoroutinefunction(engine.run_async):
                return name, await engine.run_async(self.symbol, timestamp)
            
            # Run sync engine in thread pool
            loop = asyncio.get_running_loop()
            snapshot = await loop.run_in_executor(
                self._thread_pool, 
                engine.run, 
                self.symbol, 
                timestamp
            )
            return name, snapshot
        except Exception as e:
            logger.error(f"Engine {name} failed: {e}")
            return name, e

    async def run_once_async(self, timestamp: datetime) -> PipelineResult:
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
            engine_names = [name for name in ["hedge", "liquidity", "sentiment", "elasticity"] if name in self.engines]
            tasks = [self._run_engine(name, self.engines[name], timestamp) for name in engine_names]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, snapshot in results:
                if isinstance(snapshot, Exception):
                    logger.error(f"Error running {name} engine: {snapshot}")
                    continue
                if name == "hedge":
                    result.hedge_snapshot = snapshot
                elif name == "liquidity":
                    result.liquidity_snapshot = snapshot
                elif name == "sentiment":
                    result.sentiment_snapshot = snapshot
                elif name == "elasticity":
                    result.elasticity_snapshot = snapshot

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

            # Generate trade ideas with optional watchlist gating
            if self.trade_agent:
                try:
                    # Watchlist gating: Only generate trades for active symbols
                    # HedgeSnapshot.data issue resolved - watchlist now uses proper attributes
                    skip_gating = self.config.get("skip_watchlist_gating", False)
                    
                    if not skip_gating and self.watchlist and not self.watchlist.is_symbol_active(self.symbol):
                        logger.info(
                            f"Skipping trade idea generation for {self.symbol} — not on adaptive watchlist"
                        )
                        result.trade_ideas = []
                    else:
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

    def run_once(self, timestamp: datetime) -> PipelineResult:
        """Synchronous wrapper for environments not using asyncio.
        
        Uses a persistent event loop to avoid the overhead of creating
        a new loop for each call. This is much more efficient for
        repeated pipeline runs.
        """
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, can't use run_until_complete
            # This shouldn't happen in normal usage, but handle it gracefully
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.run_once_async(timestamp))
                return future.result()
        except RuntimeError:
            # No running loop - this is the normal sync case
            pass
        
        # Use persistent event loop for efficiency
        loop = get_event_loop()
        future = asyncio.run_coroutine_threadsafe(
            self.run_once_async(timestamp),
            loop
        )
        
        # Wait for result with timeout
        try:
            return future.result(timeout=120.0)
        except TimeoutError:
            logger.error(f"Pipeline timeout for {self.symbol}")
            future.cancel()
            return PipelineResult(timestamp=timestamp, symbol=self.symbol)
