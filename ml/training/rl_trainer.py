"""
Reinforcement Learning Training Pipeline

Provides:
- Data loading and preprocessing
- Training loop with logging
- Evaluation and backtesting
- Model checkpointing
- Hyperparameter optimization

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    import random as np_random

from models.rl_agent import (
    DQNAgent,
    TradingRLEnvironment,
    RLAgentConfig,
    MarketState,
    TradingAction,
    Experience,
    ReplayBuffer,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RLTrainingConfig:
    """Training configuration for RL agent."""
    # Training parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 500
    eval_frequency: int = 50
    eval_episodes: int = 10
    
    # Early stopping
    early_stopping_patience: int = 100
    min_improvement: float = 0.01
    
    # Learning schedule
    warmup_episodes: int = 10
    lr_decay_rate: float = 0.999
    min_lr: float = 1e-5
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/rl"
    save_frequency: int = 100
    keep_last_n: int = 5
    
    # Logging
    log_frequency: int = 10
    tensorboard_dir: Optional[str] = None
    
    # Data
    train_data_ratio: float = 0.8
    validation_data_ratio: float = 0.1
    shuffle_episodes: bool = True
    
    # Environment
    initial_capital: float = 100000
    max_position_size: float = 0.2  # Max 20% of capital per position
    transaction_cost: float = 0.001
    
    # Reward shaping
    reward_scale: float = 1.0
    risk_penalty: float = 0.1
    drawdown_penalty: float = 0.5
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: int = 3


@dataclass
class TrainingMetrics:
    """Metrics from training."""
    episode: int
    total_reward: float
    avg_reward: float
    portfolio_value: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    num_trades: int
    epsilon: float
    loss: float
    steps: int
    duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EvaluationResult:
    """Results from evaluation."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    final_portfolio_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# Data Generator
# =============================================================================

class MarketDataGenerator:
    """
    Generates or loads market data for training.
    
    Can use:
    - Historical data from files
    - Synthetic data for testing
    - Real-time data streams
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ):
        """Initialize data generator."""
        self.symbols = symbols or ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
        self.start_date = start_date or (datetime.now() - timedelta(days=365))
        self.end_date = end_date or datetime.now()
        self._data_cache: Dict[str, List[float]] = {}
    
    def generate_synthetic_data(
        self,
        num_days: int = 252,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        drift: float = 0.0001,
    ) -> List[float]:
        """Generate synthetic price data using geometric Brownian motion.
        
        Args:
            num_days: Number of trading days
            initial_price: Starting price
            volatility: Daily volatility
            drift: Daily drift (expected return)
            
        Returns:
            List of prices
        """
        if HAS_NUMPY:
            np.random.seed(int(time.time()) % 10000)
            returns = np.random.normal(drift, volatility, num_days)
            prices = initial_price * np.exp(np.cumsum(returns))
            return prices.tolist()
        else:
            prices = [initial_price]
            for _ in range(num_days - 1):
                ret = random.gauss(drift, volatility)
                prices.append(prices[-1] * math.exp(ret))
            return prices
    
    def generate_training_episodes(
        self,
        num_episodes: int,
        episode_length: int = 252,
    ) -> List[List[float]]:
        """Generate multiple episodes of price data.
        
        Args:
            num_episodes: Number of episodes
            episode_length: Steps per episode
            
        Returns:
            List of price sequences
        """
        episodes = []
        
        for i in range(num_episodes):
            # Vary parameters for diversity
            vol = random.uniform(0.01, 0.04)
            drift = random.uniform(-0.0002, 0.0003)
            initial = random.uniform(80, 150)
            
            prices = self.generate_synthetic_data(
                num_days=episode_length,
                initial_price=initial,
                volatility=vol,
                drift=drift,
            )
            episodes.append(prices)
        
        return episodes
    
    def load_historical_data(self, symbol: str) -> Optional[List[float]]:
        """Load historical data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of prices or None
        """
        # Check cache
        if symbol in self._data_cache:
            return self._data_cache[symbol]
        
        # Try to load from file
        data_path = Path(f"data/historical/{symbol}.json")
        if data_path.exists():
            with open(data_path) as f:
                data = json.load(f)
                self._data_cache[symbol] = data.get("close", [])
                return self._data_cache[symbol]
        
        # Generate synthetic data as fallback
        logger.warning(f"No historical data for {symbol}, using synthetic")
        prices = self.generate_synthetic_data()
        self._data_cache[symbol] = prices
        return prices


# =============================================================================
# RL Trainer
# =============================================================================

class RLTrainer:
    """
    Training pipeline for RL trading agent.
    
    Features:
    - Multi-episode training
    - Curriculum learning
    - Evaluation and backtesting
    - Checkpointing
    - Logging and metrics
    """
    
    def __init__(
        self,
        config: Optional[RLTrainingConfig] = None,
        agent_config: Optional[RLAgentConfig] = None,
    ):
        """Initialize trainer.
        
        Args:
            config: Training configuration
            agent_config: Agent configuration
        """
        self.config = config or RLTrainingConfig()
        self.agent_config = agent_config or RLAgentConfig()
        
        # Create agent
        self.agent = DQNAgent(self.agent_config)
        
        # Create data generator
        self.data_generator = MarketDataGenerator()
        
        # Training state
        self.current_episode = 0
        self.best_reward = float("-inf")
        self.episodes_without_improvement = 0
        self.training_history: List[TrainingMetrics] = []
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RLTrainer initialized | episodes={self.config.num_episodes}")
    
    def train(
        self,
        price_data: Optional[List[List[float]]] = None,
        callback: Optional[Callable[[TrainingMetrics], None]] = None,
    ) -> Dict[str, Any]:
        """Run training loop.
        
        Args:
            price_data: Optional pre-generated price data
            callback: Optional callback for each episode
            
        Returns:
            Training summary
        """
        logger.info("Starting RL training...")
        start_time = time.time()
        
        # Generate training data if not provided
        if price_data is None:
            price_data = self.data_generator.generate_training_episodes(
                num_episodes=self.config.num_episodes,
                episode_length=self.config.max_steps_per_episode + 50,
            )
        
        # Split into train/val/test
        n = len(price_data)
        train_end = int(n * self.config.train_data_ratio)
        val_end = train_end + int(n * self.config.validation_data_ratio)
        
        train_data = price_data[:train_end]
        val_data = price_data[train_end:val_end]
        test_data = price_data[val_end:]
        
        logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        # Training loop
        for episode in range(self.config.num_episodes):
            self.current_episode = episode
            
            # Select episode data
            if self.config.shuffle_episodes:
                episode_prices = random.choice(train_data)
            else:
                episode_prices = train_data[episode % len(train_data)]
            
            # Run episode
            metrics = self._run_episode(episode_prices, training=True)
            self.training_history.append(metrics)
            
            # Logging
            if episode % self.config.log_frequency == 0:
                logger.info(
                    f"Episode {episode}/{self.config.num_episodes} | "
                    f"Reward: {metrics.total_reward:.2f} | "
                    f"Portfolio: ${metrics.portfolio_value:,.0f} | "
                    f"Sharpe: {metrics.sharpe_ratio:.2f} | "
                    f"ε: {metrics.epsilon:.3f}"
                )
            
            # Evaluation
            if episode % self.config.eval_frequency == 0 and episode > 0:
                eval_result = self.evaluate(val_data[:self.config.eval_episodes])
                logger.info(
                    f"Evaluation | Return: {eval_result.total_return:.2%} | "
                    f"Sharpe: {eval_result.sharpe_ratio:.2f} | "
                    f"MaxDD: {eval_result.max_drawdown:.2%}"
                )
            
            # Checkpointing
            if episode % self.config.save_frequency == 0 and episode > 0:
                self._save_checkpoint(episode)
            
            # Early stopping
            if metrics.total_reward > self.best_reward + self.config.min_improvement:
                self.best_reward = metrics.total_reward
                self.episodes_without_improvement = 0
                self._save_checkpoint(episode, is_best=True)
            else:
                self.episodes_without_improvement += 1
            
            if self.episodes_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at episode {episode}")
                break
            
            # Callback
            if callback:
                callback(metrics)
        
        # Final evaluation on test data
        final_eval = self.evaluate(test_data) if test_data else None
        
        training_time = time.time() - start_time
        
        summary = {
            "episodes_completed": self.current_episode + 1,
            "best_reward": self.best_reward,
            "final_epsilon": self.agent.epsilon,
            "training_time_seconds": training_time,
            "final_evaluation": final_eval.to_dict() if final_eval else None,
        }
        
        logger.info(f"Training completed in {training_time:.1f}s")
        
        return summary
    
    def _run_episode(
        self,
        prices: List[float],
        training: bool = True,
    ) -> TrainingMetrics:
        """Run a single training episode.
        
        Args:
            prices: Price data for the episode
            training: Whether to train (update weights)
            
        Returns:
            Episode metrics
        """
        start_time = time.time()
        
        # Create environment
        env = TradingRLEnvironment(
            initial_capital=self.config.initial_capital,
            max_position_pct=self.config.max_position_size,
        )
        
        state = env.reset(prices)
        
        total_reward = 0.0
        rewards = []
        losses = []
        steps = 0
        
        for step in range(min(len(prices) - 1, self.config.max_steps_per_episode)):
            # Select action
            action = self.agent.select_action(state, training=training)
            
            # Take step
            next_state, reward, done = env.step(action)
            
            # Store experience
            if training:
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
                self.agent.store_experience(experience)
                
                # Train
                loss = self.agent.train_step()
                if loss is not None:
                    losses.append(loss)
            
            total_reward += reward
            rewards.append(reward)
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Get performance from environment
        perf = env.get_performance()
        portfolio_value = perf.get('final_value', self.config.initial_capital)
        num_trades = perf.get('num_trades', 0)
        
        # Calculate max drawdown from reward history
        max_drawdown = 0.0
        if rewards:
            cumulative = 0.0
            peak = 0.0
            for r in rewards:
                cumulative += r
                if cumulative > peak:
                    peak = cumulative
                drawdown = (peak - cumulative) / (peak + 1e-8) if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio
        if rewards and len(rewards) > 1:
            if HAS_NUMPY:
                returns = np.array(rewards)
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            else:
                mean_ret = sum(rewards) / len(rewards)
                var_ret = sum((r - mean_ret) ** 2 for r in rewards) / len(rewards)
                sharpe = mean_ret / (math.sqrt(var_ret) + 1e-8) * math.sqrt(252)
        else:
            sharpe = 0.0
        
        # Win rate estimation from rewards
        win_rate = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0
        
        return TrainingMetrics(
            episode=self.current_episode,
            total_reward=total_reward,
            avg_reward=total_reward / max(steps, 1),
            portfolio_value=portfolio_value,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            num_trades=num_trades,
            epsilon=self.agent.epsilon,
            loss=sum(losses) / len(losses) if losses else 0.0,
            steps=steps,
            duration_seconds=time.time() - start_time,
        )
    
    def evaluate(
        self,
        eval_data: List[List[float]],
    ) -> EvaluationResult:
        """Evaluate agent on held-out data.
        
        Args:
            eval_data: List of price sequences
            
        Returns:
            Evaluation results
        """
        all_returns = []
        all_trades = []
        final_values = []
        max_drawdowns = []
        
        for prices in eval_data:
            env = TradingRLEnvironment(
                initial_capital=self.config.initial_capital,
            )
            state = env.reset(prices)
            episode_rewards = []
            
            for step in range(len(prices) - 1):
                action = self.agent.select_action(state, training=False)
                state, reward, done = env.step(action)
                episode_rewards.append(reward)
                if done:
                    break
            
            # Get performance from environment
            perf = env.get_performance()
            final_value = perf.get('final_value', self.config.initial_capital)
            total_return = perf.get('total_return', 0.0)
            num_trades = perf.get('num_trades', 0)
            
            all_returns.append(total_return)
            final_values.append(final_value)
            
            # Calculate max drawdown from rewards
            max_dd = 0.0
            if episode_rewards:
                cumulative = 0.0
                peak = 0.0
                for r in episode_rewards:
                    cumulative += r
                    if cumulative > peak:
                        peak = cumulative
                    drawdown = (peak - cumulative) / (peak + 1e-8) if peak > 0 else 0
                    max_dd = max(max_dd, drawdown)
            max_drawdowns.append(max_dd)
            
            # Track trades as reward events
            for r in episode_rewards:
                if r != 0:
                    all_trades.append({"pnl": r})
        
        # Aggregate metrics
        avg_return = sum(all_returns) / len(all_returns) if all_returns else 0
        
        if all_returns and len(all_returns) > 1:
            if HAS_NUMPY:
                sharpe = np.mean(all_returns) / (np.std(all_returns) + 1e-8) * math.sqrt(len(all_returns))
            else:
                mean_ret = sum(all_returns) / len(all_returns)
                var_ret = sum((r - mean_ret) ** 2 for r in all_returns) / len(all_returns)
                sharpe = mean_ret / (math.sqrt(var_ret) + 1e-8) * math.sqrt(len(all_returns))
        else:
            sharpe = 0.0
        
        max_dd = max(max_drawdowns) if max_drawdowns else 0
        
        # Trade statistics
        if all_trades:
            trade_pnls = [t.get("pnl", 0) for t in all_trades]
            wins = sum(1 for p in trade_pnls if p > 0)
            win_rate = wins / len(trade_pnls)
            
            profits = [p for p in trade_pnls if p > 0]
            losses = [abs(p) for p in trade_pnls if p < 0]
            
            gross_profit = sum(profits) if profits else 0
            gross_loss = sum(losses) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            avg_trade = sum(trade_pnls) / len(trade_pnls)
            best_trade = max(trade_pnls)
            worst_trade = min(trade_pnls)
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade = 0
            best_trade = 0
            worst_trade = 0
        
        return EvaluationResult(
            total_return=avg_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(all_trades),
            avg_trade_return=avg_trade,
            best_trade=best_trade,
            worst_trade=worst_trade,
            final_portfolio_value=sum(final_values) / len(final_values) if final_values else 0,
        )
    
    def _save_checkpoint(self, episode: int, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            episode: Current episode
            is_best: Whether this is the best model
        """
        checkpoint = {
            "episode": episode,
            "agent_stats": self.agent.get_stats(),
            "config": asdict(self.config),
            "agent_config": asdict(self.agent_config),
            "best_reward": self.best_reward,
            "training_history": [m.to_dict() for m in self.training_history[-100:]],
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        # Save agent model separately
        model_path = self.checkpoint_dir / f"model_ep{episode}.json"
        self.agent.save_model(str(model_path))
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.json"
            with open(best_path, "w") as f:
                json.dump(checkpoint, f, indent=2, default=str)
            best_model_path = self.checkpoint_dir / "best_agent.json"
            self.agent.save_model(str(best_model_path))
            logger.info(f"Saved best model at episode {episode}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_ep*.json"),
            key=lambda p: int(p.stem.split("ep")[1]),
        )
        
        if len(checkpoints) > self.config.keep_last_n:
            for old_checkpoint in checkpoints[:-self.config.keep_last_n]:
                old_checkpoint.unlink()
    
    def load_checkpoint(self, path: Optional[str] = None):
        """Load model from checkpoint.
        
        Args:
            path: Path to checkpoint (uses best if None)
        """
        if path is None:
            path = self.checkpoint_dir / "best_model.json"
        
        path = Path(path)
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return
        
        with open(path) as f:
            checkpoint = json.load(f)
        
        # Load agent model if exists
        model_path = path.parent / "best_agent.json"
        if model_path.exists():
            self.agent.load_model(str(model_path))
        
        self.current_episode = checkpoint["episode"]
        self.best_reward = checkpoint["best_reward"]
        
        logger.info(f"Loaded checkpoint from episode {self.current_episode}")
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history as list of dicts."""
        return [m.to_dict() for m in self.training_history]


# =============================================================================
# Convenience Functions
# =============================================================================

def train_rl_agent(
    num_episodes: int = 500,
    eval_frequency: int = 50,
    checkpoint_dir: str = "checkpoints/rl",
    **kwargs,
) -> Dict[str, Any]:
    """Train RL agent with default configuration.
    
    Args:
        num_episodes: Number of training episodes
        eval_frequency: Evaluation frequency
        checkpoint_dir: Directory for checkpoints
        **kwargs: Additional config parameters
        
    Returns:
        Training summary
    """
    config = RLTrainingConfig(
        num_episodes=num_episodes,
        eval_frequency=eval_frequency,
        checkpoint_dir=checkpoint_dir,
        **kwargs,
    )
    
    trainer = RLTrainer(config)
    return trainer.train()
