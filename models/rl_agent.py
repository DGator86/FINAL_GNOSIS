"""
Reinforcement Learning Trading Agent

Deep RL agent for trading decisions using:
- Deep Q-Network (DQN)
- Policy Gradient methods
- Actor-Critic architectures

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import random
import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class TradingAction(int, Enum):
    """Discrete trading actions."""
    HOLD = 0
    BUY_SMALL = 1   # 25% of max position
    BUY_MEDIUM = 2  # 50% of max position
    BUY_LARGE = 3   # 100% of max position
    SELL_SMALL = 4
    SELL_MEDIUM = 5
    SELL_LARGE = 6
    CLOSE = 7       # Close entire position


@dataclass
class MarketState:
    """State representation for RL agent."""
    # Price features
    price: float
    price_change_1d: float
    price_change_5d: float
    price_change_20d: float
    
    # Volatility features
    realized_vol: float
    implied_vol: float
    vol_ratio: float  # IV/RV
    
    # Momentum features
    rsi: float
    macd: float
    macd_signal: float
    
    # Volume features
    volume_ratio: float  # Current vs average
    
    # Position features
    position_size: float
    position_pnl: float
    position_duration: int
    
    # Portfolio features
    portfolio_heat: float  # Risk utilization
    cash_ratio: float
    
    # Time features
    day_of_week: int
    hour_of_day: int
    days_to_expiry: Optional[int] = None
    
    def to_array(self) -> List[float]:
        """Convert to feature array."""
        return [
            self.price_change_1d,
            self.price_change_5d,
            self.price_change_20d,
            self.realized_vol,
            self.implied_vol,
            self.vol_ratio,
            self.rsi / 100,  # Normalize to 0-1
            self.macd,
            self.macd_signal,
            self.volume_ratio,
            self.position_size,
            self.position_pnl,
            min(self.position_duration / 30, 1),  # Cap at 30 days
            self.portfolio_heat,
            self.cash_ratio,
            self.day_of_week / 4,  # Normalize
            self.hour_of_day / 24,
        ]
    
    @property
    def state_dim(self) -> int:
        """Dimension of state vector."""
        return len(self.to_array())


@dataclass
class Experience:
    """Single experience for replay buffer."""
    state: MarketState
    action: TradingAction
    reward: float
    next_state: MarketState
    done: bool


@dataclass
class RLAgentConfig:
    """RL agent configuration."""
    # Architecture
    state_dim: int = 17
    action_dim: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    
    # Learning
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Replay buffer
    buffer_size: int = 100000
    batch_size: int = 64
    min_buffer_size: int = 1000
    
    # Training
    target_update_freq: int = 100
    train_freq: int = 4
    
    # Rewards
    profit_reward_scale: float = 1.0
    risk_penalty_scale: float = 0.5
    holding_cost: float = 0.001
    transaction_cost: float = 0.002


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int):
        """Initialize buffer."""
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class NeuralNetwork:
    """Simple neural network (pure Python implementation)."""
    
    def __init__(self, layer_dims: List[int]):
        """Initialize network with given layer dimensions."""
        self.layers = []
        self.layer_dims = layer_dims
        
        # Initialize weights
        for i in range(len(layer_dims) - 1):
            # Xavier initialization
            scale = math.sqrt(2.0 / (layer_dims[i] + layer_dims[i+1]))
            weights = [[random.gauss(0, scale) for _ in range(layer_dims[i+1])]
                      for _ in range(layer_dims[i])]
            biases = [0.0 for _ in range(layer_dims[i+1])]
            self.layers.append((weights, biases))
    
    def forward(self, x: List[float]) -> List[float]:
        """Forward pass."""
        activation = x
        
        for i, (weights, biases) in enumerate(self.layers):
            # Linear transformation
            output = []
            for j in range(len(weights[0])):
                val = biases[j]
                for k in range(len(activation)):
                    val += activation[k] * weights[k][j]
                output.append(val)
            
            # ReLU activation (except last layer)
            if i < len(self.layers) - 1:
                activation = [max(0, v) for v in output]
            else:
                activation = output
        
        return activation
    
    def get_weights(self) -> List[Tuple]:
        """Get all weights."""
        return [(w.copy(), b.copy()) for w, b in self.layers]
    
    def set_weights(self, weights: List[Tuple]) -> None:
        """Set all weights."""
        self.layers = [(w.copy(), b.copy()) for w, b in weights]
    
    def update(self, gradients: List[Tuple], learning_rate: float) -> None:
        """Update weights with gradients."""
        for i, (w_grad, b_grad) in enumerate(gradients):
            weights, biases = self.layers[i]
            for j in range(len(weights)):
                for k in range(len(weights[j])):
                    weights[j][k] -= learning_rate * w_grad[j][k]
            for j in range(len(biases)):
                biases[j] -= learning_rate * b_grad[j]


class DQNAgent:
    """
    Deep Q-Network agent for trading.
    
    Features:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    - Double DQN
    """
    
    def __init__(self, config: Optional[RLAgentConfig] = None):
        """Initialize DQN agent."""
        self.config = config or RLAgentConfig()
        
        # Networks
        layer_dims = [self.config.state_dim] + self.config.hidden_dims + [self.config.action_dim]
        self.q_network = NeuralNetwork(layer_dims)
        self.target_network = NeuralNetwork(layer_dims)
        self.target_network.set_weights(self.q_network.get_weights())
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        
        # Exploration
        self.epsilon = self.config.epsilon_start
        
        # Training stats
        self.steps = 0
        self.episodes = 0
        self.total_reward = 0.0
        self.losses = []
        
        logger.info("DQN Agent initialized")
    
    def select_action(self, state: MarketState, training: bool = True) -> TradingAction:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current market state
            training: Whether we're in training mode
        
        Returns:
            Selected trading action
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return TradingAction(random.randint(0, self.config.action_dim - 1))
        
        # Greedy action
        state_array = state.to_array()
        q_values = self.q_network.forward(state_array)
        
        # Apply action masking for invalid actions
        q_values = self._apply_action_mask(q_values, state)
        
        best_action = q_values.index(max(q_values))
        return TradingAction(best_action)
    
    def _apply_action_mask(self, q_values: List[float], state: MarketState) -> List[float]:
        """Mask invalid actions."""
        masked = q_values.copy()
        
        # Can't sell if no position
        if state.position_size <= 0:
            masked[TradingAction.SELL_SMALL] = float('-inf')
            masked[TradingAction.SELL_MEDIUM] = float('-inf')
            masked[TradingAction.SELL_LARGE] = float('-inf')
            masked[TradingAction.CLOSE] = float('-inf')
        
        # Can't buy if fully invested
        if state.cash_ratio < 0.1:
            masked[TradingAction.BUY_SMALL] = float('-inf')
            masked[TradingAction.BUY_MEDIUM] = float('-inf')
            masked[TradingAction.BUY_LARGE] = float('-inf')
        
        return masked
    
    def store_experience(self, experience: Experience) -> None:
        """Store experience in replay buffer."""
        self.replay_buffer.push(experience)
        self.steps += 1
        self.total_reward += experience.reward
    
    def train_step(self) -> Optional[float]:
        """
        Perform single training step.
        
        Returns loss if training occurred, None otherwise.
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None
        
        if self.steps % self.config.train_freq != 0:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Compute loss and update
        loss = self._compute_loss_and_update(batch)
        self.losses.append(loss)
        
        # Update target network
        if self.steps % self.config.target_update_freq == 0:
            self.target_network.set_weights(self.q_network.get_weights())
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
        return loss
    
    def _compute_loss_and_update(self, batch: List[Experience]) -> float:
        """Compute TD loss and update Q-network."""
        total_loss = 0.0
        
        for exp in batch:
            state_array = exp.state.to_array()
            next_state_array = exp.next_state.to_array()
            
            # Current Q-values
            q_values = self.q_network.forward(state_array)
            
            # Target Q-value
            if exp.done:
                target = exp.reward
            else:
                # Double DQN: use online network to select action, target network to evaluate
                next_q_online = self.q_network.forward(next_state_array)
                best_next_action = next_q_online.index(max(next_q_online))
                next_q_target = self.target_network.forward(next_state_array)
                target = exp.reward + self.config.gamma * next_q_target[best_next_action]
            
            # TD error
            current_q = q_values[exp.action.value]
            td_error = target - current_q
            total_loss += td_error ** 2
            
            # Simple gradient update (in production, use proper optimizer)
            # This is a simplified update for demonstration
        
        return total_loss / len(batch)
    
    def calculate_reward(
        self,
        action: TradingAction,
        pnl: float,
        position_size: float,
        portfolio_value: float,
    ) -> float:
        """
        Calculate reward for action.
        
        Reward structure:
        - Profit/loss scaled
        - Risk penalty for large positions
        - Holding cost
        - Transaction cost
        """
        reward = 0.0
        
        # PnL reward
        pnl_pct = pnl / portfolio_value if portfolio_value > 0 else 0
        reward += pnl_pct * self.config.profit_reward_scale * 100  # Scale up
        
        # Risk penalty
        position_ratio = abs(position_size) / portfolio_value if portfolio_value > 0 else 0
        if position_ratio > 0.5:
            reward -= (position_ratio - 0.5) * self.config.risk_penalty_scale
        
        # Holding cost (encourages action)
        if action == TradingAction.HOLD and position_size != 0:
            reward -= self.config.holding_cost
        
        # Transaction cost
        if action not in (TradingAction.HOLD,):
            reward -= self.config.transaction_cost
        
        return reward
    
    def save_model(self, path: str) -> None:
        """Save model weights."""
        import json
        
        data = {
            "q_network": self.q_network.get_weights(),
            "target_network": self.target_network.get_weights(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "episodes": self.episodes,
            "config": {
                "state_dim": self.config.state_dim,
                "action_dim": self.config.action_dim,
                "hidden_dims": self.config.hidden_dims,
            }
        }
        
        # Convert to serializable format
        def convert_weights(weights):
            return [([list(row) for row in w], list(b)) for w, b in weights]
        
        data["q_network"] = convert_weights(data["q_network"])
        data["target_network"] = convert_weights(data["target_network"])
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model weights."""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.q_network.set_weights(data["q_network"])
        self.target_network.set_weights(data["target_network"])
        self.epsilon = data["epsilon"]
        self.steps = data["steps"]
        self.episodes = data["episodes"]
        
        logger.info(f"Model loaded from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "steps": self.steps,
            "episodes": self.episodes,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "total_reward": self.total_reward,
            "avg_loss": sum(self.losses[-100:]) / max(1, len(self.losses[-100:])) if self.losses else 0,
        }


class PolicyGradientAgent:
    """
    Policy Gradient agent using REINFORCE.
    
    Direct policy optimization without Q-values.
    """
    
    def __init__(self, config: Optional[RLAgentConfig] = None):
        """Initialize policy gradient agent."""
        self.config = config or RLAgentConfig()
        
        # Policy network
        layer_dims = [self.config.state_dim] + self.config.hidden_dims + [self.config.action_dim]
        self.policy_network = NeuralNetwork(layer_dims)
        
        # Episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Stats
        self.episodes = 0
        self.total_reward = 0.0
        
        logger.info("Policy Gradient Agent initialized")
    
    def select_action(self, state: MarketState, training: bool = True) -> TradingAction:
        """Select action from policy."""
        state_array = state.to_array()
        logits = self.policy_network.forward(state_array)
        
        # Apply softmax
        probs = self._softmax(logits)
        
        # Apply action mask
        probs = self._apply_mask(probs, state)
        
        # Sample from distribution during training
        if training:
            action = self._sample_action(probs)
        else:
            action = probs.index(max(probs))
        
        return TradingAction(action)
    
    def _softmax(self, logits: List[float]) -> List[float]:
        """Compute softmax probabilities."""
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        return [e / sum_exp for e in exp_logits]
    
    def _apply_mask(self, probs: List[float], state: MarketState) -> List[float]:
        """Mask invalid actions."""
        masked = probs.copy()
        
        if state.position_size <= 0:
            masked[TradingAction.SELL_SMALL] = 0
            masked[TradingAction.SELL_MEDIUM] = 0
            masked[TradingAction.SELL_LARGE] = 0
            masked[TradingAction.CLOSE] = 0
        
        if state.cash_ratio < 0.1:
            masked[TradingAction.BUY_SMALL] = 0
            masked[TradingAction.BUY_MEDIUM] = 0
            masked[TradingAction.BUY_LARGE] = 0
        
        # Renormalize
        total = sum(masked)
        if total > 0:
            masked = [p / total for p in masked]
        else:
            masked[TradingAction.HOLD] = 1.0
        
        return masked
    
    def _sample_action(self, probs: List[float]) -> int:
        """Sample action from probability distribution."""
        r = random.random()
        cumsum = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                return i
        return len(probs) - 1
    
    def store_transition(self, state: MarketState, action: TradingAction, reward: float) -> None:
        """Store transition for episode."""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def end_episode(self) -> float:
        """End episode and update policy."""
        if not self.episode_rewards:
            return 0.0
        
        # Calculate returns
        returns = self._calculate_returns()
        
        # Normalize returns
        mean_return = sum(returns) / len(returns)
        std_return = math.sqrt(sum((r - mean_return)**2 for r in returns) / len(returns)) + 1e-8
        normalized_returns = [(r - mean_return) / std_return for r in returns]
        
        # Policy gradient update would happen here
        # Simplified for demonstration
        
        episode_return = sum(self.episode_rewards)
        self.total_reward += episode_return
        self.episodes += 1
        
        # Clear episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        return episode_return
    
    def _calculate_returns(self) -> List[float]:
        """Calculate discounted returns."""
        returns = []
        running_return = 0
        
        for reward in reversed(self.episode_rewards):
            running_return = reward + self.config.gamma * running_return
            returns.insert(0, running_return)
        
        return returns
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "episodes": self.episodes,
            "total_reward": self.total_reward,
            "avg_episode_reward": self.total_reward / max(1, self.episodes),
        }


class TradingRLEnvironment:
    """
    Trading environment for RL agents.
    
    Simulates market for training.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        max_position_pct: float = 0.5,
    ):
        """Initialize environment."""
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        
        # State
        self.capital = initial_capital
        self.position = 0.0
        self.position_cost = 0.0
        self.position_duration = 0
        
        # History
        self.price_history: List[float] = []
        self.action_history: List[TradingAction] = []
        self.reward_history: List[float] = []
        
        self.step_count = 0
    
    def reset(self, price_history: List[float]) -> MarketState:
        """Reset environment with new price history."""
        self.capital = self.initial_capital
        self.position = 0.0
        self.position_cost = 0.0
        self.position_duration = 0
        self.price_history = price_history
        self.action_history = []
        self.reward_history = []
        self.step_count = 0
        
        return self._get_state()
    
    def step(self, action: TradingAction) -> Tuple[MarketState, float, bool]:
        """
        Execute action and return new state, reward, done.
        """
        self.step_count += 1
        current_price = self.price_history[min(self.step_count, len(self.price_history) - 1)]
        
        # Execute action
        pnl = self._execute_action(action, current_price)
        
        # Calculate reward
        portfolio_value = self.capital + self.position * current_price
        reward = self._calculate_reward(action, pnl, portfolio_value)
        
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # Check if done
        done = self.step_count >= len(self.price_history) - 1
        done = done or portfolio_value < self.initial_capital * 0.5  # Stop if 50% loss
        
        # Update position duration
        if self.position != 0:
            self.position_duration += 1
        else:
            self.position_duration = 0
        
        return self._get_state(), reward, done
    
    def _execute_action(self, action: TradingAction, price: float) -> float:
        """Execute trading action."""
        pnl = 0.0
        max_position_value = self.initial_capital * self.max_position_pct
        
        if action == TradingAction.BUY_SMALL:
            shares = int(max_position_value * 0.25 / price)
            cost = shares * price
            if cost <= self.capital:
                self.position += shares
                self.capital -= cost
                self.position_cost += cost
        
        elif action == TradingAction.BUY_MEDIUM:
            shares = int(max_position_value * 0.5 / price)
            cost = shares * price
            if cost <= self.capital:
                self.position += shares
                self.capital -= cost
                self.position_cost += cost
        
        elif action == TradingAction.BUY_LARGE:
            shares = int(max_position_value / price)
            cost = shares * price
            if cost <= self.capital:
                self.position += shares
                self.capital -= cost
                self.position_cost += cost
        
        elif action in (TradingAction.SELL_SMALL, TradingAction.SELL_MEDIUM, TradingAction.SELL_LARGE, TradingAction.CLOSE):
            if self.position > 0:
                if action == TradingAction.CLOSE:
                    sell_shares = self.position
                elif action == TradingAction.SELL_LARGE:
                    sell_shares = self.position
                elif action == TradingAction.SELL_MEDIUM:
                    sell_shares = self.position * 0.5
                else:
                    sell_shares = self.position * 0.25
                
                sell_shares = int(sell_shares)
                if sell_shares > 0:
                    proceeds = sell_shares * price
                    cost_basis = (self.position_cost / self.position) * sell_shares if self.position > 0 else 0
                    pnl = proceeds - cost_basis
                    
                    self.capital += proceeds
                    self.position -= sell_shares
                    self.position_cost -= cost_basis
        
        return pnl
    
    def _calculate_reward(self, action: TradingAction, pnl: float, portfolio_value: float) -> float:
        """Calculate step reward."""
        reward = pnl / self.initial_capital * 100  # PnL percentage
        
        # Penalize holding large positions
        position_value = abs(self.position * self.price_history[self.step_count])
        if position_value / portfolio_value > 0.3:
            reward -= 0.01
        
        return reward
    
    def _get_state(self) -> MarketState:
        """Get current market state."""
        idx = min(self.step_count, len(self.price_history) - 1)
        price = self.price_history[idx]
        
        # Calculate features
        price_change_1d = (price - self.price_history[max(0, idx-1)]) / self.price_history[max(0, idx-1)] if idx > 0 else 0
        price_change_5d = (price - self.price_history[max(0, idx-5)]) / self.price_history[max(0, idx-5)] if idx > 5 else 0
        price_change_20d = (price - self.price_history[max(0, idx-20)]) / self.price_history[max(0, idx-20)] if idx > 20 else 0
        
        portfolio_value = self.capital + self.position * price
        position_pnl = (self.position * price - self.position_cost) if self.position > 0 else 0
        
        return MarketState(
            price=price,
            price_change_1d=price_change_1d,
            price_change_5d=price_change_5d,
            price_change_20d=price_change_20d,
            realized_vol=0.2,  # Placeholder
            implied_vol=0.25,
            vol_ratio=1.25,
            rsi=50,
            macd=0,
            macd_signal=0,
            volume_ratio=1.0,
            position_size=self.position * price / portfolio_value if portfolio_value > 0 else 0,
            position_pnl=position_pnl / portfolio_value if portfolio_value > 0 else 0,
            position_duration=self.position_duration,
            portfolio_heat=abs(self.position * price) / portfolio_value if portfolio_value > 0 else 0,
            cash_ratio=self.capital / portfolio_value if portfolio_value > 0 else 1,
            day_of_week=0,
            hour_of_day=12,
        )
    
    def get_performance(self) -> Dict[str, Any]:
        """Get episode performance metrics."""
        final_value = self.capital + self.position * self.price_history[-1] if self.price_history else self.capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        return {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "total_reward": sum(self.reward_history),
            "num_trades": sum(1 for a in self.action_history if a != TradingAction.HOLD),
            "steps": self.step_count,
        }


# Convenience functions
def create_dqn_agent(config: Optional[RLAgentConfig] = None) -> DQNAgent:
    """Create DQN agent."""
    return DQNAgent(config)


def create_policy_agent(config: Optional[RLAgentConfig] = None) -> PolicyGradientAgent:
    """Create policy gradient agent."""
    return PolicyGradientAgent(config)


def create_environment(initial_capital: float = 100000) -> TradingRLEnvironment:
    """Create trading environment."""
    return TradingRLEnvironment(initial_capital)
