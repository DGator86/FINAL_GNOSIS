"""
Deep Q-Network (DQN) agent for trading decisions
Includes Double DQN, Dueling DQN, and Prioritized Experience Replay
"""

import logging
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


@dataclass
class DQNConfig:
    """Configuration for DQN Agent"""

    state_dim: int = 150
    action_dim: int = 4  # [hold, long_small, long_large, short_small, short_large]
    hidden_dims: Optional[List[int]] = None

    learning_rate: float = 1e-4
    batch_size: int = 64
    memory_size: int = 100000
    target_update_freq: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    gamma: float = 0.99

    dueling: bool = True
    double_dqn: bool = True
    prioritized_replay: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001

        self.tree_capacity = 1
        while self.tree_capacity < capacity:
            self.tree_capacity *= 2

        self.tree = np.zeros(2 * self.tree_capacity - 1)
        self.data = np.zeros(self.tree_capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0

    def add(self, experience: Experience, priority: float):
        tree_idx = self.data_pointer + self.tree_capacity - 1

        self.data[self.data_pointer] = experience
        self.update_priority(tree_idx, priority)

        self.data_pointer = (self.data_pointer + 1) % self.tree_capacity
        if self.size < self.tree_capacity:
            self.size += 1

    def update_priority(self, tree_idx: int, priority: float):
        priority = max(priority, 1e-6)
        priority = priority ** self.alpha

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        batch = []
        idxs = []
        priorities = []

        segment_size = self.tree[0] / batch_size

        for i in range(batch_size):
            a = segment_size * i
            b = segment_size * (i + 1)

            s = random.uniform(a, b)
            idx = self._retrieve(0, s)

            data_idx = idx - self.tree_capacity + 1
            batch.append(self.data[data_idx])
            idxs.append(idx)
            priorities.append(self.tree[idx])

        sampling_probs = np.array(priorities) / self.tree[0]
        weights = (self.size * sampling_probs) ** (-self.beta)
        weights = weights / weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch, np.array(idxs), weights

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def __len__(self) -> int:
        return self.size


class DuelingDQN(nn.Module):
    """Dueling DQN Network"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()

        self.action_dim = action_dim

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = hidden_dim

        self.feature_layers = nn.Sequential(*layers)

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(x)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)

        return q_values


class DQNAgent:
    """Deep Q-Network Trading Agent"""

    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.q_network = DuelingDQN(config.state_dim, config.action_dim, config.hidden_dims).to(self.device)

        self.target_network = DuelingDQN(config.state_dim, config.action_dim, config.hidden_dims).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

        if config.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(config.memory_size)
        else:
            self.memory = deque(maxlen=config.memory_size)

        self.epsilon = config.epsilon_start
        self.steps_done = 0
        self.losses: List[float] = []

        self.actions = {
            0: {"action": "hold", "size": 0.0},
            1: {"action": "long", "size": 0.25},
            2: {"action": "long", "size": 0.5},
            3: {"action": "short", "size": 0.25},
        }

    def get_state_features(self, market_state: Dict) -> np.ndarray:
        features = market_state.get("features", np.zeros(self.config.state_dim - 4))

        position = market_state.get("position", 0.0)
        pnl = market_state.get("pnl", 0.0)
        drawdown = market_state.get("drawdown", 0.0)
        volatility = market_state.get("volatility", 0.1)

        position_norm = np.clip(position, -1, 1)
        pnl_norm = np.tanh(pnl / 0.1)
        drawdown_norm = np.clip(drawdown / 0.2, -1, 0)
        vol_norm = np.clip(volatility / 0.5, 0, 2)

        state = np.concatenate([features, [position_norm, pnl_norm, drawdown_norm, vol_norm]])

        return state.astype(np.float32)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def store_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        experience = Experience(state, action, reward, next_state, done)

        if self.config.prioritized_replay:
            max_priority = max(self.memory.tree[: self.memory.tree_capacity]) if self.memory.size > 0 else 1.0
            self.memory.add(experience, max_priority)
        else:
            self.memory.append(experience)

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.config.batch_size:
            return None

        if self.config.prioritized_replay:
            batch, tree_idxs, weights = self.memory.sample(self.config.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = random.sample(self.memory, self.config.batch_size)
            weights = torch.ones(self.config.batch_size).to(self.device)

        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        if self.config.double_dqn:
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
        else:
            next_q_values = self.target_network(next_states).max(dim=1)[0].unsqueeze(1)

        target_q_values = rewards.unsqueeze(1) + (self.config.gamma * next_q_values * ~dones.unsqueeze(1))

        td_errors = target_q_values - current_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()

        if self.config.prioritized_replay:
            priorities = torch.abs(td_errors).detach().cpu().numpy().squeeze()
            for tree_idx, priority in zip(tree_idxs, priorities):
                self.memory.update_priority(tree_idx, priority)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.steps_done % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

        self.steps_done += 1
        self.losses.append(loss.item())

        return loss.item()

    def get_action_details(self, action_idx: int) -> Dict:
        if action_idx in self.actions:
            return self.actions[action_idx].copy()
        return {"action": "hold", "size": 0.0}

    def save(self, path: str):
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "epsilon": self.epsilon,
                "steps_done": self.steps_done,
                "losses": self.losses,
            },
            path,
        )
        logger.info(f"DQN agent saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]
        self.losses = checkpoint.get("losses", [])

        logger.info(f"DQN agent loaded from {path}")


class TradingEnvironment:
    """Trading environment for RL agent training"""

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        max_position: float = 1.0,
    ):
        self.features = features
        self.prices = prices
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position

        self.reset()

    def reset(self) -> np.ndarray:
        self.current_step = 60
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_trades = 0
        self.total_pnl = 0.0
        self.max_pnl = 0.0
        self.drawdown = 0.0

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        market_features = self.features[self.current_step]

        pnl_pct = self.total_pnl / self.initial_balance
        drawdown_pct = self.drawdown

        recent_returns = np.diff(self.prices[max(0, self.current_step - 20) : self.current_step + 1])
        volatility = np.std(recent_returns) if len(recent_returns) > 0 else 0.1

        state = np.concatenate([market_features, [self.position, pnl_pct, drawdown_pct, volatility]])

        return state.astype(np.float32)

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if action_idx == 0:
            target_position = self.position
        elif action_idx == 1:
            target_position = 0.25
        elif action_idx == 2:
            target_position = 0.5
        elif action_idx == 3:
            target_position = -0.25
        else:
            target_position = -0.5

        position_change = target_position - self.position

        current_price = self.prices[self.current_step]

        trade_cost = 0.0
        if abs(position_change) > 1e-6:
            trade_cost = abs(position_change) * self.transaction_cost
            self.total_trades += 1

        old_position = self.position
        self.position = target_position

        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        if not done:
            next_price = self.prices[self.current_step]

            price_change = (next_price - current_price) / current_price
            position_pnl = old_position * price_change

            step_pnl = position_pnl - trade_cost
            self.total_pnl += step_pnl

            if self.total_pnl > self.max_pnl:
                self.max_pnl = self.total_pnl

            self.drawdown = (self.max_pnl - self.total_pnl) / self.initial_balance
        else:
            step_pnl = 0.0

        reward = self._calculate_reward(step_pnl, trade_cost, position_change)

        next_state = self._get_state()

        info = {
            "step_pnl": step_pnl,
            "total_pnl": self.total_pnl,
            "position": self.position,
            "trade_cost": trade_cost,
            "drawdown": self.drawdown,
            "total_trades": self.total_trades,
        }

        return next_state, reward, done, info

    def _calculate_reward(self, step_pnl: float, trade_cost: float, position_change: float) -> float:
        reward = step_pnl * 100

        reward -= trade_cost * 50

        if self.drawdown > 0.1:
            reward -= (self.drawdown - 0.1) * 200

        reward -= abs(position_change) * 10

        if self.current_step > 100:
            recent_pnls: List[float] = []
            if len(recent_pnls) > 20:
                sharpe = np.mean(recent_pnls) / (np.std(recent_pnls) + 1e-8)
                reward += sharpe * 10

        return reward


if __name__ == "__main__":
    config = DQNConfig(state_dim=154, action_dim=4, hidden_dims=[256, 256, 128], learning_rate=1e-4, batch_size=64)

    agent = DQNAgent(config)

    n_steps = 10000
    n_features = 150

    features = np.random.randn(n_steps, n_features)
    prices = np.cumsum(np.random.randn(n_steps) * 0.01) + 100

    env = TradingEnvironment(features, prices)

    num_episodes = 100
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.select_action(state, training=True)

            next_state, reward, done, info = env.step(action)

            agent.store_experience(state, action, reward, next_state, done)

            agent.train_step()

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        if episode % 10 == 0:
            logger.info(
                f"Episode {episode}: Reward={total_reward:.2f}, "
                f"Steps={steps}, PnL={info['total_pnl']:.4f}, Trades={info['total_trades']}, "
                f"Epsilon={agent.epsilon:.3f}"
            )

    agent.save("dqn_trading_agent.pt")
    print("Training completed and model saved!")
