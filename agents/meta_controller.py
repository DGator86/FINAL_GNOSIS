"""
Meta-Controller using Reinforcement Learning for Multi-Agent Orchestration
Learns optimal weight allocation across different agents based on performance and context
"""

import logging
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AgentPerformance:
    """Track individual agent performance"""
    agent_id: str
    total_suggestions: int = 0
    successful_suggestions: int = 0
    total_pnl: float = 0.0
    avg_confidence: float = 0.0
    recent_sharpe: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
    
    @property
    def success_rate(self) -> float:
        return (
            self.successful_suggestions / self.total_suggestions
            if self.total_suggestions > 0 else 0.5
        )


@dataclass
class MetaControllerConfig:
    """Configuration for Meta-Controller"""
    state_dim: int = 20  # Agent suggestions + market context + performance metrics
    action_dim: int = 3   # Number of agents to weight
    hidden_dims: List[int] = None
    
    # RL hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # Soft update parameter
    batch_size: int = 64
    memory_size: int = 10000
    
    # Training parameters
    exploration_noise: float = 0.1
    target_update_freq: int = 2
    
    # Reward function weights
    reward_weights: Dict[str, float] = None
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]
        
        if self.reward_weights is None:
            self.reward_weights = {
                'sharpe_ratio': 0.5,
                'max_drawdown': 0.3,
                'win_rate': 0.1,
                'turnover_penalty': 0.1
            }


class ActorNetwork(nn.Module):
    """Actor network for continuous weight allocation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Output layer with softmax for weight allocation
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Softmax(dim=-1)  # Ensures weights sum to 1
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(state)
        weights = self.weight_head(features)
        return weights


class CriticNetwork(nn.Module):
    """Critic network for value estimation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        # State encoder
        state_layers = []
        prev_dim = state_dim
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            state_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.state_encoder = nn.Sequential(*state_layers)
        
        # Combined state-action processing
        combined_dim = hidden_dims[-2] + action_dim
        self.combined_layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[-1], 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_features = self.state_encoder(state)
        combined = torch.cat([state_features, action], dim=-1)
        value = self.combined_layers(combined)
        return value


class MetaController:
    """
    Meta-Controller for Multi-Agent Orchestration using Deep Deterministic Policy Gradient (DDPG)
    
    Responsibilities:
    1. Receive suggestions from multiple agents (Hedge, Liquidity, Sentiment)
    2. Assess current market context and agent performance
    3. Dynamically allocate weights to agents using RL
    4. Generate final consensus decision
    5. Learn from outcomes to improve future decisions
    """
    
    def __init__(self, config: MetaControllerConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        self.actor = ActorNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dims
        ).to(self.device)
        
        self.critic = CriticNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dims
        ).to(self.device)
        
        # Target networks
        self.target_actor = ActorNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dims
        ).to(self.device)
        
        self.target_critic = CriticNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dims
        ).to(self.device)
        
        # Copy weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate
        )
        
        # Experience replay
        self.memory = deque(maxlen=config.memory_size)
        
        # Agent performance tracking
        self.agent_performances: Dict[str, AgentPerformance] = {}
        
        # Training state
        self.steps = 0
        self.episode_rewards = deque(maxlen=1000)
        self.losses = {'actor': [], 'critic': []}
        
        # Portfolio state tracking
        self.portfolio_state = {
            'position': 0.0,
            'pnl': 0.0,
            'max_pnl': 0.0,
            'drawdown': 0.0,
            'trade_count': 0,
            'recent_returns': deque(maxlen=100)
        }
        
        logger.info("Meta-Controller initialized with DDPG")
    
    def orchestrate_agents(
        self,
        agent_suggestions: Dict[str, Dict],
        market_context: Dict[str, Any],
        training: bool = True
    ) -> Dict[str, Any]:
        """
        Main orchestration method - combines agent suggestions using learned weights
        
        Args:
            agent_suggestions: Dict with format {agent_id: suggestion_dict}
            market_context: Current market state and context
            training: Whether in training mode
            
        Returns:
            Final consensus decision with metadata
        """
        
        # 1. Prepare state vector
        state = self._prepare_state_vector(agent_suggestions, market_context)
        
        # 2. Get weight allocation from actor network
        weights = self._get_agent_weights(state, training)
        
        # 3. Combine agent suggestions using weights
        consensus = self._combine_suggestions(agent_suggestions, weights)
        
        # 4. Store experience for learning (if training)
        if training:
            self._store_experience(state, weights, consensus, market_context)
        
        # 5. Update agent performance tracking
        self._update_agent_tracking(agent_suggestions)
        
        return {
            'consensus': consensus,
            'agent_weights': weights,
            'state_vector': state.cpu().numpy() if isinstance(state, torch.Tensor) else state,
            'metadata': {
                'num_agents': len(agent_suggestions),
                'disagreement_level': self._calculate_disagreement(agent_suggestions),
                'confidence_weighted': self._calculate_weighted_confidence(agent_suggestions, weights),
                'meta_controller_step': self.steps
            }
        }
    
    def _prepare_state_vector(
        self,
        agent_suggestions: Dict[str, Dict],
        market_context: Dict[str, Any]
    ) -> torch.Tensor:
        """Prepare state vector for neural network input"""
        
        state_components = []
        
        # 1. Agent suggestion features (normalized)
        agent_ids = ['hedge_agent', 'liquidity_agent', 'sentiment_agent']
        
        for agent_id in agent_ids:
            if agent_id in agent_suggestions:
                suggestion = agent_suggestions[agent_id]
                
                # Action encoding (one-hot or numeric)
                action = suggestion.get('action', 'NEUTRAL')
                action_encoding = self._encode_action(action)
                
                # Confidence
                confidence = suggestion.get('confidence', 0.5)
                
                # Agent-specific features
                state_components.extend([
                    action_encoding,
                    confidence,
                    # Add more agent-specific features here
                ])
            else:
                # Missing agent - use default values
                state_components.extend([0.0, 0.5])
        
        # 2. Market context features
        market_features = [
            market_context.get('volatility', 0.2),
            market_context.get('regime_trending', 0.0),  # Regime as binary indicators
            market_context.get('regime_ranging', 0.0),
            market_context.get('regime_volatile', 0.0),
            market_context.get('volume_ratio', 1.0),
            market_context.get('time_of_day_norm', 0.5),  # Normalized hour
            market_context.get('vix_level', 0.2),
        ]
        state_components.extend(market_features)
        
        # 3. Agent performance features
        for agent_id in agent_ids:
            if agent_id in self.agent_performances:
                perf = self.agent_performances[agent_id]
                perf_features = [
                    perf.success_rate,
                    perf.recent_sharpe,
                    perf.avg_confidence
                ]
            else:
                perf_features = [0.5, 0.0, 0.5]  # Default values
            
            state_components.extend(perf_features)
        
        # 4. Portfolio state features
        portfolio_features = [
            np.tanh(self.portfolio_state['position']),  # Normalize position
            np.tanh(self.portfolio_state['pnl'] / 0.1),  # Normalize P&L
            -self.portfolio_state['drawdown'],  # Drawdown (negative)
        ]
        state_components.extend(portfolio_features)
        
        # Ensure fixed dimensionality
        state_vector = np.array(state_components, dtype=np.float32)
        
        # Pad or truncate to config.state_dim
        if len(state_vector) < self.config.state_dim:
            padding = np.zeros(self.config.state_dim - len(state_vector))
            state_vector = np.concatenate([state_vector, padding])
        elif len(state_vector) > self.config.state_dim:
            state_vector = state_vector[:self.config.state_dim]
        
        return torch.FloatTensor(state_vector).to(self.device)
    
    def _encode_action(self, action: str) -> float:
        """Encode action as numerical value"""
        action_map = {
            'LONG': 1.0,
            'SHORT': -1.0,
            'NEUTRAL': 0.0,
            'HOLD': 0.0
        }
        return action_map.get(action, 0.0)
    
    def _get_agent_weights(
        self,
        state: torch.Tensor,
        training: bool = True
    ) -> Dict[str, float]:
        """Get agent weights from actor network"""
        
        with torch.no_grad() if not training else torch.enable_grad():
            state_batch = state.unsqueeze(0)  # Add batch dimension
            weight_tensor = self.actor(state_batch).squeeze(0)  # Remove batch dimension
            
            if training and np.random.random() < self.config.exploration_noise:
                # Add exploration noise
                noise = torch.randn_like(weight_tensor) * 0.1
                weight_tensor = torch.clamp(weight_tensor + noise, 0.0, 1.0)
                # Re-normalize to sum to 1
                weight_tensor = weight_tensor / weight_tensor.sum()
        
        # Convert to dict
        agent_ids = ['hedge_agent', 'liquidity_agent', 'sentiment_agent']
        weights = {}
        
        for i, agent_id in enumerate(agent_ids):
            weights[agent_id] = float(weight_tensor[i].cpu().numpy())
        
        return weights
    
    def _combine_suggestions(
        self,
        agent_suggestions: Dict[str, Dict],
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Combine agent suggestions using learned weights"""
        
        # Initialize consensus
        consensus = {
            'action': 'NEUTRAL',
            'confidence': 0.0,
            'position_size': 0.0,
            'reasoning': []
        }
        
        # Weighted voting for action
        action_scores = {'LONG': 0.0, 'SHORT': 0.0, 'NEUTRAL': 0.0, 'HOLD': 0.0}
        total_confidence = 0.0
        total_weight = 0.0
        
        for agent_id, suggestion in agent_suggestions.items():
            agent_weight = weights.get(agent_id, 0.0)
            
            if agent_weight > 0:
                action = suggestion.get('action', 'NEUTRAL')
                confidence = suggestion.get('confidence', 0.5)
                
                # Weight the action vote by both agent weight and confidence
                vote_strength = agent_weight * confidence
                action_scores[action] += vote_strength
                
                total_confidence += confidence * agent_weight
                total_weight += agent_weight
                
                # Collect reasoning
                reasoning = suggestion.get('reasoning', f'{agent_id} suggestion')
                consensus['reasoning'].append(
                    f"{agent_id} ({agent_weight:.2f}): {reasoning[:100]}..."
                )
        
        # Determine final action
        if total_weight > 0:
            consensus['action'] = max(action_scores, key=action_scores.get)
            consensus['confidence'] = total_confidence / total_weight
            
            # Calculate position size based on action and confidence
            if consensus['action'] == 'LONG':
                consensus['position_size'] = consensus['confidence'] * 0.5  # Max 50% long
            elif consensus['action'] == 'SHORT':
                consensus['position_size'] = -consensus['confidence'] * 0.5  # Max 50% short
            else:
                consensus['position_size'] = 0.0
        else:
            # No valid suggestions - default to neutral
            consensus['reasoning'].append("No valid agent suggestions - defaulting to NEUTRAL")
        
        return consensus
    
    def _calculate_disagreement(self, agent_suggestions: Dict[str, Dict]) -> float:
        """Calculate level of disagreement between agents"""
        
        actions = []
        confidences = []
        
        for suggestion in agent_suggestions.values():
            action = suggestion.get('action', 'NEUTRAL')
            confidence = suggestion.get('confidence', 0.5)
            
            actions.append(self._encode_action(action))
            confidences.append(confidence)
        
        if len(actions) < 2:
            return 0.0
        
        # Calculate variance in action encodings (weighted by confidence)
        actions = np.array(actions)
        confidences = np.array(confidences)
        
        weighted_mean = np.average(actions, weights=confidences)
        weighted_variance = np.average((actions - weighted_mean) ** 2, weights=confidences)
        
        # Normalize to 0-1 range
        disagreement = min(1.0, weighted_variance / 2.0)  # Max variance is 4 (LONG vs SHORT)
        
        return disagreement
    
    def _calculate_weighted_confidence(
        self,
        agent_suggestions: Dict[str, Dict],
        weights: Dict[str, float]
    ) -> float:
        """Calculate confidence weighted by agent importance"""
        
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for agent_id, suggestion in agent_suggestions.items():
            agent_weight = weights.get(agent_id, 0.0)
            confidence = suggestion.get('confidence', 0.5)
            
            total_weighted_confidence += confidence * agent_weight
            total_weight += agent_weight
        
        return total_weighted_confidence / total_weight if total_weight > 0 else 0.5
    
    def _store_experience(
        self,
        state: torch.Tensor,
        action: Dict[str, float],
        consensus: Dict[str, Any],
        market_context: Dict[str, Any]
    ):
        """Store experience for replay buffer"""
        
        # Convert action dict to tensor
        action_tensor = torch.FloatTensor([
            action.get('hedge_agent', 0.0),
            action.get('liquidity_agent', 0.0),
            action.get('sentiment_agent', 0.0)
        ])
        
        experience = {
            'state': state.cpu(),
            'action': action_tensor,
            'consensus': consensus,
            'timestamp': datetime.now()
        }
        
        self.memory.append(experience)
    
    def _update_agent_tracking(self, agent_suggestions: Dict[str, Dict]):
        """Update agent performance tracking"""
        
        for agent_id, suggestion in agent_suggestions.items():
            if agent_id not in self.agent_performances:
                self.agent_performances[agent_id] = AgentPerformance(agent_id=agent_id)
            
            perf = self.agent_performances[agent_id]
            perf.total_suggestions += 1
            
            # Update confidence tracking
            confidence = suggestion.get('confidence', 0.5)
            current_avg = perf.avg_confidence
            total = perf.total_suggestions
            perf.avg_confidence = ((current_avg * (total - 1)) + confidence) / total
            
            perf.last_updated = datetime.now()
    
    def update_outcome(
        self,
        episode_data: Dict[str, Any],
        final_pnl: float,
        duration_minutes: int
    ):
        """
        Update with trading outcome for learning
        
        Args:
            episode_data: Data from the orchestration step
            final_pnl: Final P&L from the trade
            duration_minutes: Duration the position was held
        """
        
        # Update portfolio state
        self.portfolio_state['pnl'] += final_pnl
        self.portfolio_state['recent_returns'].append(final_pnl)
        
        if self.portfolio_state['pnl'] > self.portfolio_state['max_pnl']:
            self.portfolio_state['max_pnl'] = self.portfolio_state['pnl']
        
        self.portfolio_state['drawdown'] = max(
            0.0,
            (self.portfolio_state['max_pnl'] - self.portfolio_state['pnl'])
        )
        
        self.portfolio_state['trade_count'] += 1
        
        # Calculate reward
        reward = self._calculate_reward(final_pnl, duration_minutes)
        
        # Update agent performance (successful if profitable)
        if 'agent_weights' in episode_data:
            for agent_id, weight in episode_data['agent_weights'].items():
                if agent_id in self.agent_performances and weight > 0.1:  # Only if agent had significant influence
                    perf = self.agent_performances[agent_id]
                    if final_pnl > 0:
                        perf.successful_suggestions += 1
                    perf.total_pnl += final_pnl * weight  # Proportional attribution
        
        # Update recent Sharpe ratios
        self._update_sharpe_ratios()
        
        # Store complete experience for training
        if len(self.memory) > 0:
            latest_experience = self.memory[-1]
            latest_experience['reward'] = reward
            latest_experience['next_state'] = self._get_current_state()
            latest_experience['done'] = True  # Each trade is an episode
        
        # Train if enough experience
        if len(self.memory) >= self.config.batch_size:
            self.train_step()
        
        self.episode_rewards.append(reward)
        
        logger.debug(f"Meta-controller updated: PnL={final_pnl:.4f}, Reward={reward:.4f}, Portfolio PnL={self.portfolio_state['pnl']:.4f}")
    
    def _calculate_reward(self, pnl: float, duration_minutes: int) -> float:
        """Calculate reward for RL training"""
        
        weights = self.config.reward_weights
        
        # 1. P&L component (scaled)
        pnl_reward = pnl * 100  # Scale up for training
        
        # 2. Sharpe ratio component
        if len(self.portfolio_state['recent_returns']) > 10:
            returns = list(self.portfolio_state['recent_returns'])
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = mean_return / (std_return + 1e-8)
            sharpe_reward = sharpe * weights['sharpe_ratio']
        else:
            sharpe_reward = 0.0
        
        # 3. Drawdown penalty
        drawdown_penalty = -self.portfolio_state['drawdown'] * weights['max_drawdown'] * 10
        
        # 4. Win rate bonus (simplified)
        win_rate_bonus = (1.0 if pnl > 0 else -0.5) * weights['win_rate']
        
        # 5. Turnover penalty (encourage longer holds)
        turnover_penalty = -weights['turnover_penalty'] * (1440 / max(duration_minutes, 60))  # Penalty for short trades
        
        total_reward = (
            pnl_reward +
            sharpe_reward +
            drawdown_penalty +
            win_rate_bonus +
            turnover_penalty
        )
        
        return total_reward
    
    def _update_sharpe_ratios(self):
        """Update recent Sharpe ratios for all agents"""
        
        if len(self.portfolio_state['recent_returns']) < 20:
            return
        
        recent_returns = list(self.portfolio_state['recent_returns'])[-20:]  # Last 20 trades
        
        if len(recent_returns) > 0:
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            overall_sharpe = mean_return / (std_return + 1e-8)
            
            # Update all agent Sharpe ratios (simplified - in reality would track individual contributions)
            for perf in self.agent_performances.values():
                perf.recent_sharpe = overall_sharpe * 0.9 + perf.recent_sharpe * 0.1  # EMA
    
    def _get_current_state(self) -> torch.Tensor:
        """Get current state for next_state in experience"""
        # This would be called with current market context
        # For now, return a dummy state
        dummy_suggestions = {
            'hedge_agent': {'action': 'NEUTRAL', 'confidence': 0.5},
            'liquidity_agent': {'action': 'NEUTRAL', 'confidence': 0.5},
            'sentiment_agent': {'action': 'NEUTRAL', 'confidence': 0.5}
        }
        dummy_market = {
            'volatility': 0.2,
            'regime_trending': 0.0,
            'regime_ranging': 1.0,
            'regime_volatile': 0.0
        }
        
        return self._prepare_state_vector(dummy_suggestions, dummy_market)
    
    def train_step(self) -> Tuple[float, float]:
        """Perform one training step using DDPG"""
        
        if len(self.memory) < self.config.batch_size:
            return 0.0, 0.0
        
        # Sample batch from replay buffer
        batch_indices = np.random.choice(len(self.memory), self.config.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        
        # Extract batch data
        states = torch.stack([exp['state'] for exp in batch]).to(self.device)
        actions = torch.stack([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp.get('reward', 0.0) for exp in batch]).to(self.device)
        next_states = torch.stack([exp.get('next_state', exp['state']) for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp.get('done', True) for exp in batch]).to(self.device)
        
        # Train Critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards.unsqueeze(1) + (self.config.gamma * target_q * ~dones.unsqueeze(1))
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Train Actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        if self.steps % self.config.target_update_freq == 0:
            self._soft_update(self.target_actor, self.actor, self.config.tau)
            self._soft_update(self.target_critic, self.critic, self.config.tau)
        
        self.steps += 1
        
        # Store losses
        self.losses['actor'].append(actor_loss.item())
        self.losses['critic'].append(critic_loss.item())
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update(self, target_net: nn.Module, source_net: nn.Module, tau: float):
        """Soft update target network parameters"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'meta_controller': {
                'total_steps': self.steps,
                'portfolio_pnl': self.portfolio_state['pnl'],
                'max_drawdown': self.portfolio_state['drawdown'],
                'total_trades': self.portfolio_state['trade_count'],
                'avg_episode_reward': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0.0,
                'recent_actor_loss': np.mean(self.losses['actor'][-10:]) if self.losses['actor'] else 0.0,
                'recent_critic_loss': np.mean(self.losses['critic'][-10:]) if self.losses['critic'] else 0.0
            },
            'agent_performances': {}
        }
        
        # Add agent performance details
        for agent_id, perf in self.agent_performances.items():
            report['agent_performances'][agent_id] = {
                'success_rate': perf.success_rate,
                'total_suggestions': perf.total_suggestions,
                'total_pnl': perf.total_pnl,
                'avg_confidence': perf.avg_confidence,
                'recent_sharpe': perf.recent_sharpe
            }
        
        # Calculate overall Sharpe ratio
        if len(self.portfolio_state['recent_returns']) > 0:
            returns = list(self.portfolio_state['recent_returns'])
            report['meta_controller']['sharpe_ratio'] = (
                np.mean(returns) / (np.std(returns) + 1e-8)
            )
        
        return report
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'steps': self.steps,
            'portfolio_state': self.portfolio_state,
            'agent_performances': {k: self._serialize_agent_performance(v) for k, v in self.agent_performances.items()},
            'losses': self.losses
        }, path)
        
        logger.info(f"Meta-Controller model saved to {path}")
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.steps = checkpoint['steps']
        self.portfolio_state = checkpoint['portfolio_state']
        self.losses = checkpoint['losses']
        
        # Restore agent performances
        for agent_id, perf_dict in checkpoint['agent_performances'].items():
            perf_dict = self._deserialize_agent_performance(perf_dict)
            self.agent_performances[agent_id] = AgentPerformance(**perf_dict)
        
        logger.info(f"Meta-Controller model loaded from {path}")

    def _serialize_agent_performance(self, perf: AgentPerformance) -> Dict[str, Any]:
        """Serialize AgentPerformance for saving."""

        perf_dict = asdict(perf)
        perf_dict['last_updated'] = perf.last_updated.isoformat()
        return perf_dict

    def _deserialize_agent_performance(self, perf_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize AgentPerformance data from checkpoint."""

        last_updated = perf_dict.get('last_updated')

        if isinstance(last_updated, str):
            perf_dict['last_updated'] = datetime.fromisoformat(last_updated)
        elif last_updated is None:
            perf_dict['last_updated'] = datetime.now()

        return perf_dict


# Example usage
if __name__ == "__main__":
    # Initialize meta-controller
    config = MetaControllerConfig(
        state_dim=20,
        action_dim=3,  # hedge, liquidity, sentiment agents
        hidden_dims=[256, 256, 128]
    )
    
    meta_controller = MetaController(config)
    
    # Mock agent suggestions
    agent_suggestions = {
        'hedge_agent': {
            'action': 'LONG',
            'confidence': 0.75,
            'reasoning': 'High elasticity suggests upward movement'
        },
        'liquidity_agent': {
            'action': 'NEUTRAL',
            'confidence': 0.6,
            'reasoning': 'Moderate liquidity conditions'
        },
        'sentiment_agent': {
            'action': 'LONG',
            'confidence': 0.8,
            'reasoning': 'Positive sentiment from news analysis'
        }
    }
    
    # Mock market context
    market_context = {
        'volatility': 0.25,
        'regime_trending': 1.0,
        'regime_ranging': 0.0,
        'regime_volatile': 0.0,
        'volume_ratio': 1.2,
        'time_of_day_norm': 0.6,
        'vix_level': 0.22
    }
    
    # Get orchestrated decision
    result = meta_controller.orchestrate_agents(agent_suggestions, market_context)
    
    print("Meta-Controller Orchestration Result:")
    print(f"  Final Action: {result['consensus']['action']}")
    print(f"  Final Confidence: {result['consensus']['confidence']:.3f}")
    print(f"  Position Size: {result['consensus']['position_size']:.3f}")
    print(f"  Agent Weights: {result['agent_weights']}")
    print(f"  Disagreement Level: {result['metadata']['disagreement_level']:.3f}")
    
    # Simulate outcome
    import time
    time.sleep(1)
    
    # Update with outcome
    meta_controller.update_outcome(
        episode_data=result,
        final_pnl=0.015,  # 1.5% profit
        duration_minutes=90
    )
    
    # Get performance report
    report = meta_controller.get_performance_report()
    print(f"\nPerformance Report:")
    print(f"  Portfolio P&L: {report['meta_controller']['portfolio_pnl']:.4f}")
    print(f"  Total Trades: {report['meta_controller']['total_trades']}")
    print(f"  Avg Reward: {report['meta_controller']['avg_episode_reward']:.4f}")
    
    # Save model
    meta_controller.save_model("./models/saved/meta_controller.pt")
    
    print("\nMeta-Controller demonstration completed!")
