"""
Episodic Memory System for Intelligent Agents
Stores and retrieves past trading experiences for learning
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Single trading episode/experience"""
    episode_id: str
    timestamp: datetime
    
    # State information
    market_state: Dict[str, Any]  # Features, regime, etc.
    agent_state: Dict[str, Any]   # Agent's internal state
    
    # Action taken
    action: str
    action_params: Dict[str, Any]
    confidence: float
    
    # Outcome
    immediate_reward: float
    final_outcome: Optional[float] = None  # P&L after position closed
    duration: Optional[int] = None  # Minutes position was held
    
    # Context
    symbol: str = "SPY"
    regime: str = "unknown"
    volatility_level: str = "medium"
    
    # Metadata
    tags: List[str] = None
    notes: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class EpisodicMemory:
    """
    Episodic Memory System for Trading Agents
    
    Features:
    - Store detailed trading episodes
    - Similarity-based retrieval
    - Success rate analysis
    - Pattern recognition
    - Persistent storage (SQLite)
    """
    
    def __init__(
        self,
        memory_path: str = "./data/agent_memory/episodic_memory.db",
        max_episodes: int = 50000,
        similarity_threshold: float = 0.7
    ):
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.max_episodes = max_episodes
        self.similarity_threshold = similarity_threshold
        
        # Initialize database
        self._init_database()
        
        # In-memory cache for fast access
        self._episode_cache: Dict[str, Episode] = {}
        self._feature_cache: Dict[str, np.ndarray] = {}
        
        # Load recent episodes into cache
        self._load_recent_episodes()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        self.conn = sqlite3.connect(str(self.memory_path), check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                market_state TEXT,
                agent_state TEXT,
                action TEXT NOT NULL,
                action_params TEXT,
                confidence REAL,
                immediate_reward REAL,
                final_outcome REAL,
                duration INTEGER,
                symbol TEXT,
                regime TEXT,
                volatility_level TEXT,
                tags TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for faster queries
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp)
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol ON episodes(symbol)
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_regime ON episodes(regime)
        ''')
        
        self.conn.commit()
    
    def store_episode(self, episode: Episode):
        """
        Store a new episode in memory
        
        Args:
            episode: Episode object to store
        """
        # Convert to database format
        episode_data = (
            episode.episode_id,
            episode.timestamp.isoformat(),
            json.dumps(episode.market_state),
            json.dumps(episode.agent_state),
            episode.action,
            json.dumps(episode.action_params),
            episode.confidence,
            episode.immediate_reward,
            episode.final_outcome,
            episode.duration,
            episode.symbol,
            episode.regime,
            episode.volatility_level,
            json.dumps(episode.tags),
            episode.notes,
        )
        
        # Insert into database
        try:
            self.conn.execute('''
                INSERT OR REPLACE INTO episodes (
                    episode_id, timestamp, market_state, agent_state,
                    action, action_params, confidence, immediate_reward,
                    final_outcome, duration, symbol, regime, volatility_level,
                    tags, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', episode_data)
            self.conn.commit()
            
            # Add to cache
            self._episode_cache[episode.episode_id] = episode
            
            # Extract and cache features
            features = self._extract_episode_features(episode)
            self._feature_cache[episode.episode_id] = features
            
            logger.debug(f"Stored episode {episode.episode_id}")
            
        except Exception as e:
            logger.error(f"Error storing episode: {e}")
    
    def _extract_episode_features(self, episode: Episode) -> np.ndarray:
        """Extract numerical features from episode for similarity computation"""
        features = []
        
        # Market state features
        if 'features' in episode.market_state:
            market_features = episode.market_state['features']
            if isinstance(market_features, (list, np.ndarray)):
                normalized_features = list(market_features)[:50]
                if len(normalized_features) < 50:
                    normalized_features.extend([0.0] * (50 - len(normalized_features)))
                features.extend(normalized_features)
            else:
                features.extend([0.0] * 50)
        else:
            features.extend([0.0] * 50)
        
        # Hedge engine features (if available)
        hedge_features = [
            episode.market_state.get('elasticity', 0.0),
            episode.market_state.get('movement_energy', 0.0),
            episode.market_state.get('energy_asymmetry', 0.0),
            episode.market_state.get('pressure_net', 0.0),
            episode.market_state.get('dealer_gamma_sign', 0.0),
        ]
        features.extend(hedge_features)
        
        # Agent state features
        features.extend([
            episode.agent_state.get('position', 0.0),
            episode.agent_state.get('pnl', 0.0),
            episode.agent_state.get('drawdown', 0.0),
            episode.confidence,
        ])
        
        # Contextual features
        regime_encoding = {
            'ranging': 0.0, 'trending': 1.0, 'volatile': 2.0, 'unknown': -1.0
        }
        vol_encoding = {
            'low': 0.0, 'medium': 1.0, 'high': 2.0
        }
        action_encoding = {
            'LONG': 1.0, 'SHORT': -1.0, 'NEUTRAL': 0.0, 'HOLD': 0.0
        }
        
        features.extend([
            regime_encoding.get(episode.regime, -1.0),
            vol_encoding.get(episode.volatility_level, 1.0),
            action_encoding.get(episode.action, 0.0),
        ])
        
        # Time features (cyclical encoding)
        hour = episode.timestamp.hour
        day_of_week = episode.timestamp.weekday()
        
        features.extend([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
        ])
        
        return np.array(features, dtype=np.float32)
    
    def retrieve_similar_episodes(
        self,
        current_state: Dict[str, Any],
        current_action: Optional[str] = None,
        top_k: int = 10,
        time_window_days: Optional[int] = None
    ) -> List[Tuple[Episode, float]]:
        """
        Retrieve episodes similar to current state
        
        Args:
            current_state: Current market/agent state
            current_action: Optional action filter
            top_k: Number of similar episodes to return
            time_window_days: Only consider episodes from last N days
            
        Returns:
            List of (episode, similarity_score) tuples
        """
        if len(self._feature_cache) == 0:
            return []
        
        # Create dummy episode for feature extraction
        dummy_episode = Episode(
            episode_id="current",
            timestamp=datetime.now(),
            market_state=current_state.get('market_state', {}),
            agent_state=current_state.get('agent_state', {}),
            action=current_action or "NEUTRAL",
            action_params={},
            confidence=0.5,
            immediate_reward=0.0,
            regime=current_state.get('regime', 'unknown'),
            volatility_level=current_state.get('volatility_level', 'medium')
        )
        
        current_features = self._extract_episode_features(dummy_episode)
        
        # Get all cached features
        episode_ids = list(self._feature_cache.keys())
        feature_matrix = np.array(list(self._feature_cache.values()))
        
        # Compute similarities
        similarities = cosine_similarity([current_features], feature_matrix)[0]
        
        # Filter by time window if specified
        valid_indices = []
        if time_window_days:
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            for i, episode_id in enumerate(episode_ids):
                if episode_id in self._episode_cache:
                    episode = self._episode_cache[episode_id]
                    if episode.timestamp >= cutoff_date:
                        valid_indices.append(i)
        else:
            valid_indices = list(range(len(episode_ids)))
        
        # Filter by action if specified
        if current_action:
            action_indices = []
            for i in valid_indices:
                episode_id = episode_ids[i]
                if episode_id in self._episode_cache:
                    episode = self._episode_cache[episode_id]
                    if episode.action == current_action:
                        action_indices.append(i)
            valid_indices = action_indices
        
        # Get top-k similar episodes
        if len(valid_indices) == 0:
            return []
        
        valid_similarities = [(i, similarities[i]) for i in valid_indices]
        valid_similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, sim_score in valid_similarities[:top_k]:
            if sim_score >= self.similarity_threshold:
                episode_id = episode_ids[i]
                episode = self._episode_cache[episode_id]
                results.append((episode, sim_score))
        
        return results
    
    def get_success_rate(
        self,
        action: str,
        current_state: Optional[Dict] = None,
        min_similarity: float = 0.7,
        lookback_days: int = 90
    ) -> Dict[str, float]:
        """
        Calculate success rate for an action in similar situations
        
        Args:
            action: Action to analyze
            current_state: Current state for similarity matching
            min_similarity: Minimum similarity threshold
            lookback_days: Days to look back
            
        Returns:
            Dict with success metrics
        """
        if current_state:
            # Get similar episodes
            similar_episodes = self.retrieve_similar_episodes(
                current_state,
                current_action=action,
                top_k=100,
                time_window_days=lookback_days
            )
            episodes_to_analyze = [ep for ep, sim in similar_episodes if sim >= min_similarity]
        else:
            # Get all episodes with this action
            episodes_to_analyze = [
                ep for ep in self._episode_cache.values()
                if ep.action == action and ep.final_outcome is not None
                and ep.timestamp >= datetime.now() - timedelta(days=lookback_days)
            ]
        
        if not episodes_to_analyze:
            return {
                'success_rate': 0.5,  # Neutral default
                'avg_return': 0.0,
                'avg_duration': 0.0,
                'sample_size': 0
            }
        
        # Calculate metrics
        successful_episodes = [ep for ep in episodes_to_analyze if ep.final_outcome > 0]
        returns = [ep.final_outcome for ep in episodes_to_analyze if ep.final_outcome is not None]
        durations = [ep.duration for ep in episodes_to_analyze if ep.duration is not None]
        
        return {
            'success_rate': len(successful_episodes) / len(episodes_to_analyze),
            'avg_return': np.mean(returns) if returns else 0.0,
            'std_return': np.std(returns) if returns else 0.0,
            'avg_duration': np.mean(durations) if durations else 0.0,
            'sample_size': len(episodes_to_analyze),
            'win_rate': len(successful_episodes) / len(episodes_to_analyze),
            'best_return': np.max(returns) if returns else 0.0,
            'worst_return': np.min(returns) if returns else 0.0
        }
    
    def update_episode_outcome(
        self,
        episode_id: str,
        final_outcome: float,
        duration: int
    ):
        """Update episode with final outcome when position is closed"""
        try:
            self.conn.execute('''
                UPDATE episodes 
                SET final_outcome = ?, duration = ?
                WHERE episode_id = ?
            ''', (final_outcome, duration, episode_id))
            self.conn.commit()
            
            # Update cache
            if episode_id in self._episode_cache:
                self._episode_cache[episode_id].final_outcome = final_outcome
                self._episode_cache[episode_id].duration = duration
                
            logger.debug(f"Updated episode {episode_id} with outcome {final_outcome}")
            
        except Exception as e:
            logger.error(f"Error updating episode outcome: {e}")
    
    def _load_recent_episodes(self, days: int = 30):
        """Load recent episodes into memory cache"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor = self.conn.execute('''
                SELECT * FROM episodes 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            ''', (cutoff_date.isoformat(),))
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            for row in rows:
                row_dict = dict(zip(columns, row))
                
                # Convert back to Episode object
                episode = Episode(
                    episode_id=row_dict['episode_id'],
                    timestamp=datetime.fromisoformat(row_dict['timestamp']),
                    market_state=json.loads(row_dict['market_state']),
                    agent_state=json.loads(row_dict['agent_state']),
                    action=row_dict['action'],
                    action_params=json.loads(row_dict['action_params']),
                    confidence=row_dict['confidence'],
                    immediate_reward=row_dict['immediate_reward'],
                    final_outcome=row_dict['final_outcome'],
                    duration=row_dict['duration'],
                    symbol=row_dict['symbol'],
                    regime=row_dict['regime'],
                    volatility_level=row_dict['volatility_level'],
                    tags=json.loads(row_dict['tags']),
                    notes=row_dict['notes']
                )
                
                self._episode_cache[episode.episode_id] = episode
                
                # Extract features
                features = self._extract_episode_features(episode)
                self._feature_cache[episode.episode_id] = features
            
            logger.info(f"Loaded {len(rows)} recent episodes into cache")
            
        except Exception as e:
            logger.error(f"Error loading recent episodes: {e}")
    
    def analyze_patterns(self, lookback_days: int = 90) -> Dict[str, Any]:
        """
        Analyze patterns in stored episodes
        
        Returns:
            Analysis results including:
            - Action success rates
            - Regime performance
            - Time-of-day patterns
            - Common failure modes
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_episodes = [
            ep for ep in self._episode_cache.values()
            if ep.timestamp >= cutoff_date and ep.final_outcome is not None
        ]
        
        if len(recent_episodes) < 10:
            return {'error': 'Insufficient data for analysis'}
        
        analysis = {}
        
        # Action success rates
        action_analysis = {}
        for action in ['LONG', 'SHORT', 'NEUTRAL']:
            action_episodes = [ep for ep in recent_episodes if ep.action == action]
            if action_episodes:
                success_rate = len([ep for ep in action_episodes if ep.final_outcome > 0]) / len(action_episodes)
                avg_return = np.mean([ep.final_outcome for ep in action_episodes])
                action_analysis[action] = {
                    'success_rate': success_rate,
                    'avg_return': avg_return,
                    'count': len(action_episodes)
                }
        
        analysis['action_performance'] = action_analysis
        
        # Regime performance
        regime_analysis = {}
        for regime in ['ranging', 'trending', 'volatile']:
            regime_episodes = [ep for ep in recent_episodes if ep.regime == regime]
            if regime_episodes:
                success_rate = len([ep for ep in regime_episodes if ep.final_outcome > 0]) / len(regime_episodes)
                avg_return = np.mean([ep.final_outcome for ep in regime_episodes])
                regime_analysis[regime] = {
                    'success_rate': success_rate,
                    'avg_return': avg_return,
                    'count': len(regime_episodes)
                }
        
        analysis['regime_performance'] = regime_analysis
        
        # Time patterns
        hour_performance = {}
        for hour in range(24):
            hour_episodes = [ep for ep in recent_episodes if ep.timestamp.hour == hour]
            if hour_episodes:
                success_rate = len([ep for ep in hour_episodes if ep.final_outcome > 0]) / len(hour_episodes)
                hour_performance[hour] = {
                    'success_rate': success_rate,
                    'count': len(hour_episodes)
                }
        
        analysis['hourly_performance'] = hour_performance
        
        # Overall statistics
        analysis['overall'] = {
            'total_episodes': len(recent_episodes),
            'overall_success_rate': len([ep for ep in recent_episodes if ep.final_outcome > 0]) / len(recent_episodes),
            'avg_return': np.mean([ep.final_outcome for ep in recent_episodes]),
            'total_return': np.sum([ep.final_outcome for ep in recent_episodes]),
            'best_trade': np.max([ep.final_outcome for ep in recent_episodes]),
            'worst_trade': np.min([ep.final_outcome for ep in recent_episodes]),
            'sharpe_ratio': np.mean([ep.final_outcome for ep in recent_episodes]) / np.std([ep.final_outcome for ep in recent_episodes])
        }
        
        return analysis
    
    def cleanup_old_episodes(self, keep_days: int = 365):
        """Remove episodes older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        try:
            # Remove from database
            cursor = self.conn.execute('''
                DELETE FROM episodes WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            self.conn.commit()
            
            # Remove from cache
            episode_ids_to_remove = [
                episode_id for episode_id, episode in self._episode_cache.items()
                if episode.timestamp < cutoff_date
            ]
            
            for episode_id in episode_ids_to_remove:
                del self._episode_cache[episode_id]
                if episode_id in self._feature_cache:
                    del self._feature_cache[episode_id]
            
            logger.info(f"Cleaned up {deleted_count} old episodes")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize episodic memory
    memory = EpisodicMemory()
    
    # Create sample episode
    episode = Episode(
        episode_id="test_001",
        timestamp=datetime.now(),
        market_state={
            'features': np.random.randn(50).tolist(),
            'elasticity': 0.15,
            'movement_energy': 0.25,
            'regime': 'trending'
        },
        agent_state={
            'position': 0.0,
            'pnl': 0.0,
            'confidence': 0.8
        },
        action="LONG",
        action_params={'size': 0.5},
        confidence=0.8,
        immediate_reward=0.01,
        regime="trending",
        volatility_level="medium"
    )
    
    # Store episode
    memory.store_episode(episode)
    
    # Simulate closing position with outcome
    memory.update_episode_outcome("test_001", final_outcome=0.025, duration=120)
    
    # Query similar episodes
    current_state = {
        'market_state': {
            'features': np.random.randn(50).tolist(),
            'elasticity': 0.16,
            'movement_energy': 0.23
        },
        'agent_state': {
            'position': 0.0,
            'pnl': 0.0
        },
        'regime': 'trending',
        'volatility_level': 'medium'
    }
    
    similar_episodes = memory.retrieve_similar_episodes(current_state, current_action="LONG")
    print(f"Found {len(similar_episodes)} similar episodes")
    
    # Get success rate
    success_metrics = memory.get_success_rate("LONG", current_state)
    print(f"Success metrics for LONG: {success_metrics}")
    
    # Analyze patterns
    analysis = memory.analyze_patterns()
    print(f"Pattern analysis: {analysis}")
