"""
Enhanced Hedge Agent v3.0+ with ML capabilities, memory, and tool calling
Integrates with episodic/semantic memory and uses RL for continuous improvement
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.hedge_agent_v3 import HedgeAgentV3

# Import memory systems
from agents.memory.episodic_memory import Episode, EpisodicMemory
from agents.memory.semantic_memory import SemanticMemory

# Import ML models
from models.predictors.ensemble.xgboost_model import XGBoostEnsemble
from models.predictors.time_series.lstm_forecaster import LSTMForecastModel

# Import existing GNOSIS components
from schemas.core_schemas import AgentSuggestion

logger = logging.getLogger(__name__)


@dataclass
class ToolCallResult:
    """Result from a tool call"""
    tool_name: str
    input_params: Dict[str, Any]
    output: Dict[str, Any]
    success: bool
    execution_time: float
    timestamp: datetime


class ToolRegistry:
    """Registry of available tools for agents"""
    
    def __init__(self):
        self.tools = {}
        
        # Register default tools
        self.register_tool("web_search", self._web_search)
        self.register_tool("risk_calculator", self._risk_calculator)
        self.register_tool("historical_query", self._historical_query)
        self.register_tool("news_search", self._news_search)
        self.register_tool("options_analyzer", self._options_analyzer)
    
    def register_tool(self, name: str, function: callable):
        """Register a new tool"""
        self.tools[name] = function
        logger.debug(f"Registered tool: {name}")
    
    def call_tool(self, name: str, params: Dict[str, Any]) -> ToolCallResult:
        """Execute a tool and return results"""
        start_time = datetime.now()
        
        try:
            if name not in self.tools:
                raise ValueError(f"Tool {name} not found")
            
            result = self.tools[name](**params)
            success = True
            
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            result = {"error": str(e)}
            success = False
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ToolCallResult(
            tool_name=name,
            input_params=params,
            output=result,
            success=success,
            execution_time=execution_time,
            timestamp=start_time
        )
    
    def _web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Mock web search tool (replace with real implementation)"""
        # In real implementation, this would call actual web search API
        return {
            "query": query,
            "results": [
                {
                    "title": f"Search result {i} for: {query}",
                    "url": f"https://example.com/result_{i}",
                    "snippet": f"This is a mock search result snippet for {query}"
                }
                for i in range(min(max_results, 3))
            ],
            "total_results": 150
        }
    
    def _risk_calculator(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Monte Carlo risk simulation"""
        # Mock implementation - replace with real Monte Carlo
        position_size = scenario.get('position_size', 0.5)
        volatility = scenario.get('volatility', 0.2)
        time_horizon = scenario.get('time_horizon_days', 1)
        
        # Simple risk calculation
        daily_var = position_size * volatility * 1.65  # 95% VaR
        max_loss = daily_var * np.sqrt(time_horizon)
        
        return {
            "var_95": daily_var,
            "expected_max_loss": max_loss,
            "recommended_position_size": min(position_size, 0.1 / volatility),
            "risk_level": "high" if max_loss > 0.05 else "medium" if max_loss > 0.02 else "low"
        }
    
    def _historical_query(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Query historical data matching conditions"""
        # Mock implementation
        return {
            "matching_periods": 25,
            "avg_return": 0.015,
            "success_rate": 0.68,
            "max_drawdown": -0.08,
            "avg_duration_hours": 4.2,
            "confidence": 0.75
        }
    
    def _news_search(self, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        """Search for recent news about a symbol"""
        # Mock implementation
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "articles": [
                {
                    "title": f"Breaking: {symbol} reaches new levels",
                    "sentiment": 0.2,
                    "source": "Financial News",
                    "timestamp": datetime.now().isoformat(),
                    "relevance": 0.85
                }
            ],
            "overall_sentiment": 0.15,
            "news_volume": "high"
        }
    
    def _options_analyzer(self, symbol: str, analysis_type: str = "flow") -> Dict[str, Any]:
        """Analyze options data"""
        # Mock implementation
        return {
            "symbol": symbol,
            "unusual_activity": True,
            "flow_sentiment": 0.3,
            "large_trades": 12,
            "put_call_ratio": 0.85,
            "implied_volatility_rank": 45,
            "recommendation": "bullish_bias"
        }


class EnhancedHedgeAgentV3(HedgeAgentV3):
    """
    Enhanced Hedge Agent v3.0+ with advanced ML capabilities
    
    New Features:
    - Episodic memory (learns from past experiences)
    - Semantic memory (knowledge graph reasoning)
    - ML model integration (XGBoost, LSTM forecasts)
    - Tool calling capabilities
    - Reinforcement learning policy
    - Multi-objective decision making
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        memory_path: str = "./data/agent_memory/hedge_agent",
        enable_learning: bool = True
    ):
        # Initialize parent class
        super().__init__(config)
        
        # Memory systems
        self.episodic_memory = EpisodicMemory(f"{memory_path}_episodic.db")
        self.semantic_memory = SemanticMemory(f"{memory_path}_semantic.pkl")
        
        # ML models
        self.ml_ensemble = XGBoostEnsemble()
        self.lstm_model = None  # Will be loaded if available
        
        # Tool registry
        self.tool_registry = ToolRegistry()
        
        # Learning settings
        self.enable_learning = enable_learning
        self.learning_rate = 0.1
        self.exploration_rate = 0.1
        
        # State tracking
        self.current_episode_id = None
        self.current_episode = None
        self.decision_history = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_suggestions': 0,
            'successful_suggestions': 0,
            'avg_confidence': 0.0,
            'memory_retrievals': 0,
            'tool_calls': 0
        }
        
        # Load ML models if available
        self._load_ml_models()
        
        logger.info("Enhanced Hedge Agent v3.0+ initialized with memory and ML capabilities")
    
    def _load_ml_models(self):
        """Load pre-trained ML models if available"""
        try:
            # Try to load LSTM model
            lstm_path = "./models/saved/lstm_forecaster.pt"
            self.lstm_model = LSTMForecastModel(lstm_path)
            logger.info("Loaded LSTM forecaster model")
        except Exception as e:
            logger.warning(f"Could not load LSTM model: {e}")
        
        # ML ensemble will be trained on first use if no saved model exists
    
    def suggest(self, hedge_output: Dict) -> AgentSuggestion:
        """
        Enhanced suggestion method with memory and ML integration
        
        Process:
        1. Get base suggestion from original Hedge Engine logic
        2. Retrieve similar past experiences
        3. Get ML model predictions
        4. Query semantic memory for rules
        5. Optionally call tools for additional context
        6. Combine all inputs for final decision
        7. Store experience for future learning
        """
        
        # Generate unique episode ID
        self.current_episode_id = str(uuid.uuid4())
        
        # 1. Get base suggestion from original logic
        base_suggestion = super().suggest(hedge_output)
        
        # 2. Prepare current state for memory queries
        current_state = self._prepare_state_for_memory(hedge_output)
        
        # 3. Retrieve similar past experiences
        similar_episodes = self.episodic_memory.retrieve_similar_episodes(
            current_state,
            current_action=base_suggestion.action,
            top_k=5,
            time_window_days=90
        )
        
        # 4. Get success rate for this action in similar situations
        success_metrics = self.episodic_memory.get_success_rate(
            base_suggestion.action,
            current_state
        )
        
        # 5. Query semantic memory for rules and suggestions
        semantic_suggestions = self.semantic_memory.query_suggestions(
            current_state['market_state']
        )
        
        # 6. Get ML model predictions (if available)
        ml_predictions = self._get_ml_predictions(hedge_output, current_state)
        
        # 7. Decide whether to call tools based on uncertainty/novelty
        tool_results = []
        if self._should_call_tools(hedge_output, similar_episodes):
            tool_results = self._call_relevant_tools(hedge_output, current_state)
        
        # 8. Integrate all information sources
        enhanced_suggestion = self._integrate_all_inputs(
            base_suggestion=base_suggestion,
            similar_episodes=similar_episodes,
            success_metrics=success_metrics,
            semantic_suggestions=semantic_suggestions,
            ml_predictions=ml_predictions,
            tool_results=tool_results,
            current_state=current_state
        )
        
        # 9. Store current experience for future learning
        self._store_current_experience(
            enhanced_suggestion,
            hedge_output,
            current_state,
            similar_episodes,
            tool_results
        )
        
        # 10. Update performance metrics
        self._update_performance_metrics(enhanced_suggestion)
        
        return enhanced_suggestion
    
    def _prepare_state_for_memory(self, hedge_output: Dict) -> Dict[str, Any]:
        """Prepare current state for memory queries"""
        
        market_state = {
            # Hedge Engine outputs
            'elasticity': hedge_output.get('elasticity', 0.0),
            'movement_energy': hedge_output.get('movement_energy', 0.0),
            'energy_asymmetry': hedge_output.get('energy_asymmetry', 0.0),
            'pressure_net': hedge_output.get('pressure_net', 0.0),
            'dealer_gamma_sign': hedge_output.get('dealer_gamma_sign', 0.0),
            
            # Market context (would be provided by pipeline)
            'volatility': hedge_output.get('volatility', 0.2),
            'regime': hedge_output.get('regime', 'unknown'),
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
        }
        
        # Add technical indicators if available
        if 'technical_indicators' in hedge_output:
            market_state.update(hedge_output['technical_indicators'])
        
        agent_state = {
            'position': hedge_output.get('current_position', 0.0),
            'pnl': hedge_output.get('current_pnl', 0.0),
            'drawdown': hedge_output.get('current_drawdown', 0.0),
        }
        
        return {
            'market_state': market_state,
            'agent_state': agent_state,
            'regime': market_state['regime'],
            'volatility_level': self._categorize_volatility(market_state['volatility'])
        }
    
    def _categorize_volatility(self, volatility: float) -> str:
        """Categorize volatility level"""
        if volatility < 0.15:
            return 'low'
        elif volatility < 0.3:
            return 'medium'
        else:
            return 'high'
    
    def _get_ml_predictions(
        self,
        hedge_output: Dict,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get predictions from ML models"""
        predictions = {}
        
        # XGBoost ensemble predictions
        try:
            # Prepare features for XGBoost (simplified)
            features_dict = {**hedge_output}
            features_df = pd.DataFrame([features_dict])
            
            xgb_predictions = self.ml_ensemble.predict_all(features_df)
            predictions['xgboost'] = xgb_predictions
            
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
            predictions['xgboost'] = {
                'predicted_regime': 'unknown',
                'signal': 'hold',
                'confidence': 0.5
            }
        
        # LSTM forecaster predictions
        if self.lstm_model:
            try:
                # Prepare features for LSTM (needs feature matrix)
                features = np.array(list(hedge_output.values())[:10]).reshape(1, 1, -1)
                lstm_pred = self.lstm_model.predict(features, with_uncertainty=True)
                predictions['lstm'] = lstm_pred
                
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
        
        return predictions
    
    def _should_call_tools(
        self,
        hedge_output: Dict,
        similar_episodes: List[Tuple[Episode, float]]
    ) -> bool:
        """Decide whether to call tools for additional context"""
        
        # Call tools if:
        # 1. High uncertainty (low confidence or conflicting signals)
        # 2. Unusual market conditions
        # 3. Few similar past experiences
        
        high_energy = hedge_output.get('movement_energy', 0) > 0.5
        high_elasticity = hedge_output.get('elasticity', 0) > 0.3
        few_similar = len(similar_episodes) < 3
        
        # Random exploration
        random_exploration = np.random.random() < self.exploration_rate
        
        should_call = high_energy or high_elasticity or few_similar or random_exploration
        
        logger.debug(f"Tool calling decision: {should_call} (energy={high_energy}, elasticity={high_elasticity}, similar={len(similar_episodes)})")
        
        return should_call
    
    def _call_relevant_tools(
        self,
        hedge_output: Dict,
        current_state: Dict[str, Any]
    ) -> List[ToolCallResult]:
        """Call relevant tools based on current situation"""
        
        tool_results = []
        
        # 1. Risk calculation for current scenario
        risk_scenario = {
            'position_size': abs(hedge_output.get('suggested_position', 0.5)),
            'volatility': current_state['market_state']['volatility'],
            'time_horizon_days': 1
        }
        
        risk_result = self.tool_registry.call_tool('risk_calculator', risk_scenario)
        tool_results.append(risk_result)
        
        # 2. News search if high movement energy
        if hedge_output.get('movement_energy', 0) > 0.4:
            news_result = self.tool_registry.call_tool('news_search', {
                'symbol': 'SPY',
                'timeframe': '4h'
            })
            tool_results.append(news_result)
        
        # 3. Historical query for similar conditions
        if len(tool_results) < 3:  # Don't overload with tools
            historical_conditions = {
                'elasticity_range': [
                    hedge_output.get('elasticity', 0.2) - 0.05,
                    hedge_output.get('elasticity', 0.2) + 0.05
                ],
                'regime': current_state.get('regime', 'unknown')
            }
            
            historical_result = self.tool_registry.call_tool('historical_query', {
                'conditions': historical_conditions
            })
            tool_results.append(historical_result)
        
        self.performance_metrics['tool_calls'] += len(tool_results)
        
        return tool_results
    
    def _integrate_all_inputs(
        self,
        base_suggestion: AgentSuggestion,
        similar_episodes: List[Tuple[Episode, float]],
        success_metrics: Dict[str, float],
        semantic_suggestions: List[Dict[str, Any]],
        ml_predictions: Dict[str, Any],
        tool_results: List[ToolCallResult],
        current_state: Dict[str, Any]
    ) -> AgentSuggestion:
        """
        Integrate all information sources to make final decision
        
        Integration strategy:
        1. Start with base suggestion (Hedge Engine v3.0 logic)
        2. Adjust confidence based on historical success rate
        3. Consider ML model predictions
        4. Factor in semantic memory suggestions
        5. Incorporate tool results
        6. Apply final calibration
        """
        
        # Start with base suggestion
        final_action = base_suggestion.action
        final_confidence = base_suggestion.confidence
        reasoning_parts = [f"Base Hedge Engine: {base_suggestion.reasoning}"]
        
        # 1. Adjust confidence based on historical success rate
        if success_metrics['sample_size'] >= 5:  # Only if we have enough data
            historical_success = success_metrics['success_rate']
            confidence_adjustment = (historical_success - 0.5) * 0.3  # Max ±15% adjustment
            final_confidence += confidence_adjustment
            
            reasoning_parts.append(
                f"Historical success rate: {historical_success:.2f} "
                f"({success_metrics['sample_size']} similar cases)"
            )
        
        # 2. Consider ML predictions
        ml_confidence_sum = 0
        ml_weight = 0.2
        
        if 'xgboost' in ml_predictions:
            xgb_signal = ml_predictions['xgboost'].get('signal', 'hold')
            xgb_confidence = ml_predictions['xgboost'].get('confidence', 0.5)
            
            # Check agreement with base suggestion
            if self._signals_agree(final_action, xgb_signal):
                ml_confidence_sum += xgb_confidence * ml_weight
                reasoning_parts.append(f"XGBoost agrees: {xgb_signal} ({xgb_confidence:.2f})")
            else:
                ml_confidence_sum -= xgb_confidence * ml_weight * 0.5  # Penalty for disagreement
                reasoning_parts.append(f"XGBoost disagrees: {xgb_signal} vs {final_action}")
        
        if 'lstm' in ml_predictions and ml_predictions['lstm']:
            # LSTM provides price forecasts, convert to directional signal
            lstm_forecast = ml_predictions['lstm']
            forecast_1min = lstm_forecast.get('price_forecast_1min', {})
            
            if isinstance(forecast_1min, dict) and 'value' in forecast_1min:
                forecast_direction = 'LONG' if forecast_1min['value'] > 0.001 else 'SHORT' if forecast_1min['value'] < -0.001 else 'NEUTRAL'
                forecast_confidence = 1.0 - forecast_1min.get('uncertainty', 0.5)  # Convert uncertainty to confidence
                
                if self._signals_agree(final_action, forecast_direction):
                    ml_confidence_sum += forecast_confidence * ml_weight
                    reasoning_parts.append(f"LSTM forecast supports: {forecast_direction}")
        
        final_confidence += ml_confidence_sum
        
        # 3. Factor in semantic memory suggestions
        semantic_weight = 0.15
        if semantic_suggestions:
            top_semantic = semantic_suggestions[0]
            semantic_action = top_semantic['action']
            semantic_confidence = top_semantic['confidence']
            
            if self._actions_agree(final_action, semantic_action):
                final_confidence += semantic_confidence * semantic_weight
                reasoning_parts.append(f"Knowledge graph supports: {top_semantic['name']}")
            else:
                final_confidence -= semantic_confidence * semantic_weight * 0.3
        
        # 4. Incorporate tool results
        tool_weight = 0.1
        for tool_result in tool_results:
            if not tool_result.success:
                continue
            
            if tool_result.tool_name == 'risk_calculator':
                risk_level = tool_result.output.get('risk_level', 'medium')
                if risk_level == 'high':
                    final_confidence *= 0.8  # Reduce confidence for high risk
                    reasoning_parts.append("Risk calculator: HIGH RISK - reduced confidence")
                elif risk_level == 'low':
                    final_confidence *= 1.1  # Increase confidence for low risk
            
            elif tool_result.tool_name == 'news_search':
                news_sentiment = tool_result.output.get('overall_sentiment', 0.0)
                if abs(news_sentiment) > 0.2:  # Significant sentiment
                    if (news_sentiment > 0 and final_action == 'LONG') or \
                       (news_sentiment < 0 and final_action == 'SHORT'):
                        final_confidence += tool_weight
                        reasoning_parts.append(f"News sentiment supports ({news_sentiment:.2f})")
            
            elif tool_result.tool_name == 'historical_query':
                historical_success = tool_result.output.get('success_rate', 0.5)
                if historical_success > 0.6:
                    final_confidence += tool_weight
                    reasoning_parts.append(f"Historical analysis positive ({historical_success:.2f})")
        
        # 5. Apply final calibration and bounds
        final_confidence = np.clip(final_confidence, 0.1, 0.95)  # Keep within reasonable bounds
        
        # 6. Action override logic (if confidence in different direction is very high)
        # For now, stick with base action but this could be enhanced
        
        # Create enhanced suggestion
        enhanced_suggestion = AgentSuggestion(
            agent_id="hedge_agent_v3_enhanced",
            action=final_action,
            confidence=final_confidence,
            reasoning=" | ".join(reasoning_parts),
            timestamp=datetime.now(),
            metadata={
                'base_confidence': base_suggestion.confidence,
                'confidence_adjustments': {
                    'historical': confidence_adjustment if 'confidence_adjustment' in locals() else 0,
                    'ml_models': ml_confidence_sum,
                    'semantic': semantic_confidence * semantic_weight if semantic_suggestions else 0,
                    'tools': sum([tool_weight for _ in tool_results if _.success])
                },
                'similar_episodes_count': len(similar_episodes),
                'ml_predictions': ml_predictions,
                'tool_calls': len(tool_results),
                'episode_id': self.current_episode_id
            }
        )
        
        return enhanced_suggestion
    
    def _signals_agree(self, signal1: str, signal2: str) -> bool:
        """Check if two signals agree"""
        # Normalize signals
        normalize_map = {
            'LONG': 'bullish', 'SHORT': 'bearish', 'NEUTRAL': 'neutral', 'HOLD': 'neutral',
            'buy': 'bullish', 'sell': 'bearish', 'hold': 'neutral'
        }
        
        norm1 = normalize_map.get(signal1, signal1.lower())
        norm2 = normalize_map.get(signal2, signal2.lower())
        
        return norm1 == norm2
    
    def _actions_agree(self, action1: str, action_concept_id: str) -> bool:
        """Check if action agrees with semantic concept"""
        action_map = {
            'LONG': ['go_long', 'long'],
            'SHORT': ['go_short', 'short'],
            'NEUTRAL': ['hold', 'neutral', 'close_position']
        }
        
        compatible_concepts = action_map.get(action1, [])
        return action_concept_id in compatible_concepts
    
    def _store_current_experience(
        self,
        suggestion: AgentSuggestion,
        hedge_output: Dict,
        current_state: Dict[str, Any],
        similar_episodes: List[Tuple[Episode, float]],
        tool_results: List[ToolCallResult]
    ):
        """Store current experience in episodic memory"""
        
        if not self.enable_learning:
            return
        
        # Create episode
        episode = Episode(
            episode_id=self.current_episode_id,
            timestamp=suggestion.timestamp,
            market_state=current_state['market_state'],
            agent_state=current_state['agent_state'],
            action=suggestion.action,
            action_params={
                'confidence': suggestion.confidence,
                'base_confidence': suggestion.metadata.get('base_confidence', 0.5)
            },
            confidence=suggestion.confidence,
            immediate_reward=0.0,  # Will be updated when outcome is known
            symbol="SPY",  # Default, should be configurable
            regime=current_state.get('regime', 'unknown'),
            volatility_level=current_state.get('volatility_level', 'medium'),
            tags=[
                f"hedge_engine_v3",
                f"elasticity_{hedge_output.get('elasticity', 0):.2f}",
                f"energy_{hedge_output.get('movement_energy', 0):.2f}",
                f"similar_episodes_{len(similar_episodes)}",
                f"tool_calls_{len(tool_results)}"
            ],
            notes=f"Enhanced suggestion with {len(similar_episodes)} similar episodes"
        )
        
        # Store episode
        self.episodic_memory.store_episode(episode)
        self.current_episode = episode
        
        # Update semantic memory with current experience
        # (This will be more meaningful after we have the outcome)
        
        logger.debug(f"Stored episode {self.current_episode_id} in memory")
    
    def _update_performance_metrics(self, suggestion: AgentSuggestion):
        """Update agent performance metrics"""
        self.performance_metrics['total_suggestions'] += 1
        
        # Update rolling average confidence
        total = self.performance_metrics['total_suggestions']
        current_avg = self.performance_metrics['avg_confidence']
        new_avg = ((current_avg * (total - 1)) + suggestion.confidence) / total
        self.performance_metrics['avg_confidence'] = new_avg
        
        # Increment memory retrievals if we used similar episodes
        similar_count = suggestion.metadata.get('similar_episodes_count', 0)
        if similar_count > 0:
            self.performance_metrics['memory_retrievals'] += 1
    
    def update_outcome(
        self,
        episode_id: str,
        final_pnl: float,
        duration_minutes: int
    ):
        """
        Update episode outcome when position is closed
        This enables the agent to learn from results
        """
        
        if not self.enable_learning:
            return
        
        # Update episodic memory
        self.episodic_memory.update_episode_outcome(
            episode_id,
            final_outcome=final_pnl,
            duration=duration_minutes
        )
        
        # Learn from this experience in semantic memory
        if episode_id in self.episodic_memory._episode_cache:
            episode = self.episodic_memory._episode_cache[episode_id]
            
            outcome_type = 'profit' if final_pnl > 0.001 else 'loss' if final_pnl < -0.001 else 'breakeven'
            
            self.semantic_memory.learn_from_experience(
                market_state=episode.market_state,
                action_taken=episode.action,
                outcome=outcome_type,
                outcome_value=final_pnl
            )
        
        # Update performance metrics
        if final_pnl > 0:
            self.performance_metrics['successful_suggestions'] += 1
        
        logger.debug(f"Updated outcome for episode {episode_id}: PnL={final_pnl:.4f}, Duration={duration_minutes}min")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # Basic metrics
        total_suggestions = self.performance_metrics['total_suggestions']
        success_rate = (
            self.performance_metrics['successful_suggestions'] / total_suggestions
            if total_suggestions > 0 else 0.0
        )
        
        report = {
            'basic_metrics': {
                'total_suggestions': total_suggestions,
                'success_rate': success_rate,
                'avg_confidence': self.performance_metrics['avg_confidence'],
                'memory_usage_rate': (
                    self.performance_metrics['memory_retrievals'] / total_suggestions
                    if total_suggestions > 0 else 0.0
                ),
                'avg_tool_calls': (
                    self.performance_metrics['tool_calls'] / total_suggestions
                    if total_suggestions > 0 else 0.0
                )
            }
        }
        
        # Memory analysis
        try:
            memory_analysis = self.episodic_memory.analyze_patterns(lookback_days=30)
            report['memory_analysis'] = memory_analysis
        except Exception as e:
            logger.warning(f"Could not generate memory analysis: {e}")
            report['memory_analysis'] = {'error': str(e)}
        
        # Recent episodes
        recent_episodes = [
            {
                'episode_id': ep.episode_id,
                'timestamp': ep.timestamp.isoformat(),
                'action': ep.action,
                'confidence': ep.confidence,
                'final_outcome': ep.final_outcome,
                'regime': ep.regime
            }
            for ep in list(self.episodic_memory._episode_cache.values())[-10:]
        ]
        report['recent_episodes'] = recent_episodes
        
        return report
    
    def save_agent_state(self, path: str):
        """Save agent state including memory"""
        agent_state = {
            'performance_metrics': self.performance_metrics,
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'enable_learning': self.enable_learning
        }
        
        import json
        with open(f"{path}_agent_state.json", 'w') as f:
            json.dump(agent_state, f, indent=2, default=str)
        
        # Save memory systems
        self.semantic_memory.save_knowledge()
        
        logger.info(f"Saved enhanced agent state to {path}")


# Example usage and integration
if __name__ == "__main__":
    # Initialize enhanced hedge agent
    enhanced_agent = EnhancedHedgeAgentV3(
        memory_path="./data/test_agent_memory/hedge_agent",
        enable_learning=True
    )
    
    # Mock hedge engine output
    hedge_output = {
        'elasticity': 0.25,
        'movement_energy': 0.35,
        'energy_asymmetry': 0.15,
        'pressure_net': 0.1,
        'dealer_gamma_sign': 1.0,
        'volatility': 0.22,
        'regime': 'trending',
        'current_position': 0.0,
        'current_pnl': 0.0
    }
    
    # Get enhanced suggestion
    suggestion = enhanced_agent.suggest(hedge_output)
    
    print(f"Enhanced Suggestion:")
    print(f"  Action: {suggestion.action}")
    print(f"  Confidence: {suggestion.confidence:.3f}")
    print(f"  Reasoning: {suggestion.reasoning}")
    print(f"  Episode ID: {suggestion.metadata['episode_id']}")
    
    # Simulate outcome after some time
    import time
    time.sleep(1)  # Simulate trade duration
    
    # Update with outcome (simulate a profitable trade)
    enhanced_agent.update_outcome(
        episode_id=suggestion.metadata['episode_id'],
        final_pnl=0.025,  # 2.5% profit
        duration_minutes=120  # 2 hours
    )
    
    # Get performance report
    report = enhanced_agent.get_performance_report()
    print(f"\nPerformance Report:")
    print(f"  Total suggestions: {report['basic_metrics']['total_suggestions']}")
    print(f"  Success rate: {report['basic_metrics']['success_rate']:.2f}")
    print(f"  Avg confidence: {report['basic_metrics']['avg_confidence']:.3f}")
    print(f"  Memory usage: {report['basic_metrics']['memory_usage_rate']:.2f}")
    
    # Save agent state
    enhanced_agent.save_agent_state("./data/test_agent_memory/hedge_agent")
    
    print("\nEnhanced Hedge Agent demonstration completed!")
