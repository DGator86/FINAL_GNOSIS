"""
Semantic Memory System using Knowledge Graphs
Stores and reasons about market concepts, relationships, and learned rules
"""

from typing import Dict, List, Tuple, Optional, Set, Any
import networkx as nx
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """A concept in the knowledge graph"""
    id: str
    name: str
    type: str  # 'market_condition', 'action', 'outcome', 'indicator', 'rule'
    attributes: Dict[str, Any]
    confidence: float = 1.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Relation:
    """A relationship between concepts"""
    source: str  # Concept ID
    target: str  # Concept ID
    relation_type: str  # 'causes', 'correlates_with', 'leads_to', 'indicates', 'precedes'
    strength: float  # -1 to 1 (negative = inverse relationship)
    confidence: float = 1.0
    evidence_count: int = 1
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


class SemanticMemory:
    """
    Knowledge Graph-based Semantic Memory System
    
    Features:
    - Store market concepts and relationships
    - Learn causal relationships from experience
    - Query for decision support
    - Reasoning chains
    - Concept similarity
    """
    
    def __init__(self, memory_path: str = "./data/agent_memory/semantic_memory.pkl"):
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Knowledge graph (NetworkX directed graph)
        self.knowledge_graph = nx.DiGraph()
        
        # Concept and relation storage
        self.concepts: Dict[str, Concept] = {}
        self.relations: Dict[Tuple[str, str, str], Relation] = {}
        
        # Initialize with basic market concepts
        self._initialize_base_knowledge()
        
        # Load existing knowledge if available
        self._load_knowledge()
    
    def _initialize_base_knowledge(self):
        """Initialize with fundamental market concepts and relationships"""
        
        # Market conditions
        market_conditions = [
            ("high_volatility", "High Volatility", "market_condition", {"vix_threshold": 25}),
            ("low_volatility", "Low Volatility", "market_condition", {"vix_threshold": 15}),
            ("trending_up", "Uptrend", "market_condition", {"trend_strength": 0.6}),
            ("trending_down", "Downtrend", "market_condition", {"trend_strength": -0.6}),
            ("ranging_market", "Sideways/Ranging", "market_condition", {"trend_strength": 0.1}),
            ("high_volume", "High Volume", "market_condition", {"volume_ratio": 1.5}),
            ("low_volume", "Low Volume", "market_condition", {"volume_ratio": 0.5}),
        ]
        
        for concept_id, name, concept_type, attrs in market_conditions:
            self.add_concept(concept_id, name, concept_type, attrs)
        
        # Actions
        actions = [
            ("go_long", "Enter Long Position", "action", {"direction": 1}),
            ("go_short", "Enter Short Position", "action", {"direction": -1}),
            ("close_position", "Close Position", "action", {"direction": 0}),
            ("increase_position", "Increase Position Size", "action", {"scaling": 1.5}),
            ("reduce_position", "Reduce Position Size", "action", {"scaling": 0.5}),
            ("hold", "Hold Current Position", "action", {"direction": 0}),
        ]
        
        for concept_id, name, concept_type, attrs in actions:
            self.add_concept(concept_id, name, concept_type, attrs)
        
        # Outcomes
        outcomes = [
            ("profit", "Profitable Trade", "outcome", {"pnl_threshold": 0.01}),
            ("loss", "Losing Trade", "outcome", {"pnl_threshold": -0.01}),
            ("breakeven", "Breakeven Trade", "outcome", {"pnl_range": [-0.001, 0.001]}),
            ("large_profit", "Large Profit", "outcome", {"pnl_threshold": 0.05}),
            ("large_loss", "Large Loss", "outcome", {"pnl_threshold": -0.05}),
        ]
        
        for concept_id, name, concept_type, attrs in outcomes:
            self.add_concept(concept_id, name, concept_type, attrs)
        
        # Technical indicators
        indicators = [
            ("rsi_overbought", "RSI Overbought", "indicator", {"rsi_threshold": 70}),
            ("rsi_oversold", "RSI Oversold", "indicator", {"rsi_threshold": 30}),
            ("macd_bullish", "MACD Bullish Signal", "indicator", {"macd_signal": "positive"}),
            ("macd_bearish", "MACD Bearish Signal", "indicator", {"macd_signal": "negative"}),
            ("bollinger_squeeze", "Bollinger Band Squeeze", "indicator", {"bb_width_percentile": 20}),
        ]
        
        for concept_id, name, concept_type, attrs in indicators:
            self.add_concept(concept_id, name, concept_type, attrs)
        
        # Add some basic relationships
        basic_relations = [
            # Volatility relationships
            ("high_volatility", "large_profit", "increases_probability_of", 0.3),
            ("high_volatility", "large_loss", "increases_probability_of", 0.3),
            ("low_volatility", "ranging_market", "often_coincides_with", 0.7),
            
            # Trend relationships
            ("trending_up", "go_long", "suggests", 0.6),
            ("trending_down", "go_short", "suggests", 0.6),
            ("ranging_market", "hold", "suggests", 0.5),
            
            # Technical indicator relationships
            ("rsi_overbought", "go_short", "suggests", 0.4),
            ("rsi_oversold", "go_long", "suggests", 0.4),
            ("macd_bullish", "go_long", "suggests", 0.5),
            ("macd_bearish", "go_short", "suggests", 0.5),
        ]
        
        for source, target, relation_type, strength in basic_relations:
            self.add_relation(source, target, relation_type, strength)
    
    def add_concept(
        self,
        concept_id: str,
        name: str,
        concept_type: str,
        attributes: Dict[str, Any],
        confidence: float = 1.0
    ):
        """Add a new concept to the knowledge graph"""
        concept = Concept(
            id=concept_id,
            name=name,
            type=concept_type,
            attributes=attributes,
            confidence=confidence
        )
        
        self.concepts[concept_id] = concept
        self.knowledge_graph.add_node(
            concept_id,
            name=name,
            type=concept_type,
            attributes=attributes,
            confidence=confidence
        )
        
        logger.debug(f"Added concept: {name}")
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        strength: float,
        confidence: float = 1.0
    ):
        """Add or update a relationship between concepts"""
        
        # Check if concepts exist
        if source_id not in self.concepts or target_id not in self.concepts:
            logger.warning(f"Cannot add relation: missing concepts {source_id} or {target_id}")
            return
        
        relation_key = (source_id, target_id, relation_type)
        
        if relation_key in self.relations:
            # Update existing relation
            existing = self.relations[relation_key]
            # Weighted average of strength
            new_count = existing.evidence_count + 1
            new_strength = (
                (existing.strength * existing.evidence_count + strength) / new_count
            )
            existing.strength = new_strength
            existing.evidence_count = new_count
            existing.confidence = min(1.0, existing.confidence + 0.1)
            existing.last_updated = datetime.now()
        else:
            # Create new relation
            relation = Relation(
                source=source_id,
                target=target_id,
                relation_type=relation_type,
                strength=strength,
                confidence=confidence
            )
            self.relations[relation_key] = relation
        
        # Add to graph
        self.knowledge_graph.add_edge(
            source_id,
            target_id,
            relation_type=relation_type,
            strength=strength,
            confidence=confidence
        )
        
        logger.debug(f"Added relation: {source_id} {relation_type} {target_id} (strength: {strength})")
    
    def learn_from_experience(
        self,
        market_state: Dict[str, Any],
        action_taken: str,
        outcome: str,
        outcome_value: float
    ):
        """
        Learn relationships from trading experience
        
        Args:
            market_state: Current market conditions and indicators
            action_taken: Action that was taken
            outcome: Type of outcome ('profit', 'loss', etc.)
            outcome_value: Numerical outcome (P&L)
        """
        
        # Extract market conditions from state
        market_conditions = self._extract_market_conditions(market_state)
        
        # Create concepts for new conditions if needed
        for condition_id, condition_data in market_conditions.items():
            if condition_id not in self.concepts:
                self.add_concept(
                    condition_id,
                    condition_data['name'],
                    'market_condition',
                    condition_data['attributes']
                )
        
        # Learn relationships between market conditions and action
        action_strength = 0.6 if outcome == 'profit' else -0.3 if outcome == 'loss' else 0.0
        
        for condition_id in market_conditions:
            self.add_relation(
                condition_id,
                action_taken,
                'suggests' if action_strength > 0 else 'suggests_against',
                abs(action_strength),
                confidence=0.7
            )
        
        # Learn relationships between action and outcome
        outcome_strength = 1.0 if outcome_value > 0 else -1.0
        self.add_relation(
            action_taken,
            outcome,
            'leads_to',
            abs(outcome_strength),
            confidence=0.8
        )
        
        # Learn conditional relationships (market condition -> action -> outcome)
        for condition_id in market_conditions:
            # Create a composite rule
            rule_id = f"rule_{condition_id}_{action_taken}_{outcome}"
            
            if rule_id not in self.concepts:
                self.add_concept(
                    rule_id,
                    f"In {condition_id}, {action_taken} leads to {outcome}",
                    'rule',
                    {
                        'condition': condition_id,
                        'action': action_taken,
                        'outcome': outcome,
                        'success_rate': 0.5,
                        'sample_size': 0
                    }
                )
            
            # Update rule statistics
            rule = self.concepts[rule_id]
            current_success = rule.attributes.get('success_rate', 0.5)
            sample_size = rule.attributes.get('sample_size', 0)
            
            new_sample_size = sample_size + 1
            success_indicator = 1.0 if outcome_value > 0 else 0.0
            new_success_rate = (
                (current_success * sample_size + success_indicator) / new_sample_size
            )
            
            rule.attributes['success_rate'] = new_success_rate
            rule.attributes['sample_size'] = new_sample_size
    
    def _extract_market_conditions(self, market_state: Dict[str, Any]) -> Dict[str, Dict]:
        """Extract identifiable market conditions from state"""
        conditions = {}
        
        # Volatility conditions
        volatility = market_state.get('volatility', 0.1)
        if volatility > 0.25:
            conditions['current_high_vol'] = {
                'name': f'High Volatility ({volatility:.3f})',
                'attributes': {'volatility_level': volatility}
            }
        elif volatility < 0.1:
            conditions['current_low_vol'] = {
                'name': f'Low Volatility ({volatility:.3f})',
                'attributes': {'volatility_level': volatility}
            }
        
        # Regime conditions
        regime = market_state.get('regime', 'unknown')
        if regime != 'unknown':
            conditions[f'current_{regime}'] = {
                'name': f'Current {regime.title()} Regime',
                'attributes': {'regime_type': regime}
            }
        
        # Technical indicator conditions
        if 'rsi' in market_state:
            rsi = market_state['rsi']
            if rsi > 70:
                conditions['current_rsi_overbought'] = {
                    'name': f'RSI Overbought ({rsi:.1f})',
                    'attributes': {'rsi_value': rsi}
                }
            elif rsi < 30:
                conditions['current_rsi_oversold'] = {
                    'name': f'RSI Oversold ({rsi:.1f})',
                    'attributes': {'rsi_value': rsi}
                }
        
        # Hedge Engine conditions (if available)
        if 'elasticity' in market_state:
            elasticity = market_state['elasticity']
            if elasticity > 0.3:
                conditions['high_elasticity'] = {
                    'name': f'High Market Elasticity ({elasticity:.3f})',
                    'attributes': {'elasticity_value': elasticity}
                }
            elif elasticity < 0.1:
                conditions['low_elasticity'] = {
                    'name': f'Low Market Elasticity ({elasticity:.3f})',
                    'attributes': {'elasticity_value': elasticity}
                }
        
        if 'movement_energy' in market_state:
            energy = market_state['movement_energy']
            if energy > 0.5:
                conditions['high_movement_energy'] = {
                    'name': f'High Movement Energy ({energy:.3f})',
                    'attributes': {'energy_value': energy}
                }
        
        return conditions
    
    def query_suggestions(
        self,
        market_state: Dict[str, Any],
        relation_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query for action suggestions based on current market state
        
        Args:
            market_state: Current market conditions
            relation_types: Types of relations to consider
            
        Returns:
            List of suggested actions with confidence scores
        """
        if relation_types is None:
            relation_types = ['suggests', 'indicates', 'leads_to']
        
        # Extract current conditions
        current_conditions = self._extract_market_conditions(market_state)
        
        suggestions = {}
        
        # Find actions suggested by current conditions
        for condition_id in current_conditions:
            # Look for similar conditions in knowledge graph
            similar_conditions = self._find_similar_concepts(condition_id, concept_type='market_condition')
            
            for similar_id, similarity in similar_conditions:
                # Find outgoing relations from this condition
                if similar_id in self.knowledge_graph:
                    for neighbor in self.knowledge_graph.successors(similar_id):
                        edge_data = self.knowledge_graph[similar_id][neighbor]
                        
                        if edge_data.get('relation_type') in relation_types:
                            neighbor_concept = self.concepts.get(neighbor)
                            
                            if neighbor_concept and neighbor_concept.type == 'action':
                                strength = edge_data.get('strength', 0.5)
                                confidence = edge_data.get('confidence', 0.5)
                                
                                # Weight by similarity to current condition
                                weighted_strength = strength * similarity * confidence
                                
                                if neighbor in suggestions:
                                    suggestions[neighbor] += weighted_strength
                                else:
                                    suggestions[neighbor] = weighted_strength
        
        # Convert to sorted list
        suggestion_list = []
        for action_id, total_strength in suggestions.items():
            action_concept = self.concepts[action_id]
            suggestion_list.append({
                'action': action_id,
                'name': action_concept.name,
                'confidence': min(1.0, total_strength),
                'reasoning': self._get_reasoning_chain(current_conditions, action_id)
            })
        
        # Sort by confidence
        suggestion_list.sort(key=lambda x: x['confidence'], reverse=True)
        
        return suggestion_list[:5]  # Top 5 suggestions
    
    def _find_similar_concepts(
        self,
        concept_id: str,
        concept_type: Optional[str] = None,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """Find concepts similar to the given concept"""
        
        if concept_id not in self.concepts:
            return [(concept_id, 1.0)]  # Return self with perfect similarity
        
        target_concept = self.concepts[concept_id]
        similarities = []
        
        for other_id, other_concept in self.concepts.items():
            if concept_type and other_concept.type != concept_type:
                continue
            
            if other_id == concept_id:
                similarities.append((other_id, 1.0))
                continue
            
            # Calculate similarity based on attributes and type
            similarity = self._calculate_concept_similarity(target_concept, other_concept)
            
            if similarity >= similarity_threshold:
                similarities.append((other_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def _calculate_concept_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """Calculate similarity between two concepts"""
        
        # Same type bonus
        type_similarity = 1.0 if concept1.type == concept2.type else 0.5
        
        # Attribute similarity
        attr_similarity = 0.5  # Default
        
        if concept1.attributes and concept2.attributes:
            common_keys = set(concept1.attributes.keys()) & set(concept2.attributes.keys())
            
            if common_keys:
                similarities = []
                for key in common_keys:
                    val1 = concept1.attributes[key]
                    val2 = concept2.attributes[key]
                    
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        # Numerical similarity
                        max_val = max(abs(val1), abs(val2), 1e-6)
                        diff = abs(val1 - val2)
                        similarities.append(1.0 - diff / max_val)
                    elif val1 == val2:
                        # Exact match
                        similarities.append(1.0)
                    else:
                        # No match
                        similarities.append(0.0)
                
                attr_similarity = np.mean(similarities) if similarities else 0.5
        
        # Name similarity (simple string matching)
        name_similarity = 0.5
        if concept1.name and concept2.name:
            common_words = set(concept1.name.lower().split()) & set(concept2.name.lower().split())
            total_words = set(concept1.name.lower().split()) | set(concept2.name.lower().split())
            name_similarity = len(common_words) / len(total_words) if total_words else 0.0
        
        # Combine similarities
        overall_similarity = (
            type_similarity * 0.4 +
            attr_similarity * 0.4 +
            name_similarity * 0.2
        )
        
        return overall_similarity
    
    def _get_reasoning_chain(
        self,
        conditions: Dict[str, Dict],
        action_id: str,
        max_depth: int = 3
    ) -> List[str]:
        """Get reasoning chain for why an action is suggested"""
        
        reasoning = []
        
        for condition_id, condition_data in conditions.items():
            # Find path from condition to action
            try:
                if condition_id in self.knowledge_graph and action_id in self.knowledge_graph:
                    # Simple direct path
                    if self.knowledge_graph.has_edge(condition_id, action_id):
                        edge_data = self.knowledge_graph[condition_id][action_id]
                        relation_type = edge_data.get('relation_type', 'relates_to')
                        strength = edge_data.get('strength', 0.5)
                        
                        reasoning.append(
                            f"{condition_data['name']} {relation_type} {self.concepts[action_id].name} "
                            f"(strength: {strength:.2f})"
                        )
            except:
                continue
        
        return reasoning[:3]  # Limit to top 3 reasoning steps
    
    def get_concept_network(self, concept_id: str, max_hops: int = 2) -> Dict[str, Any]:
        """Get the network of concepts connected to a given concept"""
        
        if concept_id not in self.knowledge_graph:
            return {}
        
        # Get subgraph within max_hops
        ego_graph = nx.ego_graph(self.knowledge_graph, concept_id, radius=max_hops)
        
        # Convert to serializable format
        network = {
            'nodes': [],
            'edges': []
        }
        
        for node_id in ego_graph.nodes():
            concept = self.concepts[node_id]
            network['nodes'].append({
                'id': node_id,
                'name': concept.name,
                'type': concept.type,
                'confidence': concept.confidence
            })
        
        for source, target in ego_graph.edges():
            edge_data = ego_graph[source][target]
            network['edges'].append({
                'source': source,
                'target': target,
                'relation_type': edge_data.get('relation_type', 'unknown'),
                'strength': edge_data.get('strength', 0.5),
                'confidence': edge_data.get('confidence', 0.5)
            })
        
        return network
    
    def save_knowledge(self):
        """Save knowledge graph to disk"""
        try:
            knowledge_data = {
                'concepts': {k: asdict(v) for k, v in self.concepts.items()},
                'relations': {f"{k[0]}|{k[1]}|{k[2]}": asdict(v) for k, v in self.relations.items()},
                'graph': nx.node_link_data(self.knowledge_graph)
            }
            
            with open(self.memory_path, 'wb') as f:
                pickle.dump(knowledge_data, f)
            
            logger.info(f"Saved knowledge graph with {len(self.concepts)} concepts and {len(self.relations)} relations")
            
        except Exception as e:
            logger.error(f"Error saving knowledge: {e}")
    
    def _ensure_datetime(self, value: Any) -> Optional[datetime]:
        """Normalize stored datetime values that may have been serialized differently."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None
    
    def _load_knowledge(self):
        """Load knowledge graph from disk"""
        if not self.memory_path.exists():
            return
        
        try:
            with open(self.memory_path, 'rb') as f:
                knowledge_data = pickle.load(f)
            
            # Load concepts
            for concept_id, concept_data in knowledge_data.get('concepts', {}).items():
                concept_record = dict(concept_data)
                concept_record['created_at'] = self._ensure_datetime(concept_record.get('created_at'))
                concept = Concept(**concept_record)
                self.concepts[concept_id] = concept
            
            # Load relations
            for relation_key, relation_data in knowledge_data.get('relations', {}).items():
                source, target, relation_type = relation_key.split('|')
                relation_record = dict(relation_data)
                relation_record['last_updated'] = self._ensure_datetime(relation_record.get('last_updated'))
                relation = Relation(**relation_record)
                self.relations[(source, target, relation_type)] = relation
            
            # Load graph
            self.knowledge_graph = nx.node_link_graph(knowledge_data.get('graph', {}))
            
            logger.info(f"Loaded knowledge graph with {len(self.concepts)} concepts and {len(self.relations)} relations")
            
        except Exception as e:
            logger.error(f"Error loading knowledge: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize semantic memory
    semantic_memory = SemanticMemory()
    
    # Simulate learning from experience
    market_state = {
        'volatility': 0.3,  # High volatility
        'regime': 'volatile',
        'rsi': 75,  # Overbought
        'elasticity': 0.25,
        'movement_energy': 0.6
    }
    
    # Agent took a short action and made profit
    semantic_memory.learn_from_experience(
        market_state=market_state,
        action_taken='go_short',
        outcome='profit',
        outcome_value=0.025
    )
    
    # Query for suggestions in similar market state
    suggestions = semantic_memory.query_suggestions(market_state)
    
    print("Action Suggestions:")
    for suggestion in suggestions:
        print(f"  {suggestion['name']}: {suggestion['confidence']:.3f}")
        for reason in suggestion['reasoning']:
            print(f"    - {reason}")
    
    # Get network around a concept
    network = semantic_memory.get_concept_network('high_volatility')
    print(f"\nNetwork around 'high_volatility': {len(network.get('nodes', []))} nodes, {len(network.get('edges', []))} edges")
    
    # Save knowledge
    semantic_memory.save_knowledge()
