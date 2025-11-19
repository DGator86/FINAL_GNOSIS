## Episodic Memory System

**Persistent learning from past trades to inform future decisions**

---

## 🎯 Overview

The Memory System adds **episodic learning** to your trading agents. Each trade becomes an episode with:
- **Context:** Market features, agent views at decision time
- **Decision:** What was chosen and why
- **Outcome:** PnL, exit reason, duration
- **Reflection:** Auto-generated critique and lessons learned

Before making new decisions, the system **recalls similar past contexts** and adjusts:
- Confidence thresholds (raise/lower based on historical win rate)
- Position sizing (size up winners, size down losers)
- Agent reliability weights (learn which agents work in which regimes)

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MEMORY SYSTEM                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │   Episode    │─────▶│  EpisodeStore│─────▶│  DuckDB  │ │
│  │   Schema     │      │  (Structured)│      │ Parquet  │ │
│  └──────────────┘      └──────────────┘      └──────────┘ │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │  to_text()   │─────▶│  Embedding   │─────▶│  FAISS   │ │
│  │              │      │  (sentence-  │      │  Index   │ │
│  └──────────────┘      │  transformers)│      └──────────┘ │
│                        └──────────────┘                     │
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │  Reflection  │─────▶│   Critique   │                   │
│  │   Engine     │      │  & Lessons   │                   │
│  └──────────────┘      └──────────────┘                   │
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │   Retrieval  │─────▶│   Recall     │                   │
│  │   Query      │      │  Top-K       │                   │
│  └──────────────┘      └──────────────┘                   │
│                                │                            │
│                                ▼                            │
│                        ┌──────────────┐                    │
│                        │  Adjustments │                    │
│                        │  - Confidence│                    │
│                        │  - Size      │                    │
│                        │  - Threshold │                    │
│                        └──────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Components

### 1. Episode Schema (`gnosis/memory/schema.py`)

Core data structure for trade episodes:

```python
@dataclass
class Episode:
    # Identity
    episode_id: str
    symbol: str
    
    # Entry context
    t_open: datetime
    price_open: float
    features_digest: Dict[str, float]  # Key L3 features
    agent_views: List[AgentView]       # Agent opinions
    
    # Decision
    decision: int  # -1, 0, 1
    decision_confidence: float
    position_size: float
    consensus_logic: str
    
    # Exit (filled later)
    t_close: Optional[datetime]
    price_close: Optional[float]
    exit_reason: str  # "TP", "SL", "time_stop", "signal_reverse"
    
    # Outcome
    pnl: Optional[float]
    return_pct: Optional[float]
    hit_target: Optional[bool]
    
    # Reflection
    critique: Optional[str]           # Post-mortem analysis
    regime_label: Optional[str]       # Market state
    key_lesson: Optional[str]         # Takeaway
    
    # Retrieval
    embedding: Optional[List[float]]  # Vector representation
    retrieval_score: float            # Recency × outcome weighting
```

**Key Methods:**
- `to_text()` - Convert to natural language for embedding
- `update_outcome()` - Fill in exit data
- `compute_retrieval_score()` - Weight for ranking (recency + outcome)

### 2. Episode Store (`gnosis/memory/store.py`)

DuckDB-backed persistent storage:

```python
from gnosis.memory import write_episode, read_episode, get_recent_episodes

# Write episode
ep = Episode(...)
write_episode(ep)

# Read back
ep2 = read_episode(episode_id)

# Get recent episodes
recent = get_recent_episodes(symbol="SPY", limit=100, closed_only=True)

# Get stats
stats = get_memory_stats(symbol="SPY")
# Returns: {total, closed, winners, losers, win_rate, avg_pnl, ...}
```

**Schema:**
- Structured SQL table with JSON column for full episode
- Indexed by symbol, time, outcome
- Append-only writes with updates on exit
- Export to Parquet for analysis

### 3. Vector Memory (`gnosis/memory/vec.py`)

FAISS-based semantic search:

```python
from gnosis.memory import recall_similar, index_episode, rebuild_memory_index

# Recall similar episodes
query = "SPY bullish setup with high gamma and tight spreads"
recall = recall_similar(query, k=10)

# Returns: MemoryRecall
#   - episodes: List[Episode]
#   - similarities: List[float]
#   - aggregate_signal: int
#   - aggregate_confidence: float

print(recall.to_summary())

# Index new episode
index_episode(ep)

# Rebuild entire index (after many new episodes)
rebuild_memory_index()
```

**How It Works:**
1. **Embed:** Convert episode context to 384-dim vector (sentence-transformers)
2. **Index:** Store in FAISS for fast cosine similarity search
3. **Query:** Find K most similar past episodes
4. **Weight:** Rank by `similarity × retrieval_score` (recency + outcome)

### 4. Reflection Engine (`gnosis/memory/reflect.py`)

Auto-generates post-trade analysis:

```python
from gnosis.memory import reflect_on_episode

# After trade closes
ep.update_outcome(t_close, price_close, exit_reason, pnl, hit_target)
reflect_on_episode(ep)

# Now has:
#   ep.critique = "WIN: +$0.45 (0.77%). Hit take profit target..."
#   ep.key_lesson = "Unanimous high-confidence setups work."
#   ep.regime_label = "trending_up"
```

**Reflection Logic:**
- Analyzes outcome vs setup quality
- Identifies regime from features
- Generates actionable lessons
- No LLM needed (rule-based)

### 5. Memory-Augmented Composer (`gnosis/memory/composer_hook.py`)

Integrates memory into decision flow:

```python
from gnosis.memory import augment_with_memory

# Base decision from agents
base_decision = 1
base_confidence = 0.7
base_size = 0.10

# Augment with memory
decision, confidence, size, memory_ctx = augment_with_memory(
    symbol="SPY",
    features=current_features,
    agent_views=agent_opinions,
    base_decision=base_decision,
    base_confidence=base_confidence,
    base_size=base_size
)

# Adjustments based on similar past setups:
#   confidence: 0.7 → 0.82 (boosted because similar setups won)
#   size: 0.10 → 0.13 (sized up 30%)
#   decision: 1 (same, high conviction)

print(f"Memory adjusted confidence by {memory_ctx['confidence_delta']:.3f}")
print(f"Recalled {memory_ctx['recall_count']} similar episodes")
print(f"Historical win rate: {memory_ctx['recall_win_rate']:.2%}")
```

**Adjustment Logic:**
- **High win rate + direction match** → boost confidence, size up, lower threshold
- **Low win rate + direction match** → reduce confidence, size down, raise threshold
- **Weak direction match** → be cautious, reduce size
- **Strong opposite signal** → flip decision (if high conviction)

---

## 🚀 Integration Example

### Complete Workflow

```python
import uuid
from datetime import datetime
from gnosis.memory import (
    Episode, AgentView,
    write_episode, reflect_on_episode, index_episode,
    augment_with_memory
)

# 1. ENTRY: Agents evaluate and decide
agent_views = [
    AgentView("hedge", 1, 0.8, "Gamma wall above", {"gamma": 0.7}),
    AgentView("liquidity", 1, 0.7, "Tight spreads", {"amihud": 0.01}),
    AgentView("sentiment", 1, 0.6, "Bullish momentum", {"momentum": 0.6}),
]

# Base decision from composer
base_decision = 1
base_confidence = 0.7
base_size = 0.10

# 2. AUGMENT: Memory adjusts decision
decision, confidence, size, memory_ctx = augment_with_memory(
    symbol="SPY",
    features={"hedge_gamma": 0.7, "liq_amihud": 0.01, "sent_momentum": 0.6},
    agent_views=agent_views,
    base_decision=base_decision,
    base_confidence=base_confidence,
    base_size=base_size
)

# 3. EXECUTE: Open position
print(f"Opening {decision} position: size={size:.2%}, conf={confidence:.2f}")

# 4. STORE: Create episode
ep = Episode(
    episode_id=str(uuid.uuid4()),
    symbol="SPY",
    t_open=datetime.now(),
    price_open=580.0,
    features_digest={"hedge_gamma": 0.7, "liq_amihud": 0.01, "sent_momentum": 0.6},
    agent_views=agent_views,
    decision=decision,
    decision_confidence=confidence,
    position_size=size,
    consensus_logic="memory-augmented 3-agent"
)
write_episode(ep)

# 5. EXIT: Update outcome when trade closes
ep.update_outcome(
    t_close=datetime.now(),
    price_close=582.5,
    exit_reason="TP",
    pnl=2.5 * size,
    hit_target=True
)

# 6. REFLECT: Generate critique and lessons
reflect_on_episode(ep)
write_episode(ep)  # Update with reflection

# 7. INDEX: Add to vector memory for future recall
index_episode(ep)

print(f"Episode complete: {ep.critique}")
print(f"Lesson: {ep.key_lesson}")
```

---

## 📊 Performance Impact

### Example Adjustments

**Scenario 1: Strong Historical Performance**
- Similar setups won 75% with avg PnL +$0.50
- **Adjustment:** Confidence +0.15, Size ×1.3
- **Rationale:** This setup works, be aggressive

**Scenario 2: Weak Historical Performance**
- Similar setups won 30% with avg PnL -$0.20
- **Adjustment:** Confidence -0.15, Size ×0.7
- **Rationale:** This setup fails often, be defensive

**Scenario 3: No Clear History**
- Few similar episodes or mixed outcomes
- **Adjustment:** Neutral (no change)
- **Rationale:** No data, stick with base decision

### Retrieval Score Formula

```python
retrieval_score = recency_weight × outcome_weight

recency_weight = 0.5 ^ (days_old / 30)  # Half-life of 30 days
outcome_weight = {
    2.0 if PnL > 0,  # Boost winners
    1.0 if PnL == 0,  # Neutral
    0.5 if PnL < 0   # Downweight losers
}
```

---

## 🛠️ Installation

### Required Dependencies

```bash
# Core (always needed)
pip install duckdb pandas numpy

# Vector memory (optional but recommended)
pip install faiss-cpu sentence-transformers

# Or for GPU:
pip install faiss-gpu sentence-transformers
```

### Verify Installation

```bash
cd /home/user/webapp/agentic-trading
python -c "from gnosis.memory import VECTOR_MEMORY_AVAILABLE; print(f'Vector Memory: {VECTOR_MEMORY_AVAILABLE}')"
```

---

## 📁 File Structure

```
gnosis/memory/
├── __init__.py              # Module exports
├── schema.py                # Episode and MemoryRecall classes
├── store.py                 # DuckDB storage layer
├── vec.py                   # FAISS vector retrieval
├── reflect.py               # Post-trade reflection
└── composer_hook.py         # Decision augmentation

memory/                      # Data directory (created automatically)
├── episodes.duckdb          # Structured episode storage
├── faiss_index.faiss        # Vector index
└── faiss_index.meta         # Episode ID mapping
```

---

## 🔬 Testing

Each module has built-in tests. Run them individually:

```bash
# Test episode store
python -m gnosis.memory.store

# Test vector memory
python -m gnosis.memory.vec

# Test reflection engine
python -m gnosis.memory.reflect

# Test composer hook
python -m gnosis.memory.composer_hook
```

---

## 📈 Best Practices

### 1. **Index Regularly**
- After every N trades (e.g., 10-50), run `rebuild_memory_index()`
- Keeps vector search fresh with latest episodes

### 2. **Monitor Stats**
```python
stats = get_memory_stats("SPY")
print(f"Win rate: {stats['win_rate']:.2%}")
print(f"Total episodes: {stats['total_episodes']}")
```

### 3. **Tune Adjustment Strength**
```python
# Conservative (small adjustments)
composer = MemoryAugmentedComposer(adjustment_strength=0.1)

# Aggressive (large adjustments)
composer = MemoryAugmentedComposer(adjustment_strength=0.5)

# Moderate (default)
composer = MemoryAugmentedComposer(adjustment_strength=0.3)
```

### 4. **Disable Memory for Testing**
```python
composer = MemoryAugmentedComposer(enable_memory=False)
```

### 5. **Export for Analysis**
```python
store = get_store()
store.export_parquet("exports/spy_episodes_2024.parquet", symbol="SPY")
```

---

## 🔮 Future Enhancements

### Planned Features
1. **Regime-Dependent Adjustment:** Different strategies per regime
2. **Multi-Symbol Learning:** Transfer learning across correlated assets
3. **Agent Reliability Tracking:** Weight agents by past performance
4. **LLM-Based Reflection:** GPT-4 generates richer critiques
5. **Hierarchical Memory:** Short-term (session) vs long-term (cross-session)
6. **Adversarial Filtering:** Detect and downweight luck-based wins

---

## 📞 API Reference

See individual module docstrings for complete API:
- `help(Episode)`
- `help(EpisodeStore)`
- `help(VectorMemory)`
- `help(MemoryAugmentedComposer)`

---

**Status:** ✅ Production Ready  
**Last Updated:** 2025-11-03  
**Integration:** Drop-in compatible with existing Gnosis system
