#!/usr/bin/env python3
"""
Demo: Persistent Memory for Trading Agents

Shows how the memory system works with trading scenarios.
Based on Marktechpost Agentic AI Memory tutorial.
"""

import time
from gnosis.memory.trading_memory import TradingMemoryStore

print("="*70)
print("  TRADING MEMORY SYSTEM DEMO")
print("="*70)

# Create memory store with 1-hour half-life
memory = TradingMemoryStore(decay_half_life=3600)

print("\n📝 STEP 1: Store trading memories")
print("-" * 70)

# Add various trading events
memory.add(
    kind="signal",
    content="Hedge agent detected gamma squeeze setup",
    score=1.5,
    symbol="SPY",
    metadata={"gamma_exp": -0.8, "dealer_flow": 0.5}
)

memory.add(
    kind="trade",
    content="Entered long position at $580.50",
    score=1.0,
    symbol="SPY",
    metadata={"entry_price": 580.50, "position_size": 100}
)

memory.add(
    kind="outcome",
    content="Trade closed with +$250 profit",
    score=2.0,  # Big wins are important
    symbol="SPY",
    metadata={"pnl": 250.0, "exit_price": 583.00}
)

memory.add(
    kind="regime",
    content="Market entered trending_up regime",
    score=1.3,
    symbol="SPY",
    metadata={"markov_state": "trending_up", "confidence": 0.75}
)

memory.add(
    kind="pattern",
    content="Wyckoff spring detected at support",
    score=1.6,
    symbol="SPY",
    metadata={"support_level": 578.00}
)

# Add a memory for different symbol
memory.add(
    kind="signal",
    content="High volatility detected in tech sector",
    score=1.2,
    symbol="QQQ",
    metadata={"vix_spike": True}
)

print(f"✅ Stored {len(memory)} memories")

# Show summary
summary = memory.summarize("SPY", lookback_hours=1)
print(f"\n📊 Summary (last hour):")
print(f"   Total memories: {summary['total_memories']}")
print(f"   Recent (SPY): {summary['recent_count']}")
print(f"   Avg score: {summary['avg_score']:.2f}")
print(f"   By kind: {summary['by_kind']}")

print("\n" + "="*70)
print("\n🔍 STEP 2: Search for relevant memories")
print("-" * 70)

# Search for gamma-related memories
print("\n1️⃣  Search: 'gamma squeeze dealer'")
results = memory.search(query="gamma squeeze dealer", symbol="SPY", topk=3)
for i, item in enumerate(results, 1):
    decay = memory._decay_factor(item)
    print(f"   [{i}] {item.kind}: {item.content[:50]}... (score={item.score:.1f}, decay={decay:.3f})")

# Search by kind
print("\n2️⃣  Search: All 'outcome' memories")
outcomes = memory.search(kind="outcome", symbol="SPY", topk=5)
for i, item in enumerate(outcomes, 1):
    pnl = item.metadata.get('pnl', 0)
    print(f"   [{i}] {item.content} (PnL=${pnl:.2f})")

# Get recent memories
print("\n3️⃣  Get: Last 3 recent memories (any kind)")
recent = memory.get_recent(symbol="SPY", n=3)
for i, item in enumerate(recent, 1):
    print(f"   [{i}] {item.kind}: {item.content[:50]}...")

print("\n" + "="*70)
print("\n⏰ STEP 3: Demonstrate exponential decay")
print("-" * 70)

print("\nCreating memories at different times...")
memory.clear()  # Start fresh

# Add memory "now"
memory.add("test", "Recent memory (0 seconds old)", 1.0, "SPY")
time.sleep(1)

# Add memory "1 second ago" (simulate by manually setting timestamp)
old_item = memory.add("test", "Old memory (simulated 1800s old)", 1.0, "SPY")
old_item.timestamp = time.time() - 1800  # 30 minutes ago (half-life = 60 min)

# Add very old memory
ancient_item = memory.add("test", "Ancient memory (simulated 7200s old)", 1.0, "SPY")
ancient_item.timestamp = time.time() - 7200  # 2 hours ago

print("\n📈 Decay factors:")
for item in memory.items:
    decay = memory._decay_factor(item)
    age = time.time() - item.timestamp
    print(f"   {item.content[:30]:40s} → decay={decay:.4f} (age={age:.0f}s)")

print("\n" + "="*70)
print("\n🧹 STEP 4: Cleanup stale memories")
print("-" * 70)

print(f"\nBefore cleanup: {len(memory)} memories")
removed = memory.cleanup(min_score=0.3)
print(f"After cleanup:  {len(memory)} memories (removed {removed})")

# Add back memories for next demo
memory.clear()
memory.add("signal", "Hedge gamma squeeze", 1.5, "SPY")
memory.add("trade", "Long entry $580.50", 1.0, "SPY")
memory.add("outcome", "Profit $250", 2.0, "SPY", {"pnl": 250})
memory.add("outcome", "Loss $-100", 1.5, "SPY", {"pnl": -100})
memory.add("outcome", "Profit $150", 1.8, "SPY", {"pnl": 150})

print("\n" + "="*70)
print("\n📊 STEP 5: Calculate win rate from outcomes")
print("-" * 70)

outcomes = memory.search(kind="outcome", symbol="SPY", topk=10)
if outcomes:
    wins = sum(1 for item in outcomes if item.metadata.get('pnl', 0) > 0)
    total = len(outcomes)
    win_rate = wins / total if total > 0 else 0
    avg_pnl = sum(item.metadata.get('pnl', 0) for item in outcomes) / total
    
    print(f"\n   Total trades: {total}")
    print(f"   Wins: {wins}")
    print(f"   Win rate: {win_rate:.1%}")
    print(f"   Avg PnL: ${avg_pnl:.2f}")
    
    print("\n   Recent outcomes:")
    for item in outcomes:
        pnl = item.metadata.get('pnl', 0)
        result = "✅ WIN" if pnl > 0 else "❌ LOSS"
        print(f"      {result}: ${pnl:+.2f} - {item.content}")

print("\n" + "="*70)
print("\n✨ DEMO COMPLETE")
print("="*70)
print("\n💡 Key Takeaways:")
print("   • Memories decay exponentially over time (recent = more important)")
print("   • Hybrid search combines importance, recency, and text similarity")
print("   • Auto-cleanup prevents memory bloat")
print("   • Outcome tracking enables performance-based adaptation")
print("   • Memory can be filtered by kind, symbol, or query")
print("\n🚀 Ready to integrate into trading agents!")
print()
