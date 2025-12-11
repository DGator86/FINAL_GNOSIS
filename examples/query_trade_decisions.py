#!/usr/bin/env python3
"""
Example: How to query trade decisions for analysis and ML.

This shows common query patterns for:
- Analytics dashboards
- Performance analysis
- ML dataset construction
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db import SessionLocal
from crud.trade_decision import list_trade_decisions, get_trade_decision_by_id


def query_recent_decisions():
    """Query recent trade decisions."""
    db = SessionLocal()
    try:
        decisions = list_trade_decisions(db, limit=10)

        print("=" * 60)
        print(f"Recent Trade Decisions ({len(decisions)} found)")
        print("=" * 60)

        for d in decisions:
            print()
            print(f"ID: {d.id}")
            print(f"Symbol: {d.symbol}")
            print(f"Direction: {d.direction}")
            print(f"Structure: {d.structure}")
            print(f"Mode: {d.mode}")
            print(f"Timestamp: {d.timestamp}")
            print(f"Status: {d.status or 'pending'}")

            # Show agent consensus
            hedge_conf = d.hedge_agent_vote.get('confidence', 0)
            liq_conf = d.liquidity_agent_vote.get('confidence', 0)
            sent_conf = d.sentiment_agent_vote.get('confidence', 0)
            avg_conf = (hedge_conf + liq_conf + sent_conf) / 3

            print(f"Agent Confidence: {avg_conf:.2f}")
            print(f"  Hedge: {hedge_conf:.2f}")
            print(f"  Liquidity: {liq_conf:.2f}")
            print(f"  Sentiment: {sent_conf:.2f}")

    finally:
        db.close()


def query_by_symbol(symbol: str):
    """Query decisions for a specific symbol."""
    db = SessionLocal()
    try:
        decisions = list_trade_decisions(db, symbol=symbol, limit=50)

        print()
        print("=" * 60)
        print(f"Decisions for {symbol} ({len(decisions)} found)")
        print("=" * 60)

        if not decisions:
            print("No decisions found")
            return

        # Analyze direction bias
        long_count = sum(1 for d in decisions if d.direction == "long")
        short_count = sum(1 for d in decisions if d.direction == "short")
        neutral_count = sum(1 for d in decisions if d.direction == "neutral")

        print()
        print(f"Direction Distribution:")
        print(f"  Long: {long_count} ({long_count/len(decisions)*100:.1f}%)")
        print(f"  Short: {short_count} ({short_count/len(decisions)*100:.1f}%)")
        print(f"  Neutral: {neutral_count} ({neutral_count/len(decisions)*100:.1f}%)")

        # Analyze structure distribution
        from collections import Counter
        structure_counts = Counter(d.structure for d in decisions)

        print()
        print(f"Structure Distribution:")
        for structure, count in structure_counts.most_common():
            print(f"  {structure}: {count} ({count/len(decisions)*100:.1f}%)")

        # Analyze mode distribution
        mode_counts = Counter(d.mode for d in decisions)

        print()
        print(f"Mode Distribution:")
        for mode, count in mode_counts.most_common():
            print(f"  {mode}: {count} ({count/len(decisions)*100:.1f}%)")

    finally:
        db.close()


def query_for_ml_dataset():
    """Query decisions for ML dataset construction."""
    db = SessionLocal()
    try:
        # Example: Get backtest decisions from last 30 days
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30)

        decisions = list_trade_decisions(
            db,
            mode="backtest",
            limit=1000,
        )

        print()
        print("=" * 60)
        print(f"ML Dataset Query ({len(decisions)} samples)")
        print("=" * 60)

        if not decisions:
            print("No decisions found")
            return

        # Analyze feature availability
        feature_counts = {
            'dealer_features': 0,
            'liquidity_features': 0,
            'sentiment_features': 0,
            'has_execution': 0,
        }

        for d in decisions:
            if d.dealer_features:
                feature_counts['dealer_features'] += 1
            if d.liquidity_features:
                feature_counts['liquidity_features'] += 1
            if d.sentiment_features:
                feature_counts['sentiment_features'] += 1
            if d.entry_price is not None:
                feature_counts['has_execution'] += 1

        print()
        print(f"Feature Availability:")
        for feature, count in feature_counts.items():
            print(f"  {feature}: {count} ({count/len(decisions)*100:.1f}%)")

        # Sample feature extraction
        if decisions:
            sample = decisions[0]
            print()
            print(f"Sample Decision (ID: {sample.id}):")
            print(f"  Dealer Features: {list(sample.dealer_features.keys())}")
            print(f"  Liquidity Features: {list(sample.liquidity_features.keys())}")
            print(f"  Sentiment Features: {list(sample.sentiment_features.keys())}")

    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("GNOSIS Trade Decision Query Examples")
    print("=" * 60)

    # Query recent decisions
    query_recent_decisions()

    # Query by symbol
    query_by_symbol("SPY")

    # Query for ML
    query_for_ml_dataset()
