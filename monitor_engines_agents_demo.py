#!/usr/bin/env python3
"""
DEMO: Engine & Agent Monitor - Shows example output
"""

import os
import time
from datetime import datetime
from collections import defaultdict


class DemoEngineMonitor:
    """Demo monitor showing what real engine/agent output looks like."""

    def __init__(self):
        # Sample ticker data
        self.ticker_data = {
            'SPY': {
                'hedge_engine': {'status': 'ACTIVE', 'last_check': '14:23:45'},
                'liquidity_engine': {'status': 'ACTIVE', 'last_check': '14:23:46'},
                'sentiment_engine': {'direction': 'BULLISH', 'last_check': '14:23:47'},
                'elasticity_engine': {'last_check': '14:23:48'},
                'greek_exposure': {'has_gex': True, 'has_vex': True, 'has_charm': True},
                'dark_pool': {'detected': True, 'time': '14:24:00'},
                'agent_decisions': {'action': 'BUY', 'time': '14:25:00'},
                'opportunity_score': 0.87,
                'last_updated': '14:25:00'
            },
            'QQQ': {
                'hedge_engine': {'status': 'ACTIVE', 'last_check': '14:23:50'},
                'liquidity_engine': {'status': 'ACTIVE', 'last_check': '14:23:51'},
                'sentiment_engine': {'direction': 'BEARISH', 'last_check': '14:23:52'},
                'elasticity_engine': {'last_check': '14:23:53'},
                'greek_exposure': {'has_gex': True, 'has_vex': False, 'has_charm': True},
                'agent_decisions': {'action': 'SELL', 'time': '14:25:05'},
                'opportunity_score': 0.72,
                'last_updated': '14:25:05'
            },
            'NVDA': {
                'hedge_engine': {'status': 'ACTIVE', 'last_check': '14:24:10'},
                'liquidity_engine': {'status': 'ACTIVE', 'last_check': '14:24:11'},
                'sentiment_engine': {'direction': 'BULLISH', 'last_check': '14:24:12'},
                'elasticity_engine': {'last_check': '14:24:13'},
                'greek_exposure': {'has_gex': True, 'has_vex': True, 'has_charm': False},
                'dark_pool': {'detected': True, 'time': '14:24:20'},
                'agent_decisions': {'action': 'BUY', 'time': '14:25:10'},
                'opportunity_score': 0.94,
                'last_updated': '14:25:10'
            },
            'TSLA': {
                'hedge_engine': {'status': 'ACTIVE', 'last_check': '14:24:15'},
                'liquidity_engine': {'status': 'ACTIVE', 'last_check': '14:24:16'},
                'sentiment_engine': {'direction': 'NEUTRAL', 'last_check': '14:24:17'},
                'elasticity_engine': {'last_check': '14:24:18'},
                'agent_decisions': {'action': 'HOLD', 'time': '14:25:15'},
                'opportunity_score': 0.45,
                'last_updated': '14:25:15'
            },
        }
        self.last_universe_update = "14:23:00"

    def display_dashboard(self):
        """Display the comprehensive dashboard."""
        os.system('clear')

        print("=" * 120)
        print("🔬 COMPREHENSIVE ENGINE & AGENT MONITOR [DEMO MODE]".center(120))
        print("=" * 120)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    Universe Update: {self.last_universe_update}")
        print("=" * 120)
        print()

        # Display each ticker
        for ticker, data in sorted(self.ticker_data.items()):
            last_update = data.get('last_updated', 'N/A')
            opp_score = data.get('opportunity_score')

            print(f"\n┌{'─' * 118}┐")
            print(f"│ 📊 {ticker:6} │ Last Update: {last_update:8} │ Opportunity Score: {opp_score if opp_score else 'N/A':6} │")
            print(f"├{'─' * 118}┤")

            # Engine Status
            print(f"│ {'ENGINES':^118} │")
            print(f"├{'─' * 118}┤")

            # Hedge Engine
            hedge = data.get('hedge_engine', {})
            hedge_status = hedge.get('status', 'IDLE')
            hedge_time = hedge.get('last_check', 'N/A')
            print(f"│   🛡️  Hedge Engine     : {hedge_status:10} │ Last: {hedge_time:8}                                           │")

            # Liquidity Engine
            liquidity = data.get('liquidity_engine', {})
            liq_status = liquidity.get('status', 'IDLE')
            liq_time = liquidity.get('last_check', 'N/A')
            print(f"│   💧 Liquidity Engine : {liq_status:10} │ Last: {liq_time:8}                                           │")

            # Sentiment Engine
            sentiment = data.get('sentiment_engine', {})
            sent_direction = sentiment.get('direction', 'NEUTRAL')
            sent_time = sentiment.get('last_check', 'N/A')
            print(f"│   💭 Sentiment Engine : {sent_direction:10} │ Last: {sent_time:8}                                           │")

            # Elasticity Engine
            elasticity = data.get('elasticity_engine', {})
            elast_time = elasticity.get('last_check', 'N/A')
            print(f"│   ⚡ Elasticity Engine: ACTIVE     │ Last: {elast_time:8}                                           │")

            # Greek Exposure
            greek = data.get('greek_exposure', {})
            if greek:
                indicators = []
                if greek.get('has_gex'): indicators.append('GEX')
                if greek.get('has_vex'): indicators.append('VEX')
                if greek.get('has_charm'): indicators.append('Charm')
                greek_str = ', '.join(indicators) if indicators else 'N/A'
                print(f"│   📊 Greek Exposure   : {greek_str:85} │")

            # Dark Pool
            dark_pool = data.get('dark_pool', {})
            if dark_pool.get('detected'):
                dp_time = dark_pool.get('time', 'N/A')
                print(f"│   🌑 Dark Pool        : DETECTED   │ Time: {dp_time:8}                                           │")

            # Agent Decisions
            decision = data.get('agent_decisions', {})
            if decision:
                action = decision.get('action', 'N/A')
                dec_time = decision.get('time', 'N/A')
                print(f"├{'─' * 118}┤")
                print(f"│ {'AGENT DECISION':^118} │")
                print(f"├{'─' * 118}┤")
                action_emoji = '🟢' if action == 'BUY' else '🔴' if action == 'SELL' else '🟡'
                print(f"│   {action_emoji} Action: {action:10} │ Time: {dec_time:8}                                                      │")

            print(f"└{'─' * 118}┘")

        print()
        print("=" * 120)
        print("Legend: 🛡️ Hedge  💧 Liquidity  💭 Sentiment  ⚡ Elasticity  📊 Greeks  🌑 Dark Pool  🟢 Buy  🔴 Sell  🟡 Hold")
        print("=" * 120)
        print()
        print("💡 This is DEMO mode showing sample data. To see real data:")
        print("   1. Configure Alpaca Paper Trading credentials in .env")
        print("   2. Run: python3 start_dynamic_trading.py")
        print("   3. Run: python3 monitor_engines_agents.py (in another terminal)")
        print()

    def run_demo(self):
        """Run the demo monitor."""
        print("🎬 Starting Engine & Agent Monitor Demo...")
        print("Press Ctrl+C to stop")
        print()
        time.sleep(2)

        try:
            while True:
                self.display_dashboard()
                time.sleep(3)  # Update every 3 seconds
        except KeyboardInterrupt:
            print("\n\n✅ Demo stopped")


if __name__ == "__main__":
    demo = DemoEngineMonitor()
    demo.run_demo()
