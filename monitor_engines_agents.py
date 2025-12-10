#!/usr/bin/env python3
"""
COMPREHENSIVE ENGINE & AGENT MONITOR
Shows detailed processing for each ticker across all timeframes
"""

import os
import json
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()


class EngineAgentMonitor:
    """Monitor engine processing and agent thinking across all tickers and timeframes."""

    def __init__(self):
        self.ticker_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'hedge_engine': {},
            'liquidity_engine': {},
            'sentiment_engine': {},
            'elasticity_engine': {},
            'agent_decisions': {},
            'opportunity_score': None,
            'last_updated': None,
        })
        self.current_universe: List[str] = []
        self.last_universe_update: str = ""

    def parse_log_line(self, line: str) -> None:
        """Extract engine and agent data from log lines."""

        # Track universe changes
        if "Universe update" in line or "current universe" in line:
            if "symbols" in line.lower():
                self.last_universe_update = datetime.now().strftime("%H:%M:%S")

        # Track ticker being processed
        if "Testing:" in line or "Scanning" in line or "Evaluating" in line:
            # Extract ticker symbol
            parts = line.split()
            for part in parts:
                if part.isupper() and len(part) <= 5 and part.isalpha():
                    ticker = part
                    self.ticker_data[ticker]['last_updated'] = datetime.now().strftime("%H:%M:%S")

        # Parse engine outputs
        if "HedgeEngine" in line or "hedge" in line.lower():
            self._parse_hedge_data(line)
        elif "LiquidityEngine" in line or "liquidity" in line.lower():
            self._parse_liquidity_data(line)
        elif "SentimentEngine" in line or "sentiment" in line.lower():
            self._parse_sentiment_data(line)
        elif "ElasticityEngine" in line or "elasticity" in line.lower():
            self._parse_elasticity_data(line)

        # Parse greek exposure
        if "greek exposure" in line.lower() or "GEX" in line or "VEX" in line:
            self._parse_greek_exposure(line)

        # Parse dark pool
        if "dark pool" in line.lower():
            self._parse_dark_pool(line)

        # Parse opportunity scores
        if "opportunity" in line.lower() and "score" in line.lower():
            self._parse_opportunity_score(line)

        # Parse agent decisions
        if "agent" in line.lower() and ("signal" in line.lower() or "decision" in line.lower()):
            self._parse_agent_decision(line)

    def _parse_hedge_data(self, line: str) -> None:
        """Parse hedge engine output."""
        # Extract volatility, hedge ratio, etc.
        if "volatility" in line.lower() or "hedge" in line.lower():
            # Simple extraction - in production you'd parse JSON or structured logs
            for ticker in self.ticker_data.keys():
                if ticker in line:
                    self.ticker_data[ticker]['hedge_engine']['last_check'] = datetime.now().strftime("%H:%M:%S")
                    if "ratio" in line.lower():
                        self.ticker_data[ticker]['hedge_engine']['status'] = 'ACTIVE'

    def _parse_liquidity_data(self, line: str) -> None:
        """Parse liquidity engine output."""
        for ticker in self.ticker_data.keys():
            if ticker in line:
                self.ticker_data[ticker]['liquidity_engine']['last_check'] = datetime.now().strftime("%H:%M:%S")
                if "score" in line.lower():
                    self.ticker_data[ticker]['liquidity_engine']['status'] = 'ACTIVE'

    def _parse_sentiment_data(self, line: str) -> None:
        """Parse sentiment engine output."""
        for ticker in self.ticker_data.keys():
            if ticker in line:
                self.ticker_data[ticker]['sentiment_engine']['last_check'] = datetime.now().strftime("%H:%M:%S")
                if "bullish" in line.lower() or "bearish" in line.lower():
                    sentiment = "BULLISH" if "bullish" in line.lower() else "BEARISH"
                    self.ticker_data[ticker]['sentiment_engine']['direction'] = sentiment

    def _parse_elasticity_data(self, line: str) -> None:
        """Parse elasticity engine output."""
        for ticker in self.ticker_data.keys():
            if ticker in line:
                self.ticker_data[ticker]['elasticity_engine']['last_check'] = datetime.now().strftime("%H:%M:%S")

    def _parse_greek_exposure(self, line: str) -> None:
        """Parse greek exposure data."""
        for ticker in self.ticker_data.keys():
            if ticker in line:
                data = {}
                if "GEX" in line:
                    # Extract GEX value
                    parts = line.split("GEX")
                    if len(parts) > 1:
                        data['has_gex'] = True
                if "VEX" in line or "vanna" in line.lower():
                    data['has_vex'] = True
                if "charm" in line.lower():
                    data['has_charm'] = True

                if data:
                    self.ticker_data[ticker]['greek_exposure'] = data

    def _parse_dark_pool(self, line: str) -> None:
        """Parse dark pool data."""
        for ticker in self.ticker_data.keys():
            if ticker in line:
                self.ticker_data[ticker]['dark_pool'] = {
                    'detected': True,
                    'time': datetime.now().strftime("%H:%M:%S")
                }

    def _parse_opportunity_score(self, line: str) -> None:
        """Parse opportunity score."""
        for ticker in self.ticker_data.keys():
            if ticker in line:
                # Try to extract numeric score
                import re
                scores = re.findall(r'\d+\.\d+', line)
                if scores:
                    self.ticker_data[ticker]['opportunity_score'] = float(scores[0])

    def _parse_agent_decision(self, line: str) -> None:
        """Parse agent decisions."""
        for ticker in self.ticker_data.keys():
            if ticker in line:
                decision = {}
                if "buy" in line.lower():
                    decision['action'] = 'BUY'
                elif "sell" in line.lower():
                    decision['action'] = 'SELL'
                elif "hold" in line.lower():
                    decision['action'] = 'HOLD'

                if decision:
                    decision['time'] = datetime.now().strftime("%H:%M:%S")
                    self.ticker_data[ticker]['agent_decisions'] = decision

    def display_dashboard(self) -> None:
        """Display comprehensive dashboard."""
        os.system('clear')

        print("=" * 120)
        print("🔬 COMPREHENSIVE ENGINE & AGENT MONITOR".center(120))
        print("=" * 120)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    Universe Update: {self.last_universe_update}")
        print("=" * 120)
        print()

        if not self.ticker_data:
            print("⏳ Waiting for trading system activity...")
            print()
            print("💡 Start the trading system in another terminal:")
            print("   python3 start_dynamic_trading.py")
            return

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

    def monitor_live(self, log_file: str) -> None:
        """Monitor log file in real-time."""

        print(f"📡 Monitoring: {log_file}")
        print("Press Ctrl+C to stop")
        print()
        time.sleep(2)

        last_pos = 0

        try:
            while True:
                with open(log_file, 'r') as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()

                    for line in new_lines:
                        self.parse_log_line(line)

                # Update display every second
                self.display_dashboard()
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n✅ Monitor stopped")


def main():
    """Run the comprehensive monitor."""

    # Find latest log file
    log_dir = "logs"
    if not os.path.exists(log_dir):
        print(f"❌ Log directory '{log_dir}' not found.")
        print("Start the trading system first:")
        print("  python3 start_dynamic_trading.py")
        return

    log_files = [f for f in os.listdir(log_dir) if f.startswith("dynamic_trading")]
    if not log_files:
        print(f"❌ No log files found in '{log_dir}'.")
        print("Start the trading system first:")
        print("  python3 start_dynamic_trading.py")
        return

    latest_log = os.path.join(log_dir, sorted(log_files)[-1])

    monitor = EngineAgentMonitor()
    monitor.monitor_live(latest_log)


if __name__ == "__main__":
    main()
