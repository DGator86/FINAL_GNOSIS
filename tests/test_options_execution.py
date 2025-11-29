import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime
from gnosis.unified_trading_bot import UnifiedTradingBot
from schemas.core_schemas import OptionsOrderRequest, OptionsLeg


class TestOptionsExecution(unittest.IsolatedAsyncioTestCase):
    async def test_options_sizing(self):
        # Config
        config = {"risk": {"risk_per_trade_pct": 0.02, "max_positions": 5}}

        # Initialize Bot
        bot = UnifiedTradingBot(config, enable_trading=True, paper_mode=True)

        # Mock Adapters
        bot.adapter = MagicMock()
        bot.options_adapter = AsyncMock()

        # Mock Account Info
        mock_account = MagicMock()
        mock_account.equity = 100000.0
        mock_account.buying_power = 200000.0
        bot.adapter.get_account.return_value = mock_account

        # Create Options Strategy
        leg1 = OptionsLeg(
            symbol="SPY230616C00400000",
            ratio=1,
            side="buy",
            type="call",
            strike=400.0,
            expiration="2023-06-16",
            action="buy_to_open",
        )

        strategy = OptionsOrderRequest(
            symbol="SPY",
            strategy_name="Long Call",
            legs=[leg1],
            max_loss=500.0,
            max_profit=1000.0,
            bpr=500.0,
            rationale="Test",
            confidence=0.8,
        )

        # Execute
        await bot.open_position("SPY", strategy, current_price=400.0)

        # Verify
        # Risk = 100,000 * 0.02 = 2,000
        # Cost = 500
        # Quantity = 2000 / 500 = 4

        expected_qty = 4

        # Check if place_multileg_order was called
        bot.options_adapter.place_multileg_order.assert_called_once()

        # Check arguments
        call_args = bot.options_adapter.place_multileg_order.call_args
        legs_payload = call_args.kwargs["legs"]

        self.assertEqual(len(legs_payload), 1)
        self.assertEqual(legs_payload[0]["qty"], expected_qty)
        self.assertEqual(legs_payload[0]["symbol"], "SPY230616C00400000")

        print(
            f"SUCCESS: Calculated quantity {legs_payload[0]['qty']} matches expected {expected_qty}"
        )

        # Verify Position Tracking
        self.assertIn("SPY", bot.positions)
        pos = bot.positions["SPY"]
        self.assertEqual(pos.quantity, expected_qty)
        self.assertEqual(pos.asset_class, "option_strategy")


if __name__ == "__main__":
    unittest.main()
