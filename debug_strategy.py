from datetime import datetime, timedelta
from trade.trade_agent_v3 import TradeStrategy

try:
    s = TradeStrategy(
        symbol="AAPL",
        direction="LONG",
        entry_price=100.0,
        quantity=1,
        stop_loss_price=90.0,
        take_profit_price=120.0,
        trailing_stop_config={"enabled": False},
        timeframe="1Hour",
        max_hold_time=timedelta(days=1),
        min_hold_time=timedelta(minutes=30),
        risk_amount=10.0,
        reward_amount=20.0,
        risk_reward_ratio=2.0,
        position_size_pct=0.05,
        confidence=0.8,
        reasoning="Test",
        timestamp=datetime.now(),
        asset_class="option",
        option_symbol="AAPL...",
        strike=100.0,
        expiration=datetime.now(),
        option_type="call",
    )
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
