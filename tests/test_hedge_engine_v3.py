from datetime import datetime

from engines.hedge.hedge_engine_v3 import HedgeEngineV3


class DummyContract:
    def __init__(self, option_type: str, open_interest: float, gamma: float, delta: float):
        self.option_type = option_type
        self.open_interest = open_interest
        self.gamma = gamma
        self.delta = delta
        self.vega = 0.1
        self.theta = -0.01


def test_directional_elasticity_uses_oi_weights():
    contracts = [
        DummyContract("call", 100, 0.2, 0.5),
        DummyContract("call", 50, 0.1, 0.4),
        DummyContract("put", 70, -0.15, -0.5),
    ]
    engine = HedgeEngineV3(options_adapter=None, config={"ledger_flows": []})
    up = engine._directional_elasticity(contracts, option_type="call")
    down = engine._directional_elasticity(contracts, option_type="put")
    assert up > 0
    assert down > 0


def test_flow_history_limited_to_30():
    flows = [{"flow": i, "price": i * 0.1} for i in range(40)]
    engine = HedgeEngineV3(options_adapter=None, config={"ledger_flows": flows})
    history = engine._load_flow_history()
    assert len(history) == 30
