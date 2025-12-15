"""Order simulator v2 with market impact and async throttling."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from schemas.core_schemas import OrderResult, OrderStatus

try:  # pragma: no cover
    from ib_insync import IB
except Exception:  # pragma: no cover
    IB = None


@dataclass
class SimulatedOrder:
    symbol: str
    size: float
    price: float
    adv: float
    vol: float
    side: str
    order_type: str = "limit"


class OrderSimulatorV2:
    def __init__(self, risk_free: float = 0.02):
        self.risk_free = risk_free
        self.ib: Optional[Any] = None

    def _impact_price(self, order: SimulatedOrder) -> float:
        impact = np.sign(order.size) * (abs(order.size) / (order.adv + 1e-8)) ** 0.5 * order.vol
        return float(order.price + impact)

    def execute(self, order: SimulatedOrder) -> OrderResult:
        fill_price = self._impact_price(order)
        return OrderResult(
            timestamp=None,
            symbol=order.symbol,
            status=OrderStatus.FILLED,
            filled_qty=order.size,
            filled_price=fill_price,
            message="simulated",
        )

    async def execute_batch(self, orders: List[SimulatedOrder]) -> List[OrderResult]:
        results: List[OrderResult] = []
        for order in orders:
            retries = 0
            while retries < 3:
                try:
                    await asyncio.sleep(1)
                    results.append(self.execute(order))
                    break
                except Exception as exc:  # pragma: no cover
                    retries += 1
                    logger.warning(f"Retry {retries} for order {order.symbol}: {exc}")
        return results

    def connect_ib(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1) -> None:
        if IB is None:
            logger.info("ib_insync not available; skipping IB connection")
            return
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id)

    def place_ib_order(self, contract: Any, order: Any) -> Optional[Any]:
        if not self.ib:
            logger.warning("IB connection not established")
            return None
        return self.ib.placeOrder(contract, order)


# Test: ensure execute returns OrderResult
