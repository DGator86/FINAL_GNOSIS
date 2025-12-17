"""Greeks Calculator - Fetch and calculate option Greeks."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from scipy import stats


class GreeksCalculator:
    """
    Calculate or fetch option Greeks.

    Greeks measure sensitivity of option prices to various factors:
    - Delta: Price sensitivity (0-1 for calls, -1-0 for puts)
    - Gamma: Delta sensitivity (rate of change of delta)
    - Theta: Time decay (daily P&L from time passage)
    - Vega: IV sensitivity (P&L from 1% IV change)
    """

    def __init__(self, options_adapter=None):
        """Initialize with optional options adapter for API fetching."""
        self.options_adapter = options_adapter
        logger.info("GreeksCalculator initialized")

    def fetch_greeks_from_alpaca(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Fetch Greeks from Alpaca API for a specific option symbol.

        Args:
            symbol: OCC option symbol (e.g., AAPL230616C00150000)

        Returns:
            Dictionary with delta, gamma, theta, vega or None if unavailable
        """
        if not self.options_adapter:
            logger.warning("No options adapter available for Greeks fetching")
            return None

        try:
            # Alpaca options API returns Greeks in the snapshot/quote
            # Get latest option quote which includes Greeks
            quote = self.options_adapter.get_latest_quote(symbol)

            if quote and hasattr(quote, "greeks"):
                greeks = quote.greeks
                return {
                    "delta": greeks.delta,
                    "gamma": greeks.gamma,
                    "theta": greeks.theta,
                    "vega": greeks.vega,
                    "rho": getattr(greeks, "rho", 0.0),
                }

            logger.warning(f"No Greeks available from API for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error fetching Greeks for {symbol}: {e}")
            return None

    def calculate_position_greeks(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate aggregate Greeks for a list of option positions.

        Args:
            positions: List of position dicts with 'symbol', 'quantity', 'greeks'

        Returns:
            Dictionary with net_delta, net_gamma, net_theta, net_vega
        """
        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0
        net_vega = 0.0

        for pos in positions:
            quantity = pos.get("quantity", 0)
            greeks = pos.get("greeks", {})

            if greeks:
                # Multiply by quantity and contract multiplier (100)
                net_delta += greeks.get("delta", 0.0) * quantity * 100
                net_gamma += greeks.get("gamma", 0.0) * quantity * 100
                net_theta += greeks.get("theta", 0.0) * quantity * 100
                net_vega += greeks.get("vega", 0.0) * quantity * 100

        return {
            "net_delta": round(net_delta, 2),
            "net_gamma": round(net_gamma, 4),
            "net_theta": round(net_theta, 2),
            "net_vega": round(net_vega, 2),
        }

    def calculate_portfolio_greeks(
        self, equity_positions: List[Dict[str, Any]], options_positions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate total portfolio Greeks including equity and options.

        Args:
            equity_positions: List of equity position dicts
            options_positions: List of options position dicts

        Returns:
            Dictionary with total portfolio Greeks
        """
        # Equity positions contribute to delta only (1 delta per share)
        equity_delta = sum(pos.get("quantity", 0) for pos in equity_positions)

        # Options Greeks
        options_greeks = self.calculate_position_greeks(options_positions)

        return {
            "net_delta": round(equity_delta + options_greeks["net_delta"], 2),
            "net_gamma": options_greeks["net_gamma"],
            "net_theta": options_greeks["net_theta"],
            "net_vega": options_greeks["net_vega"],
            "timestamp": datetime.now().isoformat(),
        }

    def calculate_black_scholes_greeks(
        self,
        option_type: str,
        spot_price: float,
        strike: float,
        time_to_expiration: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate Greeks using Black-Scholes-Merton model.

        Args:
            option_type: 'call' or 'put'
            spot_price: Current underlying price
            strike: Strike price
            time_to_expiration: Time to expiration in years
            risk_free_rate: Risk-free interest rate (annual)
            volatility: Implied volatility (annual, e.g., 0.20 for 20%)
            dividend_yield: Continuous dividend yield (annual)

        Returns:
            Dictionary with delta, gamma, theta, vega, rho
        """
        try:
            # Prevent division by zero
            if time_to_expiration <= 0:
                # At expiration, options have intrinsic value only
                # Intrinsic calculation: max(0, spot - strike) or max(0, strike - spot)
                # But Greeks are derivatives, so at expiration delta is 0 or 1.
                return {
                    "delta": 1.0 if (option_type == "call" and spot_price > strike) else 0.0,
                    "gamma": 0.0,
                    "theta": 0.0,
                    "vega": 0.0,
                    "rho": 0.0,
                }

            # Calculate d1 and d2
            # Calculate d1 and d2
            numerator = (
                math.log(spot_price / strike)
                + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiration
            )
            d1 = numerator / (volatility * math.sqrt(time_to_expiration))
            d2 = d1 - volatility * math.sqrt(time_to_expiration)

            # Standard normal CDF and PDF
            nd1 = stats.norm.cdf(d1)
            nd2 = stats.norm.cdf(d2)
            npd1 = stats.norm.pdf(d1)

            # Delta
            if option_type == "call":
                delta = math.exp(-dividend_yield * time_to_expiration) * nd1
            else:
                delta = -math.exp(-dividend_yield * time_to_expiration) * (1 - nd1)

            # Gamma (same for calls and puts)
            # Gamma (same for calls and puts)
            gamma = (math.exp(-dividend_yield * time_to_expiration) * npd1) / (
                spot_price * volatility * math.sqrt(time_to_expiration)
            )

            # Theta (daily decay, divide by 365)
            if option_type == "call":
                theta = (
                    -spot_price
                    * npd1
                    * volatility
                    * math.exp(-dividend_yield * time_to_expiration)
                    / (2 * math.sqrt(time_to_expiration))
                    - risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiration) * nd2
                    + dividend_yield
                    * spot_price
                    * math.exp(-dividend_yield * time_to_expiration)
                    * nd1
                ) / 365.0
            else:
                theta = (
                    -spot_price
                    * npd1
                    * volatility
                    * math.exp(-dividend_yield * time_to_expiration)
                    / (2 * math.sqrt(time_to_expiration))
                    + risk_free_rate
                    * strike
                    * math.exp(-risk_free_rate * time_to_expiration)
                    * (1 - nd2)
                    - dividend_yield
                    * spot_price
                    * math.exp(-dividend_yield * time_to_expiration)
                    * (1 - nd1)
                ) / 365.0

            # Vega (per 1% change in volatility, divide by 100)
            # Vega (per 1% change in volatility, divide by 100)
            vega = (
                spot_price
                * math.exp(-dividend_yield * time_to_expiration)
                * npd1
                * math.sqrt(time_to_expiration)
            ) / 100.0

            # Rho (per 1% change in interest rate, divide by 100)
            if option_type == "call":
                rho = (
                    strike
                    * time_to_expiration
                    * math.exp(-risk_free_rate * time_to_expiration)
                    * nd2
                ) / 100.0
            else:
                rho = (
                    -strike
                    * time_to_expiration
                    * math.exp(-risk_free_rate * time_to_expiration)
                    * (1 - nd2)
                ) / 100.0

            return {
                "delta": round(delta, 4),
                "gamma": round(gamma, 6),
                "theta": round(theta, 4),
                "vega": round(vega, 4),
                "rho": round(rho, 4),
            }

        except Exception as e:
            logger.error(f"Error calculating Black-Scholes Greeks: {e}")
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "rho": 0.0,
            }

    def estimate_greeks_simple(
        self,
        option_type: str,
        strike: float,
        current_price: float,
        days_to_expiration: int,
    ) -> Dict[str, float]:
        """
        Simple estimation of Greeks without full Black-Scholes.

        DEPRECATED: Use calculate_black_scholes_greeks() instead for accurate Greeks.
        This method is kept for backwards compatibility only.

        Args:
            option_type: 'call' or 'put'
            strike: Strike price
            current_price: Current underlying price
            days_to_expiration: Days until expiration

        Returns:
            Dictionary with estimated Greeks
        """
        logger.warning(
            "estimate_greeks_simple() is DEPRECATED. "
            "Use calculate_black_scholes_greeks() for accurate Greeks."
        )

        # Use Black-Scholes with assumed volatility and risk-free rate
        assumed_vol = 0.25  # 25% annual volatility
        assumed_rate = 0.05  # 5% risk-free rate
        time_to_exp = days_to_expiration / 365.0

        return self.calculate_black_scholes_greeks(
            option_type=option_type,
            spot_price=current_price,
            strike=strike,
            time_to_expiration=time_to_exp,
            risk_free_rate=assumed_rate,
            volatility=assumed_vol,
        )

    # =========================================================================
    # Strategy Analysis Methods (PoP, Breakevens, P&L)
    # =========================================================================

    def calculate_option_payoff(
        self,
        option_type: str,
        strike: float,
        premium: float,
        quantity: int,
        stock_price: float,
    ) -> float:
        """
        Calculate option P&L at expiration for a given stock price.

        Args:
            option_type: 'call' or 'put'
            strike: Strike price
            premium: Premium paid (positive) or received (negative)
            quantity: Number of contracts (positive=long, negative=short)
            stock_price: Stock price at expiration

        Returns:
            P&L per contract in dollars (multiply by 100 for actual value)
        """
        if option_type == "call":
            intrinsic = max(0, stock_price - strike)
        else:
            intrinsic = max(0, strike - stock_price)

        # Long position: pay premium, receive intrinsic
        # Short position: receive premium, pay intrinsic
        if quantity > 0:
            return (intrinsic - premium) * abs(quantity)
        else:
            return (premium - intrinsic) * abs(quantity)

    def calculate_strategy_payoff(
        self,
        legs: List[Dict[str, Any]],
        stock_price: float,
    ) -> float:
        """
        Calculate total strategy P&L at expiration.

        Args:
            legs: List of option legs, each with:
                - type: 'call' or 'put'
                - strike: Strike price
                - premium: Premium per share
                - quantity: Contracts (positive=long, negative=short)
            stock_price: Stock price at expiration

        Returns:
            Total P&L in dollars per share (multiply by 100 for contract value)
        """
        total_pnl = 0.0
        for leg in legs:
            total_pnl += self.calculate_option_payoff(
                option_type=leg["type"],
                strike=leg["strike"],
                premium=leg["premium"],
                quantity=leg["quantity"],
                stock_price=stock_price,
            )
        return total_pnl

    def calculate_probability_of_profit(
        self,
        legs: List[Dict[str, Any]],
        spot_price: float,
        volatility: float,
        days_to_expiration: int,
        risk_free_rate: float = 0.05,
        simulations: int = 10000,
    ) -> Dict[str, float]:
        """
        Calculate probability of profit using Monte Carlo simulation.

        Args:
            legs: List of option legs (see calculate_strategy_payoff)
            spot_price: Current stock price
            volatility: Implied volatility (annual, e.g., 0.25 for 25%)
            days_to_expiration: Days until expiration
            risk_free_rate: Risk-free rate (annual)
            simulations: Number of Monte Carlo paths

        Returns:
            Dictionary with probability metrics
        """
        if days_to_expiration <= 0:
            # At expiration, just calculate current P&L
            pnl = self.calculate_strategy_payoff(legs, spot_price)
            return {
                "probability_of_profit": 1.0 if pnl > 0 else 0.0,
                "expected_profit": max(0, pnl) * 100,
                "expected_loss": min(0, pnl) * 100,
                "average_pnl": pnl * 100,
                "max_profit": pnl * 100,
                "max_loss": pnl * 100,
            }

        # GBM simulation for terminal prices
        dt = days_to_expiration / 252.0
        drift = (risk_free_rate - 0.5 * volatility**2) * dt
        shock = volatility * np.sqrt(dt) * np.random.randn(simulations)
        terminal_prices = spot_price * np.exp(drift + shock)

        # Calculate P&L for each simulation
        pnls = np.array(
            [self.calculate_strategy_payoff(legs, price) * 100 for price in terminal_prices]
        )

        profitable = pnls > 0
        winners = pnls[profitable]
        losers = pnls[~profitable]

        return {
            "probability_of_profit": float(np.mean(profitable)),
            "expected_profit": float(np.mean(winners)) if len(winners) > 0 else 0.0,
            "expected_loss": float(np.mean(losers)) if len(losers) > 0 else 0.0,
            "average_pnl": float(np.mean(pnls)),
            "max_profit": float(np.max(pnls)),
            "max_loss": float(np.min(pnls)),
            "pnl_std_dev": float(np.std(pnls)),
        }

    def find_breakevens(
        self,
        legs: List[Dict[str, Any]],
        spot_price: float,
        price_range_pct: float = 0.50,
        precision: float = 0.01,
    ) -> List[float]:
        """
        Find breakeven points for a strategy.

        Args:
            legs: List of option legs
            spot_price: Current stock price
            price_range_pct: Range to search (e.g., 0.50 = ±50%)
            precision: Price increment for search

        Returns:
            List of breakeven prices (sorted)
        """
        low = spot_price * (1 - price_range_pct)
        high = spot_price * (1 + price_range_pct)
        prices = np.arange(low, high, precision)

        breakevens = []
        prev_pnl = None

        for price in prices:
            pnl = float(self.calculate_strategy_payoff(legs, price))

            if prev_pnl is not None:
                # Check for sign change (crossing zero)
                if (prev_pnl < 0 and pnl >= 0) or (prev_pnl >= 0 and pnl < 0):
                    breakevens.append(round(price, 2))

            prev_pnl = pnl

        return breakevens

    def analyze_profit_zones(
        self,
        legs: List[Dict[str, Any]],
        spot_price: float,
        price_range_pct: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Analyze profit and loss zones for a strategy.

        Args:
            legs: List of option legs
            spot_price: Current stock price
            price_range_pct: Range to analyze (e.g., 0.50 = ±50%)

        Returns:
            Dictionary with zone analysis
        """
        low = spot_price * (1 - price_range_pct)
        high = spot_price * (1 + price_range_pct)
        prices = np.linspace(low, high, 500)

        pnls = np.array(
            [float(self.calculate_strategy_payoff(legs, price)) * 100 for price in prices]
        )

        # Find profit zones (contiguous regions where P&L > 0)
        profit_zones = []
        in_profit = False
        zone_start = None

        for i, (price, pnl) in enumerate(zip(prices, pnls)):
            if pnl > 0 and not in_profit:
                zone_start = price
                in_profit = True
            elif pnl <= 0 and in_profit:
                profit_zones.append((round(zone_start, 2), round(prices[i - 1], 2)))
                in_profit = False

        if in_profit:
            profit_zones.append((round(zone_start, 2), round(high, 2)))

        # Find max profit and loss points
        max_profit_idx = np.argmax(pnls)
        max_loss_idx = np.argmin(pnls)

        return {
            "profit_zones": profit_zones,
            "max_profit": round(float(pnls[max_profit_idx]), 2),
            "max_profit_at": round(float(prices[max_profit_idx]), 2),
            "max_loss": round(float(pnls[max_loss_idx]), 2),
            "max_loss_at": round(float(prices[max_loss_idx]), 2),
            "breakevens": self.find_breakevens(legs, spot_price, price_range_pct),
            "pnl_at_current": round(
                float(self.calculate_strategy_payoff(legs, spot_price) * 100), 2
            ),
        }
