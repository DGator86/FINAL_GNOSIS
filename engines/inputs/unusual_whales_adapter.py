"""Unusual Whales API adapter for options chain and flow data - FIXED Nov 2025."""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import httpx
import pandas as pd
from loguru import logger

from engines.inputs.options_chain_adapter import OptionContract


class UnusualWhalesAdapter:
    """
    Unusual Whales API adapter for options data.
    
    Fixed endpoints as of November 2025:
    - Base URL: https://api.unusualwhales.com
    - Auth: Bearer token in Authorization header
    - Token: UNUSUAL_WHALES_TOKEN or UNUSUAL_WHALES_API_KEY
    - Endpoint: /api/options/contracts/{symbol} for full chain + greeks
    """
    
    BASE_URL = "https://api.unusualwhales.com"
    
    def __init__(self):
        """Initialize Unusual Whales adapter with proper authentication."""
        # Support both env var names
        self.api_token = os.getenv("UNUSUAL_WHALES_TOKEN") or os.getenv("UNUSUAL_WHALES_API_KEY")
        
        if not self.api_token:
            logger.warning("⚠️  UNUSUAL_WHALES_TOKEN not set → using stub data fallback")
            self.client = None
            self.use_stub = True
        else:
            # Try Bearer token auth (for JWT tokens)
            self.headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_token}"
            }
            self.client = httpx.Client(headers=self.headers, timeout=30.0)
            self.use_stub = False
            logger.info("UnusualWhalesAdapter initialized with API token")
    
    def get_chain(self, symbol: str, timestamp: datetime) -> List[OptionContract]:
        """
        Get options chain for a symbol with greeks.
        
        Args:
            symbol: Underlying symbol
            timestamp: Data timestamp
            
        Returns:
            List of option contracts with full greeks
        """
        # Use stub if no token or already detected API unavailable
        if not self.client or not self.api_token or self.use_stub:
            return self._get_stub_chain(symbol, timestamp)
        
        try:
            # CORRECT ENDPOINT (Nov 2025): /api/options/contracts/{symbol}
            url = f"{self.BASE_URL}/api/options/contracts/{symbol}"
            
            response = self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for contracts in response
            contracts_data = data.get("contracts", [])
            if not contracts_data:
                logger.warning(f"No options chain data for {symbol} - using stub")
                return self._get_stub_chain(symbol, timestamp)
            
            contracts = []
            
            for option in contracts_data:
                try:
                    # Parse expiration date
                    exp_str = option.get("expiration_date", "")
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d") if exp_str else timestamp
                    
                    # Get option type
                    option_type = option.get("type", "").lower()
                    if option_type not in ["call", "put"]:
                        continue
                    
                    # Get greeks
                    greeks = option.get("greeks", {})
                    
                    # Create contract
                    contract = OptionContract(
                        symbol=option.get("symbol", ""),
                        strike=float(option.get("strike_price", 0)),
                        expiration=exp_date,
                        option_type=option_type,
                        bid=float(option.get("bid", 0)),
                        ask=float(option.get("ask", 0)),
                        last=float(option.get("last", 0)),
                        volume=float(option.get("volume", 0)),
                        open_interest=float(option.get("open_interest", 0)),
                        implied_volatility=float(option.get("implied_volatility", 0)),
                        delta=float(greeks.get("delta", 0)),
                        gamma=float(greeks.get("gamma", 0)),
                        theta=float(greeks.get("theta", 0)),
                        vega=float(greeks.get("vega", 0)),
                        rho=float(greeks.get("rho", 0)),
                    )
                    
                    contracts.append(contract)
                    
                except (ValueError, KeyError) as e:
                    logger.debug(f"Error parsing option contract: {e}")
                    continue
            
            if contracts:
                logger.info(f"✅ Retrieved {len(contracts)} real option contracts for {symbol}")
                return contracts
            else:
                logger.warning(f"No valid contracts parsed for {symbol} - using stub")
                return self._get_stub_chain(symbol, timestamp)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("❌ Invalid Unusual Whales token (401 Unauthorized) - switching to stub mode")
                self.use_stub = True
            elif e.response.status_code == 403:
                logger.error("❌ Unusual Whales API access forbidden (403) - subscription may not include API access - switching to stub mode")
                self.use_stub = True
            elif e.response.status_code == 404:
                logger.warning(f"⚠️  Unusual Whales API endpoint not found (404) - switching to stub mode permanently")
                self.use_stub = True
            else:
                logger.error(f"❌ Unusual Whales HTTP error {e.response.status_code}: {e}")
                self.use_stub = True
            return self._get_stub_chain(symbol, timestamp)
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error getting options chain for {symbol}: {e}")
            return self._get_stub_chain(symbol, timestamp)
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return self._get_stub_chain(symbol, timestamp)
    
    def _get_stub_chain(self, symbol: str, timestamp: datetime) -> List[OptionContract]:
        """
        High-quality stub options chain for fallback.
        
        Returns deterministic, realistic options chain so Hedge Engine v3
        still runs perfectly even without API access.
        """
        logger.info(f"📊 Using stub options chain for {symbol}")
        
        # Import stub adapter for fallback
        try:
            from engines.inputs.stub_adapters import StaticOptionsAdapter
            stub = StaticOptionsAdapter()
            return stub.get_chain(symbol, timestamp)
        except (ImportError, Exception) as e:
            # Emergency fallback if stub not available
            logger.warning(f"Stub adapter error for {symbol}: {e} - returning empty chain")
            return []
    
    def get_unusual_activity(self, symbol: Optional[str] = None) -> List[dict]:
        """
        Get unusual options activity.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of unusual activity records
        """
        if not self.client or not self.api_token:
            logger.debug("No API token - skipping unusual activity")
            return []
        
        try:
            # Try multiple possible endpoints
            endpoints = [
                f"{self.BASE_URL}/api/activity",
                f"{self.BASE_URL}/api/options/activity",
            ]
            
            params = {}
            if symbol:
                params["ticker"] = symbol
            
            for url in endpoints:
                try:
                    response = self.client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    activity = data.get("data", []) or data.get("activity", [])
                    if activity:
                        logger.info(f"Retrieved {len(activity)} unusual activity records")
                        return activity
                except httpx.HTTPStatusError:
                    continue
            
            logger.debug("No unusual activity endpoint working")
            return []
            
        except Exception as e:
            logger.error(f"Error getting unusual activity: {e}")
            return []
    
    def get_flow_summary(self, symbol: str) -> dict:
        """
        Get options flow summary for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Flow summary dict
        """
        if not self.client or not self.api_token:
            return {}
        
        try:
            endpoints = [
                f"{self.BASE_URL}/api/flow/{symbol}",
                f"{self.BASE_URL}/api/options/flow/{symbol}",
            ]
            
            for url in endpoints:
                try:
                    response = self.client.get(url)
                    response.raise_for_status()
                    data = response.json()
                    return data.get("data", {})
                except httpx.HTTPStatusError:
                    continue
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting flow summary for {symbol}: {e}")
            return {}
    
    def get_implied_volatility(self, symbol: str) -> Optional[float]:
        """
        Get current implied volatility for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            IV as a decimal (e.g., 0.25 for 25%)
        """
        if not self.client or not self.api_token:
            return None
        
        try:
            endpoints = [
                f"{self.BASE_URL}/api/stock/{symbol}/iv",
                f"{self.BASE_URL}/api/options/{symbol}/iv",
            ]
            
            for url in endpoints:
                try:
                    response = self.client.get(url)
                    response.raise_for_status()
                    data = response.json()
                    iv = data.get("data", {}).get("iv")
                    if iv is not None:
                        return float(iv)
                except httpx.HTTPStatusError:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting IV for {symbol}: {e}")
            return None
    
    def close(self):
        """Close the HTTP client."""
        if self.client:
            self.client.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
