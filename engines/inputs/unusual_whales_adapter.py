"""Unusual Whales API adapter for options chain and flow data."""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import httpx
from loguru import logger

from engines.inputs.options_chain_adapter import OptionContract


class UnusualWhalesAdapter:
    """Unusual Whales API adapter for options data."""
    
    BASE_URL = "https://api.unusualwhales.com"
    
    def __init__(self):
        """Initialize Unusual Whales adapter."""
        self.api_key = os.getenv("UNUSUAL_WHALES_API_KEY")
        
        if not self.api_key:
            raise ValueError("Unusual Whales API key not found. Set UNUSUAL_WHALES_API_KEY.")
        
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        self.client = httpx.Client(headers=self.headers, timeout=30.0)
        
        logger.info("UnusualWhalesAdapter initialized")
    
    def get_chain(self, symbol: str, timestamp: datetime) -> List[OptionContract]:
        """
        Get options chain for a symbol.
        
        Args:
            symbol: Underlying symbol
            timestamp: Data timestamp
            
        Returns:
            List of option contracts
        """
        try:
            # Get option chain from Unusual Whales
            # API v2 endpoint
            url = f"{self.BASE_URL}/v2/options/chain/{symbol}"
            
            response = self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or "data" not in data:
                logger.warning(f"No options chain data for {symbol}")
                return []
            
            contracts = []
            
            for option in data.get("data", []):
                try:
                    # Parse expiration date
                    exp_date = datetime.strptime(option.get("expiration_date", ""), "%Y-%m-%d")
                    
                    # Determine option type
                    option_type = option.get("option_type", "").lower()
                    if option_type not in ["call", "put"]:
                        continue
                    
                    # Create contract
                    contract = OptionContract(
                        symbol=option.get("option_symbol", ""),
                        strike=float(option.get("strike", 0)),
                        expiration=exp_date,
                        option_type=option_type,
                        bid=float(option.get("bid", 0)),
                        ask=float(option.get("ask", 0)),
                        last=float(option.get("last", 0)),
                        volume=float(option.get("volume", 0)),
                        open_interest=float(option.get("open_interest", 0)),
                        implied_volatility=float(option.get("implied_volatility", 0)),
                        delta=float(option.get("delta", 0)),
                        gamma=float(option.get("gamma", 0)),
                        theta=float(option.get("theta", 0)),
                        vega=float(option.get("vega", 0)),
                        rho=float(option.get("rho", 0)),
                    )
                    
                    contracts.append(contract)
                    
                except (ValueError, KeyError) as e:
                    logger.debug(f"Error parsing option contract: {e}")
                    continue
            
            logger.info(f"Retrieved {len(contracts)} option contracts for {symbol}")
            return contracts
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error getting options chain for {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return []
    
    def get_unusual_activity(self, symbol: Optional[str] = None) -> List[dict]:
        """
        Get unusual options activity.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of unusual activity records
        """
        try:
            url = f"{self.BASE_URL}/v2/activity"
            params = {}
            if symbol:
                params["ticker"] = symbol
            
            response = self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            activity = data.get("data", [])
            logger.info(f"Retrieved {len(activity)} unusual activity records")
            
            return activity
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error getting unusual activity: {e}")
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
        try:
            url = f"{self.BASE_URL}/v2/flow/{symbol}"
            
            response = self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            return data.get("data", {})
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error getting flow summary for {symbol}: {e}")
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
        try:
            url = f"{self.BASE_URL}/v2/stock/{symbol}/iv"
            
            response = self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            iv = data.get("data", {}).get("iv")
            if iv is not None:
                return float(iv)
            
            return None
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error getting IV for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting IV for {symbol}: {e}")
            return None
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
