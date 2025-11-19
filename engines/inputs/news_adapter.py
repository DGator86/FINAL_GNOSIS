"""News data adapter protocol."""

from __future__ import annotations

from datetime import datetime
from typing import List, Protocol

from pydantic import BaseModel


class NewsArticle(BaseModel):
    """News article."""
    
    timestamp: datetime
    headline: str
    summary: str = ""
    source: str = ""
    url: str = ""
    sentiment: float = 0.0  # -1 to 1


class NewsAdapter(Protocol):
    """Protocol for news data providers."""
    
    def get_news(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime
    ) -> List[NewsArticle]:
        """
        Get news articles for a symbol.
        
        Args:
            symbol: Trading symbol
            start: Start timestamp
            end: End timestamp
            
        Returns:
            List of news articles
        """
        ...
