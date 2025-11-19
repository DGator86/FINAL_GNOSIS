"""Base agent protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from schemas.core_schemas import AgentSuggestion, PipelineResult


class Agent(Protocol):
    """Protocol for all agents."""
    
    def suggest(self, pipeline_result: PipelineResult, timestamp: datetime) -> AgentSuggestion:
        """
        Generate a suggestion based on pipeline results.
        
        Args:
            pipeline_result: Complete pipeline result
            timestamp: Suggestion timestamp
            
        Returns:
            Agent suggestion
        """
        ...
