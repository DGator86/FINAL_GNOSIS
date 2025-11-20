# Adaptive Watchlist Verification (Nov 20, 2025 claim)

## Summary of claim
The adaptive watchlist is described as a dynamic, self-optimizing list that re-ranks ~400 symbols every 5–15 minutes using Hedge Engine v3 signals and multiple adaptive filters. Only symbols on the watchlist are supposed to be traded, and the dashboard is expected to show a live “Adaptive Watchlist” tab.

## What is in the current codebase
- `engines/orchestration/pipeline_runner.py` orchestrates engine runs and passes results to agents, but it does not perform any watchlist ranking or filtering. There is no hook for a dynamic universe change per cycle.
- `trade/trade_agent_v1.py` generates trade ideas directly from consensus outputs without checking any watchlist membership or adaptive filters.
- There is no `watchlist` package or module in the repository, nor any scoring logic for the described composite Opportunity Score.
- Launch scripts such as `start_full_trading_system.py` attempt to load a static `config/watchlist.yaml` file that is not present in the repository, which would fail at startup.

## Conclusion
The repository currently lacks the adaptive watchlist functionality described in the claim. Trading logic does not reference a dynamic watchlist, and required configuration files for even a static watchlist are missing. The described ranking, filtering, and dashboard updates are not implemented in the checked codebase.
