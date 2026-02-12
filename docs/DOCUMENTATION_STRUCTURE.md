# Documentation Organization

## Overview

Documentation has been reorganized from 60+ scattered markdown files into a structured hierarchy.

## New Structure

```
docs/
├── README.md                    # Documentation home
├── DOCUMENTATION_STRUCTURE.md   # This file
├── guides/                      # User guides
│   ├── MULTI_LEG_OPTIONS_GUIDE.md
│   └── GNOSIS_ANTIGRAVITY_AGENT.md
├── implementation/              # Implementation details
│   ├── HEDGE_ENGINE_V3.md
│   ├── SENTIMENT_ENGINE_V3.md
│   ├── LIQUIDITY_ENGINE_V3.md
│   ├── ELASTICITY_ENGINE_V3.md
│   ├── TRADE_EXECUTION_V3.md
│   └── BACKTEST_ENGINE_V3.md
├── history/                     # Historical logs (moved from root)
│   ├── SESSION_SUMMARY*.md
│   ├── MILESTONE*.md
│   ├── V3_TRANSFORMATION*.md
│   └── [other historical logs]
└── api/                         # API documentation (future)
    ├── SCHEMAS.md
    ├── ENGINES.md
    └── AGENTS.md

Root-level docs (kept for visibility):
├── README.md                    # Main project README
├── QUICKSTART.md                # Quick start guide
├── QUICK_REFERENCE.md           # Command cheat sheet
├── ARCHITECTURE_OVERVIEW.md     # System architecture
├── DEV_GUIDE.md                 # Development guide
├── DASHBOARD_GUIDE.md           # Dashboard usage
├── OPERATIONS_RUNBOOK.md        # Production operations
├── LIVE_TRADING.md              # Live trading setup
├── ALPACA_SETUP.md              # Broker integration
├── UNUSUAL_WHALES_INTEGRATION.md # Options data setup
└── ML_FEATURE_MATRIX.md         # ML features
```

## File Movement

### Moved to docs/history/
- CELEBRATION.md
- SESSION_SUMMARY*.md
- MILESTONE_ACHIEVED.md
- INTEGRATION_COMPLETE.md
- IMPLEMENTATION_COMPLETE.md
- V3_TRANSFORMATION*.md
- FINAL_COMPLETION*.md

### Moved to docs/implementation/
- *_ENGINE_V3_IMPLEMENTATION.md
- TRADE_EXECUTION_V3_IMPLEMENTATION.md
- BACKTEST_ENGINE_V3_IMPLEMENTATION.md

### Moved to docs/guides/
- MULTI_LEG_OPTIONS_GUIDE.md
- GNOSIS_ANTIGRAVITY_AGENT.md

### Kept in Root
Essential user-facing docs remain in root for discoverability:
- README.md, QUICKSTART.md, QUICK_REFERENCE.md
- ARCHITECTURE_OVERVIEW.md, DEV_GUIDE.md
- DASHBOARD_GUIDE.md, OPERATIONS_RUNBOOK.md
- Integration guides (ALPACA_SETUP.md, etc.)

## Usage

### For New Users
1. Start with [README.md](../README.md)
2. Follow [QUICKSTART.md](../QUICKSTART.md)
3. Reference [QUICK_REFERENCE.md](../QUICK_REFERENCE.md)

### For Developers
1. Read [DEV_GUIDE.md](../DEV_GUIDE.md)
2. Study [ARCHITECTURE_OVERVIEW.md](../ARCHITECTURE_OVERVIEW.md)
3. Explore [implementation/](implementation/) for deep dives

### For Operators
1. Review [OPERATIONS_RUNBOOK.md](../OPERATIONS_RUNBOOK.md)
2. Setup following [LIVE_TRADING.md](../LIVE_TRADING.md)
3. Monitor with [DASHBOARD_GUIDE.md](../DASHBOARD_GUIDE.md)

### For Historical Context
1. Browse [history/](history/) for development logs
2. Review milestone documents for major achievements

## Maintenance

When adding new documentation:

- **User guides** → `docs/guides/`
- **Implementation details** → `docs/implementation/`
- **API documentation** → `docs/api/`
- **Session logs** → `docs/history/`
- **Essential user-facing** → Root level

Keep root documentation concise and focused on discoverability.

Last Updated: 2025-12-03
