## TODO Items Tracking

This document tracks all TODO comments in the codebase with their status and priority.

### High Priority (P0) - Production Blockers

| File | Line | TODO | Status | Notes |
|------|------|------|--------|-------|
| `execution/broker_adapters/alpaca_options_adapter.py` | 79 | Implement atomic multi-leg when supported/verified | ⏳ PENDING | Multi-leg options orders need atomic execution |
| `gnosis/unified_trading_bot.py` | 155 | Calculate real PnL | ⏳ PENDING | Currently returns 0.0, need actual P&L tracking |
| `gnosis/unified_trading_bot.py` | 170 | Get last price | ⏳ PENDING | Price fetch needed for position valuation |
| `gnosis/unified_trading_bot.py` | 176 | Check clock | ⏳ PENDING | Market hours validation needed |

### Medium Priority (P1) - Feature Completeness

| File | Line | TODO | Status | Notes |
|------|------|------|--------|-------|
| `agents/composer/composer_agent_v3.py` | 162 | Apply express weights to signals | ⏳ PENDING | Would enhance signal metadata handling |
| `engines/engine_factory.py` | 70 | Initialize actual sentiment processors | ⏳ PENDING | Placeholder comment, verify implementation |
| `engines/sentiment/sentiment_engine_v3.py` | 175 | Integrate with social media APIs | ⏳ PENDING | Twitter/Reddit integration for sentiment |

### Low Priority (P2) - Enhancements

| File | Line | TODO | Status | Notes |
|------|------|------|--------|-------|
| `gnosis/scanner/__init__.py` | 49 | Implement full scanning logic | ⏳ PENDING | Integration with OpportunityScanner |
| `engines/orchestration/pipeline_runner.py` | 131 | Fix HedgeSnapshot.data attribute error | ⏳ PENDING | Commented out code, needs investigation |

### Completed ✅

| File | Line | TODO | Status | Completion Date |
|------|------|------|--------|-----------------|
| `gnosis/utils/greeks_calculator.py` | 46 | Implement actual Alpaca API call | ✅ DONE | 2025-12-03 |

---

## Action Items

1. **For P0 items**: These must be completed before live trading
   - Real P&L calculation is critical for risk management
   - Market hours check prevents trading when closed
   - Multi-leg atomic execution prevents partial fills

2. **For P1 items**: Should be completed within next sprint
   - Social media integration would enhance sentiment accuracy
   - Express weights would improve signal handling

3. **For P2 items**: Nice-to-haves for future releases
   - Can be deferred to later versions

---

## Review Process

When completing a TODO:
1. Update this tracker with completion date
2. Add unit tests for the new functionality
3. Update relevant documentation
4. Create PR referencing this tracker

Last Updated: 2025-12-03
