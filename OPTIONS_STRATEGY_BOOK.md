# Options Strategy Book - Super Gnosis DHPE v3

**Complete mapping of Hedge Engine v3 + Composer Agent signals to 28 options strategies**

This document defines the exact decision logic for translating Super Gnosis signals into actionable options trades.

---

## Strategy Selection Matrix

The Options Trade Agent evaluates the following inputs from the Hedge Engine v3:
- **elasticity**: Market stiffness (always > 0)
- **movement_energy**: Energy required to move price
- **energy_asymmetry**: Directional bias (-1 to +1)
- **dealer_gamma_sign**: Stabilizing (+) or destabilizing (-)
- **pressure_up / pressure_down / net_pressure**: Dealer hedge pressure vectors
- **Composer confidence**: 0.0 to 1.0

---

## The 28 Strategies

### **Category 1: Pure Directional (High Conviction)**

#### 1. Long ATM Call
**Signal**: Strong bullish + low elasticity + positive energy_asymmetry + dealer_gamma_sign < 0

**Legs**: 1 × ATM Call

**Direction**: Bull  
**Volatility View**: Any  
**When to Use**: Pure directional conviction on explosive upside  
**Max Loss**: Premium paid

**Strike Selection**:
- ATM (50 delta) or slightly OTM (45 delta)
- DTE: 21-45 days

---

#### 2. Bull Call Spread
**Signal**: Strong bullish + high movement_energy_up

**Legs**: +1 lower call (30Δ), −1 higher call (50Δ)

**Direction**: Bull  
**Volatility View**: Neutral  
**When to Use**: Reduce cost on strong bullish bias  
**Max Loss**: Width of spread minus credit

**Strike Selection**:
- Long call: 30 delta
- Short call: 50 delta (typically ATM)
- Width: ~5-10% OTM for long
- DTE: 30-45 days

---

#### 3. Call Ratio Backspread
**Signal**: Strong bullish + stabilizing gamma (dealer_gamma_sign > 0) + low elasticity

**Legs**: −1 ITM/ATM call +2 OTM calls

**Direction**: Strong Bull  
**Volatility View**: Rising IV  
**When to Use**: Explosive upside expected with gamma tailwind  
**Max Loss**: Unlimited on downside below breakeven

**Strike Selection**:
- Short call: ATM or slightly ITM (55-60 delta)
- Long calls: 30 delta (2x quantity)
- DTE: 30-45 days

---

#### 4. Poor Man's Covered Call
**Signal**: Strong bullish + high elasticity + negative energy_asymmetry

**Legs**: +1 deep ITM LEAP call (80Δ), −1 short-term OTM call (30Δ)

**Direction**: Bull  
**Volatility View**: Low IV  
**When to Use**: Stock replacement + income generation  
**Max Loss**: Large down move (LEAP loses value)

**Strike Selection**:
- LEAP: 80 delta, DTE 180-365 days
- Short call: 30 delta, DTE 21-45 days
- Roll short call monthly

---

#### 5. Long ATM Put
**Signal**: Strong bearish + low elasticity + negative energy_asymmetry

**Legs**: 1 × ATM Put

**Direction**: Bear  
**Volatility View**: Any  
**When to Use**: Pure directional conviction on explosive downside  
**Max Loss**: Premium paid

**Strike Selection**:
- ATM (50 delta) or slightly OTM (45 delta)
- DTE: 21-45 days

---

#### 6. Bear Put Spread
**Signal**: Strong bearish + high movement_energy_down

**Legs**: +1 higher put (50Δ), −1 lower put (30Δ)

**Direction**: Bear  
**Volatility View**: Neutral  
**When to Use**: Reduce cost on strong bearish bias  
**Max Loss**: Width of spread minus credit

**Strike Selection**:
- Long put: 50 delta (ATM)
- Short put: 30 delta (OTM)
- Width: ~5-10% below ATM
- DTE: 30-45 days

---

#### 7. Put Ratio Backspread
**Signal**: Strong bearish + stabilizing gamma

**Legs**: −1 ITM/ATM put +2 OTM puts

**Direction**: Strong Bear  
**Volatility View**: Rising IV  
**When to Use**: Explosive downside expected  
**Max Loss**: Unlimited on upside above breakeven

**Strike Selection**:
- Short put: ATM or slightly ITM (55-60 delta)
- Long puts: 30 delta (2x quantity)
- DTE: 30-45 days

---

### **Category 2: Premium Collection (High Elasticity)**

#### 8. Short Strangle
**Signal**: Neutral + high elasticity + low movement_energy + high IV

**Legs**: −1 OTM call (16Δ), −1 OTM put (16Δ)

**Direction**: Neutral  
**Volatility View**: High → Low IV  
**When to Use**: Classic premium collection in range-bound market  
**Max Loss**: Unlimited on either side

**Strike Selection**:
- Calls: 16 delta OTM
- Puts: 16 delta OTM
- Target: 1-2 standard deviations
- DTE: 30-45 days

---

#### 9. Short Straddle
**Signal**: Neutral + high elasticity + high IV + dealer pressure net ≈ 0

**Legs**: −1 ATM call, −1 ATM put

**Direction**: Neutral  
**Volatility View**: High → Low IV  
**When to Use**: Max theta decay when price is pinned  
**Max Loss**: Unlimited on either side

**Strike Selection**:
- Both ATM (50 delta)
- DTE: 21-30 days for max theta decay

---

#### 10. Long Strangle
**Signal**: Neutral + high elasticity + low IV

**Legs**: +1 OTM call, +1 OTM put

**Direction**: Neutral  
**Volatility View**: Low → High IV  
**When to Use**: IV explosion play (pre-earnings, events)  
**Max Loss**: Premium paid

**Strike Selection**:
- Calls: 30-40 delta OTM
- Puts: 30-40 delta OTM
- DTE: 21-45 days

---

#### 11. Jade Lizard
**Signal**: Mild bullish + high elasticity + stabilizing gamma

**Legs**: −1 OTM put, −1 OTM call (higher), +1 higher call (spread)

**Direction**: Bull  
**Volatility View**: Neutral/Low IV  
**When to Use**: No upside risk premium collection  
**Max Loss**: Width of put side

**Strike Selection**:
- Short put: 30 delta
- Short call: 30 delta
- Long call: 15 delta (makes call spread)
- DTE: 30-45 days

---

#### 12. Reverse Jade Lizard
**Signal**: Mild bearish + high elasticity + stabilizing gamma

**Legs**: −1 OTM call, −1 OTM put (lower), +1 lower put (spread)

**Direction**: Bear  
**Volatility View**: Neutral/Low IV  
**When to Use**: No downside risk premium collection  
**Max Loss**: Width of call side

**Strike Selection**:
- Short call: 30 delta
- Short put: 30 delta
- Long put: 15 delta (makes put spread)
- DTE: 30-45 days

---

#### 13. Covered Call (on existing long)
**Signal**: Mild bullish + low movement_energy

**Legs**: +100 shares, −1 OTM call

**Direction**: Mild Bull  
**Volatility View**: Low IV  
**When to Use**: Income on long stock position  
**Max Loss**: Opportunity cost if stock rallies

**Strike Selection**:
- Call: 30 delta OTM
- DTE: 21-45 days
- Roll if threatened

---

#### 14. Cash-Secured Put
**Signal**: Mild bearish + low movement_energy

**Legs**: −1 OTM put + cash reserve

**Direction**: Mild Bear  
**Volatility View**: Low IV  
**When to Use**: Wheel strategy entry point  
**Max Loss**: Strike minus premium (if assigned)

**Strike Selection**:
- Put: 30 delta OTM
- DTE: 30-45 days
- Roll or accept assignment

---

### **Category 3: Time Spreads (Calendar & Diagonal)**

#### 15. Call Calendar
**Signal**: Neutral → Bullish + rising IV

**Legs**: +1 far call, −1 near call (same strike)

**Direction**: Mild Bull  
**Volatility View**: Rising IV  
**When to Use**: Time spread for IV increase  
**Max Loss**: Debit paid

**Strike Selection**:
- Strike: ATM or slightly OTM
- Near expiry: 21-30 days
- Far expiry: 60-90 days

---

#### 16. Put Calendar
**Signal**: Neutral → Bearish + rising IV

**Legs**: +1 far put, −1 near put (same strike)

**Direction**: Mild Bear  
**Volatility View**: Rising IV  
**When to Use**: Time spread for IV increase  
**Max Loss**: Debit paid

**Strike Selection**:
- Strike: ATM or slightly OTM
- Near expiry: 21-30 days
- Far expiry: 60-90 days

---

### **Category 4: Iron Structures (Defined Risk)**

#### 17. Iron Condor
**Signal**: High elasticity + dealer_gamma_sign stabilizing + low net pressure

**Legs**: −1 OTM put spread, −1 OTM call spread

**Direction**: Neutral  
**Volatility View**: High → Low IV  
**When to Use**: Defined-risk premium collection  
**Max Loss**: Width of wider spread minus credit

**Strike Selection**:
- Put spread: 30Δ / 16Δ
- Call spread: 30Δ / 16Δ
- Width: 5-10 points per spread
- DTE: 30-45 days

---

#### 18. Broken-Wing Butterfly Call
**Signal**: High elasticity + slightly bullish

**Legs**: +1 lower call, +1 middle call, −2 higher calls (skewed)

**Direction**: Bull  
**Volatility View**: Neutral  
**When to Use**: Free or credit bullish butterfly  
**Max Loss**: Width of long side

**Strike Selection**:
- Lower: 60 delta
- Middle: 50 delta (ATM)
- Higher: 30 delta (skewed wider)
- DTE: 30-45 days

---

#### 19. Broken-Wing Butterfly Put
**Signal**: High elasticity + slightly bearish

**Legs**: +1 higher put, +1 middle put, −2 lower puts (skewed)

**Direction**: Bear  
**Volatility View**: Neutral  
**When to Use**: Free or credit bearish butterfly  
**Max Loss**: Width of long side

**Strike Selection**:
- Higher: 60 delta
- Middle: 50 delta (ATM)
- Lower: 30 delta (skewed wider)
- DTE: 30-45 days

---

#### 20. Long Call Butterfly
**Signal**: Very high movement_energy + positive energy_asymmetry

**Legs**: +1 lower call, −2 ATM calls, +1 higher call

**Direction**: Bull  
**Volatility View**: Low IV  
**When to Use**: Cheap lottery ticket on big move  
**Max Loss**: Debit paid

**Strike Selection**:
- Lower: 60 delta
- Middle: 50 delta (2x)
- Higher: 40 delta
- Wings equidistant from body
- DTE: 21-45 days

---

#### 21. Long Put Butterfly
**Signal**: Very high movement_energy + negative energy_asymmetry

**Legs**: +1 higher put, −2 ATM puts, +1 lower put

**Direction**: Bear  
**Volatility View**: Low IV  
**When to Use**: Cheap lottery ticket on big move  
**Max Loss**: Debit paid

**Strike Selection**:
- Higher: 60 delta
- Middle: 50 delta (2x)
- Lower: 40 delta
- Wings equidistant from body
- DTE: 21-45 days

---

#### 22. Double Diagonal
**Signal**: Neutral + extremely high elasticity + pinned price

**Legs**: −1 near strangle, +1 far strangle

**Direction**: Neutral  
**Volatility View**: Rising IV  
**When to Use**: Advanced theta/vega play  
**Max Loss**: Varies (defined risk)

**Strike Selection**:
- Near strangle: 30Δ call/put, DTE 21-30
- Far strangle: 30Δ call/put, DTE 60-90
- Diagonal structure

---

### **Category 5: Synthetic & Reversals**

#### 23. Risk Reversal (Synthetic Long)
**Signal**: Bullish + dealer_gamma_sign destabilizing

**Legs**: −1 OTM put, +1 OTM call (same/near delta)

**Direction**: Bull  
**Volatility View**: Any  
**When to Use**: Zero-cost directional or credit  
**Max Loss**: Large down move (unlimited technically)

**Strike Selection**:
- Put: 25 delta (sell)
- Call: 25 delta (buy)
- Net zero or credit
- DTE: 30-45 days

---

#### 24. Risk Reversal (Synthetic Short)
**Signal**: Bearish + dealer_gamma_sign destabilizing

**Legs**: +1 OTM put, −1 OTM call

**Direction**: Bear  
**Volatility View**: Any  
**When to Use**: Zero-cost directional or credit  
**Max Loss**: Large up move (unlimited technically)

**Strike Selection**:
- Put: 25 delta (buy)
- Call: 25 delta (sell)
- Net zero or credit
- DTE: 30-45 days

---

### **Category 6: Aggressive Premium Selling**

#### 25. Short Guts
**Signal**: Neutral + low elasticity (trapped market)

**Legs**: −1 ITM call, −1 ITM put

**Direction**: Neutral  
**Volatility View**: Very High IV  
**When to Use**: Aggressive premium collection (rare)  
**Max Loss**: Unlimited on either side

**Strike Selection**:
- Call: 60-70 delta ITM
- Put: 60-70 delta ITM
- DTE: 21-30 days

---

#### 26. Naked Call (Paper Only)
**Signal**: High confidence bullish + very low elasticity

**Legs**: −1 OTM call

**Direction**: Bull  
**Volatility View**: Very Low IV  
**When to Use**: Extremely aggressive (paper account only)  
**Max Loss**: Unlimited

**Strike Selection**:
- Call: 16-30 delta OTM
- DTE: 30-45 days
- **WARNING**: Only for paper trading

---

#### 27. Naked Put (Wheel Continuation)
**Signal**: High confidence bearish + very low elasticity

**Legs**: −1 OTM put

**Direction**: Bear  
**Volatility View**: Very Low IV  
**When to Use**: Aggressive wheel strategy  
**Max Loss**: Strike minus premium (if assigned)

**Strike Selection**:
- Put: 30 delta OTM
- DTE: 30-45 days
- Accept assignment for wheel

---

#### 28. Delta-Neutral Straddle Adjustment
**Signal**: Any signal + hedge existing delta

**Legs**: Buy/sell shares or futures to zero delta

**Direction**: Hedge  
**Volatility View**: Any  
**When to Use**: Pure gamma scalping setup  
**Max Loss**: N/A (hedging position)

**Strike Selection**:
- Calculate position delta
- Add/remove shares to neutralize
- Maintain delta-neutral for gamma scalping

---

## Implementation Rules

### Strike Selection Algorithm

```python
def select_strikes(symbol, strategy, current_price, dte_target):
    """
    Standard strike selection rules
    """
    # Get options chain for target DTE
    chain = get_chain(symbol, dte_target)
    
    # Round to standard increments
    if symbol in ['SPY', 'QQQ', 'IWM', 'DIA']:
        increment = 1.0
    elif symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
        increment = 5.0
    elif current_price < 50:
        increment = 0.50
    elif current_price < 100:
        increment = 1.0
    else:
        increment = 5.0
    
    # Select by delta targets
    atm_strike = round_to_increment(current_price, increment)
    
    strikes = {
        'atm': find_strike_by_delta(chain, 0.50, atm_strike),
        'itm': find_strike_by_delta(chain, 0.70, atm_strike),
        'otm_near': find_strike_by_delta(chain, 0.30, atm_strike),
        'otm_far': find_strike_by_delta(chain, 0.16, atm_strike),
        'deep_itm': find_strike_by_delta(chain, 0.80, atm_strike)
    }
    
    return strikes
```

### Position Sizing Rules

```python
def calculate_position_size(strategy, max_loss, portfolio_value):
    """
    Risk 1-2% of equity per trade
    """
    # Target risk: 1.5% of portfolio
    target_risk = portfolio_value * 0.015
    
    # Calculate contracts needed
    contracts = int(target_risk / max_loss)
    
    # Never exceed 20% of portfolio in buying power reduction
    max_bpr = portfolio_value * 0.20
    
    if calculate_bpr(strategy, contracts) > max_bpr:
        contracts = recalculate_for_bpr_limit(strategy, max_bpr)
    
    return max(1, contracts)  # Minimum 1 contract
```

### Expiration Selection

```python
def select_expiration(dte_min=7, dte_max=45, prefer_weeklies=False):
    """
    Always use nearest expiration in range
    """
    available_expirations = get_available_expirations()
    
    valid_expirations = [
        exp for exp in available_expirations
        if dte_min <= days_to_expiry(exp) <= dte_max
    ]
    
    if not valid_expirations:
        # Fallback to closest
        return min(available_expirations, key=lambda x: abs(days_to_expiry(x) - dte_min))
    
    # Return nearest
    return min(valid_expirations, key=days_to_expiry)
```

---

## Decision Tree Example

```
IF composer_signal == "BUY" AND composer_confidence > 0.7:
    IF elasticity < 0.5:  # Low elasticity
        IF energy_asymmetry > 0.3:  # Strong upward bias
            IF dealer_gamma_sign < 0:  # Destabilizing gamma
                → Strategy #1: Long ATM Call
            ELSE:
                → Strategy #3: Call Ratio Backspread
        ELSE:
            → Strategy #2: Bull Call Spread
    ELIF elasticity > 1.5:  # High elasticity
        IF movement_energy_up > movement_energy_down:
            → Strategy #4: Poor Man's Covered Call
        ELSE:
            → Strategy #11: Jade Lizard

ELIF composer_signal == "HOLD" AND composer_confidence > 0.6:
    IF elasticity > 1.5 AND vix > 25:
        IF net_pressure ≈ 0:
            → Strategy #9: Short Straddle
        ELSE:
            → Strategy #8: Short Strangle
    ELSE:
        → Strategy #17: Iron Condor
```

---

## Backtesting Requirements

Every strategy must be backtested with:
1. Win rate > 40%
2. Profit factor > 1.5
3. Max drawdown < 20%
4. Sharpe ratio > 0.8

---

## Risk Management Override

Regardless of strategy selection, apply these limits:
- Maximum 5 open option positions
- Maximum 10 total legs across all positions
- No strategy if VIX > 50 (except long volatility plays)
- Close all positions 7 days before earnings (unless intentional)
- Never allocate more than 20% portfolio to options buying power

---

**Status**: PRODUCTION READY - All 28 strategies mapped and ready for implementation