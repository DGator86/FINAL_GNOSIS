-- Migration: Create trade_decisions table
-- Description: Table for tracking trade decisions with full GNOSIS context
-- Author: GNOSIS ML Pipeline
-- Date: 2025-12-11

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create trade_decisions table
CREATE TABLE IF NOT EXISTS trade_decisions (
    -- ========== Identity ==========
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- ========== Meta ==========
    timestamp        TIMESTAMPTZ NOT NULL,
    mode             TEXT NOT NULL CHECK (mode IN ('live', 'paper', 'backtest')),
    symbol           TEXT NOT NULL,
    direction        TEXT NOT NULL CHECK (direction IN ('long', 'short', 'neutral')),
    structure        TEXT NOT NULL,
    config_version   TEXT NOT NULL,

    -- ========== Universe/Filter State ==========
    universe_eligible    BOOLEAN NOT NULL,
    universe_reasons     TEXT[] NOT NULL,
    price                NUMERIC(18,8) NOT NULL,
    adv                  NUMERIC(18,2) NOT NULL,
    iv_rank              NUMERIC(6,3) NOT NULL,
    realized_vol_30d     NUMERIC(8,4) NOT NULL,
    options_liq_score    NUMERIC(8,4) NOT NULL,

    -- ========== Engine Snapshots ==========
    dealer_features      JSONB NOT NULL,
    liquidity_features   JSONB NOT NULL,
    sentiment_features   JSONB NOT NULL,

    -- ========== Agent Logic ==========
    hedge_agent_vote     JSONB NOT NULL,
    liquidity_agent_vote JSONB NOT NULL,
    sentiment_agent_vote JSONB NOT NULL,
    composer_decision    JSONB NOT NULL,

    -- ========== Portfolio Context ==========
    portfolio_context    JSONB NOT NULL,

    -- ========== Execution Outcome ==========
    order_id        TEXT,
    entry_price     NUMERIC(18,8),
    target_price    NUMERIC(18,8),
    stop_price      NUMERIC(18,8),
    slippage_bps    NUMERIC(10,4),
    status          TEXT,

    -- ========== Audit Timestamps ==========
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ========== Indexes ==========

-- Basic indexes for typical queries
CREATE INDEX IF NOT EXISTS idx_trade_decisions_symbol_time
    ON trade_decisions (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_trade_decisions_mode_time
    ON trade_decisions (mode, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_trade_decisions_status
    ON trade_decisions (status);

CREATE INDEX IF NOT EXISTS idx_trade_decisions_timestamp
    ON trade_decisions (timestamp DESC);

-- Optional: GIN indexes on JSONB if you want to query inside engine/agent fields
CREATE INDEX IF NOT EXISTS idx_trade_decisions_dealer_features_gin
    ON trade_decisions USING GIN (dealer_features);

CREATE INDEX IF NOT EXISTS idx_trade_decisions_sentiment_features_gin
    ON trade_decisions USING GIN (sentiment_features);

CREATE INDEX IF NOT EXISTS idx_trade_decisions_liquidity_features_gin
    ON trade_decisions USING GIN (liquidity_features);

-- ========== Trigger for updated_at ==========
-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger
DROP TRIGGER IF EXISTS update_trade_decisions_updated_at ON trade_decisions;
CREATE TRIGGER update_trade_decisions_updated_at
    BEFORE UPDATE ON trade_decisions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ========== Comments ==========
COMMENT ON TABLE trade_decisions IS 'Trade decisions with full GNOSIS context for ML training';
COMMENT ON COLUMN trade_decisions.dealer_features IS 'Dealer Hedge Pressure Field Engine snapshot: GEX, vanna, charm, gamma pivot, etc.';
COMMENT ON COLUMN trade_decisions.liquidity_features IS 'Liquidity Engine snapshot: liquidity zones, dark pool, HVN/LVN, VWAP, etc.';
COMMENT ON COLUMN trade_decisions.sentiment_features IS 'Sentiment & Regime Engine snapshot: Wyckoff, micro/macro regime, sentiment scores';
COMMENT ON COLUMN trade_decisions.hedge_agent_vote IS 'Hedge Agent vote: bias, direction_bias, confidence';
COMMENT ON COLUMN trade_decisions.liquidity_agent_vote IS 'Liquidity Agent vote: zone, confidence';
COMMENT ON COLUMN trade_decisions.sentiment_agent_vote IS 'Sentiment Agent vote: risk_posture, trend_alignment, confidence';
COMMENT ON COLUMN trade_decisions.composer_decision IS 'Composer Agent decision: final_direction, structure, sizing, invalidation, reason_codes';
COMMENT ON COLUMN trade_decisions.portfolio_context IS 'Portfolio state: exposure_before/after, risk_per_trade, max_dd_limit, etc.';
