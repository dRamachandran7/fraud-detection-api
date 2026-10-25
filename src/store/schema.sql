-- Scored transactions, one row per transaction_id.
--
-- The scoring consumer is at-least-once, so the same score can arrive more
-- than once; transaction_id is the primary key and writes upsert, making
-- redelivery a no-op rather than a duplicate row.
CREATE TABLE IF NOT EXISTS fraud_scores (
    transaction_id TEXT PRIMARY KEY,
    is_fraud       BOOLEAN          NOT NULL,
    score          DOUBLE PRECISION NOT NULL,
    model_version  TEXT             NOT NULL,
    scored_at      TIMESTAMPTZ      NOT NULL DEFAULT now()
);

-- Supports the analyst query this store exists for: what got flagged, newest
-- first. Partial, because the flagged rows are ~0.5% of the table.
CREATE INDEX IF NOT EXISTS fraud_scores_flagged_idx
    ON fraud_scores (scored_at DESC)
    WHERE is_fraud;
