-- Migration: Add gate_weights table to store gated fusion analysis
-- This enables visualization of when the model trusts image vs text modalities

CREATE TABLE IF NOT EXISTS gate_weights (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES model_registry(id) ON DELETE CASCADE,
    split TEXT DEFAULT 'val',  -- 'train', 'val'
    gate_weights_data JSONB NOT NULL,  -- Contains per-class gate weight statistics
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, split)
);

CREATE INDEX idx_gate_weights_model ON gate_weights(model_id);
CREATE INDEX idx_gate_weights_split ON gate_weights(split);

-- Comment:
-- This table stores aggregated gate weight statistics computed during training.
-- gate_weights_data JSON structure:
-- {
--   "per_class": {
--     "Safe": {"mean": 0.65, "std": 0.12, "count": 100},
--     "Aggressive": {"mean": 0.42, "std": 0.18, "count": 50},
--     ...
--   },
--   "overall": {"mean": 0.55, "std": 0.15, "total": 200},
--   "interpretation": {
--     "image_dominant_classes": ["Safe"],
--     "text_dominant_classes": ["Aggressive"],
--     "balanced_classes": ["Sexual"]
--   }
-- }
-- Gate weight interpretation: ~1.0 = trust image, ~0.0 = trust text, ~0.5 = balanced
