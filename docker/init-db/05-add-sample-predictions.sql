-- Migration: Add sample_predictions table to store per-sample predictions from training
-- This enables error analysis by storing the predicted label for each training sample

CREATE TABLE IF NOT EXISTS sample_predictions (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES model_registry(id) ON DELETE CASCADE,
    sample_id TEXT NOT NULL,  -- References training_samples_v2.sample_id
    true_label TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence FLOAT,
    p_text JSONB,      -- Text model probabilities
    p_img JSONB,       -- Image model probabilities  
    p_final JSONB,     -- Final fused probabilities
    split TEXT DEFAULT 'val',  -- 'train', 'val', 'test'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, sample_id)
);

CREATE INDEX idx_sample_predictions_model ON sample_predictions(model_id);
CREATE INDEX idx_sample_predictions_sample ON sample_predictions(sample_id);
CREATE INDEX idx_sample_predictions_labels ON sample_predictions(true_label, predicted_label);
CREATE INDEX idx_sample_predictions_split ON sample_predictions(split);

-- Comment: 
-- This table stores individual predictions for each sample after training.
-- Used for error analysis to see which specific samples were misclassified.
