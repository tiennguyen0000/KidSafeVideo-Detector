-- Add training metrics table for detailed checkpoint evaluation
CREATE TABLE IF NOT EXISTS training_metrics (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES model_registry(id) ON DELETE CASCADE,
    epoch INTEGER NOT NULL,
    split TEXT NOT NULL,  -- 'train', 'val', 'test'
    
    -- Overall metrics
    loss FLOAT NOT NULL,
    accuracy FLOAT NOT NULL,
    precision FLOAT NOT NULL,
    recall FLOAT NOT NULL,
    f1 FLOAT NOT NULL,
    
    -- Per-class metrics (JSONB)
    per_class_metrics JSONB,  -- {class_name: {precision, recall, f1}}
    
    -- Confusion matrix
    confusion_matrix JSONB,  -- [[tp, fp], [fn, tn]] for each class
    
    -- Training info
    learning_rate FLOAT,
    batch_size INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_id, epoch, split)
);

CREATE INDEX idx_training_metrics_model_id ON training_metrics(model_id);
CREATE INDEX idx_training_metrics_split ON training_metrics(split);
CREATE INDEX idx_training_metrics_epoch ON training_metrics(epoch);

-- Add training configuration to model_registry
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS training_config JSONB;

-- Add epoch history to model_registry (for quick access)
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS best_epoch INTEGER;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS total_epochs INTEGER;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS training_time_seconds FLOAT;
