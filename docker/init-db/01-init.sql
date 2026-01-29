-- Create videos table
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    label TEXT,  -- NULL for inference-only videos (not used for training)
    title TEXT,  -- Video title
    transcript TEXT,
    storage_path TEXT,
    status_preprocess TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_videos_status ON videos(status_preprocess);
CREATE INDEX idx_videos_label ON videos(label);
CREATE INDEX idx_videos_created_at ON videos(created_at);

-- Create video_preprocess table for tracking preprocessing steps
CREATE TABLE IF NOT EXISTS video_preprocess (
    id SERIAL PRIMARY KEY,
    video_id TEXT REFERENCES videos(video_id) ON DELETE CASCADE,
    frames_extracted BOOLEAN DEFAULT FALSE,
    audio_extracted BOOLEAN DEFAULT FALSE,
    transcript_generated BOOLEAN DEFAULT FALSE,
    frames_sampled BOOLEAN DEFAULT FALSE,
    frame_indices JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(video_id)
);

CREATE INDEX idx_video_preprocess_video_id ON video_preprocess(video_id);

-- Create training_samples table
CREATE TABLE IF NOT EXISTS training_samples (
    id SERIAL PRIMARY KEY,
    video_id TEXT REFERENCES videos(video_id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    frame_indices JSONB,
    transcript_available BOOLEAN DEFAULT FALSE,
    mode_ready TEXT,
    split TEXT DEFAULT 'train',  -- 'train', 'val', 'test'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_training_samples_video_id ON training_samples(video_id);
CREATE INDEX idx_training_samples_split ON training_samples(split);
CREATE INDEX idx_training_samples_label ON training_samples(label);

-- Create fusion_config table
CREATE TABLE IF NOT EXISTS fusion_config (
    id SERIAL PRIMARY KEY,
    mode TEXT NOT NULL,  -- 'balanced' or 'ultra_light'
    class_name TEXT NOT NULL,
    w_text FLOAT NOT NULL,
    w_img FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(mode, class_name, is_active)
);

CREATE INDEX idx_fusion_config_mode ON fusion_config(mode);
CREATE INDEX idx_fusion_config_active ON fusion_config(is_active);

-- Create model_registry table
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    mode TEXT NOT NULL,  -- 'balanced' or 'ultra_light'
    model_type TEXT NOT NULL,  -- 'text' or 'image'
    version TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE,
    metrics JSONB
);

CREATE INDEX idx_model_registry_mode ON model_registry(mode);
CREATE INDEX idx_model_registry_type ON model_registry(model_type);
CREATE INDEX idx_model_registry_active ON model_registry(is_active);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    video_id TEXT REFERENCES videos(video_id) ON DELETE CASCADE,
    mode TEXT NOT NULL,
    y_pred TEXT NOT NULL,
    p_text JSONB,
    p_img JSONB,
    p_final JSONB,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_video_id ON predictions(video_id);
CREATE INDEX idx_predictions_mode ON predictions(mode);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);

-- Insert default fusion weights
INSERT INTO fusion_config (mode, class_name, w_text, w_img, is_active)
VALUES
    ('ultra_light', 'Aggressive', 0.5, 0.5, TRUE),
    ('ultra_light', 'Superstition', 0.6, 0.4, TRUE),
    ('ultra_light', 'Safe', 0.5, 0.5, TRUE),
    ('ultra_light', 'Sexual', 0.2, 0.8, TRUE),
    ('balanced', 'Aggressive', 0.5, 0.5, TRUE),
    ('balanced', 'Superstition', 0.6, 0.4, TRUE),
    ('balanced', 'Safe', 0.5, 0.5, TRUE),
    ('balanced', 'Sexual', 0.2, 0.8, TRUE)
ON CONFLICT (mode, class_name, is_active) DO NOTHING;

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_videos_updated_at BEFORE UPDATE ON videos
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_video_preprocess_updated_at BEFORE UPDATE ON video_preprocess
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create training_samples_v2 table (pre-generated with augmentation)
CREATE TABLE IF NOT EXISTS training_samples_v2 (
    sample_id TEXT PRIMARY KEY,
    video_id TEXT REFERENCES videos(video_id) ON DELETE CASCADE,
    label TEXT,  -- NULL for inference samples
    augment_idx INTEGER DEFAULT 0,
    selected_frames TEXT,  -- JSON array of frame indices
    selected_chunks TEXT,  -- JSON array of chunk indices
    split TEXT DEFAULT 'train',  -- 'train', 'val', 'test', 'inference'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_training_samples_v2_video ON training_samples_v2(video_id);
CREATE INDEX idx_training_samples_v2_label ON training_samples_v2(label);
CREATE INDEX idx_training_samples_v2_split ON training_samples_v2(split);

CREATE TRIGGER update_training_samples_v2_updated_at BEFORE UPDATE ON training_samples_v2
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
