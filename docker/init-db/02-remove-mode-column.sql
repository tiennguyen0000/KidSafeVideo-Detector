-- Migration: Remove mode column from training_samples_v2
-- This makes samples reusable across ultra_light and balanced modes

-- Drop the mode column and its index
ALTER TABLE training_samples_v2 DROP COLUMN IF EXISTS mode;
DROP INDEX IF EXISTS idx_training_samples_v2_mode;

-- Note: Existing samples remain valid as they were identical regardless of mode
