-- Create search_results table for caching video search results
-- This is needed because Airflow and API run in different containers
-- So in-memory cache doesn't work across containers

CREATE TABLE IF NOT EXISTS search_results (
    dag_run_id TEXT PRIMARY KEY,
    videos JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_search_results_expires ON search_results(expires_at);

-- Create function to auto-cleanup expired search results
CREATE OR REPLACE FUNCTION cleanup_expired_search_results()
RETURNS TRIGGER AS $$
BEGIN
    DELETE FROM search_results WHERE expires_at < CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to cleanup on insert (lazy cleanup)
CREATE TRIGGER cleanup_search_results_on_insert
    AFTER INSERT ON search_results
    FOR EACH STATEMENT EXECUTE FUNCTION cleanup_expired_search_results();
