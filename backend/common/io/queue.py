"""
Queue client - Search results cache only.

Note: Inference and training queues migrated to Kafka.
This module handles search results caching using PostgreSQL database.

Important: In-memory cache doesn't work across containers (Airflow vs API).
So we use the database for cross-container communication.
"""

import json
from typing import Any, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class QueueClient:
    """
    Client for caching search results using PostgreSQL.
    
    Note: All queue operations (inference, training) now use Kafka.
    This class only handles temporary search results cache via database.
    """
    
    def __init__(self):
        logger.info("QueueClient initialized (using PostgreSQL for cross-container cache)")
    
    def _get_db_connection(self):
        """Get database connection (lazy import to avoid circular imports)."""
        from common.io.database import db
        return db
    
    def cache_search_results(self, dag_run_id: str, videos: list, expire: int = 600) -> bool:
        """
        Cache video search results with expiration in database.
        
        Args:
            dag_run_id: Unique identifier for the search
            videos: List of video dicts
            expire: Expiration time in seconds (default: 10 minutes)
        """
        try:
            db = self._get_db_connection()
            expiry_time = datetime.now() + timedelta(seconds=expire)
            
            # Upsert search results
            query = """
                INSERT INTO search_results (dag_run_id, videos, expires_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (dag_run_id) DO UPDATE
                SET videos = EXCLUDED.videos,
                    expires_at = EXCLUDED.expires_at,
                    created_at = CURRENT_TIMESTAMP
            """
            
            import psycopg2.extras
            db.execute(query, (dag_run_id, psycopg2.extras.Json(videos), expiry_time))
            
            logger.info(f"Cached {len(videos)} search results for {dag_run_id} in database")
            return True
        except Exception as e:
            logger.error(f"Error caching search results in database: {e}")
            return False
    
    def get_search_results(self, dag_run_id: str) -> Optional[list]:
        """
        Get cached video search results from database.
        
        Returns None if not found or expired.
        """
        try:
            db = self._get_db_connection()
            
            query = """
                SELECT videos FROM search_results 
                WHERE dag_run_id = %s AND expires_at > CURRENT_TIMESTAMP
            """
            results = db.execute(query, (dag_run_id,), fetch=True)
            
            if results and len(results) > 0:
                videos = results[0]['videos']
                logger.info(f"Retrieved {len(videos)} search results for {dag_run_id}")
                return videos
            
            # Check if expired vs not found
            check_query = "SELECT dag_run_id FROM search_results WHERE dag_run_id = %s"
            exists = db.execute(check_query, (dag_run_id,), fetch=True)
            
            if exists:
                logger.info(f"Search results expired for {dag_run_id}")
            else:
                logger.debug(f"No search results found for {dag_run_id}")
            
            return None
        except Exception as e:
            logger.error(f"Error getting search results from database: {e}")
            return None
    
    def cleanup_expired_results(self) -> int:
        """Manually cleanup expired search results. Returns count of deleted rows."""
        try:
            db = self._get_db_connection()
            query = "DELETE FROM search_results WHERE expires_at < CURRENT_TIMESTAMP RETURNING dag_run_id"
            results = db.execute(query, fetch=True)
            count = len(results) if results else 0
            if count > 0:
                logger.info(f"Cleaned up {count} expired search results")
            return count
        except Exception as e:
            logger.error(f"Error cleaning up expired results: {e}")
            return 0


# Global queue client instance
queue = QueueClient()

