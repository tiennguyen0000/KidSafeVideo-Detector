"""
Database cleanup script - Keep only videos from labels.csv

This script:
1. Reads labels.csv to get list of valid video URLs
2. Backs up current database to JSON files
3. Deletes records for videos NOT in labels.csv
4. Clears Kafka topics (optional)

Tables to clean (per-video records):
- videos (main table, video_id = link from CSV)
- video_preprocess (FK → videos, ON DELETE CASCADE)
- training_samples (FK → videos, ON DELETE CASCADE)
- training_samples_v2 (FK → videos, ON DELETE CASCADE)
- predictions (FK → videos, ON DELETE CASCADE)

Tables NOT cleaned (not per-video or handled by CASCADE):
- fusion_config (global config)
- model_registry (model artifacts)
- training_metrics (FK → model_registry)
- sample_predictions (FK → model_registry, refs sample_id)
- search_results (temporary cache)

Usage:
    # Inside airflow container:
    python /app/scripts/cleanup_database.py
    
    # With options:
    python /app/scripts/cleanup_database.py --dry-run  # Preview only
    python /app/scripts/cleanup_database.py --skip-kafka  # Skip Kafka cleanup
    python /app/scripts/cleanup_database.py --csv-path /custom/path/labels.csv
"""

import os
import sys
import csv
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add common modules to path (support both /opt/airflow and /app)
for path in ['/opt/airflow', '/app']:
    if path not in sys.path:
        sys.path.insert(0, path)

from common.io import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_labels_csv(csv_path: str) -> set:
    """Read labels.csv and return set of valid video URLs."""
    valid_urls = set()
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_url = row.get('link', '').strip()
                if video_url:
                    valid_urls.add(video_url)
        
        logger.info(f"Found {len(valid_urls)} valid video URLs in labels.csv")
        return valid_urls
    except Exception as e:
        logger.error(f"Error reading labels.csv: {e}")
        raise


def backup_table(table_name: str, backup_dir: Path):
    """Backup a table to JSON file."""
    try:
        query = f"SELECT * FROM {table_name}"
        rows = db.execute(query, fetch=True)
        
        if rows:
            backup_file = backup_dir / f"{table_name}.json"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(rows, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"✅ Backed up {table_name}: {len(rows)} records")
            return len(rows)
        else:
            logger.info(f"⚠️  {table_name} is empty, skipping backup")
            return 0
    except Exception as e:
        logger.error(f"❌ Error backing up {table_name}: {e}")
        return 0


def backup_database(backup_dir: Path) -> dict:
    """Backup all important tables."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    tables_to_backup = [
        'videos',
        'video_preprocess',
        'training_samples',
        'training_samples_v2',
        'predictions',
        'sample_predictions',
        'fusion_config',
        'model_registry',
        'training_metrics',
    ]
    
    logger.info("=" * 80)
    logger.info(f"BACKING UP DATABASE to {backup_dir}")
    logger.info("=" * 80)
    
    backup_counts = {}
    for table in tables_to_backup:
        count = backup_table(table, backup_dir)
        backup_counts[table] = count
    
    # Save backup summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'backup_dir': str(backup_dir),
        'tables': backup_counts
    }
    summary_file = backup_dir / 'backup_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("✅ Database backup completed")
    return backup_counts


def get_invalid_video_ids(valid_urls: set) -> list:
    """Get list of video_ids that are NOT in labels.csv."""
    query = "SELECT video_id FROM videos"
    all_videos = db.execute(query, fetch=True) or []
    
    invalid_ids = []
    for video in all_videos:
        video_id = video['video_id']
        if video_id not in valid_urls:
            invalid_ids.append(video_id)
    
    logger.info(f"Found {len(all_videos)} total videos")
    logger.info(f"Found {len(valid_urls)} valid videos (in labels.csv)")
    logger.info(f"Found {len(invalid_ids)} invalid videos (will be deleted)")
    
    return invalid_ids


def clean_table(table_name: str, invalid_ids: list, video_id_column: str = 'video_id'):
    """Delete records for invalid video IDs."""
    if not invalid_ids:
        logger.info(f"⚠️  No invalid IDs, skipping {table_name}")
        return 0
    
    try:
        # Delete in batches to avoid long queries
        batch_size = 100
        total_deleted = 0
        
        for i in range(0, len(invalid_ids), batch_size):
            batch = invalid_ids[i:i + batch_size]
            placeholders = ','.join(['%s'] * len(batch))
            
            query = f"""
                DELETE FROM {table_name}
                WHERE {video_id_column} IN ({placeholders})
            """
            
            db.execute(query, tuple(batch), fetch=False)
            total_deleted += len(batch)
            
            logger.info(f"  Deleted batch {i//batch_size + 1}: {len(batch)} records")
        
        logger.info(f"✅ Cleaned {table_name}: deleted {total_deleted} records")
        return total_deleted
    except Exception as e:
        logger.error(f"❌ Error cleaning {table_name}: {e}")
        return 0


def clean_database(valid_urls: set, dry_run: bool = False):
    """Clean all tables - keep only videos from labels.csv."""
    logger.info("=" * 80)
    logger.info("CLEANING DATABASE" + (" [DRY RUN]" if dry_run else ""))
    logger.info("=" * 80)
    
    # Get invalid video IDs
    invalid_ids = get_invalid_video_ids(valid_urls)
    
    if not invalid_ids:
        logger.info("✅ No invalid videos found, database is already clean!")
        return
    
    if dry_run:
        logger.info(f"\n[DRY RUN] Would delete {len(invalid_ids)} videos and related records:")
        for vid in invalid_ids[:10]:
            logger.info(f"  - {vid[:80]}...")
        if len(invalid_ids) > 10:
            logger.info(f"  ... and {len(invalid_ids) - 10} more")
        return
    
    # First, clean sample_predictions that reference training_samples_v2
    # (since sample_predictions.sample_id → training_samples_v2.sample_id is not CASCADE)
    logger.info("\n--- Cleaning sample_predictions (linked via sample_id) ---")
    try:
        # Get sample_ids from training_samples_v2 that will be deleted
        if invalid_ids:
            placeholders = ','.join(['%s'] * len(invalid_ids))
            query = f"""
                DELETE FROM sample_predictions
                WHERE sample_id IN (
                    SELECT sample_id FROM training_samples_v2 
                    WHERE video_id IN ({placeholders})
                )
            """
            db.execute(query, tuple(invalid_ids), fetch=False)
            logger.info("✅ Cleaned sample_predictions")
    except Exception as e:
        logger.warning(f"⚠️  Error cleaning sample_predictions: {e}")
    
    # Clean tables in order (children first to avoid FK constraint errors)
    # Note: With ON DELETE CASCADE, we only need to delete from 'videos'
    # but doing explicit deletes gives us better logging
    tables_to_clean = [
        ('predictions', 'video_id'),
        ('training_samples_v2', 'video_id'),
        ('training_samples', 'video_id'),
        ('video_preprocess', 'video_id'),
        ('videos', 'video_id'),
    ]
    
    total_deleted = 0
    for table_name, column in tables_to_clean:
        deleted = clean_table(table_name, invalid_ids, column)
        total_deleted += deleted
    
    logger.info("=" * 80)
    logger.info(f"✅ Database cleaning completed: {total_deleted} total records deleted")
    logger.info("=" * 80)


def clear_kafka_topics():
    """Clear all Kafka topics."""
    logger.info("=" * 80)
    logger.info("CLEARING KAFKA TOPICS")
    logger.info("=" * 80)
    
    try:
        from kafka.admin import KafkaAdminClient, NewTopic
        import os
        import time
        
        bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        
        # Topics to clear
        topics_to_clear = [
            'video-preprocessing',
            'video-inference',
            'video-training',
            'video-status',
        ]
        
        admin_client = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id='cleanup-script'
        )
        
        # Delete topics
        try:
            admin_client.delete_topics(topics_to_clear, timeout_ms=10000)
            logger.info(f"✅ Deleted Kafka topics: {topics_to_clear}")
        except Exception as e:
            logger.warning(f"Could not delete topics (may not exist): {e}")
        
        # Wait a bit for deletion to propagate
        time.sleep(3)
        
        # Recreate topics
        new_topics = [
            NewTopic(name=topic, num_partitions=3, replication_factor=1)
            for topic in topics_to_clear
        ]
        
        try:
            admin_client.create_topics(new_topics, validate_only=False)
            logger.info(f"✅ Recreated Kafka topics: {topics_to_clear}")
        except Exception as e:
            logger.warning(f"Could not create topics (may already exist): {e}")
        
        admin_client.close()
        
    except ImportError:
        logger.warning("⚠️  kafka-python not installed, skipping Kafka cleanup")
    except Exception as e:
        logger.error(f"❌ Error clearing Kafka topics: {e}")


def show_database_stats():
    """Show current database statistics."""
    logger.info("\n--- Database Statistics ---")
    
    tables = [
        'videos',
        'video_preprocess',
        'training_samples',
        'training_samples_v2',
        'predictions',
        'sample_predictions',
        'model_registry',
        'training_metrics',
        'fusion_config',
    ]
    
    for table in tables:
        try:
            query = f"SELECT COUNT(*) as count FROM {table}"
            result = db.execute(query, fetch=True)
            count = result[0]['count'] if result else 0
            logger.info(f"  {table}: {count} records")
        except Exception as e:
            logger.info(f"  {table}: Error - {e}")


def main():
    """Main cleanup process."""
    parser = argparse.ArgumentParser(
        description='Database cleanup - keep only videos from labels.csv'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be deleted without actually deleting'
    )
    parser.add_argument(
        '--skip-kafka',
        action='store_true',
        help='Skip Kafka topic cleanup'
    )
    parser.add_argument(
        '--skip-backup',
        action='store_true',
        help='Skip database backup (not recommended)'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default=None,
        help='Path to labels.csv (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect CSV path
    csv_paths = [
        args.csv_path,
        '/opt/airflow/data/raw/labels.csv',
        '/app/data/raw/labels.csv',
        './data/raw/labels.csv',
    ]
    
    labels_csv_path = None
    for path in csv_paths:
        if path and os.path.exists(path):
            labels_csv_path = path
            break
    
    if not labels_csv_path:
        logger.error(f"❌ labels.csv not found. Tried: {csv_paths}")
        return 1
    
    # Auto-detect backup directory
    backup_base = '/opt/airflow/backups' if os.path.exists('/opt/airflow') else '/app/backups'
    backup_dir = Path(f'{backup_base}/db_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    logger.info("=" * 80)
    logger.info("DATABASE CLEANUP SCRIPT")
    logger.info("=" * 80)
    logger.info(f"Labels CSV: {labels_csv_path}")
    logger.info(f"Backup dir: {backup_dir}")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    logger.info("=" * 80)
    
    # Show current stats
    show_database_stats()
    
    # Read valid URLs from labels.csv
    valid_urls = read_labels_csv(labels_csv_path)
    
    # Backup database
    if not args.skip_backup and not args.dry_run:
        backup_database(backup_dir)
    elif args.dry_run:
        logger.info("\n[DRY RUN] Skipping backup")
    
    # Clean database
    clean_database(valid_urls, dry_run=args.dry_run)
    
    # Clear Kafka topics
    if not args.skip_kafka and not args.dry_run:
        clear_kafka_topics()
    elif args.dry_run:
        logger.info("\n[DRY RUN] Skipping Kafka cleanup")
    
    # Show final stats
    if not args.dry_run:
        logger.info("\n--- After Cleanup ---")
        show_database_stats()
    
    logger.info("=" * 80)
    logger.info("✅ CLEANUP COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
