"""
Data Cleanup DAG - Clear data from MinIO and PostgreSQL.

Use with caution! This DAG deletes data permanently.

Trigger with config:
{
    "clear_videos": true,        # Clear videos table and MinIO videos
    "clear_samples": true,       # Clear training samples v2 and MinIO samples
    "clear_frames": true,        # Clear MinIO frames directory
    "clear_predictions": true,   # Clear predictions table
    "clear_models": true,        # Clear model registry and MinIO models
    "clear_all": false           # Clear everything (overrides other options)
}
"""

import os
import sys
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, '/opt/airflow')

from common.io import db, storage

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}


def clear_videos(**context):
    """Clear videos table and MinIO raw videos."""
    conf = context.get('dag_run').conf or {}
    
    if not (conf.get('clear_videos') or conf.get('clear_all')):
        logger.info("Skipping videos cleanup (not requested)")
        return 0
    
    logger.info("=" * 60)
    logger.info("CLEARING VIDEOS")
    logger.info("=" * 60)
    
    deleted_db = 0
    deleted_minio = 0
    
    try:
        # Get all videos first
        videos = db.execute("SELECT video_id, storage_path FROM videos", fetch=True)
        logger.info(f"Found {len(videos)} videos in database")
        
        # Delete from MinIO
        for video in videos:
            storage_path = video.get('storage_path')
            if storage_path:
                try:
                    storage.delete_file(storage_path)
                    deleted_minio += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {storage_path}: {e}")
        
        # Clear raw/ directory in MinIO
        try:
            objects = list(storage.client.list_objects(storage.bucket, prefix='raw/', recursive=True))
            for obj in objects:
                storage.client.remove_object(storage.bucket, obj.object_name)
                deleted_minio += 1
        except Exception as e:
            logger.warning(f"Failed to clear raw/: {e}")
        
        # Delete from database (cascade will handle related tables)
        db.execute("DELETE FROM videos")
        deleted_db = len(videos)
        
        logger.info(f"Deleted {deleted_db} videos from DB, {deleted_minio} files from MinIO")
        
    except Exception as e:
        logger.error(f"Error clearing videos: {e}")
        raise
    
    return deleted_db


def clear_samples(**context):
    """Clear training samples v2 and MinIO samples directory."""
    conf = context.get('dag_run').conf or {}
    
    if not (conf.get('clear_samples') or conf.get('clear_all')):
        logger.info("Skipping samples cleanup (not requested)")
        return 0
    
    logger.info("=" * 60)
    logger.info("CLEARING TRAINING SAMPLES")
    logger.info("=" * 60)
    
    deleted_db = 0
    deleted_minio = 0
    
    try:
        # Count samples
        count = db.execute("SELECT COUNT(*) as cnt FROM training_samples_v2", fetch=True)
        sample_count = count[0]['cnt'] if count else 0
        logger.info(f"Found {sample_count} samples in database")
        
        # Clear samples/ directory in MinIO
        try:
            objects = list(storage.client.list_objects(storage.bucket, prefix='samples/', recursive=True))
            logger.info(f"Found {len(objects)} objects in samples/")
            for obj in objects:
                storage.client.remove_object(storage.bucket, obj.object_name)
                deleted_minio += 1
        except Exception as e:
            logger.warning(f"Failed to clear samples/: {e}")
        
        # Delete from database
        db.execute("DELETE FROM training_samples_v2")
        db.execute("DELETE FROM training_samples")
        deleted_db = sample_count
        
        logger.info(f"Deleted {deleted_db} samples from DB, {deleted_minio} files from MinIO")
        
    except Exception as e:
        logger.error(f"Error clearing samples: {e}")
        raise
    
    return deleted_db


def clear_frames(**context):
    """Clear MinIO frames directory."""
    conf = context.get('dag_run').conf or {}
    
    if not (conf.get('clear_frames') or conf.get('clear_all')):
        logger.info("Skipping frames cleanup (not requested)")
        return 0
    
    logger.info("=" * 60)
    logger.info("CLEARING FRAMES")
    logger.info("=" * 60)
    
    deleted = 0
    
    try:
        objects = list(storage.client.list_objects(storage.bucket, prefix='frames/', recursive=True))
        logger.info(f"Found {len(objects)} frame files")
        
        for obj in objects:
            storage.client.remove_object(storage.bucket, obj.object_name)
            deleted += 1
            
            if deleted % 1000 == 0:
                logger.info(f"  Deleted {deleted} files...")
        
        # Also clear video_preprocess table
        db.execute("DELETE FROM video_preprocess")
        
        logger.info(f"Deleted {deleted} frame files from MinIO")
        
    except Exception as e:
        logger.error(f"Error clearing frames: {e}")
        raise
    
    return deleted


def clear_predictions(**context):
    """Clear predictions table."""
    conf = context.get('dag_run').conf or {}
    
    if not (conf.get('clear_predictions') or conf.get('clear_all')):
        logger.info("Skipping predictions cleanup (not requested)")
        return 0
    
    logger.info("=" * 60)
    logger.info("CLEARING PREDICTIONS")
    logger.info("=" * 60)
    
    try:
        count = db.execute("SELECT COUNT(*) as cnt FROM predictions", fetch=True)
        pred_count = count[0]['cnt'] if count else 0
        
        db.execute("DELETE FROM predictions")
        
        logger.info(f"Deleted {pred_count} predictions from DB")
        return pred_count
        
    except Exception as e:
        logger.error(f"Error clearing predictions: {e}")
        raise


def clear_models(**context):
    """Clear model registry and MinIO models."""
    conf = context.get('dag_run').conf or {}
    
    if not (conf.get('clear_models') or conf.get('clear_all')):
        logger.info("Skipping models cleanup (not requested)")
        return 0
    
    logger.info("=" * 60)
    logger.info("CLEARING MODELS")
    logger.info("=" * 60)
    
    deleted_db = 0
    deleted_minio = 0
    
    try:
        # Count models
        count = db.execute("SELECT COUNT(*) as cnt FROM model_registry", fetch=True)
        model_count = count[0]['cnt'] if count else 0
        
        # Clear models/ directory in MinIO
        try:
            objects = list(storage.client.list_objects(storage.bucket, prefix='models/', recursive=True))
            logger.info(f"Found {len(objects)} model files")
            for obj in objects:
                storage.client.remove_object(storage.bucket, obj.object_name)
                deleted_minio += 1
        except Exception as e:
            logger.warning(f"Failed to clear models/: {e}")
        
        # Delete from database
        db.execute("DELETE FROM model_registry")
        deleted_db = model_count
        
        logger.info(f"Deleted {deleted_db} models from DB, {deleted_minio} files from MinIO")
        
    except Exception as e:
        logger.error(f"Error clearing models: {e}")
        raise
    
    return deleted_db


def clear_cache(**context):
    """Clear embedding cache directory in MinIO."""
    conf = context.get('dag_run').conf or {}
    
    if not (conf.get('clear_cache') or conf.get('clear_all')):
        logger.info("Skipping cache cleanup (not requested)")
        return 0
    
    logger.info("=" * 60)
    logger.info("CLEARING CACHE")
    logger.info("=" * 60)
    
    deleted = 0
    
    try:
        # Clear cache/ directory
        objects = list(storage.client.list_objects(storage.bucket, prefix='cache/', recursive=True))
        logger.info(f"Found {len(objects)} cache files")
        
        for obj in objects:
            storage.client.remove_object(storage.bucket, obj.object_name)
            deleted += 1
        
        logger.info(f"Deleted {deleted} cache files from MinIO")
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise
    
    return deleted


def print_summary(**context):
    """Print cleanup summary."""
    conf = context.get('dag_run').conf or {}
    
    logger.info("=" * 60)
    logger.info("CLEANUP SUMMARY")
    logger.info("=" * 60)
    
    # Get current counts
    try:
        video_count = db.execute("SELECT COUNT(*) as cnt FROM videos", fetch=True)[0]['cnt']
        sample_count = db.execute("SELECT COUNT(*) as cnt FROM training_samples_v2", fetch=True)[0]['cnt']
        pred_count = db.execute("SELECT COUNT(*) as cnt FROM predictions", fetch=True)[0]['cnt']
        model_count = db.execute("SELECT COUNT(*) as cnt FROM model_registry", fetch=True)[0]['cnt']
        
        logger.info(f"Remaining videos:     {video_count}")
        logger.info(f"Remaining samples:    {sample_count}")
        logger.info(f"Remaining predictions: {pred_count}")
        logger.info(f"Remaining models:     {model_count}")
        
        # Count MinIO objects
        try:
            raw_objs = len(list(storage.client.list_objects(storage.bucket, prefix='raw/', recursive=True)))
            sample_objs = len(list(storage.client.list_objects(storage.bucket, prefix='samples/', recursive=True)))
            frame_objs = len(list(storage.client.list_objects(storage.bucket, prefix='frames/', recursive=True)))
            model_objs = len(list(storage.client.list_objects(storage.bucket, prefix='models/', recursive=True)))
            
            logger.info(f"MinIO raw/:     {raw_objs} objects")
            logger.info(f"MinIO samples/: {sample_objs} objects")
            logger.info(f"MinIO frames/:  {frame_objs} objects")
            logger.info(f"MinIO models/:  {model_objs} objects")
        except Exception as e:
            logger.warning(f"Could not count MinIO objects: {e}")
        
    except Exception as e:
        logger.error(f"Error getting counts: {e}")
    
    logger.info("=" * 60)
    logger.info("✅ CLEANUP COMPLETE")
    logger.info("=" * 60)


# ============================================================================
# DAG DEFINITION
# ============================================================================
with DAG(
    'data_cleanup',
    default_args=default_args,
    description='Clear data from MinIO and PostgreSQL',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=['maintenance', 'cleanup', 'admin'],
) as dag:
    
    task_videos = PythonOperator(
        task_id='clear_videos',
        python_callable=clear_videos,
    )
    
    task_samples = PythonOperator(
        task_id='clear_samples',
        python_callable=clear_samples,
    )
    
    task_frames = PythonOperator(
        task_id='clear_frames',
        python_callable=clear_frames,
    )
    
    task_predictions = PythonOperator(
        task_id='clear_predictions',
        python_callable=clear_predictions,
    )
    
    task_models = PythonOperator(
        task_id='clear_models',
        python_callable=clear_models,
    )
    
    task_cache = PythonOperator(
        task_id='clear_cache',
        python_callable=clear_cache,
    )
    
    task_summary = PythonOperator(
        task_id='print_summary',
        python_callable=print_summary,
        trigger_rule='all_done',
    )
    
    # All cleanup tasks run in parallel, then summary
    [task_videos, task_samples, task_frames, task_predictions, task_models, task_cache] >> task_summary
