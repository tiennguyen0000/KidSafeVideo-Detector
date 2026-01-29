"""
Advanced Fusion Inference Pipeline DAG.

Scans PostgreSQL for videos pending inference and runs predictions using the unified fusion model.
Replaces legacy 4-model inference for better speed and accuracy.

Workflow:
    1. Load fusion models (text encoder + image encoder + fusion model)
    2. Scan PostgreSQL for videos with status='pending_inference'
    3. Process batch of videos (lock, predict, save to DB, update status to 'classified')

Expected performance:
    - 2x faster than legacy (40-90ms vs 80-120ms per video)
    - +2-6% accuracy improvement
    - Single unified model (10-50MB vs 50-100MB)

Spark Integration:
    - If USE_SPARK=true and batch_size >= 20, uses Spark for parallel inference
    - Otherwise falls back to sequential processing

Database-driven:
    - Uses PostgreSQL status_preprocess field to track inference tasks
    - No Kafka dependency (only used for video search → ingestion)
"""

import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
import logging

# Setup logger
logger = logging.getLogger(__name__)


def is_spark_enabled() -> bool:
    """Check if Spark is enabled via environment."""
    return os.environ.get('USE_SPARK', 'true').lower() == 'true'


# Import after airflow context is ready
def get_clients():
    """Lazy import to avoid circular dependencies."""
    from common.io import db, storage, queue, config
    from common.pipelines.inference_advanced_fusion import (
        load_advanced_fusion_models,
        predict_advanced_fusion,
        get_cached_models,
        clear_model_cache
    )
    return db, storage, queue, config, load_advanced_fusion_models, predict_advanced_fusion, get_cached_models, clear_model_cache


# ============================================================================
# TASK FUNCTIONS
# ============================================================================

# Note: wait_for_inference_tasks sensor REMOVED - DAG is triggered by preprocessing
# when messages are pushed to Kafka. No polling needed, prevents message loss.


def load_models(**context):
    """
    Load advanced fusion models and cache them.
    
    Stores models in XCom for reuse by downstream tasks.
    """
    db, _, _, config, load_advanced_fusion_models, _, get_cached_models, _ = get_clients()
    
    mode = context['dag_run'].conf.get('mode', config.get('system.mode', 'ultra_light'))
    logger.info(f"Loading advanced fusion models for mode={mode}")
    
    # Check cache first
    cached_models = get_cached_models(mode)
    if cached_models:
        logger.info("Using cached models")
        return {
            'status': 'loaded_from_cache',
            'mode': mode,
            'fusion_type': cached_models.get('fusion_type', 'gated')
        }
    
    # Load models from DB + MinIO
    try:
        models = load_advanced_fusion_models(mode)
        logger.info(f"✅ Loaded {models['fusion_type']} fusion model successfully")
        
        return {
            'status': 'loaded',
            'mode': mode,
            'fusion_type': models['fusion_type'],
            'd_img': models.get('text_encoder').embedding_dim if hasattr(models['text_encoder'], 'embedding_dim') else 'N/A',
            'd_txt': models.get('image_encoder').embedding_dim if hasattr(models['image_encoder'], 'embedding_dim') else 'N/A'
        }
    
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


def process_inference_batch(**context):
    """
    Process a batch of inference tasks from PostgreSQL.
    
    For each video:
        1. Scan DB for videos with status='pending_inference'
        2. Lock videos (update to 'processing')
        3. Run prediction with fusion model
        4. Save prediction to database
        5. Update video status to 'classified'
    
    Batch size: 10 videos (configurable)
    
    If USE_SPARK=true and batch_size >= 20, uses Spark for parallel processing.
    """
    db, storage, queue, config, _, predict_advanced_fusion, get_cached_models, _ = get_clients()
    
    mode = context['dag_run'].conf.get('mode', config.get('system.mode', 'ultra_light'))
    batch_size = context['dag_run'].conf.get('batch_size', 10)
    
    logger.info(f"Processing inference batch (mode={mode}, batch_size={batch_size})")
    
    # =========================================================================
    # Collect tasks from PostgreSQL (videos with status='pending_inference')
    # =========================================================================
    logger.info(f"Scanning PostgreSQL for videos with status='pending_inference'...")
    
    # Debug: Check videos with pending_inference status
    debug_query = "SELECT COUNT(*) as count FROM videos WHERE status_preprocess = 'pending_inference'"
    debug_result = db.execute(debug_query, fetch=True)
    if debug_result:
        logger.info(f"Debug: Found {debug_result[0].get('count', 0)} videos with status='pending_inference' in videos table")
    
    # Debug: Check inference samples
    debug_samples_query = "SELECT COUNT(*) as count FROM training_samples_v2 WHERE split = 'inference'"
    debug_samples_result = db.execute(debug_samples_query, fetch=True)
    if debug_samples_result:
        logger.info(f"Debug: Found {debug_samples_result[0].get('count', 0)} inference samples in training_samples_v2")
    
    videos = db.get_videos_pending_inference(limit=batch_size)
    
    if not videos:
        logger.warning("No videos pending inference found by get_videos_pending_inference()")
        logger.info("This could mean:")
        logger.info("  1. Videos don't have status='pending_inference'")
        logger.info("  2. Videos don't have inference samples in training_samples_v2")
        logger.info("  3. Preprocessing hasn't completed yet")
        return {'processed': 0, 'failed': 0, 'mode': mode}
    
    logger.info(f"Found {len(videos)} videos pending inference")
    
    # Lock videos (update status to 'processing')
    video_ids = []
    tasks = []
    for video in videos:
        video_id = video['video_id']
        sample_id = video.get('sample_id')
        
        if not sample_id:
            logger.warning(f"Video {video_id} has no sample_id, skipping")
            continue
        
        try:
            db.update_video_status(video_id, 'processing')
            video_ids.append(video_id)
            tasks.append({
                'video_id': video_id,
                'sample_id': sample_id,
            })
            logger.info(f"  Locked video: {video_id[:50]}... (sample={sample_id})")
        except Exception as e:
            logger.error(f"Failed to lock video {video_id}: {e}")
    
    if not tasks:
        logger.warning("No valid tasks after locking videos")
        return {'processed': 0, 'failed': 0, 'mode': mode}
    
    # =========================================================================
    # Try Spark processing (similar to training - always try if enabled)
    # =========================================================================
    if is_spark_enabled():
        try:
            logger.info("💡 Using Spark for parallel inference (similar to training)")
            from common.spark import SparkBatchProcessor
            
            sample_ids = [t.get('sample_id') for t in tasks if t.get('sample_id')]
            
            if not sample_ids:
                logger.warning("No sample_ids found, cannot use Spark")
            else:
                storage_config = {
                    'host': os.environ.get('MINIO_HOST', 'minio'),
                    'port': int(os.environ.get('MINIO_PORT', '9000')),
                    'access_key': os.environ.get('MINIO_ACCESS_KEY', 'minioadmin'),
                    'secret_key': os.environ.get('MINIO_SECRET_KEY', 'minioadmin'),
                    'bucket': os.environ.get('MINIO_BUCKET', 'video-classifier'),
                }
                
                # Get model path
                active_models = db.get_active_models(mode)
                fusion_model = active_models.get('fusion') or active_models.get('fusion_gated')
                
                if not fusion_model:
                    # Try to find any fusion model
                    fusion_models = {k: v for k, v in active_models.items() if k.startswith('fusion')}
                    if fusion_models:
                        fusion_model = list(fusion_models.values())[0]
                        logger.info(f"Using fusion model: {list(fusion_models.keys())[0]}")
                
                if fusion_model:
                    processor = SparkBatchProcessor()
                    processor.set_config(storage_config, mode)
                    
                    # Calculate optimal partitions (similar to training logic)
                    # Use more partitions for larger batches
                    num_partitions = min(8, max(1, len(sample_ids) // 8))
                    batch_size = 32 if len(sample_ids) >= 20 else 16
                    
                    logger.info(f"Spark config: {len(sample_ids)} samples, {num_partitions} partitions, batch_size={batch_size}")
                    
                    results = processor.batch_inference(
                        sample_ids=sample_ids,
                        model_path=fusion_model['artifact_path'],
                        batch_size=batch_size,
                        num_partitions=num_partitions
                    )
                    
                    # Save predictions
                    processed = 0
                    failed_count = 0
                    for pred in results:
                        try:
                            # Find video_id for this sample
                            task = next((t for t in tasks if t.get('sample_id') == pred.get('sample_id')), None)
                            
                            if not task:
                                logger.warning(f"Could not find task for sample_id: {pred.get('sample_id')}")
                                failed_count += 1
                                continue
                            
                            video_id = task.get('video_id')
                            
                            if pred.get('status') != 'success':
                                logger.warning(f"Prediction failed for {video_id}: {pred.get('error', 'unknown error')}")
                                db.update_video_status(video_id, 'error')
                                failed_count += 1
                                continue
                            
                            db.insert_prediction(
                                video_id=video_id,
                                mode=mode,
                                y_pred=pred['y_pred'],
                                p_text={},
                                p_img={},
                                p_final=pred.get('probabilities', {}),
                                confidence=pred.get('confidence', 0.0)
                            )
                            db.update_video_status(video_id, 'classified')
                            processed += 1
                            logger.debug(f"✅ Spark processed: {video_id[:50]}... → {pred['y_pred']}")
                        except Exception as e:
                            logger.error(f"Failed to save prediction: {e}", exc_info=True)
                            failed_count += 1
                    
                    logger.info(f"✅ Spark inference complete: {processed} succeeded, {failed_count} failed out of {len(tasks)} total")
                    return {'processed': processed, 'failed': failed_count, 'mode': mode}
                else:
                    logger.warning("No fusion model found, cannot use Spark inference")
                    raise ValueError("No fusion model found")
        
        except Exception as e:
            logger.warning(f"⚠️ Spark inference failed, falling back to sequential: {e}")
            logger.debug(f"Spark error details: {e}", exc_info=True)
    
    # =========================================================================
    # Sequential processing (fallback)
    # =========================================================================
    models = get_cached_models(mode)
    if not models:
        logger.error("Models not loaded! Run load_models task first.")
        raise ValueError("Models not loaded")
    
    processed = 0
    failed = 0
    
    for i, task in enumerate(tasks):
        video_id = task.get('video_id')
        sample_id = task.get('sample_id')
        
        logger.info(f"[{i+1}/{len(tasks)}] Processing: video={video_id}, sample={sample_id}")
        
        try:
            result = predict_advanced_fusion(video_id, models, sample_id=sample_id)
            
            if not result:
                raise ValueError(f"Prediction returned None for {video_id}")
            
            db.insert_prediction(
                video_id=video_id,
                mode=mode,
                y_pred=result['prediction'],
                p_text={},
                p_img={},
                p_final=result.get('class_probs', {}),
                confidence=result['confidence']
            )
            
            db.update_video_status(video_id, 'classified')
            
            logger.info(f"✅ {video_id}: {result['prediction']} (conf={result['confidence']:.3f})")
            processed += 1
        
        except Exception as e:
            logger.error(f"❌ Failed to process {video_id}: {e}")
            failed += 1
            
            try:
                db.update_video_status(video_id, 'error')
            except:
                pass
    
    logger.info(f"Batch complete: {processed} succeeded, {failed} failed")
    
    # =========================================================================
    # CHECK FOR MORE PENDING VIDEOS AND TRIGGER ANOTHER RUN
    # =========================================================================
    remaining_query = """
        SELECT COUNT(*) as count FROM videos 
        WHERE status_preprocess = 'pending_inference'
    """
    try:
        result = db.execute(remaining_query, fetch=True)
        remaining_count = result[0]['count'] if result else 0
        
        if remaining_count > 0:
            logger.info(f"⚠️  Still {remaining_count} videos pending inference. Triggering another run...")
            
            from airflow.api.common.trigger_dag import trigger_dag as trigger_dag_run
            run_id = f"auto_continue_{datetime.now().isoformat()}"
            
            try:
                trigger_dag_run(
                    dag_id='inference_advanced_fusion',
                    run_id=run_id,
                    conf={"mode": mode, "batch_size": min(remaining_count, batch_size), "auto_triggered": True}
                )
                logger.info(f"✅ Triggered next inference run for {remaining_count} remaining videos")
            except Exception as e:
                logger.warning(f"Could not auto-trigger next run: {e}")
                logger.info("You may need to manually trigger the inference DAG again")
        else:
            logger.info("✅ All videos classified! No pending inference remaining.")
    except Exception as e:
        logger.warning(f"Could not check remaining videos: {e}")
    
    return {
        'processed': processed,
        'failed': failed,
        'mode': mode
    }


def cleanup_models(**context):
    """
    Optional cleanup: Clear model cache if needed.
    
    Usually not needed as models stay cached for performance.
    """
    _, _, _, _, _, _, _, clear_model_cache = get_clients()
    
    # Only clear cache if explicitly requested
    if context['dag_run'].conf.get('clear_cache', False):
        clear_model_cache()
        logger.info("Model cache cleared")
    else:
        logger.info("Model cache retained for next run")


# ============================================================================
# DAG DEFINITION
# ============================================================================

default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=30),
}

with DAG(
    'inference_advanced_fusion',
    default_args=default_args,
    description='Advanced fusion inference pipeline - Single unified model',
    schedule_interval=None,  # Triggered manually or by preprocessing
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['inference', 'ml', 'advanced_fusion', 'production'],
) as dag:
    
    # Task 1: Load models
    task_load = PythonOperator(
        task_id='load_models',
        python_callable=load_models,
    )
    
    # Task 2: Process batch
    task_process = PythonOperator(
        task_id='process_inference_batch',
        python_callable=process_inference_batch,
    )
    
    # Task 3: Cleanup (optional)
    task_cleanup = PythonOperator(
        task_id='cleanup_models',
        python_callable=cleanup_models,
        trigger_rule='all_done',  # Run even if previous tasks failed
    )
    
    # Define dependencies
    task_load >> task_process >> task_cleanup