"""
Data Ingestion DAG
Import videos from:
1. labels.csv into database and storage (for training)
2. YouTube search results for inference-only videos
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import logging
import sys

sys.path.insert(0, '/opt/airflow')

logger = logging.getLogger(__name__)

default_args = {
    "owner": "video_classifier",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def decide_ingestion_type(**context):
    """
    Decide which ingestion path to take based on DAG config.
    
    Returns:
        'ingest_from_csv' or 'ingest_from_youtube'
    """
    conf = context["dag_run"].conf or {}
    
    # If videos list is provided, it's from YouTube search
    if conf.get("videos"):
        logger.info("YouTube search results detected, routing to YouTube ingestion")
        return "ingest_from_youtube"
    else:
        logger.info("No videos in config, routing to CSV ingestion")
        return "ingest_from_csv"


def run_ingestion(**context):
    """Run data ingestion from labels.csv.
    
    Supports custom CSV path via DAG config: {"csv_path": "/path/to/file.csv"}
    """
    from common.data import DataIngestion
    
    conf = context["dag_run"].conf or {}
    auto_train = conf.get("auto_train", True)
    csv_path = conf.get("csv_path")  # Optional custom CSV path
    
    logger.info("=" * 80)
    logger.info("DATA INGESTION: Importing videos from CSV")
    if csv_path:
        logger.info(f"Using custom CSV path: {csv_path}")
    else:
        logger.info("Using default CSV path: labels.csv")
    logger.info("=" * 80)
    
    # Create DataIngestion with optional custom CSV path
    if csv_path:
        ingestion = DataIngestion(csv_path=csv_path)
    else:
        ingestion = DataIngestion()
    
    results = ingestion.ingest_all()
    
    logger.info(f"Ingestion completed: {results}")
    
    # Store results for downstream tasks
    context["ti"].xcom_push(key="ingestion_results", value=results)
    context["ti"].xcom_push(key="auto_train", value=auto_train)
    context["ti"].xcom_push(key="source", value="csv")
    
    return results


def ingest_from_youtube_search(**context):
    """
    Ingest videos from YouTube search results.
    
    This handles videos that need inference but aren't in the database yet.
    Videos are downloaded, uploaded to MinIO, and added to DB without labels.
    """
    from common.io import db, storage
    from common.data.youtube_downloader import download_and_ingest_videos
    
    conf = context["dag_run"].conf or {}
    videos = conf.get("videos", [])
    
    if not videos:
        logger.info("No videos provided in config")
        context["ti"].xcom_push(key="ingestion_results", value={"total": 0})
        return {"total": 0}
    
    logger.info("=" * 80)
    logger.info(f"YOUTUBE INGESTION: Processing {len(videos)} videos")
    logger.info("=" * 80)
    
    # Check which videos already exist in DB AND have been fully ingested (have storage_path)
    new_videos = []
    existing_videos = []
    
    for video in videos:
        video_url = video.get("videoUrl", "")
        if not video_url:
            continue
            
        existing = db.get_video(video_url)
        # Only consider as existing if video has storage_path (fully ingested)
        if existing and existing.get("storage_path"):
            logger.info(f"Video already ingested: {video_url[:50]}...")
            
            existing_videos.append({
                "video_url": video_url,
                "label": existing.get("label"),
                "status": existing.get("status_preprocess"),
            })
            
        else:
            # Either doesn't exist or exists but not fully ingested
            if existing:
                logger.info(f"Video in DB but not ingested (no storage_path), re-downloading: {video_url[:50]}...")
            new_videos.append(video)
    
    logger.info(f"Existing in DB: {len(existing_videos)}, New to download: {len(new_videos)}")
    
    # Download and ingest new videos
    results = {
        "total": len(videos),
        "existing": len(existing_videos),
        "new_videos": len(new_videos),
        "downloaded": 0,
        "ingested": 0,
        "failed": 0,
    }
    
    # Handle empty new_videos list
    if not new_videos:
        logger.info("No new videos to download and ingest")
        context["ti"].xcom_push(key="ingestion_results", value=results)
        context["ti"].xcom_push(key="total_videos", value=videos)
        context["ti"].xcom_push(key="new_videos", value=new_videos)
        context["ti"].xcom_push(key="existing_videos", value=existing_videos)
        context["ti"].xcom_push(key="source", value="internet")
        context["ti"].xcom_push(key="ingested_video_urls", value=[])
        logger.info(f"YouTube ingestion completed: {results}")
        return results
    
    # Download and ingest new videos
    download_results = download_and_ingest_videos(
        videos=new_videos,
        db_client=db,
        storage_client=storage,
    )
    results = {
        "total": results["total"],
        "existing": results["existing"],
        "new_videos": results["new_videos"],
        "downloaded": download_results.get("downloaded", 0),
        "ingested": download_results.get("ingested", 0),
        "failed": download_results.get("failed", 0),
    }
    
    # Store results
    context["ti"].xcom_push(key="ingestion_results", value=results)
    context["ti"].xcom_push(key="total_videos", value=videos)
    context["ti"].xcom_push(key="new_videos", value=new_videos)
    context["ti"].xcom_push(key="existing_videos", value=existing_videos)
    context["ti"].xcom_push(key="source", value="internet")
    
    # Store video URLs for preprocessing
    ingested_urls = [v.get("videoUrl") for v in new_videos[:results.get("ingested", 0)]]
    context["ti"].xcom_push(key="ingested_video_urls", value=ingested_urls)
    
    logger.info(f"YouTube ingestion completed: {results}")
    return results


def _safe_trigger_dag(dag_id: str, run_id_prefix: str, conf: dict):
    """Helper to trigger a DAG with error handling."""
    from airflow.api.common.trigger_dag import trigger_dag
    
    try:
        trigger_dag(
            dag_id=dag_id,
            run_id=f"{run_id_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            conf=conf,
            replace_microseconds=False
        )
        logger.info(f"DAG '{dag_id}' triggered successfully")
        return True
    except Exception as e:
        logger.warning(f"Could not trigger {dag_id}: {e}")
        return False


def _queue_preprocessed_videos(preprocessed_videos: list) -> int:
    """Mark preprocessed videos as ready for inference by updating DB status. Returns count of videos marked."""
    from common.io import db
    
    added_count = 0
    for v in preprocessed_videos:
        video_url = v["video_url"]
        samples = db.execute(
            "SELECT sample_id FROM training_samples_v2 WHERE video_id = %s AND split = 'inference'",
            (video_url,),
            fetch=True
        ) or []
        
        if samples:
            sample_id = samples[0]['sample_id']
            # Update status to pending_inference (inference DAG will pick it up)
            db.update_video_status(video_url, 'pending_inference')
            added_count += 1
            logger.info(f"Marked as pending_inference: {video_url[:30]}... (sample={sample_id})")
        else:
            logger.warning(f"No sample found for {video_url[:30]}..., marking as pending_preprocessing")
            # Mark as pending_preprocessing (preprocessing DAG will pick it up)
            db.update_video_status(video_url, 'pending_preprocessing')
    
    return added_count


def _handle_internet_trigger(ingested_count: int, existing_videos: list, total_videos: list):
    """Handle internet source: trigger preprocessing or inference based on video state."""
    PREPROCESS_CONF = {"source": "internet", "trigger_inference": True}
        
    # Case 3: Handle existing videos
    preprocessed_videos = [v for v in existing_videos if v.get("status") == "preprocessed"]
    
    
    # Case 4: Some videos already preprocessed - try to queue for inference
    added_count = _queue_preprocessed_videos(preprocessed_videos)
    
    if added_count > 0:
        logger.info(f"Added {added_count} videos to inference queue, triggering inference DAG")
        _safe_trigger_dag("inference_advanced_fusion", "inference_existing", 
                          {"source": "internet"})
    

    if added_count < len(total_videos):
        logger.info(f"{len(total_videos) - added_count} videos not preprocessed yet, triggering preprocessing DAG")
        _safe_trigger_dag("video_preprocessing_pipeline", "internet_reprocess", PREPROCESS_CONF)

    


def trigger_preprocessing(**context):
    """Trigger preprocessing pipeline if ingestion was successful."""
    results = context["ti"].xcom_pull(key="ingestion_results")
    source = context["ti"].xcom_pull(key="source")
    
    if source == "internet":
        ingested_count = results.get("ingested", 0)
        existing_videos = context["ti"].xcom_pull(key="existing_videos") or []
        total_videos = context["ti"].xcom_pull(key="total_videos") or []

        _handle_internet_trigger(ingested_count, existing_videos, total_videos)
    
    elif source == "csv":
        success_count = results.get("success", 0)
        if success_count > 0:
            logger.info(f"Triggering preprocessing for {success_count} CSV videos")
            _safe_trigger_dag("video_preprocessing_pipeline", "csv_ingestion", {
                "source": "csv",
                "trigger_training": context["ti"].xcom_pull(key="auto_train"),
            })
        else:
            logger.info("No CSV videos to process")


with DAG(
    dag_id="data_ingestion",
    default_args=default_args,
    description="Import videos from CSV or YouTube search for training/inference",
    schedule_interval=None,  # Triggered manually via API
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "training", "data", "internet", "youtube", "tiktok"],
    max_active_runs=1,
) as dag:
    
    # Branching: decide ingestion type
    decide_task = BranchPythonOperator(
        task_id="decide_ingestion_type",
        python_callable=decide_ingestion_type,
    )
    
    # Path 1: CSV ingestion (for training data)
    csv_ingest_task = PythonOperator(
        task_id="ingest_from_csv",
        python_callable=run_ingestion,
    )
    
    # Path 2: Internet ingestion (for inference from YouTube/TikTok)
    youtube_ingest_task = PythonOperator(
        task_id="ingest_from_youtube",
        python_callable=ingest_from_youtube_search,
    )
    
    # Join paths
    join_task = EmptyOperator(
        task_id="join_ingestion",
        trigger_rule="none_failed_min_one_success",
    )
    
    # Trigger preprocessing
    preprocess_task = PythonOperator(
        task_id="trigger_preprocessing",
        python_callable=trigger_preprocessing,
    )
    
    # Define dependencies
    decide_task >> [csv_ingest_task, youtube_ingest_task]
    [csv_ingest_task, youtube_ingest_task] >> join_task >> preprocess_task
