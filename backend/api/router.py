"""
FastAPI Routers for Video Classifier API.

This API service is a lightweight proxy that:
1. Routes requests to Airflow DAGs for heavy processing
2. Queries database for results
3. Provides health checks and statistics

All heavy processing (video search, inference, training) is done by Airflow workers.
"""
import os
import logging
from datetime import datetime
from typing import Optional
import httpx

from fastapi import APIRouter, HTTPException, Form
from pydantic import BaseModel

from common.io import db, queue

logger = logging.getLogger(__name__)

router = APIRouter()

APP_BASE_DIR = os.environ.get("APP_BASE_DIR", "/app")
AIRFLOW_API_URL = os.environ.get("AIRFLOW_API_URL", "http://airflow:8080")
AIRFLOW_USERNAME = os.environ.get("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "admin")


# ===== Helper Functions =====

async def trigger_airflow_dag(dag_id: str, conf: dict = None) -> dict:
    """
    Trigger Airflow DAG via REST API.
    
    Args:
        dag_id: DAG ID to trigger
        conf: Optional configuration dict to pass to the DAG
        
    Returns:
        Response from Airflow API
    """
    url = f"{AIRFLOW_API_URL}/api/v1/dags/{dag_id}/dagRuns"
    
    payload = {
        "dag_run_id": f"api_trigger_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
    }
    if conf:
        payload["conf"] = conf
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
                headers={"Content-Type": "application/json"},
                timeout=10.0,
            )
            
            if response.status_code in (200, 201):
                return {"status": "triggered", "response": response.json()}
            else:
                logger.warning(f"Airflow API returned {response.status_code}: {response.text}")
                return {"status": "failed", "error": response.text}
                
    except Exception as e:
        logger.warning(f"Could not trigger Airflow DAG {dag_id}: {e}")
        return {"status": "failed", "error": str(e)}


async def call_airflow_api(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """
    Call Airflow REST API endpoint.
    
    Args:
        endpoint: API endpoint (e.g., "/api/v1/dags")
        method: HTTP method
        data: Request body
        
    Returns:
        Response from Airflow API
    """
    url = f"{AIRFLOW_API_URL}{endpoint}"
    
    try:
        async with httpx.AsyncClient() as client:
            if method == "GET":
                response = await client.get(
                    url,
                    auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
                    timeout=10.0,
                )
            else:
                response = await client.post(
                    url,
                    json=data,
                    auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
                    headers={"Content-Type": "application/json"},
                    timeout=30.0,
                )
            
            return {"status_code": response.status_code, "data": response.json()}
                
    except Exception as e:
        logger.error(f"Error calling Airflow API: {e}")
        return {"status_code": 500, "error": str(e)}


# ===== Request/Response Models =====

class SearchVideosRequest(BaseModel):
    keyword: str
    video_type: str = "short"  # 'regular' hoặc 'short' (áp dụng cho YouTube)
    max_results: int = 24


class InferenceRequest(BaseModel):
    video_id: str
    pipeline: str = "local"  # 'local' hoặc 'colab'


# ===== Health & Status Endpoints =====

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    db_status = "disconnected"
    try:
        db.connect()
        db_status = "connected"
    except Exception as e:
        logger.warning(f"Database not available: {e}")
    
    return {
        "status": "healthy",
        "mode": os.environ.get("MODEL_MODE", "ultra_light"),
        "database": db_status,
    }


@router.get("/api/statistics")
async def get_statistics():
    """Get system statistics."""
    try:
        stats = db.get_statistics()
        
        # Note: Inference queue is now in Kafka, not easily queryable
        # Use Kafka admin API if needed, or remove this metric
        stats["inference_queue_length"] = None  # Kafka doesn't provide easy queue length
        
        return stats
    except Exception as e:
        logger.warning(f"Database unavailable for statistics: {e}")
        return {
            "total_videos": 0,
            "total_predictions": 0,
            "videos_by_status": {},
            "videos_by_label": {},
            "inference_queue_length": 0,
            "database_status": "unavailable",
        }


# ===== Video Search (proxied to Airflow) =====

@router.post("/api/search-videos")
async def search_videos(request: SearchVideosRequest):
    """
    Search videos from YouTube.
    Triggers Airflow DAG to perform the actual search.
    """
    # Trigger video search DAG (searches both YouTube and TikTok)
    conf = {
        "keyword": request.keyword,
        "video_type": request.video_type,
        "max_results": request.max_results
    }
    
    result = await trigger_airflow_dag("video_search", conf)
    
    if result["status"] == "triggered":
        dag_run_id = result.get("response", {}).get("dag_run_id")
        return {
            "status": "searching",
            "message": "Video search started",
            "dag_run_id": dag_run_id,
            "check_results_at": f"/api/search-results/{dag_run_id}"
        }
    else:
        # Fallback: return empty results if DAG trigger fails
        return {
            "status": "error",
            "message": "Could not start video search",
            "error": result.get("error"),
            "videos": []
        }


@router.get("/api/search-results/{dag_run_id}")
async def get_search_results(dag_run_id: str):
    """Get video search results from a previous search."""
    try:
        # Query database for cached results (cross-container via PostgreSQL)
        results = queue.get_search_results(dag_run_id)
        if results:
            return {"status": "completed", "videos": results}
        
        return {"status": "pending", "message": "Search still in progress"}
    except Exception as e:
        logger.error(f"Error getting search results: {e}")
        return {"status": "error", "error": str(e)}


# ===== Inference Endpoints =====

class InferenceBatchRequest(BaseModel):
    """Request for batch inference on multiple videos."""
    videos: list  # List of video dicts from search results
    pipeline: str = "local"


@router.post("/api/inference/batch")
async def run_inference_batch(request: InferenceBatchRequest):
    """
    Trigger inference for a batch of videos from YouTube search.
    
    This endpoint:
    1. Checks which videos already exist in DB (feature store)
    2. For existing videos with labels: return prediction immediately
    3. For new videos: trigger ingestion DAG to download + preprocess + inference
    
    Returns:
        - existing_predictions: Videos with labels already in DB
        - queued_for_processing: Videos that need download/inference
    """
    try:
        videos = request.videos
        mode = "ultra_light" if request.pipeline == "local" else "balanced"
        
        existing_predictions = []
        need_processing = []
        need_inference = []  # Videos already preprocessed, just need inference
        
        for video in videos:
            video_url = video.get("videoUrl", "")
            if not video_url:
                continue
            
            video_id_frontend = video.get("id")  # Frontend video ID (e.g., yt_XXX)
            
            # Check if video exists in DB
            existing_video = db.get_video(video_url)
            
            if existing_video:
                # Video exists in DB
                label = existing_video.get("label")
                
                if label:
                    # Has label (training data) - return immediately as prediction
                    existing_predictions.append({
                        "video_id": video_id_frontend,
                        "video_url": video_url,
                        "status": "completed",
                        "prediction": label,
                        "confidence": 1.0,  # Ground truth
                        "source": "feature_store"
                    })
                else:
                    # No label - check if prediction exists
                    prediction = db.get_latest_prediction(video_url, mode)
                    if prediction:
                        existing_predictions.append({
                            "video_id": video_id_frontend,
                            "video_url": video_url,
                            "status": "completed",
                            "prediction": prediction["y_pred"],
                            "confidence": prediction["confidence"],
                            "source": "inference_cache"
                        })
                    else:
                        # Has video but no prediction - check if preprocessed
                        samples = db.execute(
                            "SELECT sample_id FROM training_samples_v2 WHERE video_id = %s AND split = 'inference'",
                            (video_url,),
                            fetch=True
                        ) or []
                        
                        if samples:
                            # Video already preprocessed - just needs inference
                            # Mark as pending_inference (ingestion DAG will trigger inference)
                            current_status = existing_video.get('status_preprocess', '')
                            if current_status != 'pending_inference':
                                db.update_video_status(video_url, 'pending_inference')
                                logger.info(f"Marked preprocessed video as pending_inference: {video_url[:50]}...")
                            
                            need_inference.append(video)
                        else:
                            # Has video but no preprocessing - need full processing
                            need_processing.append(video)
            else:
                # Video not in DB - need full processing
                need_processing.append(video)
        
        # Combine all videos that need processing (both new and preprocessed)
        all_queued_videos = need_processing + need_inference
        
        # If there are videos that need processing, trigger ingestion DAG
        dag_run_id = None
        if need_processing:
            conf = {
                "videos": need_processing,
                "mode": mode,
            }
            result = await trigger_airflow_dag("data_ingestion", conf)
            
            if result["status"] == "triggered":
                dag_run_id = result.get("response", {}).get("dag_run_id")
                logger.info(f"Triggered ingestion for {len(need_processing)} videos: {dag_run_id}")
        
        # If there are preprocessed videos needing inference, trigger inference DAG directly
        if need_inference:
            logger.info(f"Found {len(need_inference)} preprocessed videos ready for inference")
            # Trigger inference DAG directly for preprocessed videos
            inference_conf = {
                "mode": mode,
                "batch_size": len(need_inference),
                "source": "api_batch"
            }
            inference_result = await trigger_airflow_dag("inference_advanced_fusion", inference_conf)
            if inference_result["status"] == "triggered":
                logger.info(f"Triggered inference DAG for {len(need_inference)} preprocessed videos")
        
        # Combine all queued video IDs (both new and preprocessed) for frontend polling
        all_queued_ids = [v.get("id") for v in all_queued_videos]
        
        return {
            "status": "processing",
            "existing_predictions": existing_predictions,
            "queued_count": len(all_queued_videos),
            "queued_videos": all_queued_ids,
            "dag_run_id": dag_run_id,
            "message": f"Found {len(existing_predictions)} cached, {len(need_processing)} new videos, {len(need_inference)} preprocessed videos queued"
        }
        
    except Exception as e:
        logger.error(f"Error in batch inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/inference")
async def run_inference(request: InferenceRequest):
    """
    Trigger inference for a single video.
    
    First checks if video exists in DB:
    - If has label: return immediately (feature store)
    - If has prediction: return cached prediction
    - If has sample_id: queue for inference with sample_id
    - Otherwise: return pending status (video may be in ingestion/preprocessing pipeline)
    """
    try:
        video_id_request = request.video_id  # Original ID from frontend (e.g., yt_XXX)
        mode = "ultra_light" if request.pipeline == "local" else "balanced"
        
        # Normalize video_id: convert yt_XXX to full URL for DB lookup
        video_url_in_db = None  # The actual video_id stored in DB (full URL)
        
        if video_id_request.startswith("yt_"):
            # YouTube video with prefix
            youtube_id = video_id_request[3:]
            video_url_in_db = f"https://www.youtube.com/watch?v={youtube_id}"
        elif video_id_request.startswith("https://"):
            # Already a full URL (YouTube or TikTok)
            video_url_in_db = video_id_request
        else:
            # Try as-is (could be TikTok ID or other format)
            # For TikTok, we'd need to construct URL, but for now try direct lookup
            video_url_in_db = video_id_request
        
        # Try to find video in DB with normalized URL
        existing = None
        if video_url_in_db:
            existing = db.get_video(video_url_in_db)
        
        # If not found with normalized URL, try original ID (for backward compatibility)
        if not existing:
            existing = db.get_video(video_id_request)
            if existing:
                video_url_in_db = video_id_request
        
        if existing:
            # Video exists in DB - use the DB video_id for all queries
            video_id = existing.get("video_id", video_url_in_db or video_id_request)
            
            # Check if has label (training data)
            if existing.get("label"):
                return {
                    "video_id": video_id_request,  # Return original ID to frontend
                    "status": "completed",
                    "prediction": existing["label"],
                    "confidence": 1.0,
                    "source": "feature_store",
                    "message": "Label from training data"
                }
            
            # Check if has prediction
            prediction = db.get_latest_prediction(video_id, mode)
            if prediction:
                return {
                    "video_id": video_id_request,  # Return original ID to frontend
                    "status": "completed",
                    "prediction": prediction["y_pred"],
                    "confidence": prediction["confidence"],
                    "source": "inference_cache"
                }
            
            # Check if has sample_id in database (preprocessed)
            samples = db.execute(
                "SELECT sample_id FROM training_samples_v2 WHERE video_id = %s AND split = 'inference'",
                (video_id,),
                fetch=True
            ) or []
            
            if samples:
                # Video has been preprocessed and has sample_id
                # Check if status is already 'pending_inference' or should be set
                current_status = existing.get('status_preprocess', '')
                if current_status != 'pending_inference':
                    db.update_video_status(video_id, 'pending_inference')
                    logger.info(f"Marked video as pending_inference: {video_id[:50]}...")
                
                return {
                    "status": "pending_inference",
                    "video_id": video_id_request,
                    "pipeline": request.pipeline,
                    "message": "Video is ready for inference. Processing will start soon..."
                }
            
            # No sample found - video needs preprocessing
            # Update status to pending_preprocessing (preprocessing DAG will pick it up)
            current_status = existing.get('status_preprocess', '')
            if current_status not in ['pending_preprocessing', 'processing']:
                db.update_video_status(video_id, 'pending_preprocessing')
                logger.info(f"Marked video as pending_preprocessing: {video_id[:50]}...")
            
            return {
                "status": "pending_preprocessing",
                "video_id": video_id_request,
                "pipeline": request.pipeline,
                "message": "Video is queued for preprocessing. Processing will start soon..."
            }
        else:
            # Video not in DB - likely still being ingested
            # Return "pending" status so frontend continues polling
            # Once ingestion completes, video will appear in DB and be processed
            return {
                "status": "pending",
                "video_id": video_id_request,
                "pipeline": request.pipeline,
                "message": "Video is being processed (ingestion/preprocessing). Please wait..."
            }
        
    except Exception as e:
        logger.error(f"Error queueing inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/inference/{video_id}")
async def get_inference_result(video_id: str):
    """Get inference result for a video."""
    try:
        mode = os.environ.get("MODEL_MODE", "ultra_light")
        video_id_request = video_id
        
        # Normalize video_id: convert yt_XXX to full URL for DB lookup (same logic as POST endpoint)
        video_url_in_db = None
        
        if video_id.startswith("yt_"):
            # YouTube video with prefix
            youtube_id = video_id[3:]
            video_url_in_db = f"https://www.youtube.com/watch?v={youtube_id}"
        elif video_id.startswith("https://"):
            # Already a full URL (YouTube or TikTok)
            video_url_in_db = video_id
        else:
            # Try as-is (could be TikTok ID or other format)
            video_url_in_db = video_id
        
        # Try to find video in DB with normalized URL first
        existing = None
        if video_url_in_db:
            existing = db.get_video(video_url_in_db)
        
        # If not found with normalized URL, try original ID (for backward compatibility)
        if not existing:
            existing = db.get_video(video_id)
            if existing:
                video_url_in_db = video_id
        
        # Use the correct video_id from DB for prediction lookup
        db_video_id = video_url_in_db if existing else video_id_request
        
        # Try with normalized video_id first
        prediction = db.get_latest_prediction(db_video_id, mode)
        
        # If not found, try with original video_id (for backward compatibility)
        if not prediction and video_url_in_db != video_id:
            prediction = db.get_latest_prediction(video_id, mode)
            if prediction:
                db_video_id = video_id
        
        
        if not prediction:
            # No prediction yet - check video status to return appropriate status
            if existing:
                video_status = existing.get('status_preprocess', 'pending')
                
                # If status is 'classified' but no prediction found, try to wait a bit (race condition)
                # Or return pending_inference to continue polling
                if video_status == 'classified':
                    # Video marked as classified but no prediction found - could be race condition
                    # Try one more time with a small delay, or return pending to keep polling
                    logger.warning(f"Video {video_id_request} marked as classified but no prediction found - may be race condition")
                    return {
                        "video_id": video_id_request,
                        "status": "pending_inference",
                        "message": "Prediction may be in progress, please wait"
                    }
                elif video_status == 'pending_inference':
                    return {
                        "video_id": video_id_request,
                        "status": "pending_inference",
                        "message": "Video is queued for inference"
                    }
                elif video_status == 'processing':
                    return {
                        "video_id": video_id_request,
                        "status": "processing",
                        "message": "Video is currently being processed"
                    }
                elif video_status == 'pending_preprocessing':
                    return {
                        "video_id": video_id_request,
                        "status": "pending_preprocessing",
                        "message": "Video is queued for preprocessing"
                    }
                else:
                    return {
                        "video_id": video_id_request,
                        "status": video_status if video_status else "pending",
                        "message": f"Video status: {video_status}" if video_status else "Video status unknown"
                    }
            else:
                # Video not in DB - likely still being ingested
                return {
                    "video_id": video_id_request,
                    "status": "pending",
                    "message": "Video not found in database. It may still be being ingested."
                }
        
        # Return prediction with full details
        return {
            "video_id": video_id_request,
            "status": "completed",
            "prediction": prediction["y_pred"],
            "confidence": prediction["confidence"],
            "source": "inference_cache",
            "probabilities": {
                "text": prediction.get("p_text") or {},
                "image": prediction.get("p_img") or {},
                "fused": prediction.get("p_final") or {},
            }
        }
    except Exception as e:
        logger.error(f"Error getting inference result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Training & Ingestion Endpoints =====

@router.post("/api/ingest")
async def trigger_ingestion(auto_train: bool = True, csv_path: str = None):
    """
    Trigger data ingestion from labels.csv or custom CSV file.
    Triggers Airflow DAG for the actual ingestion.
    
    Args:
        auto_train: Automatically trigger training after preprocessing
        csv_path: Optional custom CSV file path (relative to data/raw/)
    """
    conf = {"auto_train": auto_train}
    
    # Handle custom CSV path
    if csv_path:
        # If relative path, prepend base directory
        if not csv_path.startswith('/'):
            csv_path = f"/opt/airflow/data/raw/{csv_path}"
        conf["csv_path"] = csv_path
        logger.info(f"Using custom CSV path: {csv_path}")
    
    result = await trigger_airflow_dag("data_ingestion", conf)
    
    if result["status"] == "triggered":
        return {
            "status": "triggered",
            "message": f"Data ingestion started{' from ' + csv_path if csv_path else ''}",
            "dag_run_id": result.get("response", {}).get("dag_run_id"),
            "note": "Check Airflow UI for progress"
        }
    else:
        return {
            "status": "failed",
            "message": "Could not start ingestion",
            "error": result.get("error")
        }


@router.get("/api/csv-files")
async def list_csv_files():
    """
    List available CSV files in data/raw/ directory.
    Returns list of CSV files that can be used for ingestion.
    """
    import os
    from pathlib import Path
    
    # Check both possible paths
    base_dirs = [
        Path("/opt/airflow/data/raw"),
        Path("/app/data/raw"),
        Path("./data/raw"),
    ]
    
    csv_files = []
    
    for base_dir in base_dirs:
        if base_dir.exists():
            for f in base_dir.glob("*.csv"):
                csv_files.append({
                    "name": f.name,
                    "path": str(f),
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime,
                })
            break
    
    return {
        "status": "success",
        "files": csv_files,
        "default": "labels.csv"
    }


@router.post("/api/trigger-preprocessing")
async def trigger_preprocessing():
    """Trigger preprocessing pipeline."""
    result = await trigger_airflow_dag("video_preprocessing_pipeline")
    
    if result["status"] == "triggered":
        return {
            "status": "success",
            "message": "Preprocessing DAG triggered successfully",
            "dag_run_id": result.get("response", {}).get("dag_run_id"),
        }
    else:
        return {
            "status": "failed",
            "message": "Failed to trigger DAG - will start within 1 minute",
            "error": result.get("error"),
        }


@router.post("/api/train")
async def trigger_training(
    mode: str = Form("ultra_light"),
    epochs: int = Form(50),
    fusion_type: str = Form("gated"),
):
    """Trigger training pipeline."""
    # Validate mode
    valid_modes = ["ultra_light", "balanced"]
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of: {valid_modes}",
        )

    # Validate fusion_type
    valid_fusion_types = ["gated", "attention"]
    if fusion_type not in valid_fusion_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid fusion_type. Must be one of: {valid_fusion_types}",
        )

    conf = {"mode": mode, "fusion_type": fusion_type, "epochs": epochs, "use_pregenerated_samples": True}
    result = await trigger_airflow_dag("training_advanced_fusion", conf)

    if result["status"] == "triggered":
        return {
            "status": "triggered",
            "training_type": "advanced_fusion",
            "fusion_type": fusion_type,
            "mode": mode,
            "epochs": epochs,
            "message": "Training triggered successfully!",
            "note": "Check Airflow UI at http://localhost:8080 for progress.",
        }
    else:
        return {
            "status": "scheduled",
            "message": "Training scheduled (will start within 1 minute)",
            "error": result.get("error"),
        }


# ===== Model Management Endpoints =====

@router.get("/api/models")
async def list_models():
    """List registered models."""
    try:
        query = """
            SELECT id, mode, model_type, version, artifact_path, 
                   is_active, metrics, created_at, training_config,
                   best_epoch, total_epochs, training_time_seconds
            FROM model_registry
            ORDER BY created_at DESC
            LIMIT 50
        """
        models = db.execute(query, fetch=True)

        return {
            "count": len(models) if models else 0,
            "models": models or [],
        }
    except Exception as e:
        logger.warning(f"Database unavailable for models: {e}")
        return {
            "count": 0,
            "models": [],
            "database_status": "unavailable",
        }


@router.get("/api/models/{model_id}/metrics")
async def get_model_metrics(model_id: int):
    """Get detailed training metrics for a specific model checkpoint."""
    try:
        # Get model info
        model_query = """
            SELECT id, mode, model_type, version, metrics, created_at,
                   training_config, best_epoch, total_epochs, training_time_seconds
            FROM model_registry
            WHERE id = %s
        """
        model = db.execute(model_query, (model_id,), fetch=True)
        
        if not model or len(model) == 0:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = model[0]
        
        # Get all training metrics
        metrics = db.get_training_metrics(model_id)
        
        # Organize metrics by epoch and split
        epochs_data = {}
        for m in metrics:
            epoch = m['epoch']
            split = m['split']
            
            if epoch not in epochs_data:
                epochs_data[epoch] = {}
            
            epochs_data[epoch][split] = {
                'loss': m['loss'],
                'accuracy': m['accuracy'],
                'precision': m['precision'],
                'recall': m['recall'],
                'f1': m['f1'],
                'per_class_metrics': m['per_class_metrics'],
                'confusion_matrix': m['confusion_matrix'],
                'learning_rate': m['learning_rate'],
                'batch_size': m['batch_size'],
            }
        
        # Convert to list sorted by epoch
        epochs_list = [
            {
                'epoch': epoch,
                'train': data.get('train'),
                'val': data.get('val'),
                'test': data.get('test'),
            }
            for epoch, data in sorted(epochs_data.items())
        ]
        
        return {
            "model": model_info,
            "epochs": epochs_list,
            "summary": {
                "best_epoch": model_info.get('best_epoch'),
                "total_epochs": model_info.get('total_epochs'),
                "training_time_seconds": model_info.get('training_time_seconds'),
                "best_accuracy": model_info.get('metrics', {}).get('accuracy') if model_info.get('metrics') else None,
                "best_f1": model_info.get('metrics', {}).get('f1') if model_info.get('metrics') else None,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Error Analysis Endpoints =====

CLASS_NAMES = ['Safe', 'Aggressive', 'Sexual', 'Superstition']


@router.get("/api/models/{model_id}/confusion-samples")
async def get_confusion_matrix_samples(
    model_id: int,
    true_label: str,
    pred_label: str,
    split: str = "val",
    limit: int = 50
):
    """
    Get samples for a specific cell in the confusion matrix.
    
    Args:
        model_id: Model ID
        true_label: True/actual label (row in matrix)
        pred_label: Predicted label (column in matrix)
        split: Data split (train, val, test)
        limit: Max samples to return
    
    Returns:
        List of samples with metadata
    """
    try:
        # Validate labels
        if true_label not in CLASS_NAMES:
            raise HTTPException(status_code=400, detail=f"Invalid true_label: {true_label}")
        if pred_label not in CLASS_NAMES:
            raise HTTPException(status_code=400, detail=f"Invalid pred_label: {pred_label}")
        
        # Get model info
        model_query = "SELECT mode FROM model_registry WHERE id = %s"
        model = db.execute(model_query, (model_id,), fetch=True)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        mode = model[0]['mode']
        
        # Get samples from sample_predictions table (stores per-sample predictions from training)
        query = """
            SELECT 
                sp.sample_id,
                s.video_id,
                sp.true_label,
                sp.predicted_label as pred_label,
                sp.confidence,
                sp.p_text,
                sp.p_img,
                sp.p_final,
                sp.split,
                s.selected_frames,
                s.selected_chunks,
                s.augment_idx,
                v.title,
                v.transcript,
                v.filename
            FROM sample_predictions sp
            JOIN training_samples_v2 s ON sp.sample_id = s.sample_id
            JOIN videos v ON s.video_id = v.video_id
            WHERE sp.model_id = %s
              AND sp.true_label = %s
              AND sp.predicted_label = %s
              AND sp.split = %s
            ORDER BY sp.confidence ASC
            LIMIT %s
        """
        
        samples = db.execute(query, (model_id, true_label, pred_label, split, limit), fetch=True)
        
        # Process samples
        result_samples = []
        for s in samples or []:
            result_samples.append({
                "sample_id": s['sample_id'],
                "video_id": s['video_id'],
                "true_label": s['true_label'],
                "pred_label": s.get('pred_label') or true_label,  # If no prediction yet, assume correct
                "title": s.get('title') or "",
                "has_transcript": bool(s.get('transcript')),
                "transcript_preview": (s.get('transcript') or "")[:200],
                "augment_idx": s['augment_idx'],
                "confidence": s.get('confidence'),
                "p_text": s.get('p_text'),
                "p_img": s.get('p_img'),
                "p_final": s.get('p_final'),
            })
        
        return {
            "model_id": model_id,
            "true_label": true_label,
            "pred_label": pred_label,
            "split": split,
            "count": len(result_samples),
            "samples": result_samples
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting confusion matrix samples: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/samples/{sample_id}/details")
async def get_sample_details(sample_id: str):
    """
    Get detailed information about a sample including frame URLs.
    
    Returns sample metadata, frame URLs from MinIO, and transcript.
    """
    try:
        from common.io.storage import storage
        
        # Get sample info
        query = """
            SELECT 
                s.sample_id,
                s.video_id,
                s.label,
                s.selected_frames,
                s.selected_chunks,
                s.augment_idx,
                s.split,
                v.title,
                v.transcript,
                v.filename,
                v.storage_path
            FROM training_samples_v2 s
            JOIN videos v ON s.video_id = v.video_id
            WHERE s.sample_id = %s
        """
        results = db.execute(query, (sample_id,), fetch=True)
        
        if not results:
            raise HTTPException(status_code=404, detail="Sample not found")
        
        sample = results[0]
        video_id = sample['video_id']
        
        # Get frame URLs from MinIO
        # Frames are stored at: samples/{sample_id}/frame_{idx:02d}.jpg
        frame_urls = []
        selected_frames = sample.get('selected_frames')
        if selected_frames:
            import json
            try:
                frame_indices = json.loads(selected_frames) if isinstance(selected_frames, str) else selected_frames
            except:
                frame_indices = []
            
            # Enumerate through selected frames (0-indexed within selection)
            for i, original_idx in enumerate(frame_indices):
                # Frame path format: samples/{sample_id}/frame_{i:02d}.jpg
                # i is the position in selection (0, 1, 2, ...), not the original frame index
                frame_path = f"samples/{sample_id}/frame_{i:02d}.jpg"
                
                if storage.object_exists(frame_path):
                    # Use public URL (bucket is public, no need for presigned)
                    url = storage.get_public_url(frame_path)
                    frame_urls.append({
                        "index": i,
                        "original_index": original_idx,
                        "url": url
                    })
        
        # If no frames found with selected_frames, try listing from MinIO directly
        if not frame_urls:
            # List all frames in the sample folder
            sample_prefix = f"samples/{sample_id}/"
            objects = storage.list_objects(prefix=sample_prefix)
            for obj_name in objects:
                if obj_name.endswith('.jpg') and 'frame_' in obj_name:
                    url = storage.get_public_url(obj_name)
                    # Extract frame index from filename (e.g., frame_00.jpg -> 0)
                    try:
                        idx = int(obj_name.split('frame_')[-1].replace('.jpg', ''))
                        frame_urls.append({
                            "index": idx,
                            "url": url
                        })
                    except:
                        pass
            # Sort by index
            frame_urls.sort(key=lambda x: x['index'])
        
        # Get transcript chunks
        transcript_chunks = []
        selected_chunks = sample.get('selected_chunks')
        
        # Load transcript JSON from MinIO
        transcript_path = f"samples/{sample_id}/transcript.json"
        transcript_data = storage.download_data(transcript_path, silent=True)
        
        if transcript_data:
            import json
            try:
                transcript_json = json.loads(transcript_data.decode('utf-8'))
                
                # Handle both formats:
                # 1. {"chunks": [...], "start_idx": ..., ...}
                # 2. Direct list [...]
                if isinstance(transcript_json, dict):
                    all_chunks = transcript_json.get('chunks', [])
                else:
                    all_chunks = transcript_json
                
                if selected_chunks:
                    try:
                        chunk_indices = json.loads(selected_chunks) if isinstance(selected_chunks, str) else selected_chunks
                        for idx in chunk_indices:
                            if idx < len(all_chunks):
                                transcript_chunks.append({
                                    "index": idx,
                                    "text": all_chunks[idx]
                                })
                    except:
                        # If parsing fails, show all chunks
                        for idx, chunk in enumerate(all_chunks):
                            transcript_chunks.append({
                                "index": idx,
                                "text": chunk
                            })
                else:
                    # No specific chunks selected, show all
                    for idx, chunk in enumerate(all_chunks):
                        transcript_chunks.append({
                            "index": idx,
                            "text": chunk
                        })
            except Exception as e:
                logger.warning(f"Error parsing transcript JSON: {e}")
        
        return {
            "sample_id": sample_id,
            "video_id": video_id,
            "label": sample['label'],
            "title": sample.get('title') or "",
            "raw_transcript": sample.get('transcript') or "",  # Original from videos table
            "augment_idx": sample['augment_idx'],
            "split": sample['split'],
            "frames": frame_urls,
            "transcript_chunks": transcript_chunks,
            "frame_count": len(frame_urls),
            "chunk_count": len(transcript_chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sample details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/models/{model_id}/error-analysis")
async def get_error_analysis(model_id: int, split: str = "val"):
    """
    Get comprehensive error analysis for a model.
    
    Returns statistics about errors, data quality, and model behavior.
    """
    try:
        # Get model info
        model_query = "SELECT mode, metrics FROM model_registry WHERE id = %s"
        model = db.execute(model_query, (model_id,), fetch=True)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        mode = model[0]['mode']
        
        # 1. Data quality statistics (from videos table - original data)
        data_quality_query = """
            SELECT 
                COUNT(*) as total_videos,
                SUM(CASE WHEN transcript IS NULL OR transcript = '' THEN 1 ELSE 0 END) as empty_transcript,
                SUM(CASE WHEN title IS NULL OR title = '' THEN 1 ELSE 0 END) as empty_title,
                SUM(CASE WHEN transcript IS NOT NULL AND transcript != '' AND title IS NOT NULL AND title != '' THEN 1 ELSE 0 END) as complete_data
            FROM videos v
            JOIN training_samples_v2 s ON v.video_id = s.video_id
            WHERE s.split = %s AND s.label IS NOT NULL
        """
        quality_stats = db.execute(data_quality_query, (split,), fetch=True)
        quality = quality_stats[0] if quality_stats else {}
        
        # 2. Per-class statistics
        class_stats_query = """
            SELECT 
                s.label,
                COUNT(DISTINCT s.video_id) as video_count,
                COUNT(*) as sample_count,
                SUM(CASE WHEN v.transcript IS NULL OR v.transcript = '' THEN 1 ELSE 0 END) as empty_transcript,
                AVG(LENGTH(v.transcript)) as avg_transcript_length
            FROM training_samples_v2 s
            JOIN videos v ON s.video_id = v.video_id
            WHERE s.split = %s AND s.label IS NOT NULL
            GROUP BY s.label
            ORDER BY s.label
        """
        class_stats = db.execute(class_stats_query, (split,), fetch=True) or []
        
        # 3. Prediction confidence analysis
        confidence_query = """
            SELECT 
                p.y_pred,
                v.label as true_label,
                AVG(p.confidence) as avg_confidence,
                MIN(p.confidence) as min_confidence,
                MAX(p.confidence) as max_confidence,
                COUNT(*) as count
            FROM predictions p
            JOIN videos v ON p.video_id = v.video_id
            WHERE p.mode = %s AND v.label IS NOT NULL
            GROUP BY p.y_pred, v.label
            ORDER BY p.y_pred, v.label
        """
        confidence_stats = db.execute(confidence_query, (mode,), fetch=True) or []
        
        # 4. Misclassification patterns
        misclass_query = """
            SELECT 
                v.label as true_label,
                p.y_pred as pred_label,
                COUNT(*) as count,
                AVG(p.confidence) as avg_confidence,
                AVG(LENGTH(v.transcript)) as avg_transcript_length,
                SUM(CASE WHEN v.transcript IS NULL OR v.transcript = '' THEN 1 ELSE 0 END) as no_transcript_count
            FROM predictions p
            JOIN videos v ON p.video_id = v.video_id
            WHERE p.mode = %s AND v.label IS NOT NULL AND v.label != p.y_pred
            GROUP BY v.label, p.y_pred
            ORDER BY count DESC
        """
        misclass_stats = db.execute(misclass_query, (mode,), fetch=True) or []
        
        # 5. Modality contribution analysis (from p_text and p_img)
        modality_query = """
            SELECT 
                v.label as true_label,
                p.y_pred,
                AVG(
                    CASE 
                        WHEN p.p_text IS NOT NULL AND p.p_img IS NOT NULL THEN
                            (p.p_text->>'max_prob')::float - (p.p_img->>'max_prob')::float
                        ELSE 0
                    END
                ) as text_vs_image_diff,
                COUNT(*) as count
            FROM predictions p
            JOIN videos v ON p.video_id = v.video_id
            WHERE p.mode = %s AND v.label IS NOT NULL
            GROUP BY v.label, p.y_pred
        """
        # Skip modality analysis if structure is different
        modality_stats = []
        try:
            modality_stats = db.execute(modality_query, (mode,), fetch=True) or []
        except:
            pass
        
        # 6. Error rate by data completeness
        error_by_quality_query = """
            SELECT 
                CASE 
                    WHEN v.transcript IS NULL OR v.transcript = '' THEN 'no_transcript'
                    WHEN LENGTH(v.transcript) < 50 THEN 'short_transcript'
                    ELSE 'normal_transcript'
                END as transcript_quality,
                COUNT(*) as total,
                SUM(CASE WHEN v.label = p.y_pred THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN v.label != p.y_pred THEN 1 ELSE 0 END) as incorrect
            FROM predictions p
            JOIN videos v ON p.video_id = v.video_id
            WHERE p.mode = %s AND v.label IS NOT NULL
            GROUP BY transcript_quality
        """
        quality_error_stats = db.execute(error_by_quality_query, (mode,), fetch=True) or []
        
        return {
            "model_id": model_id,
            "mode": mode,
            "split": split,
            "data_quality": {
                "total_videos": quality.get('total_videos', 0),
                "empty_transcript_count": quality.get('empty_transcript', 0),
                "empty_title_count": quality.get('empty_title', 0),
                "complete_data_count": quality.get('complete_data', 0),
                "empty_transcript_ratio": (quality.get('empty_transcript', 0) / quality.get('total_videos', 1)) if quality.get('total_videos') else 0,
            },
            "per_class_stats": [
                {
                    "label": stat['label'],
                    "video_count": stat['video_count'],
                    "sample_count": stat['sample_count'],
                    "empty_transcript": stat['empty_transcript'],
                    "avg_transcript_length": round(stat['avg_transcript_length'] or 0, 1)
                }
                for stat in class_stats
            ],
            "misclassification_patterns": [
                {
                    "true_label": m['true_label'],
                    "pred_label": m['pred_label'],
                    "count": m['count'],
                    "avg_confidence": round(m['avg_confidence'] or 0, 3),
                    "no_transcript_ratio": round((m['no_transcript_count'] / m['count']) if m['count'] else 0, 3)
                }
                for m in misclass_stats
            ],
            "error_by_transcript_quality": [
                {
                    "quality": q['transcript_quality'],
                    "total": q['total'],
                    "correct": q['correct'],
                    "incorrect": q['incorrect'],
                    "accuracy": round(q['correct'] / q['total'], 3) if q['total'] else 0
                }
                for q in quality_error_stats
            ],
            "confidence_by_prediction": [
                {
                    "true_label": c['true_label'],
                    "pred_label": c['y_pred'],
                    "avg_confidence": round(c['avg_confidence'] or 0, 3),
                    "min_confidence": round(c['min_confidence'] or 0, 3),
                    "max_confidence": round(c['max_confidence'] or 0, 3),
                    "count": c['count']
                }
                for c in confidence_stats
            ],
            "insights": _generate_error_insights(quality, class_stats, misclass_stats, quality_error_stats)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in error analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_error_insights(quality, class_stats, misclass_stats, quality_error_stats):
    """Generate human-readable insights from error analysis data."""
    insights = []
    
    # Data quality insights
    if quality:
        empty_ratio = quality.get('empty_transcript', 0) / quality.get('total_videos', 1) if quality.get('total_videos') else 0
        if empty_ratio > 0.3:
            insights.append({
                "type": "warning",
                "category": "data_quality",
                "message": f"High ratio of empty transcripts ({empty_ratio:.1%}). Consider improving transcript extraction or using image-only fallback."
            })
        elif empty_ratio > 0.1:
            insights.append({
                "type": "info",
                "category": "data_quality", 
                "message": f"Some videos ({empty_ratio:.1%}) have no transcript. Model relies more on visual features for these."
            })
    
    # Class imbalance insights
    if class_stats:
        counts = [s['sample_count'] for s in class_stats]
        if counts:
            max_count, min_count = max(counts), min(counts)
            if max_count > min_count * 3:
                insights.append({
                    "type": "warning",
                    "category": "class_imbalance",
                    "message": f"Significant class imbalance detected (ratio {max_count/min_count:.1f}:1). Consider data augmentation or weighted loss."
                })
    
    # Misclassification insights
    if misclass_stats:
        for m in misclass_stats[:3]:  # Top 3 error patterns
            if m['count'] >= 5:
                insights.append({
                    "type": "error",
                    "category": "misclassification",
                    "message": f"Common confusion: {m['true_label']} → {m['pred_label']} ({m['count']} cases, {m['avg_confidence']:.1%} avg confidence)"
                })
    
    # Transcript quality impact
    if quality_error_stats:
        for q in quality_error_stats:
            if q['transcript_quality'] == 'no_transcript' and q['total'] > 0:
                acc = q['correct'] / q['total']
                if acc < 0.6:
                    insights.append({
                        "type": "warning",
                        "category": "modality_dependency",
                        "message": f"Model struggles with videos without transcript (accuracy: {acc:.1%}). Visual features alone may be insufficient."
                    })
    
    if not insights:
        insights.append({
            "type": "success",
            "category": "general",
            "message": "No major issues detected in error analysis."
        })
    
    return insights


@router.get("/api/models/{model_id}/gate-weights")
async def get_gate_weights_analysis(model_id: int, split: str = "val"):
    """
    Get gate weights analysis for a model.
    
    Gate weights show how the gated fusion mechanism balances between modalities:
    - Gate weight close to 1.0 = trusts image more
    - Gate weight close to 0.0 = trusts text more
    - Gate weight around 0.5 = balanced between modalities
    
    Returns per-class statistics of gate weights (pre-computed during training).
    """
    try:
        # Get model info
        model_query = """
            SELECT id, mode, model_type, metrics
            FROM model_registry
            WHERE id = %s
        """
        model_result = db.execute(model_query, (model_id,), fetch=True)
        
        if not model_result:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = model_result[0]
        mode = model_info['mode']
        metrics = model_info.get('metrics') or {}
        
        # Check if this is a gated fusion model
        if 'gated' not in (model_info.get('model_type') or ''):
            return {
                "model_id": model_id,
                "error": "Gate weights analysis only available for gated fusion models",
                "model_type": model_info.get('model_type')
            }
        
        # Get gate weights from metrics
        gate_weights_data = metrics.get('gate_weights')
        
        if not gate_weights_data:
            return {
                "model_id": model_id,
                "mode": mode,
                "error": "Gate weights not computed for this model. Re-train the model to generate gate weights analysis.",
                "per_class": {},
                "overall": {}
            }
        
        # Get the split data (prefer requested split, fallback to available)
        split_data = gate_weights_data.get(split)
        if not split_data:
            # Try to get any available split
            for available_split in ['val', 'train', 'test']:
                if available_split in gate_weights_data and isinstance(gate_weights_data[available_split], dict):
                    split_data = gate_weights_data[available_split]
                    split = available_split
                    break
        
        if not split_data or not isinstance(split_data, dict):
            return {
                "model_id": model_id,
                "mode": mode,
                "error": "Gate weights data not available for the requested split",
                "per_class": {},
                "overall": {}
            }
        
        # Return the pre-computed gate weights
        return {
            "model_id": model_id,
            "mode": mode,
            "split": split,
            "samples_processed": split_data.get('overall', {}).get('total_samples', 0),
            "per_class": split_data.get('per_class', {}),
            "overall": split_data.get('overall', {}),
            "insights": split_data.get('insights', []),
            "explanation": split_data.get('explanation', {
                "gate_weight_meaning": "Gate weight indicates how much the model trusts each modality",
                "interpretation": {
                    "1.0": "Fully trust image features",
                    "0.5": "Equal trust between image and text",
                    "0.0": "Fully trust text features"
                }
            }),
            "analyzed_at": gate_weights_data.get('analyzed_at')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting gate weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/fusion-config")
async def get_fusion_config():
    """Get fusion configuration for all modes."""
    try:
        query = """
            SELECT mode, config, updated_at
            FROM fusion_config
        """
        rows = db.execute(query, fetch=True)

        config = {}
        for row in rows or []:
            config[row["mode"]] = row["config"]

        return config
    except Exception as e:
        logger.warning(f"Database unavailable for fusion config: {e}")
        return {
            "ultra_light": {},
            "balanced": {},
            "database_status": "unavailable",
        }
