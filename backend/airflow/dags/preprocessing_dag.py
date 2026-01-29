"""
Preprocessing DAG for video classification pipeline.

Flow:
    scan_new_videos 
        → [pre_augmentation_sample, create_inference_plan] (parallel planning)
        → [extract_frames, extract_transcript] (shared extraction)
        → [create_training_samples, create_inference_samples] (parallel creation)
        → [decide_training_trigger → trigger_training, trigger_inference] (parallel triggers)
        → finalize_preprocessing

Training videos (with label):
    - Augmented based on class balance
    - Split into train/val/test
    - Saved to database

Inference videos (no label):
    - No augmentation (1 sample per video)
    - Saved to database with split='inference'
    - Status updated to 'pending_inference' (inference DAG will pick it up)
    - Triggers inference DAG

Database-driven:
    - Uses PostgreSQL status_preprocess field to track progress
    - No Kafka dependency (Kafka only used for video search → ingestion)

Triggered by: data_ingestion_dag
"""

import os
import sys
import json
import random
import hashlib
import logging
import tempfile
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

sys.path.insert(0, '/opt/airflow')

from common.io import db, storage, queue, config
from common.data.transcript_cleaner import clean_transcript

logger = logging.getLogger(__name__)

# ============================================================================
# SPARK CONFIGURATION
# ============================================================================
def is_spark_enabled() -> bool:
    """Check if Spark processing is enabled via environment."""
    return os.environ.get('USE_SPARK', 'true').lower() == 'true'

def get_spark_config() -> dict:
    """Get Spark-related configuration for workers."""
    return {
        'storage': {
            'host': os.environ.get('MINIO_HOST', 'minio'),
            'port': int(os.environ.get('MINIO_PORT', '9000')),
            'access_key': os.environ.get('MINIO_ACCESS_KEY', 'minioadmin'),
            'secret_key': os.environ.get('MINIO_SECRET_KEY', 'minioadmin'),
            'bucket': os.environ.get('MINIO_BUCKET', 'video-classifier'),
        },
        'db': {
            'host': os.environ.get('POSTGRES_HOST', 'postgres'),
            'port': int(os.environ.get('POSTGRES_PORT', '5432')),
            'user': os.environ.get('POSTGRES_USER', 'video_classifier'),
            'password': os.environ.get('POSTGRES_PASSWORD', 'changeme123'),
            'database': os.environ.get('POSTGRES_DB', 'video_classifier'),
        }
    }

# ============================================================================
# CONSTANTS
# ============================================================================
NUM_FRAME_SEGMENTS = 16   # Video divided into K segments
NUM_TRANSCRIPT_CHUNKS = 5  # Select 5 consecutive chunks
WORDS_PER_CHUNK = 50      # Words per chunk
BATCH_SIZE_WHISPER = 4    # Whisper batch size (16GB RAM)
BATCH_SIZE_FRAMES = 20    # Frame extraction batch size

default_args = {
    'owner': 'video_classifier',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def sanitize_video_id(video_id: str) -> str:
    """Convert video_id to safe storage key using MD5 hash."""
    return hashlib.md5(video_id.encode()).hexdigest()


def generate_sample_id(video_id: str, augment_idx: int) -> str:
    """Generate unique sample ID."""
    return hashlib.md5(f"{video_id}_{augment_idx}_{random.random()}".encode()).hexdigest()[:16]


def select_random_frames_per_segment(total_frames: int, num_segments: int = NUM_FRAME_SEGMENTS) -> List[int]:
    """
    Divide video into segments and select 1 random frame per segment.
    
    Returns:
        List of frame indices (one per segment)
    """
    if total_frames <= 0:
        return []
    
    if total_frames <= num_segments:
        return list(range(total_frames))
    
    frames_per_segment = total_frames / num_segments
    selected = []
    
    for i in range(num_segments):
        start = int(i * frames_per_segment)
        end = int((i + 1) * frames_per_segment)
        frame_idx = random.randint(start, min(end - 1, total_frames - 1))
        selected.append(frame_idx)
    
    return selected


def chunk_transcript(text: str, words_per_chunk: int = WORDS_PER_CHUNK) -> List[str]:
    """Split transcript into chunks of N words each."""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i:i + words_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def select_consecutive_chunks(chunks: List[str], num_chunks: int = NUM_TRANSCRIPT_CHUNKS) -> Tuple[List[str], int]:
    """
    Randomly select consecutive chunks from transcript.
    
    Returns:
        Tuple of (selected chunks, start index)
    """
    if not chunks:
        return [], 0
    
    if len(chunks) <= num_chunks:
        return chunks, 0
    
    # Random start position
    max_start = len(chunks) - num_chunks
    start_idx = random.randint(0, max_start)
    
    selected = chunks[start_idx:start_idx + num_chunks]
    return selected, start_idx


def compute_augmentation_factors(class_counts: Dict[str, int]) -> Dict[str, int]:
    """
    Compute augmentation factors to balance classes.
    
    Logic:
        - Class with MAX count: factor = 1 (keep as-is)
        - Other classes: choose factor so count*factor is CLOSEST to max_count
        - If tie (equal distance), prefer larger (more samples)
    
    Example: 
        Input:  {Safe: 1500, Aggressive: 1200, Sexual: 600, Superstition: 400}
        Output: {Safe: 1, Aggressive: 1, Sexual: 3, Superstition: 4}
        
        Final counts: 1500, 1200, 1800, 1600 (balanced around max)
        
        Why?
        - 1200: |1200*1 - 1500| = 300 < |1200*2 - 1500| = 900 → x1
        - 600:  |600*2 - 1500| = 300 = |600*3 - 1500| = 300 → x3 (tie, prefer larger)
        - 400:  |400*3 - 1500| = 300 > |400*4 - 1500| = 100 → x4
    """
    if not class_counts:
        return {}
    
    max_count = max(class_counts.values())
    factors = {}
    
    for label, count in class_counts.items():
        if count == max_count:
            factors[label] = 1
        else:
            # Find factor that makes count*factor closest to max_count
            low = max(1, int(max_count / count))
            high = low + 1
            
            dist_low = abs(count * low - max_count)
            dist_high = abs(count * high - max_count)
            
            # If tie, prefer larger (more samples = better balance)
            if dist_high <= dist_low:
                factors[label] = high
            else:
                factors[label] = low
    
    return factors


# ============================================================================
# DAG TASKS
# ============================================================================
def scan_new_videos(**context):
    """
    Scan for videos to preprocess.
    
    Priority:
    1. Consume from Kafka preprocessing queue (if available)
    2. Fallback to polling pending videos in Postgres
    
    Note: LIMIT 100 is used to prevent memory/timeout issues.
    If more videos are pending, finalize_preprocessing will auto-trigger another run.
    """
    logger.info("=" * 80)
    logger.info("SCANNING FOR NEW VIDEOS")
    logger.info("=" * 80)
    
    videos = []
    
    # =========================================================================
    # First, count total pending videos
    # =========================================================================
    total_pending_query = "SELECT COUNT(*) as count FROM videos WHERE status_preprocess = 'pending_preprocessing'"
    try:
        result = db.execute(total_pending_query, fetch=True)
        total_pending = result[0]['count'] if result else 0
        logger.info(f"Total pending videos in database: {total_pending}")
    except:
        total_pending = "unknown"
    
    # =========================================================================
    # Polling Postgres for pending_preprocessing videos (created/updated in last 2 hours)
    # Note: Kafka is only used for video search → ingestion trigger
    # The source of truth is the 'pending_preprocessing' status in Postgres
    # =========================================================================
    logger.info("Polling Postgres for recently pending videos...")
    
    # Get videos that are pending_preprocessing AND were either:
    # 1. Created in the last 2 hours (new videos), OR
    # 2. Updated to 'pending_preprocessing' status recently
    recent_pending_query = """
        SELECT * FROM videos 
        WHERE status_preprocess = 'pending_preprocessing' 
        AND (
            created_at >= NOW() - INTERVAL '2 hours'
            OR updated_at >= NOW() - INTERVAL '2 hours'
        )
        ORDER BY created_at DESC
        LIMIT 100
    """
    
    try:
        videos = db.execute(recent_pending_query, fetch=True) or []
        logger.info(f"Found {len(videos)} videos with status='pending_preprocessing' in last 2 hours")
        
        # If no recent videos, check if there are old pending_preprocessing videos (edge case)
        if len(videos) == 0:
            old_pending = db.get_videos_by_status('pending_preprocessing', limit=100)
            if old_pending:
                logger.warning(f"No recent pending_preprocessing videos, but found {len(old_pending)} old pending_preprocessing videos")
                logger.info("Processing old pending_preprocessing videos as fallback...")
                videos = old_pending
    except Exception as e:
        logger.warning(f"Recent query failed: {e}, falling back to all pending_preprocessing")
        videos = db.get_videos_by_status('pending_preprocessing', limit=100)
        logger.info(f"Found {len(videos)} videos with status='pending_preprocessing' in Postgres")
    
    # Log batch info
    if isinstance(total_pending, int) and total_pending > len(videos):
        logger.info(f"📦 Processing batch: {len(videos)}/{total_pending} videos (remaining will be processed in next run)")
    
    if len(videos) == 0:
        logger.info("No pending videos found. Skipping preprocessing.")
        context['task_instance'].xcom_push(key='video_ids', value=[])
        context['task_instance'].xcom_push(key='videos', value=[])
        return 0
    
    # Lock videos (update status to processing)
    video_ids = [v['video_id'] for v in videos]
    
    logger.info(f"Locking {len(video_ids)} videos...")
    for video_id in video_ids:
        try:
            db.update_video_status(video_id, 'processing')
        except Exception as e:
            logger.warning(f"Failed to lock video {video_id}: {e}")
    
    # Separate train (has label) vs inference (no label)
    train_videos = [v for v in videos if v.get('label')]
    inference_videos = [v for v in videos if not v.get('label')]
    
    logger.info(f"  Training videos (with label):   {len(train_videos)}")
    logger.info(f"  Inference videos (no label):    {len(inference_videos)}")
    
    # Serialize videos for XCom (convert to simple dicts)
    videos_data = [dict(v) for v in videos]
    
    context['task_instance'].xcom_push(key='video_ids', value=video_ids)
    context['task_instance'].xcom_push(key='videos', value=videos_data)
    context['task_instance'].xcom_push(key='train_video_ids', value=[v['video_id'] for v in train_videos])
    context['task_instance'].xcom_push(key='inference_video_ids', value=[v['video_id'] for v in inference_videos])
    context['task_instance'].xcom_push(key='total_videos', value=len(video_ids))
    
    if train_videos != [] and train_videos is not None:
        logger.info(f"Training video example: {train_videos[0]}")
        context['task_instance'].xcom_push(key='is_training_pipeline', value=True)
    else:
        logger.info("No training videos found.")
        context['task_instance'].xcom_push(key='is_training_pipeline', value=False)


    logger.info("=" * 80)
    return len(video_ids)


def pre_augmentation_sample(**context):
    """
    Compute augmentation factors for training data.
    
    Split strategy:
        - Val:   150 samples per class (no augmentation)
        - Test:  150 samples per class (no augmentation)
        - Train: remaining videos (with augmentation to balance classes)
    """
    logger.info("=" * 80)
    logger.info("COMPUTING AUGMENTATION FACTORS")
    logger.info("=" * 80)
    
    train_video_ids = context['task_instance'].xcom_pull(task_ids='scan_new_videos', key='train_video_ids') or []
    videos = context['task_instance'].xcom_pull(task_ids='scan_new_videos', key='videos') or []
    
    if not train_video_ids:
        logger.info("No training videos. Skipping augmentation.")
        context['task_instance'].xcom_push(key='augmentation_plan', value={})
        context['task_instance'].xcom_push(key='sample_plan', value=[])
        context['task_instance'].xcom_push(key='val_video_ids', value=[])
        context['task_instance'].xcom_push(key='test_video_ids', value=[])
        return 0
    
    mode = config.get('system.mode', 'ultra_light')
    
    # Split ratio: 80% train, 10% val, 10% test
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # Get new training videos
    train_videos = {v['video_id']: v for v in videos if v.get('label')}
    
    # =========================================================================
    # STEP 1: Split videos into train/val/test (80/10/10 per label)
    # =========================================================================
    videos_by_label = defaultdict(list)
    for video_id in train_video_ids:
        video = train_videos.get(video_id)
        if video:
            videos_by_label[video['label']].append(video_id)
    
    val_video_ids = set()
    test_video_ids = set()
    train_split_ids = []
    
    logger.info("\nSplitting videos by class (80/10/10):")
    
    for label, video_ids in videos_by_label.items():
        random.shuffle(video_ids)
        total = len(video_ids)
        
        # Calculate split counts based on ratio
        n_val = max(1, int(total * VAL_RATIO))
        n_test = max(1, int(total * TEST_RATIO))
        n_train = total - n_val - n_test
        
        # Ensure at least 1 video for train
        if n_train < 1:
            n_train = max(1, total - 2)
            n_val = min(1, total - n_train)
            n_test = total - n_train - n_val
        
        val_video_ids.update(video_ids[:n_val])
        test_video_ids.update(video_ids[n_val:n_val + n_test])
        train_split_ids.extend(video_ids[n_val + n_test:])
        
        logger.info(f"  {label:15s}: {total:4d} total → train={n_train} ({n_train/total*100:.0f}%), val={n_val}, test={n_test}")
    
    val_split_ids = list(val_video_ids)
    test_split_ids = list(test_video_ids)
    
    logger.info(f"\nTrain/Val/Test split:")
    logger.info(f"  Train videos: {len(train_split_ids)}")
    logger.info(f"  Val videos:   {len(val_split_ids)}")
    logger.info(f"  Test videos:  {len(test_split_ids)}")
    
    # =========================================================================
    # STEP 2: Count samples for TRAIN split only (existing + new)
    # =========================================================================
    existing_samples = db.get_training_samples_v2(split='train')
    existing_counts = Counter(s['label'] for s in existing_samples)
    
    # Count new TRAIN videos only
    new_train_counts = Counter(train_videos[vid]['label'] for vid in train_split_ids if vid in train_videos)
    
    # Total = existing + new train
    total_counts = defaultdict(int)
    for label in set(list(existing_counts.keys()) + list(new_train_counts.keys())):
        total_counts[label] = existing_counts.get(label, 0) + new_train_counts.get(label, 0)
    
    logger.info("\nTrain class distribution (existing + new):")
    for label, count in sorted(total_counts.items()):
        logger.info(f"  {label:15s}: {count:4d} (existing: {existing_counts.get(label, 0)}, new: {new_train_counts.get(label, 0)})")
    
    # Compute augmentation factors for TRAIN only
    augment_factors = compute_augmentation_factors(dict(total_counts))
    
    logger.info("\nAugmentation factors (train only):")
    for label, factor in augment_factors.items():
        logger.info(f"  {label:15s}: x{factor}")
    
    # =========================================================================
    # STEP 3: Create sample plan
    # =========================================================================
    sample_plan = []
    
    # TRAIN samples - with augmentation
    for video_id in train_split_ids:
        video = train_videos.get(video_id)
        if not video:
            continue
        label = video['label']
        factor = augment_factors.get(label, 1)
        
        for aug_idx in range(factor):
            sample_plan.append({
                'video_id': video_id,
                'augment_idx': aug_idx,
                'label': label,
                'split': 'train',
                'sample_id': generate_sample_id(video_id, aug_idx)
            })
    
    # VAL samples - NO augmentation (factor=1)
    for video_id in val_split_ids:
        video = train_videos.get(video_id)
        if not video:
            continue
        sample_plan.append({
            'video_id': video_id,
            'augment_idx': 0,
            'label': video['label'],
            'split': 'val',
            'sample_id': generate_sample_id(video_id, 0)
        })
    
    # TEST samples - NO augmentation (factor=1)
    for video_id in test_split_ids:
        video = train_videos.get(video_id)
        if not video:
            continue
        sample_plan.append({
            'video_id': video_id,
            'augment_idx': 0,
            'label': video['label'],
            'split': 'test',
            'sample_id': generate_sample_id(video_id, 0)
        })
    
    # NOTE: Inference videos are NOT included here - they go through create_inference_plan
    
    train_samples = [s for s in sample_plan if s['split'] == 'train']
    val_samples = [s for s in sample_plan if s['split'] == 'val']
    test_samples = [s for s in sample_plan if s['split'] == 'test']
    
    logger.info(f"\nTotal samples to generate: {len(sample_plan)}")
    logger.info(f"  Training samples:   {len(train_samples)} (augmented)")
    logger.info(f"  Validation samples: {len(val_samples)} (no augmentation)")
    logger.info(f"  Test samples:       {len(test_samples)} (no augmentation)")
    
    context['task_instance'].xcom_push(key='augmentation_plan', value=augment_factors)
    context['task_instance'].xcom_push(key='sample_plan', value=sample_plan)
    context['task_instance'].xcom_push(key='val_video_ids', value=list(val_video_ids))
    context['task_instance'].xcom_push(key='test_video_ids', value=list(test_video_ids))
    
    logger.info("=" * 80)
    return len(sample_plan)


def create_inference_plan(**context):
    """
    Create sample plan for inference videos (no augmentation).
    
    This is the inference-only path when no training videos are present.
    """
    logger.info("=" * 80)
    logger.info("CREATING INFERENCE PLAN")
    logger.info("=" * 80)
    
    inference_video_ids = context['task_instance'].xcom_pull(task_ids='scan_new_videos', key='inference_video_ids') or []
    
    if not inference_video_ids:
        logger.info("No inference videos.")
        context['task_instance'].xcom_push(key='sample_plan', value=[])
        return 0
    
    sample_plan = []
    
    for video_id in inference_video_ids:
        sample_plan.append({
            'video_id': video_id,
            'augment_idx': 0,
            'label': None,
            'split': 'inference',
            'sample_id': generate_sample_id(video_id, 0)
        })
    
    logger.info(f"Created {len(sample_plan)} inference samples")
    
    context['task_instance'].xcom_push(key='sample_plan', value=sample_plan)
    
    logger.info("=" * 80)
    return len(sample_plan)



def extract_frames(**context):
    """
    Extract frames for all samples (both training and inference).
    
    For each sample:
        1. Divide video into K segments
        2. Randomly select 1 frame per segment
        3. Save frame paths to MinIO
    
    Augmented samples get different random frames.
    
    If USE_SPARK=true, uses Spark for parallel processing.
    """
    import subprocess
    import glob
    
    logger.info("=" * 80)
    logger.info("EXTRACTING FRAMES")
    logger.info("=" * 80)
    
    # Combine sample_plan from BOTH training and inference paths
    train_plan = context['task_instance'].xcom_pull(task_ids='pre_augmentation_sample', key='sample_plan') or []
    infer_plan = context['task_instance'].xcom_pull(task_ids='create_inference_plan', key='sample_plan') or []
    sample_plan = train_plan + infer_plan
    
    logger.info(f"Sample plan: {len(train_plan)} training + {len(infer_plan)} inference = {len(sample_plan)} total")
    
    if not sample_plan:
        logger.info("No samples to extract frames from")
        return 0
    
    # =========================================================================
    # Sequential processing (Spark disabled for preprocessing)
    # Note: Spark is only used in training and inference for better reliability
    # =========================================================================
    mode = config.get('system.mode', 'ultra_light')
    frame_width = config.get('data.frame_width', 224)
    frame_height = config.get('data.frame_height', 224)
    
    # Group samples by video_id
    samples_by_video = defaultdict(list)
    for s in sample_plan:
        samples_by_video[s['video_id']].append(s)
    
    processed = 0
    errors = 0
    frame_results = {}
    
    for video_id, samples in samples_by_video.items():
        logger.info(f"Processing video: {video_id} ({len(samples)} samples)")
        
        video = db.get_video(video_id)
        if not video or not video.get('storage_path'):
            logger.warning(f"  Video not found: {video_id}")
            errors += len(samples)
            continue
        
        try:
            # Download video once
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_path = temp_video.name
                storage.download_file(video['storage_path'], temp_path)
            
            # Get total frames
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-count_frames',
                '-show_entries', 'stream=nb_read_frames',
                '-of', 'csv=p=0',
                temp_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            total_frames = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 100
            
            logger.info(f"  Total frames: {total_frames}")
            
            # Extract ALL frames to temp dir
            with tempfile.TemporaryDirectory() as temp_dir:
                extract_cmd = [
                    'ffmpeg', '-y', '-i', temp_path,
                    '-vf', f'fps=1,scale={frame_width}:{frame_height}',
                    '-q:v', '2',
                    f'{temp_dir}/frame_%04d.jpg'
                ]
                subprocess.run(extract_cmd, capture_output=True)
                
                extracted_frames = sorted(glob.glob(f'{temp_dir}/frame_*.jpg'))
                num_extracted = len(extracted_frames)
                
                logger.info(f"  Extracted {num_extracted} frames")
                
                # For each sample, select random frames
                for sample in samples:
                    sample_id = sample['sample_id']
                    
                    selected_indices = select_random_frames_per_segment(num_extracted, NUM_FRAME_SEGMENTS)
                    
                    # Save selected frames to MinIO
                    saved_indices = []
                    for i, frame_idx in enumerate(selected_indices):
                        if frame_idx < len(extracted_frames):
                            frame_path = extracted_frames[frame_idx]
                            minio_path = f"samples/{sample_id}/frame_{i:02d}.jpg"
                            storage.upload_file(minio_path, frame_path)
                            saved_indices.append(frame_idx)
                    
                    frame_results[sample_id] = saved_indices
                    processed += 1
                    
                    if processed % 50 == 0:
                        logger.info(f"  Progress: {processed} samples")
            
            os.unlink(temp_path)
            
        except Exception as e:
            logger.error(f"  Error: {e}")
            errors += len(samples)
    
    context['task_instance'].xcom_push(key='frame_results', value=frame_results)
    
    logger.info("=" * 80)
    logger.info(f"FRAME EXTRACTION: {processed} samples, {errors} errors")
    logger.info("=" * 80)
    
    return processed


def extract_transcript(**context):
    """
    Extract transcript chunks for all samples (both training and inference).
    
    For each sample:
        1. Get/generate transcript (Whisper if needed)
        2. Chunk transcript into N-word segments
        3. Select 5 consecutive random chunks
        4. Save to MinIO
    
    Note: STT (Whisper/Groq) runs sequentially as it's API-based.
    Spark is used only for parallel chunk distribution if enabled.
    """
    import whisper
    from groq import Groq
    
    logger.info("=" * 80)
    logger.info("EXTRACTING TRANSCRIPTS")
    logger.info("=" * 80)
    
    # Combine sample_plan from BOTH training and inference paths
    train_plan = context['task_instance'].xcom_pull(task_ids='pre_augmentation_sample', key='sample_plan') or []
    infer_plan = context['task_instance'].xcom_pull(task_ids='create_inference_plan', key='sample_plan') or []
    sample_plan = train_plan + infer_plan
    
    logger.info(f"Sample plan: {len(train_plan)} training + {len(infer_plan)} inference = {len(sample_plan)} total")
    
    if not sample_plan:
        logger.info("No samples")
        return 0
    
    mode = config.get('system.mode', 'ultra_light')
    whisper_model_name = config.get('preprocessing.whisper_model', 'small')
    whisper_device = config.get('preprocessing.whisper_device', 'cpu')
    
    # Group by video
    samples_by_video = defaultdict(list)
    for s in sample_plan:
        samples_by_video[s['video_id']].append(s)
    
    # Check which need STT
    videos_need_stt = []
    video_transcripts = {}
    
    for video_id in samples_by_video.keys():
        video = db.get_video(video_id)
        if video and video.get('transcript'):
            video_transcripts[video_id] = video['transcript']
            if video.get('transcript') == "Nội dung rỗng new":
                videos_need_stt.append(video_id)
        else:
            videos_need_stt.append(video_id)
    
    logger.info(f"Videos with transcript: {len(video_transcripts)}")
    logger.info(f"Videos needing STT: {len(videos_need_stt)}")
    
    # Run STT based on video type
    if videos_need_stt:
        # Initialize Groq API keys rotation
        import time
        groq_api_keys = os.getenv('GROQ_API_KEYS', '').split(',')
        groq_api_keys = [k.strip() for k in groq_api_keys if k.strip()]
        
        if not groq_api_keys:
            # Fallback to single key
            single_key = os.getenv('GROQ_API_KEY')
            if single_key:
                groq_api_keys = [single_key]
        
        if not groq_api_keys:
            logger.warning("No GROQ_API_KEY(S) found, inference videos will have empty transcripts")
            for video_id in videos_need_stt:
                video_transcripts[video_id] = "Nội dung rỗng"
        else:
            logger.info(f"Groq API initialized with {len(groq_api_keys)} keys")
            current_key_index = 0
            
            # Process videos one by one
            for i, video_id in enumerate(videos_need_stt):
                logger.info(f"  STT progress: {i+1}/{len(videos_need_stt)} videos")
                
                video = db.get_video(video_id)
                if not video or not video.get('storage_path'):
                    logger.warning(f"    Video not found or no storage_path: {video_id[:60]}...")
                    video_transcripts[video_id] = "Nội dung rỗng"
                    continue
                
                # Determine if this is training or inference
                video_samples = samples_by_video.get(video_id, [])
                is_inference = any(s.get('split') == 'inference' for s in video_samples)
                
                # if not is_inference:
                #     # Training videos: use placeholder
                #     logger.info(f"    Training video, using placeholder: {video_id[:60]}...")
                #     transcript = "Nội dung rỗng"
                #     db.update_video_transcript(video_id, transcript)
                #     video_transcripts[video_id] = transcript
                #     continue
                
                # Inference videos: use Groq API
                transcript = None
                temp_path = None
                
                try:
                    # Download video
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                        temp_path = temp_file.name
                        storage.download_file(video['storage_path'], temp_path)
                    
                    # Check file size (Groq limit: 25MB)
                    import os as os_module
                    file_size_mb = os_module.path.getsize(temp_path) / (1024 * 1024)
                    logger.info(f"    Video size: {file_size_mb:.1f}MB")
                    
                    # If > 25MB, extract audio and compress
                    if file_size_mb > 20:
                        logger.info(f"    File too large, extracting audio...")
                        audio_path = temp_path.replace('.mp4', '.mp3')
                        import subprocess
                        subprocess.run([
                            'ffmpeg', '-y', '-i', temp_path,
                            '-vn', '-acodec', 'libmp3lame', '-b:a', '64k',
                            audio_path
                        ], capture_output=True)
                        
                        if os_module.path.exists(audio_path):
                            os_module.unlink(temp_path)
                            temp_path = audio_path
                            file_size_mb = os_module.path.getsize(temp_path) / (1024 * 1024)
                            logger.info(f"    Compressed audio size: {file_size_mb:.1f}MB")
                    
                    # Try multiple API keys with rotation
                    max_key_attempts = len(groq_api_keys)
                    key_attempt = 0
                    
                    while key_attempt < max_key_attempts and transcript is None:
                        api_key = groq_api_keys[current_key_index]
                        logger.info(f"    Using Groq API key #{current_key_index + 1}/{len(groq_api_keys)}")
                        
                        try:
                            groq_client = Groq(api_key=api_key)
                            
                            with open(temp_path, 'rb') as audio_file:
                                transcription = groq_client.audio.transcriptions.create(
                                    file=(os_module.path.basename(temp_path), audio_file.read()),
                                    model="whisper-large-v3",
                                    language="vi",
                                    response_format="text"
                                )
                            
                            transcript = clean_transcript(transcription) if transcription else ""
                            
                            # Success - wait before next request
                            time.sleep(2)
                            break
                            
                        except Exception as api_error:
                            error_str = str(api_error).lower()
                            
                            if "429" in str(api_error) or "rate" in error_str or "quota" in error_str:
                                # Rate limit - try next key
                                logger.warning(f"    API key #{current_key_index + 1} rate limited, trying next key...")
                                current_key_index = (current_key_index + 1) % len(groq_api_keys)
                                key_attempt += 1
                                time.sleep(1)
                            else:
                                # Other error - log and try next key
                                logger.error(f"    API error with key #{current_key_index + 1}: {api_error}")
                                current_key_index = (current_key_index + 1) % len(groq_api_keys)
                                key_attempt += 1
                                time.sleep(1)
                    
                    # If all keys failed, set empty transcript
                    if transcript is None:
                        logger.error(f"    All API keys failed for video {video_id[:60]}...")
                        transcript = "Nội dung rỗng"
                    
                except Exception as e:
                    logger.error(f"    STT error {video_id[:60]}...: {e}")
                    transcript = "Nội dung rỗng"
                
                finally:
                    # Cleanup temp file
                    if temp_path and os_module.path.exists(temp_path):
                        try:
                            os_module.unlink(temp_path)
                        except:
                            pass
                
                # Ensure transcript is never None
                if transcript is None:
                    transcript = "Nội dung rỗng"
                
                # Save to database
                db.update_video_transcript(video_id, transcript)
                video_transcripts[video_id] = transcript
                
                logger.info(f"    {video_id[:60]}...: {len(transcript)} chars")
    
    # Print summary of all transcripts
    logger.info("=" * 80)
    logger.info("TRANSCRIPT SUMMARY")
    logger.info("=" * 80)
    for video_id, transcript in video_transcripts.items():
        preview = transcript[:100] + "..." if len(transcript) > 100 else transcript
        logger.info(f"  {video_id[:50]}... ({len(transcript)} chars): {preview}")
    logger.info("=" * 80)
    
    # Process chunks
    processed = 0
    chunk_results = {}
    
    for video_id, samples in samples_by_video.items():
        # Get video info to extract title
        video_info = db.get_video(video_id)
        title = video_info.get('title', '') if video_info else ''
        
        transcript = video_transcripts.get(video_id, '')
        all_chunks = chunk_transcript(transcript)
        
        for sample in samples:
            sample_id = sample['sample_id']
            
            # Select NUM_TRANSCRIPT_CHUNKS-1 consecutive chunks (reserve 1 slot for title)
            num_content_chunks = NUM_TRANSCRIPT_CHUNKS - 1 if title else NUM_TRANSCRIPT_CHUNKS
            selected_chunks, start_idx = select_consecutive_chunks(all_chunks, num_content_chunks)
            
            # ALWAYS prepend title as first chunk (if available)
            if title:
                final_chunks = [title] + selected_chunks
            else:
                final_chunks = selected_chunks
            
            chunk_data = {
                'chunks': final_chunks,
                'start_idx': start_idx,
                'total_chunks': len(all_chunks),
                'video_id': video_id,
                'has_title': bool(title)
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(chunk_data, f, ensure_ascii=False)
                temp_path = f.name
            
            minio_path = f"samples/{sample_id}/transcript.json"
            storage.upload_file(minio_path, temp_path)
            os.unlink(temp_path)
            
            chunk_results[sample_id] = {'start_idx': start_idx, 'num_chunks': len(final_chunks), 'has_title': bool(title)}
            processed += 1
    
    context['task_instance'].xcom_push(key='chunk_results', value=chunk_results)
    
    logger.info("=" * 80)
    logger.info(f"TRANSCRIPT EXTRACTION: {processed} samples")
    logger.info("=" * 80)
    
    return processed


def create_training_samples(**context):
    """Create training sample records in database (train/val/test only)."""
    logger.info("=" * 80)
    logger.info("CREATING TRAINING SAMPLES")
    logger.info("=" * 80)
    
    sample_plan = context['task_instance'].xcom_pull(task_ids='pre_augmentation_sample', key='sample_plan') or []
    
    # Get frame/chunk results from unified extraction tasks
    frame_results = context['task_instance'].xcom_pull(task_ids='extract_frames', key='frame_results') or {}
    chunk_results = context['task_instance'].xcom_pull(task_ids='extract_transcript', key='chunk_results') or {}
    
    if not sample_plan:
        logger.info("No samples")
        context['task_instance'].xcom_push(key='train_count', value=0)
        context['task_instance'].xcom_push(key='val_count', value=0)
        context['task_instance'].xcom_push(key='test_count', value=0)
        return 0
    
    mode = config.get('system.mode', 'ultra_light')
    
    # Only process train/val/test samples (no inference here)
    train_samples = [s for s in sample_plan if s.get('split') == 'train']
    val_samples = [s for s in sample_plan if s.get('split') == 'val']
    test_samples = [s for s in sample_plan if s.get('split') == 'test']
    
    train_count = 0
    val_count = 0
    test_count = 0
    
    # Process train + val + test samples
    for sample in train_samples + val_samples + test_samples:
        sample_id = sample['sample_id']
        video_id = sample['video_id']
        label = sample['label']
        augment_idx = sample['augment_idx']
        split = sample['split']
        
        frame_info = frame_results.get(sample_id, [])
        chunk_info = chunk_results.get(sample_id, {})
        
        db.upsert_samples(
            sample_id=sample_id,
            video_id=video_id,
            label=label,
            augment_idx=augment_idx,
            selected_frames=json.dumps(frame_info),
            selected_chunks=json.dumps(chunk_info)
        )
        db.update_sample_split_v2(sample_id, split)
        
        if split == 'train':
            train_count += 1
        elif split == 'val':
            val_count += 1
        else:
            test_count += 1
    
    logger.info(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")
    
    context['task_instance'].xcom_push(key='train_count', value=train_count)
    context['task_instance'].xcom_push(key='val_count', value=val_count)
    context['task_instance'].xcom_push(key='test_count', value=test_count)
    
    logger.info("=" * 80)
    return train_count + val_count + test_count


def create_inference_samples(**context):
    """
    Create inference samples and mark videos as ready for inference.
    
    Saves samples to database and updates video status to 'pending_inference'.
    Inference DAG will scan PostgreSQL for videos with this status.
    """
    logger.info("=" * 80)
    logger.info("CREATING INFERENCE SAMPLES")
    logger.info("=" * 80)
    
    sample_plan = context['task_instance'].xcom_pull(task_ids='create_inference_plan', key='sample_plan') or []
    
    # Get frame/chunk results from unified extraction tasks
    frame_results = context['task_instance'].xcom_pull(task_ids='extract_frames', key='frame_results') or {}
    chunk_results = context['task_instance'].xcom_pull(task_ids='extract_transcript', key='chunk_results') or {}
    
    if not sample_plan:
        logger.info("No inference samples")
        context['task_instance'].xcom_push(key='inference_count', value=0)
        return 0
    
    inference_count = 0
    skipped_count = 0
    
    for sample in sample_plan:
        video_id = sample['video_id']
        sample_id = sample['sample_id']
        augment_idx = sample.get('augment_idx', 0)
        
        # Verify sample data exists - MUST have both frames and chunks
        frame_info = frame_results.get(sample_id, [])
        chunk_info = chunk_results.get(sample_id, {})
        
        has_frames = sample_id in frame_results
        has_chunks = sample_id in chunk_results
        
        if has_frames and has_chunks:
            # Save to database (like training samples, with split='inference')
            db.upsert_samples(
                sample_id=sample_id,
                video_id=video_id,
                label=None, 
                augment_idx=augment_idx,
                selected_frames=json.dumps(frame_info),
                selected_chunks=json.dumps(chunk_info)
            )
            db.update_sample_split_v2(sample_id, 'inference')
            
            # Update video status to 'pending_inference' (replaces Kafka queue)
            db.update_video_status(video_id, 'pending_inference')
            inference_count += 1
            logger.info(f"  ✅ Ready for inference: video={video_id[:50]}..., sample={sample_id}")
        else:
            # Skip this video - preprocessing failed
            logger.error(f"  ❌ Skipped: video={video_id} (frames={has_frames}, chunks={has_chunks})")
            db.update_video_status(video_id, 'error')
            skipped_count += 1
    
    logger.info(f"Total inference tasks: {inference_count} ready, {skipped_count} skipped")
    
    context['task_instance'].xcom_push(key='inference_count', value=inference_count)
    context['task_instance'].xcom_push(key='skipped_count', value=skipped_count)
    
    logger.info("=" * 80)
    return inference_count


def decide_training_trigger(**context):
    """Decide whether to trigger training after create_training_samples."""
    train_count = context['task_instance'].xcom_pull(task_ids='create_training_samples', key='train_count') or 0
    
    mode = config.get('system.mode', 'ultra_light')
    total_train = len(db.get_training_samples_v2(split='train'))
    
    logger.info(f"Training decision: train_count={train_count}, total_train={total_train}")
    
    if total_train >= 20:
        return 'trigger_training_pipeline'
    else:
        return 'finalize_preprocessing'


def trigger_training_pipeline(**context):
    """Trigger training DAG."""
    from airflow.api.common.trigger_dag import trigger_dag as trigger_dag_run
    
    mode = config.get('system.mode', 'ultra_light')
    run_id = f"triggered_by_preprocessing_{datetime.now().isoformat()}"
    
    try:
        trigger_dag_run(
            dag_id='training_advanced_fusion',
            run_id=run_id,
            conf={"mode": mode, "use_pregenerated_samples": True}
        )
        logger.info("✅ Triggered training DAG")
        return 1
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 0


def trigger_inference_pipeline(**context):
    """Trigger inference DAG after create_inference_samples."""
    from airflow.api.common.trigger_dag import trigger_dag as trigger_dag_run
    
    
    inference_count = context['task_instance'].xcom_pull(task_ids='create_inference_samples', key='inference_count') or 0
    
    if inference_count == 0:
        logger.info("No inference tasks, skipping trigger")
        return 0
    
    mode = config.get('system.mode', 'ultra_light')
    run_id = f"triggered_by_preprocessing_{datetime.now().isoformat()}"
    
    try:
        trigger_dag_run(
            dag_id='inference_advanced_fusion',
            run_id=run_id,
            conf={"mode": mode, "batch_size": inference_count}
        )
        logger.info(f"✅ Triggered inference DAG for {inference_count} videos")
        return 1
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 0


def finalize_preprocessing(**context):
    """
    Finalize: update video statuses.
    
    Only updates training videos to 'preprocessed'.
    Inference videos should keep status='pending_inference' (set by create_inference_samples).
    
    Also checks if there are more pending videos and triggers another run if needed.
    """
    video_ids = context['task_instance'].xcom_pull(task_ids='scan_new_videos', key='video_ids') or []
    inference_video_ids = context['task_instance'].xcom_pull(task_ids='scan_new_videos', key='inference_video_ids') or []
    
    if not video_ids:
        return 0
    
    logger.info("=" * 80)
    logger.info("FINALIZING")
    logger.info("=" * 80)
    
    # Only finalize training videos (inference videos should keep 'pending_inference' status)
    inference_video_ids_set = set(inference_video_ids)
    training_video_ids = [vid for vid in video_ids if vid not in inference_video_ids_set]
    
    completed = 0
    for video_id in training_video_ids:
        try:
            db.update_video_status(video_id, 'preprocessed')
            completed += 1
        except Exception as e:
            logger.error(f"Failed {video_id}: {e}")
    
    logger.info(f"Finalized {completed}/{len(training_video_ids)} training videos")
    logger.info(f"Skipped {len(inference_video_ids)} inference videos (keep status='pending_inference')")
    
    # =========================================================================
    # CHECK FOR MORE PENDING VIDEOS AND TRIGGER ANOTHER RUN
    # =========================================================================
    remaining_query = """
        SELECT COUNT(*) as count FROM videos 
        WHERE status_preprocess = 'pending_preprocessing'
    """
    try:
        result = db.execute(remaining_query, fetch=True)
        remaining_count = result[0]['count'] if result else 0
        
        if remaining_count > 0:
            logger.info(f"⚠️  Still {remaining_count} videos pending. Triggering another preprocessing run...")
            
            from airflow.api.common.trigger_dag import trigger_dag as trigger_dag_run
            run_id = f"auto_continue_{datetime.now().isoformat()}"
            
            try:
                trigger_dag_run(
                    dag_id='video_preprocessing_pipeline',
                    run_id=run_id,
                    conf={"auto_triggered": True, "remaining": remaining_count}
                )
                logger.info(f"✅ Triggered next preprocessing run for {remaining_count} remaining videos")
            except Exception as e:
                logger.warning(f"Could not auto-trigger next run: {e}")
                logger.info("You may need to manually trigger the preprocessing DAG again")
        else:
            logger.info("✅ All videos processed! No pending videos remaining.")
    except Exception as e:
        logger.warning(f"Could not check remaining videos: {e}")
    
    logger.info("=" * 80)
    
    return completed


# ============================================================================
# DAG DEFINITION
# ============================================================================
with DAG(
    'video_preprocessing_pipeline',
    default_args=default_args,
    description='Preprocessing with parallel training and inference paths',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=['preprocessing', 'augmentation'],
) as dag:
    
    # ========================================================================
    # SCAN & PLANNING TASKS
    # ========================================================================
    task_scan = PythonOperator(
        task_id='scan_new_videos',
        python_callable=scan_new_videos,
    )
    
    # Planning tasks run in parallel - each handles its own video type
    task_augment = PythonOperator(
        task_id='pre_augmentation_sample',
        python_callable=pre_augmentation_sample,
    )
    
    task_inference_plan = PythonOperator(
        task_id='create_inference_plan',
        python_callable=create_inference_plan,
    )
    
    # ========================================================================
    # SHARED EXTRACTION TASKS
    # ========================================================================
    task_extract_frames = PythonOperator(
        task_id='extract_frames',
        python_callable=extract_frames,
        trigger_rule='none_failed_min_one_success',
    )
    
    task_extract_transcript = PythonOperator(
        task_id='extract_transcript',
        python_callable=extract_transcript,
        trigger_rule='none_failed_min_one_success',
    )
    
    # ========================================================================
    # SAMPLE CREATION TASKS
    # ========================================================================
    # Decide which sample creation path to run (train or inference)
    def choose_sample_creation(**context):
        is_training = context['task_instance'].xcom_pull(task_ids='scan_new_videos', key='is_training_pipeline')
        if is_training:
            return 'create_training_samples'
        else:
            return 'create_inference_samples'

    task_choose_sample_creation = BranchPythonOperator(
        task_id='choose_sample_creation',
        python_callable=choose_sample_creation,
    )

    task_create_train = PythonOperator(
        task_id='create_training_samples',
        python_callable=create_training_samples,
        trigger_rule='none_failed_min_one_success',
    )

    task_create_infer = PythonOperator(
        task_id='create_inference_samples',
        python_callable=create_inference_samples,
        trigger_rule='none_failed_min_one_success',
    )
    
    # ========================================================================
    # TRIGGER TASKS
    # ========================================================================
    task_decide_train = BranchPythonOperator(
        task_id='decide_training_trigger',
        python_callable=decide_training_trigger,
    )
    
    task_trigger_train = PythonOperator(
        task_id='trigger_training_pipeline',
        python_callable=trigger_training_pipeline,
    )
    
    task_trigger_infer = PythonOperator(
        task_id='trigger_inference_pipeline',
        python_callable=trigger_inference_pipeline,
    )
    
    # ========================================================================
    # FINALIZE
    # ========================================================================
    task_finalize = PythonOperator(
        task_id='finalize_preprocessing',
        python_callable=finalize_preprocessing,
        trigger_rule='none_failed_min_one_success',
    )
    
    # ========================================================================
    # DEPENDENCIES
    # ========================================================================
    # Flow:
    #   scan → [augment, inference_plan] (parallel planning)
    #        → [extract_frames, extract_transcript] (shared extraction)
    #        → [create_training, create_inference] (parallel creation)
    #        → [decide→trigger_train, trigger_infer] (parallel triggers)
    #        → finalize
    
    # Scan → parallel planning
    task_scan >> [task_augment, task_inference_plan]
    
    # Planning → shared extraction (runs after EITHER plan completes)
    [task_augment, task_inference_plan] >> task_extract_frames
    [task_augment, task_inference_plan] >> task_extract_transcript
    
    # Extraction → branch to only one sample creation path
    [task_extract_frames, task_extract_transcript] >> task_choose_sample_creation
    task_choose_sample_creation >> [task_create_train, task_create_infer]
    
    # Training path: create → decide → trigger
    task_create_train >> task_decide_train
    task_decide_train >> task_trigger_train >> task_finalize
    task_decide_train >> task_finalize
    
    # Inference path: create → trigger
    task_create_infer >> task_trigger_infer >> task_finalize
