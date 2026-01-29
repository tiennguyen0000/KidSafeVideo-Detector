"""
Spark-based Batch Processor for Video Classification Inference.

Provides parallel batch inference using Spark for:
- Loading preprocessed samples from MinIO
- Computing embeddings in parallel
- Running fusion model inference
- Storing predictions

Optimizations:
- Batch processing within partitions to maximize GPU/CPU utilization
- Broadcast model weights to avoid serialization overhead
- Coalesce results to reduce I/O
"""

import os
import json
import logging
import tempfile
from typing import List, Dict, Iterator, Optional, Tuple
from io import BytesIO
from dataclasses import dataclass

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, '/app')

from .session import get_spark


@dataclass
class BatchInferenceConfig:
    """Configuration for batch inference."""
    mode: str = 'ultra_light'
    batch_size: int = 16
    num_frames: int = 16
    num_chunks: int = 5
    confidence_threshold: float = 0.7


class SparkBatchProcessor:
    """
    Spark-based batch processor for parallel inference.
    
    Uses mapPartitions to process batches within each partition,
    minimizing model loading overhead.
    """
    
    def __init__(self, spark=None):
        """
        Initialize the batch processor.
        
        Args:
            spark: SparkSession instance (optional)
        """
        if spark is None:
            self.spark = get_spark()
        else:
            self.spark = spark
        
        self.mode = 'ultra_light'
        self.storage_config = {}
    
    def set_config(self, storage_config: dict, mode: str):
        """
        Set configuration for storage and mode.
        
        Args:
            storage_config: MinIO storage config
            mode: Model mode ('ultra_light' or 'balanced')
        """
        self.storage_config = storage_config
        self.mode = mode
        logger.info(f"SparkBatchProcessor configured for mode: {mode}")
    
    def batch_inference(
        self,
        sample_ids: List[str],
        model_path: str,
        batch_size: int = 32,
        num_partitions: int = 4
    ) -> List[Dict]:
        """
        Run batch inference on samples using Spark.
        
        Args:
            sample_ids: List of sample IDs to process
            model_path: Path to model weights in MinIO
            batch_size: Batch size for inference
            num_partitions: Number of Spark partitions
            
        Returns:
            List of prediction dicts with y_pred, probabilities, confidence
        """
        if not sample_ids:
            return []
        
        logger.info(f"Spark batch inference: {len(sample_ids)} samples, mode={self.mode}")
        
        # Broadcast configuration
        config_bc = self.spark.sparkContext.broadcast({
            'mode': self.mode,
            'batch_size': batch_size,
            'num_frames': 16,
            'num_chunks': 5,
            'storage': self.storage_config,
            'model_path': model_path,
        })
        
        # Create RDD
        samples_rdd = self.spark.sparkContext.parallelize(sample_ids, num_partitions)
        
        def process_partition(partition: Iterator[str]) -> Iterator[Dict]:
            """Process a partition of samples."""
            sample_list = list(partition)
            if not sample_list:
                return
            
            import sys
            sys.path.insert(0, '/app')
            
            from PIL import Image
            from io import BytesIO
            import json
            import torch
            import tempfile
            import os
            from minio import Minio
            
            cfg = config_bc.value
            mode = cfg['mode']
            storage_cfg = cfg['storage']
            model_path = cfg['model_path']
            num_frames = cfg['num_frames']
            num_chunks = cfg['num_chunks']
            
            # Create MinIO client
            minio_client = Minio(
                f"{storage_cfg['host']}:{storage_cfg['port']}",
                access_key=storage_cfg['access_key'],
                secret_key=storage_cfg['secret_key'],
                secure=False
            )
            bucket = storage_cfg.get('bucket', 'video-storage')
            
            # Import model components (same as training)
            from common.io import config as app_config
            from common.io.database import DatabaseClient
            from common.models.image_encoder import get_image_encoder
            from common.models.text_encoder import get_text_encoder
            from common.models.advanced_fusion import LABELS
            from common.pipelines.train_with_advanced_fusion import VideoClassifierWithAttnPool
            
            # Load model config
            model_cfg = app_config.models.get(mode, {})
            d_img = model_cfg.get('image_embedding_dim', 1280)
            d_txt = model_cfg.get('text_embedding_dim', 768)
            
            # Get model metadata to extract training config (for d_fused, dropout, etc.)
            # Try to load from database, fallback to defaults
            d_fused = 256
            dropout = 0.3
            num_heads = 4
            use_multihead_pool = (mode == 'balanced')
            
            if model_path:
                try:
                    # Get model from database to access training_config
                    app_db = DatabaseClient()
                    all_models = app_db.execute(
                        "SELECT training_config FROM model_registry WHERE artifact_path = %s AND is_active = true LIMIT 1",
                        (model_path,),
                        fetch=True
                    )
                    if all_models and all_models[0].get('training_config'):
                        import json
                        training_config = all_models[0]['training_config']
                        if isinstance(training_config, str):
                            training_config = json.loads(training_config) if training_config else {}
                        elif training_config is None:
                            training_config = {}
                        
                        d_fused = training_config.get('d_fused', 256)
                        dropout = training_config.get('dropout', 0.3)
                        num_heads = training_config.get('num_heads', 4)
                        use_multihead_pool = training_config.get('use_multihead_pool', (mode == 'balanced'))
                        logger.debug(f"Loaded model config: d_fused={d_fused}, dropout={dropout}, num_heads={num_heads}, multihead={use_multihead_pool}")
                except Exception as e:
                    logger.warning(f"Could not load model metadata, using defaults: {e}")
            
            # Initialize encoders (once per partition) - same as training
            img_encoder = get_image_encoder(mode, num_classes=None, config=model_cfg)
            txt_encoder = get_text_encoder(mode, num_classes=None, config=model_cfg)
            img_encoder.eval()
            txt_encoder.eval()
            
            # Load fusion model using VideoClassifierWithAttnPool (same as training)
            fusion_model = None
            if model_path:
                try:
                    # Download model
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                        minio_client.fget_object(bucket, model_path, f.name)
                        state = torch.load(f.name, map_location='cpu', weights_only=True)
                        os.unlink(f.name)
                    
                    # Build model using VideoClassifierWithAttnPool (same as training)
                    fusion_model = VideoClassifierWithAttnPool(
                        d_img=d_img,
                        d_txt=d_txt,
                        d_fused=d_fused,
                        dropout=dropout,
                        num_heads=num_heads,
                        num_layers=1,
                        use_multihead_pool=use_multihead_pool
                    )
                    fusion_model.load_state_dict(state)
                    fusion_model.eval()
                    
                    logger.debug(f"Loaded fusion model: d_fused={d_fused}, dropout={dropout}, multihead={use_multihead_pool}")
                except Exception as e:
                    logger.error(f"Failed to load fusion model: {e}", exc_info=True)
            
            # Process each sample
            for sample_id in sample_list:
                try:
                    # Load frames
                    frames = []
                    for j in range(num_frames):
                        path = f"samples/{sample_id}/frame_{j:02d}.jpg"
                        try:
                            response = minio_client.get_object(bucket, path)
                            img = Image.open(BytesIO(response.read())).convert('RGB')
                            frames.append(img)
                            response.close()
                            response.release_conn()
                        except:
                            pass
                    
                    while len(frames) < num_frames:
                        frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))
                    
                    # Load transcript
                    chunks = []
                    path = f"samples/{sample_id}/transcript.json"
                    try:
                        response = minio_client.get_object(bucket, path)
                        data = json.loads(response.read().decode('utf-8'))
                        chunks = data.get('chunks', [])
                        response.close()
                        response.release_conn()
                    except:
                        pass
                    
                    while len(chunks) < num_chunks:
                        chunks.append("")
                    chunks = chunks[:num_chunks]
                    
                    # Compute embeddings
                    with torch.no_grad():
                        img_embs = img_encoder.get_embeddings(frames)
                        txt_embs = txt_encoder.get_embeddings(chunks)
                    
                    # Run fusion using VideoClassifierWithAttnPool (same as training)
                    if fusion_model:
                        with torch.no_grad():
                            # Pad/truncate to correct sequence lengths
                            if img_embs.size(0) < num_frames:
                                pad = torch.zeros(num_frames - img_embs.size(0), d_img)
                                img_embs = torch.cat([img_embs, pad], dim=0)
                            elif img_embs.size(0) > num_frames:
                                img_embs = img_embs[:num_frames]
                            
                            if txt_embs.size(0) < num_chunks:
                                pad = torch.zeros(num_chunks - txt_embs.size(0), d_txt)
                                txt_embs = torch.cat([txt_embs, pad], dim=0)
                            elif txt_embs.size(0) > num_chunks:
                                txt_embs = txt_embs[:num_chunks]
                            
                            # Add batch dimension: (1, seq_len, dim)
                            img_embs_batch = img_embs.unsqueeze(0)
                            txt_embs_batch = txt_embs.unsqueeze(0)
                            
                            # Forward pass (same as training inference)
                            logits = fusion_model(img_embs_batch, txt_embs_batch)  # (1, num_classes)
                            
                            probs = torch.softmax(logits, dim=-1)[0]
                            pred_idx = probs.argmax().item()
                            confidence = probs[pred_idx].item()
                            
                            yield {
                                'sample_id': sample_id,
                                'y_pred': LABELS[pred_idx],
                                'probabilities': {LABELS[i]: probs[i].item() for i in range(len(LABELS))},
                                'confidence': confidence,
                                'status': 'success',
                            }
                    else:
                        yield {
                            'sample_id': sample_id,
                            'y_pred': 'Safe',
                            'probabilities': {},
                            'confidence': 0.5,
                            'status': 'no_model',
                        }
                        
                except Exception as e:
                    logger.error(f"Failed to process {sample_id}: {e}")
                    yield {
                        'sample_id': sample_id,
                        'y_pred': 'Unknown',
                        'probabilities': {},
                        'confidence': 0.0,
                        'status': 'failed',
                        'error': str(e),
                    }
        
        results = samples_rdd.mapPartitions(process_partition).collect()
        
        success = sum(1 for r in results if r.get('status') == 'success')
        logger.info(f"Batch inference complete: {success}/{len(sample_ids)} success")
        
        return results
    
    def run_batch_inference(
        self,
        sample_ids: List[str],
        mode: str = 'ultra_light',
        batch_size: int = 16,
        model_path: str = None
    ) -> List[Dict]:
        """
        Alias for batch_inference with mode setting.
        
        Args:
            sample_ids: List of sample IDs
            mode: Model mode
            batch_size: Batch size
            model_path: Path to model
            
        Returns:
            List of predictions
        """
        self.mode = mode
        return self.batch_inference(sample_ids, model_path, batch_size)
    
    def compute_embeddings_batch(
        self,
        sample_ids: List[str],
        mode: str = 'ultra_light',
        output_dir: str = None
    ) -> Dict[str, any]:
        """
        Compute and cache embeddings using Spark.
        
        Args:
            sample_ids: List of sample IDs
            mode: Model mode
            output_dir: Cache directory
            
        Returns:
            Dict with status counts
        """
        if not sample_ids:
            return {'success': 0, 'failed': 0}
        
        logger.info(f"Computing embeddings for {len(sample_ids)} samples")
        
        config_bc = self.spark.sparkContext.broadcast({
            'mode': mode,
            'output_dir': output_dir or '/tmp/embedding_cache',
            'storage': self.storage_config,
        })
        
        num_partitions = min(len(sample_ids), self.spark.sparkContext.defaultParallelism)
        samples_rdd = self.spark.sparkContext.parallelize(sample_ids, num_partitions)
        
        def compute_partition(partition: Iterator[str]) -> Iterator[Dict]:
            import sys
            sys.path.insert(0, '/app')
            
            from PIL import Image
            from io import BytesIO
            import json
            import torch
            import os
            from minio import Minio
            
            cfg = config_bc.value
            mode = cfg['mode']
            cache_dir = f"{cfg['output_dir']}/{mode}"
            storage_cfg = cfg['storage']
            
            os.makedirs(cache_dir, exist_ok=True)
            
            minio_client = Minio(
                f"{storage_cfg['host']}:{storage_cfg['port']}",
                access_key=storage_cfg['access_key'],
                secret_key=storage_cfg['secret_key'],
                secure=False
            )
            bucket = storage_cfg.get('bucket', 'video-storage')
            
            from common.io import config as app_config
            from common.models.image_encoder import get_image_encoder
            from common.models.text_encoder import get_text_encoder
            
            model_cfg = app_config.models.get(mode, {})
            
            img_encoder = get_image_encoder(mode, num_classes=None, config=model_cfg)
            txt_encoder = get_text_encoder(mode, num_classes=None, config=model_cfg)
            img_encoder.eval()
            txt_encoder.eval()
            
            for sample_id in partition:
                cache_path = os.path.join(cache_dir, f"{sample_id}.pt")
                
                if os.path.exists(cache_path):
                    yield {'sample_id': sample_id, 'status': 'cached'}
                    continue
                
                try:
                    # Load frames
                    frames = []
                    for j in range(16):
                        path = f"samples/{sample_id}/frame_{j:02d}.jpg"
                        try:
                            response = minio_client.get_object(bucket, path)
                            img = Image.open(BytesIO(response.read())).convert('RGB')
                            frames.append(img)
                            response.close()
                            response.release_conn()
                        except:
                            pass
                    
                    while len(frames) < 16:
                        frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))
                    
                    # Load transcript
                    chunks = []
                    path = f"samples/{sample_id}/transcript.json"
                    try:
                        response = minio_client.get_object(bucket, path)
                        data = json.loads(response.read().decode('utf-8'))
                        chunks = data.get('chunks', [])
                        response.close()
                        response.release_conn()
                    except:
                        pass
                    
                    while len(chunks) < 5:
                        chunks.append("")
                    chunks = chunks[:5]
                    
                    # Compute embeddings
                    with torch.no_grad():
                        img_embs = img_encoder.get_embeddings(frames).cpu()
                        txt_embs = txt_encoder.get_embeddings(chunks).cpu()
                    
                    torch.save({'img_embs': img_embs, 'txt_embs': txt_embs}, cache_path)
                    
                    yield {'sample_id': sample_id, 'status': 'success'}
                    
                except Exception as e:
                    yield {'sample_id': sample_id, 'status': 'failed', 'error': str(e)}
        
        results = samples_rdd.mapPartitions(compute_partition).collect()
        
        success = sum(1 for r in results if r.get('status') in ('success', 'cached'))
        failed = sum(1 for r in results if r.get('status') == 'failed')
        cached = sum(1 for r in results if r.get('status') == 'cached')
        
        logger.info(f"Embedding computation: {success} success ({cached} cached), {failed} failed")
        
        return {'success': success, 'failed': failed, 'cached': cached}
    
    def upload_embeddings_to_minio(
        self,
        mode: str = 'balanced',
        local_cache_dir: str = None,
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Upload local cached embeddings to MinIO for persistence.
        
        Args:
            mode: Model mode (ultra_light or balanced)
            local_cache_dir: Local cache directory (default: /tmp/embedding_cache/{mode})
            batch_size: Number of files to upload per batch
            
        Returns:
            Dict with upload statistics
        """
        import sys
        sys.path.insert(0, '/app')
        from common.io import storage
        
        cache_dir = local_cache_dir or f'/tmp/embedding_cache/{mode}'
        minio_prefix = f'embeddings/{mode}'
        
        if not os.path.exists(cache_dir):
            logger.warning(f"Cache directory not found: {cache_dir}")
            return {'local': 0, 'uploaded': 0, 'already_in_minio': 0}
        
        # Get local files
        local_files = [f for f in os.listdir(cache_dir) if f.endswith('.pt')]
        logger.info(f"Found {len(local_files)} local embeddings for mode: {mode}")
        
        # Check what's already in MinIO
        existing_in_minio = set()
        try:
            objects = storage.list_objects(prefix=minio_prefix)
            for obj_path in objects:
                if obj_path.endswith('.pt'):
                    filename = os.path.basename(obj_path)
                    existing_in_minio.add(filename)
        except Exception as e:
            logger.warning(f"Failed to list MinIO objects: {e}")
        
        # Filter files to upload
        to_upload = [f for f in local_files if f not in existing_in_minio]
        logger.info(f"To upload: {len(to_upload)} (already in MinIO: {len(existing_in_minio)})")
        
        uploaded = 0
        for i in range(0, len(to_upload), batch_size):
            batch = to_upload[i:i + batch_size]
            
            for filename in batch:
                local_path = os.path.join(cache_dir, filename)
                minio_path = f"{minio_prefix}/{filename}"
                
                try:
                    if storage.upload_file(minio_path, local_path):
                        uploaded += 1
                except Exception as e:
                    logger.debug(f"Failed to upload {filename}: {e}")
            
            if (i + batch_size) % 500 == 0:
                logger.info(f"Upload progress: {min(i + batch_size, len(to_upload))}/{len(to_upload)}")
        
        logger.info(f"Uploaded {uploaded} embeddings to MinIO")
        
        return {
            'local': len(local_files),
            'uploaded': uploaded,
            'already_in_minio': len(existing_in_minio)
        }
    
    def download_embeddings_from_minio(
        self,
        sample_ids: List[str],
        mode: str = 'balanced',
        local_cache_dir: str = None,
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Download embeddings from MinIO to local cache.
        
        Args:
            sample_ids: List of sample IDs to download
            mode: Model mode
            local_cache_dir: Local cache directory
            batch_size: Batch size for downloads
            
        Returns:
            Dict with download statistics
        """
        import sys
        sys.path.insert(0, '/app')
        from common.io import storage
        
        cache_dir = local_cache_dir or f'/tmp/embedding_cache/{mode}'
        minio_prefix = f'embeddings/{mode}'
        
        os.makedirs(cache_dir, exist_ok=True)
        
        downloaded = 0
        skipped = 0
        not_found = 0
        
        for i in range(0, len(sample_ids), batch_size):
            batch = sample_ids[i:i + batch_size]
            
            for sample_id in batch:
                local_path = os.path.join(cache_dir, f"{sample_id}.pt")
                
                # Skip if exists locally
                if os.path.exists(local_path):
                    skipped += 1
                    continue
                
                minio_path = f"{minio_prefix}/{sample_id}.pt"
                
                try:
                    data = storage.download_data(minio_path, silent=True)
                    if data:
                        with open(local_path, 'wb') as f:
                            f.write(data)
                        downloaded += 1
                    else:
                        not_found += 1
                except Exception as e:
                    not_found += 1
            
            if (i + batch_size) % 500 == 0:
                logger.info(f"Download progress: {min(i + batch_size, len(sample_ids))}/{len(sample_ids)}")
        
        logger.info(f"Downloaded {downloaded} embeddings (skipped: {skipped}, not found: {not_found})")
        
        return {
            'downloaded': downloaded,
            'skipped': skipped,
            'not_found': not_found
        }
