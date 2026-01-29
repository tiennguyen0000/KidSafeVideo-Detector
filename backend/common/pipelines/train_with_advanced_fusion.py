"""
Video Classification Training Pipeline.

Architecture:
    Image: 16 frames → FrozenEncoder → [16, 1280] → AttentionPool → [1280]
    Text:  5 chunks  → FrozenEncoder → [5, 768]   → AttentionPool → [768]
    Fusion: GatedFusion(img_emb, txt_emb) → logits (4 classes)

Classes: Safe, Aggressive, Sexual, Superstition

Key design:
    - Encoders FROZEN (pretrained models are strong enough)
    - AttentionPooling learns to weight important frames/chunks
    - Lightweight fusion head (~500K params)
"""

import os
import sys
import time
import json
import random
import logging
import tempfile
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, field
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from io import BytesIO
import numpy as np

sys.path.insert(0, '/app')

from common.io import db, storage, config
from common.models.text_encoder import get_text_encoder
from common.models.image_encoder import get_image_encoder
from common.models.advanced_fusion import (
    GatedFusionClassifier, AttentionFusionClassifier, 
    compute_classification_loss, LABELS, NUM_CLASSES
)

# ============================================================================
# CONSTANTS
# ============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

NUM_FRAMES = 16      # Matches preprocessing
NUM_CHUNKS = 5       # Matches preprocessing


# ============================================================================
# MIXUP AUGMENTATION (at embedding level)
# ============================================================================
def mixup_embeddings(img_embs: torch.Tensor, txt_embs: torch.Tensor, labels: torch.Tensor, 
                     alpha: float = 0.4) -> tuple:
    """
    Apply Mixup augmentation at the embedding level.
    
    This helps with regularization by creating virtual training examples
    through linear interpolation of embeddings and labels.
    
    Args:
        img_embs: (batch, seq_len, d_img) image embeddings
        txt_embs: (batch, seq_len, d_txt) text embeddings
        labels: (batch,) integer class labels
        alpha: Beta distribution parameter (higher = more mixing)
    
    Returns:
        mixed_img_embs, mixed_txt_embs, labels_a, labels_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = img_embs.size(0)
    index = torch.randperm(batch_size, device=img_embs.device)
    
    # Mix embeddings
    mixed_img = lam * img_embs + (1 - lam) * img_embs[index]
    mixed_txt = lam * txt_embs + (1 - lam) * txt_embs[index]
    
    # Get both sets of labels for loss computation
    labels_a, labels_b = labels, labels[index]
    
    return mixed_img, mixed_txt, labels_a, labels_b, lam


def mixup_criterion(pred: torch.Tensor, labels_a: torch.Tensor, labels_b: torch.Tensor, 
                   lam: float, label_smoothing: float = 0.1) -> torch.Tensor:
    """Compute mixup loss as weighted combination of two CE losses."""
    from common.models.advanced_fusion import compute_classification_loss
    
    loss_a = compute_classification_loss(pred, labels_a, label_smoothing=label_smoothing, 
                                        use_focal_loss=True, focal_gamma=2.0)['loss']
    loss_b = compute_classification_loss(pred, labels_b, label_smoothing=label_smoothing,
                                        use_focal_loss=True, focal_gamma=2.0)['loss']
    
    return lam * loss_a + (1 - lam) * loss_b


# ============================================================================
# MULTI-HEAD ATTENTION POOLING
# ============================================================================
class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling for more expressive sequence aggregation.
    
    Uses multiple attention heads to capture different aspects of the sequence,
    then combines them. Better than single-head for high-dimensional embeddings.
    
    Input:  (batch, seq_len, d_model)
    Output: (batch, d_model)
    """
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Learnable query for each head
        self.queries = nn.Parameter(torch.randn(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.queries)
        
        # Project input to keys
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) - 1 for valid, 0 for padding
        
        Returns:
            pooled: (batch, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to keys and values
        keys = self.key_proj(x)  # (batch, seq_len, d_model)
        values = self.value_proj(x)  # (batch, seq_len, d_model)
        
        # Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose: (batch, num_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Compute attention with learnable queries
        # queries: (num_heads, head_dim) -> (1, num_heads, 1, head_dim)
        queries = self.queries.unsqueeze(0).unsqueeze(2)  # (1, num_heads, 1, head_dim)
        
        # Attention scores: (batch, num_heads, 1, seq_len)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.squeeze(2)  # (batch, num_heads, seq_len)
        
        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, seq_len)
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)  # (batch, num_heads, seq_len)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention: (batch, num_heads, head_dim)
        pooled = torch.einsum('bhs,bhsd->bhd', attn_weights, values)
        
        # Reshape back: (batch, d_model)
        pooled = pooled.reshape(batch_size, self.d_model)
        
        # Output projection with residual (using mean of input as residual)
        pooled = self.out_proj(pooled)
        pooled = self.layer_norm(pooled + x.mean(dim=1))
        
        return pooled


# ============================================================================
# SIMPLE ATTENTION POOLING (backward compatible)
# ============================================================================
class AttentionPooling(nn.Module):
    """
    Learnable attention pooling to aggregate sequence of embeddings.
    
    Input:  (batch, seq_len, d_model)
    Output: (batch, d_model)
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Linear(d_model, 1, bias=False)
        nn.init.xavier_uniform_(self.query.weight)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) - 1 for valid, 0 for padding
        """
        # Compute attention scores
        scores = self.query(x).squeeze(-1)  # (batch, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Weighted sum
        output = (x * weights.unsqueeze(-1)).sum(dim=1)  # (batch, d_model)
        return output


# ============================================================================
# METRICS
# ============================================================================
@dataclass
class Metrics:
    loss: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: np.ndarray = None
    
    def to_dict(self) -> Dict:
        return {
            'loss': self.loss, 'accuracy': self.accuracy,
            'precision': self.precision, 'recall': self.recall,
            'f1': self.f1, 'per_class': self.per_class
        }


def compute_metrics(y_true: List[int], y_pred: List[int], loss: float) -> Metrics:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    accuracy = (y_true == y_pred).mean()
    
    per_class = {}
    precisions, recalls, f1s = [], [], []
    
    for idx, label in IDX_TO_LABEL.items():
        tp = ((y_pred == idx) & (y_true == idx)).sum()
        fp = ((y_pred == idx) & (y_true != idx)).sum()
        fn = ((y_pred != idx) & (y_true == idx)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        per_class[label] = {'precision': prec, 'recall': rec, 'f1': f1}
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    return Metrics(
        loss=loss, accuracy=accuracy,
        precision=np.mean(precisions), recall=np.mean(recalls), f1=np.mean(f1s),
        per_class=per_class, confusion_matrix=cm
    )


# ============================================================================
# PRINTING UTILITIES
# ============================================================================
def print_header(title: str):
    print(f"\n{'='*70}\n {title}\n{'='*70}")

def print_config(cfg: Dict):
    print_header("TRAINING CONFIGURATION")
    for k, v in cfg.items():
        print(f"  {k:20s}: {v}")

def print_metrics_table(train_m: Metrics, val_m: Metrics):
    gap = train_m.accuracy - val_m.accuracy
    gap_str = f"{gap:+.3f} {'⚠️ OVERFIT' if gap > 0.10 else '✅'}"
    
    print(f"\n┌{'─'*68}┐")
    print(f"│ {'Metric':<15} │ {'Train':>12} │ {'Val':>12} │ {'Gap':>20} │")
    print(f"├{'─'*68}┤")
    print(f"│ {'Loss':<15} │ {train_m.loss:>12.4f} │ {val_m.loss:>12.4f} │ {'':<20} │")
    print(f"│ {'Accuracy':<15} │ {train_m.accuracy:>12.3f} │ {val_m.accuracy:>12.3f} │ {gap_str:<20} │")
    print(f"│ {'F1 Score':<15} │ {train_m.f1:>12.3f} │ {val_m.f1:>12.3f} │ {'':<20} │")
    print(f"└{'─'*68}┘")

def print_per_class(metrics: Metrics):
    print(f"\n  Validation Per-Class:")
    print(f"  ┌{'─'*50}┐")
    print(f"  │ {'Class':<12} │ {'Prec':>8} │ {'Recall':>8} │ {'F1':>8} │")
    print(f"  ├{'─'*50}┤")
    for label, m in metrics.per_class.items():
        print(f"  │ {label:<12} │ {m['precision']:>8.3f} │ {m['recall']:>8.3f} │ {m['f1']:>8.3f} │")
    print(f"  └{'─'*50}┘")

def print_confusion_matrix(cm: np.ndarray):
    print(f"\n  Confusion Matrix:")
    print(f"  ┌{'─'*52}┐")
    header = "│ " + f"{'':>10}" + " │ " + " │ ".join(f"{l[:4]:>6}" for l in LABELS) + " │"
    print(f"  {header}")
    print(f"  ├{'─'*52}┤")
    for i, label in enumerate(LABELS):
        row = " │ ".join(f"{cm[i,j]:>6}" for j in range(NUM_CLASSES))
        print(f"  │ {label[:10]:>10} │ {row} │")
    print(f"  └{'─'*52}┘")

def print_class_distribution(samples: List[Dict], split: str):
    counts = Counter(s['label'] for s in samples)
    print(f"\n  {split} distribution:")
    for label in LABELS:
        print(f"    {label:<12}: {counts.get(label, 0)}")


# ============================================================================
# DATASET
# ============================================================================
# EMBEDDING CACHE MANAGER - Hybrid MinIO + Local
# ============================================================================
CACHE_DIR = "/tmp/embedding_cache"
MINIO_EMBEDDING_PREFIX = "embeddings"  # embeddings/{mode}/{sample_id}.pt


class EmbeddingCacheManager:
    """
    Hybrid Approach: MinIO (persistent) + Local (fast training)
    
    Flow:
        1. Check MinIO trước khi compute
        2. Download cached embeddings nếu có
        3. Compute những cái chưa có
        4. Upload embeddings mới lên MinIO
    
    Benefits:
        - Persistent: không mất khi container restart
        - Shared: dùng chung giữa workers
        - Tiết kiệm 2+ giờ mỗi lần train mới
        - Versioned theo mode (ultra_light vs balanced)
    """
    
    def __init__(self, mode: str, local_cache_dir: str = None):
        self.mode = mode
        self.local_cache_dir = local_cache_dir or f"{CACHE_DIR}/{mode}"
        self.minio_prefix = f"{MINIO_EMBEDDING_PREFIX}/{mode}"
        
        os.makedirs(self.local_cache_dir, exist_ok=True)
    
    def _get_minio_path(self, sample_id: str) -> str:
        """Get MinIO object path for a sample."""
        return f"{self.minio_prefix}/{sample_id}.pt"
    
    def _get_local_path(self, sample_id: str) -> str:
        """Get local cache path for a sample."""
        return os.path.join(self.local_cache_dir, f"{sample_id}.pt")
    
    def get_cached_in_minio(self) -> set:
        """
        Get set of sample_ids cached in MinIO.
        Uses list_objects with prefix for efficiency.
        """
        try:
            objects = storage.list_objects(prefix=self.minio_prefix)
            # Extract sample_id from path: embeddings/{mode}/{sample_id}.pt
            sample_ids = set()
            for obj_path in objects:
                if obj_path.endswith('.pt'):
                    filename = os.path.basename(obj_path)
                    sample_id = filename.replace('.pt', '')
                    sample_ids.add(sample_id)
            return sample_ids
        except Exception as e:
            logger.warning(f"Failed to list MinIO embeddings: {e}")
            return set()
    
    def get_cached_locally(self) -> set:
        """Get set of sample_ids cached locally."""
        if not os.path.exists(self.local_cache_dir):
            return set()
        return set(
            f.replace('.pt', '') 
            for f in os.listdir(self.local_cache_dir) 
            if f.endswith('.pt')
        )
    
    def download_from_minio(self, sample_ids: List[str], batch_size: int = 100) -> int:
        """
        Download embeddings from MinIO to local cache.
        
        Uses batched downloads to avoid I/O explosion.
        
        Returns: Number of successfully downloaded embeddings
        """
        if not sample_ids:
            return 0
        
        downloaded = 0
        total = len(sample_ids)
        
        logger.info(f"Downloading {total} embeddings from MinIO...")
        
        for i in range(0, total, batch_size):
            batch = sample_ids[i:i + batch_size]
            
            for sample_id in batch:
                minio_path = self._get_minio_path(sample_id)
                local_path = self._get_local_path(sample_id)
                
                # Skip if already exists locally
                if os.path.exists(local_path):
                    downloaded += 1
                    continue
                
                try:
                    data = storage.download_data(minio_path, silent=True)
                    if data:
                        with open(local_path, 'wb') as f:
                            f.write(data)
                        downloaded += 1
                except Exception as e:
                    logger.debug(f"Failed to download {sample_id}: {e}")
            
            # Progress
            if (i + batch_size) % 500 == 0 or (i + batch_size) >= total:
                logger.info(f"  Download progress: {min(i + batch_size, total)}/{total}")
        
        return downloaded
    
    def upload_to_minio(self, sample_ids: List[str] = None, batch_size: int = 100) -> int:
        """
        Upload embeddings from local cache to MinIO.
        
        If sample_ids is None, uploads all locally cached embeddings.
        Uses batched uploads to avoid I/O explosion.
        
        Returns: Number of successfully uploaded embeddings
        """
        if sample_ids is None:
            sample_ids = list(self.get_cached_locally())
        
        if not sample_ids:
            return 0
        
        # Check what's already in MinIO
        existing_in_minio = self.get_cached_in_minio()
        to_upload = [sid for sid in sample_ids if sid not in existing_in_minio]
        
        if not to_upload:
            logger.info(f"All {len(sample_ids)} embeddings already in MinIO")
            return 0
        
        uploaded = 0
        total = len(to_upload)
        
        logger.info(f"Uploading {total} embeddings to MinIO...")
        
        for i in range(0, total, batch_size):
            batch = to_upload[i:i + batch_size]
            
            for sample_id in batch:
                local_path = self._get_local_path(sample_id)
                minio_path = self._get_minio_path(sample_id)
                
                if not os.path.exists(local_path):
                    continue
                
                try:
                    if storage.upload_file(minio_path, local_path, content_type='application/octet-stream'):
                        uploaded += 1
                except Exception as e:
                    logger.debug(f"Failed to upload {sample_id}: {e}")
            
            # Progress
            if (i + batch_size) % 500 == 0 or (i + batch_size) >= total:
                logger.info(f"  Upload progress: {min(i + batch_size, total)}/{total}")
        
        return uploaded
    
    def sync_before_compute(self, needed_sample_ids: List[str]) -> List[str]:
        """
        Sync embeddings from MinIO before computing.
        
        Flow:
            1. Check what's in local cache
            2. Check what's in MinIO
            3. Download from MinIO what we need but don't have locally
            4. Return list of sample_ids that still need computing
        
        Returns: List of sample_ids that need to be computed
        """
        print(f"  🔄 Syncing embeddings from MinIO...")
        
        # What we already have locally
        local_cached = self.get_cached_locally()
        needed_set = set(needed_sample_ids)
        
        # What we need but don't have locally
        missing_locally = needed_set - local_cached
        
        if not missing_locally:
            print(f"  ✅ All {len(needed_set)} embeddings already in local cache")
            return []
        
        # Check what's in MinIO
        minio_cached = self.get_cached_in_minio()
        
        # Download from MinIO what we need
        can_download = missing_locally & minio_cached
        
        if can_download:
            print(f"  ⬇️  Downloading {len(can_download)} embeddings from MinIO...")
            downloaded = self.download_from_minio(list(can_download))
            print(f"  ✅ Downloaded {downloaded} embeddings from MinIO")
        else:
            print(f"  ℹ️  No embeddings found in MinIO for this batch")
        
        # What still needs computing
        local_cached_after = self.get_cached_locally()
        to_compute = [sid for sid in needed_sample_ids if sid not in local_cached_after]
        
        print(f"  📊 Status: {len(local_cached_after)}/{len(needed_set)} cached, {len(to_compute)} to compute")
        
        return to_compute
    
    def sync_after_compute(self, computed_sample_ids: List[str]) -> int:
        """
        Upload newly computed embeddings to MinIO.
        
        Returns: Number of embeddings uploaded
        """
        if not computed_sample_ids:
            return 0
        
        print(f"  ⬆️  Uploading {len(computed_sample_ids)} new embeddings to MinIO...")
        uploaded = self.upload_to_minio(computed_sample_ids)
        print(f"  ✅ Uploaded {uploaded} embeddings to MinIO")
        
        return uploaded
    
    def invalidate_cache(self):
        """
        Invalidate cache for this mode (when model changes).
        Removes from both local and MinIO.
        """
        # Clear local
        import shutil
        if os.path.exists(self.local_cache_dir):
            shutil.rmtree(self.local_cache_dir)
            os.makedirs(self.local_cache_dir, exist_ok=True)
        
        # Clear MinIO
        try:
            objects = storage.list_objects(prefix=self.minio_prefix)
            for obj_path in objects:
                storage.delete_object(obj_path)
            logger.info(f"Invalidated cache for mode: {self.mode}")
        except Exception as e:
            logger.warning(f"Failed to clear MinIO cache: {e}")


def upload_existing_cache_to_minio(mode: str) -> Dict:
    """
    Upload existing local cache to MinIO.
    
    Useful for migrating existing cache after implementing this feature.
    
    Returns: Dict with upload statistics
    """
    print_header(f"UPLOADING EXISTING CACHE TO MINIO (mode={mode})")
    
    cache_manager = EmbeddingCacheManager(mode)
    
    local_cached = cache_manager.get_cached_locally()
    minio_cached = cache_manager.get_cached_in_minio()
    
    print(f"  Local cache: {len(local_cached)} embeddings")
    print(f"  MinIO cache: {len(minio_cached)} embeddings")
    
    to_upload = local_cached - minio_cached
    print(f"  To upload: {len(to_upload)} embeddings")
    
    if to_upload:
        uploaded = cache_manager.upload_to_minio(list(to_upload))
        print(f"  ✅ Uploaded {uploaded} embeddings to MinIO")
    else:
        print(f"  ✅ All embeddings already in MinIO")
        uploaded = 0
    
    return {
        'local_cached': len(local_cached),
        'minio_cached_before': len(minio_cached),
        'uploaded': uploaded,
        'minio_cached_after': len(minio_cached) + uploaded
    }


# ============================================================================
# DATASET - Disk-based embeddings (memory efficient)
# ============================================================================


class PrecomputedDataset(Dataset):
    """
    Dataset with pre-computed embeddings stored on DISK.
    
    Embeddings are computed ONCE, saved to disk, then loaded on-demand.
    This prevents OOM errors with large datasets.
    """
    
    def __init__(self, cache_dir: str, samples: List[Dict], d_img: int, d_txt: int):
        """
        Args:
            cache_dir: Directory containing cached embeddings
            samples: List of sample metadata with 'sample_id', 'label'
            d_img: Image embedding dimension
            d_txt: Text embedding dimension
        """
        self.cache_dir = cache_dir
        self.samples = samples
        self.d_img = d_img
        self.d_txt = d_txt
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        sample_id = s['sample_id']
        
        # Load from disk
        cache_path = os.path.join(self.cache_dir, f"{sample_id}.pt")
        
        if os.path.exists(cache_path):
            data = torch.load(cache_path, weights_only=True)
            img_embs = data['img_embs']
            txt_embs = data['txt_embs']
        else:
            # Fallback to zeros
            img_embs = torch.zeros(NUM_FRAMES, self.d_img)
            txt_embs = torch.zeros(NUM_CHUNKS, self.d_txt)
        
        return {
            'sample_id': sample_id,
            'video_id': s['video_id'],
            'img_embs': img_embs,
            'txt_embs': txt_embs,
            'label': s['label'],
            'y': LABEL_TO_IDX.get(s['label'], 0)
        }


def _check_and_invalidate_stale_cache(mode: str, d_img: int, d_txt: int, cache_dir: str) -> bool:
    """
    Check if cached embeddings are compatible with current config.
    
    Invalidates cache if:
    - d_img changed (e.g., 512 -> 2048 for balanced mode)
    - d_txt changed
    
    Returns: True if cache was invalidated
    """
    metadata_path = os.path.join(cache_dir, "_cache_metadata.json")
    
    current_meta = {
        'mode': mode,
        'd_img': d_img,
        'd_txt': d_txt,
    }
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                saved_meta = json.load(f)
            
            # Check compatibility (only dimension changes, not version)
            if (saved_meta.get('d_img') != d_img or 
                saved_meta.get('d_txt') != d_txt):
                
                print(f"  ⚠️ Cache metadata mismatch! Invalidating cache...")
                print(f"     Cached: d_img={saved_meta.get('d_img')}, d_txt={saved_meta.get('d_txt')}")
                print(f"     Current: d_img={d_img}, d_txt={d_txt}")
                
                # Invalidate local cache
                import shutil
                for f in os.listdir(cache_dir):
                    if f.endswith('.pt'):
                        os.remove(os.path.join(cache_dir, f))
                
                # Update metadata
                with open(metadata_path, 'w') as f:
                    json.dump(current_meta, f)
                
                return True
        except Exception as e:
            logger.warning(f"Failed to read cache metadata: {e}")
    
    # Save/update metadata
    os.makedirs(cache_dir, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(current_meta, f)
    
    return False


def precompute_embeddings(
    samples: List[Dict], 
    mode: str, 
    img_encoder, 
    txt_encoder,
    d_img: int,
    d_txt: int,
    cache_dir: str = CACHE_DIR,
    use_minio_cache: bool = True
) -> str:
    """
    Pre-compute all embeddings and SAVE TO DISK.
    
    Hybrid Approach:
        1. Check MinIO trước khi compute
        2. Download cached embeddings nếu có
        3. Compute những cái chưa có
        4. Upload embeddings mới lên MinIO
    
    Uses Spark if available and enabled for parallel processing.
    Falls back to sequential processing otherwise.
    
    Returns cache directory path.
    """
    print_header("PRE-COMPUTING EMBEDDINGS")
    print(f"  Total samples: {len(samples)}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Expected dims: d_img={d_img}, d_txt={d_txt}")
    print(f"  MinIO cache: {'enabled' if use_minio_cache else 'disabled'}")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check and invalidate stale cache (e.g., if d_img changed from 512 to 2048)
    cache_invalidated = _check_and_invalidate_stale_cache(mode, d_img, d_txt, cache_dir)
    
    # Also invalidate MinIO cache if local was invalidated
    if cache_invalidated and use_minio_cache:
        print(f"  🗑️  Invalidating MinIO cache for mode={mode}...")
        cache_manager_tmp = EmbeddingCacheManager(mode, cache_dir)
        cache_manager_tmp.invalidate_cache()
    
    # Initialize cache manager for hybrid approach
    cache_manager = EmbeddingCacheManager(mode, cache_dir) if use_minio_cache else None
    
    # Get all needed sample_ids
    needed_sample_ids = [s['sample_id'] for s in samples]
    
    # =========================================================================
    # STEP 1: Sync from MinIO (download existing embeddings)
    # =========================================================================
    if cache_manager:
        to_compute_ids = cache_manager.sync_before_compute(needed_sample_ids)
        to_compute = [s for s in samples if s['sample_id'] in to_compute_ids]
    else:
        # Check existing local cache only
        existing = set(f.replace('.pt', '') for f in os.listdir(cache_dir) if f.endswith('.pt'))
        to_compute = [s for s in samples if s['sample_id'] not in existing]
    
    if len(to_compute) == 0:
        print(f"  ✅ All {len(samples)} samples already cached!")
        return cache_dir
    
    print(f"  To compute: {len(to_compute)}")
    
    # =========================================================================
    # Try Spark for parallel embedding computation (>=100 samples)
    # =========================================================================
    use_spark = os.environ.get('USE_SPARK', 'true').lower() == 'true'
    
    if use_spark and len(to_compute) >= 100:
        try:
            print(f"  💡 Using Spark for parallel embedding computation")
            from common.spark import SparkBatchProcessor
            
            # Configure storage
            storage_config = {
                'host': os.environ.get('MINIO_HOST', 'minio'),
                'port': int(os.environ.get('MINIO_PORT', '9000')),
                'access_key': os.environ.get('MINIO_ACCESS_KEY', 'minioadmin'),
                'secret_key': os.environ.get('MINIO_SECRET_KEY', 'minioadmin'),
                'bucket': os.environ.get('MINIO_BUCKET', 'video-storage'),
            }
            
            processor = SparkBatchProcessor()
            processor.set_config(storage_config, mode)
            
            # Extract sample_ids
            sample_ids = [s['sample_id'] for s in to_compute]
            
            # Compute in parallel
            start_time = time.time()
            result = processor.compute_embeddings_batch(
                sample_ids=sample_ids,
                mode=mode,
                output_dir=os.path.dirname(cache_dir)
            )
            
            elapsed = time.time() - start_time
            print(f"  ✅ Spark computed {result['success']} embeddings in {elapsed:.1f}s")
            print(f"     Speed: {result['success']/elapsed:.1f} samples/sec")
            
            # Upload newly computed embeddings to MinIO
            if cache_manager:
                computed_ids = [s['sample_id'] for s in to_compute]
                cache_manager.sync_after_compute(computed_ids)
            
            return cache_dir
            
        except Exception as e:
            logger.warning(f"  ⚠️ Spark failed: {e}")
            print(f"  ⚠️ Falling back to sequential processing")
    
    # =========================================================================
    # Sequential processing (fallback or small batches)
    # =========================================================================
    print(f"  Using sequential processing")
    
    start_time = time.time()
    
    for i, s in enumerate(to_compute):
        sample_id = s['sample_id']
        cache_path = os.path.join(cache_dir, f"{sample_id}.pt")
        
        # === Load & encode frames ===
        frames = []
        for j in range(NUM_FRAMES):
            path = f"samples/{sample_id}/frame_{j:02d}.jpg"
            try:
                data = storage.download_data(path, silent=True)
                if data:
                    img = Image.open(BytesIO(data)).convert('RGB')
                    frames.append(img)
            except:
                pass
        
        # Pad with black images
        while len(frames) < NUM_FRAMES:
            frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))
        
        with torch.no_grad():
            img_embs = img_encoder.get_embeddings(frames).cpu()  # (16, d_img)
        
        # Clear frames from memory
        del frames
        
        # === Load & encode chunks ===
        chunks = []
        try:
            path = f"samples/{sample_id}/transcript.json"
            data = storage.download_data(path, silent=True)
            if data:
                chunk_data = json.loads(data.decode('utf-8'))
                chunks = chunk_data.get('chunks', [])
        except:
            pass
        
        # Pad with empty strings
        while len(chunks) < NUM_CHUNKS:
            chunks.append("")
        chunks = chunks[:NUM_CHUNKS]
        
        with torch.no_grad():
            txt_embs = txt_encoder.get_embeddings(chunks).cpu()  # (5, d_txt)
        
        # Save to disk immediately (don't keep in memory!)
        torch.save({
            'img_embs': img_embs,
            'txt_embs': txt_embs
        }, cache_path)
        
        # Clear from memory
        del img_embs, txt_embs
        
        # Progress
        if (i + 1) % 50 == 0 or i == len(to_compute) - 1:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(to_compute) - i - 1)
            print(f"  [{i+1}/{len(to_compute)}] Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"  ✅ Pre-computed {len(to_compute)} samples in {total_time:.1f}s")
    
    # =========================================================================
    # STEP 3: Upload newly computed embeddings to MinIO
    # =========================================================================
    if cache_manager:
        computed_ids = [s['sample_id'] for s in to_compute]
        cache_manager.sync_after_compute(computed_ids)
    
    return cache_dir


# ============================================================================
# MODEL WITH ATTENTION POOLING + ATTENTION FUSION
# ============================================================================
class VideoClassifierWithAttnPool(nn.Module):
    """
    Video classifier with attention pooling and gated fusion.
    
    Architecture:
        Image: [16, d_img] → AttnPool → [d_img]
        Text:  [5, d_txt]  → AttnPool → [d_txt]
        Fusion: GatedFusion(img_emb, txt_emb) → logits
        
    Note: Encoding is done BEFORE training (pre-computed embeddings).
    This model only handles pooling and fusion (trainable parts).
    """
    
    def __init__(self, d_img: int, d_txt: int, 
                 d_fused: int = 256, dropout: float = 0.3, 
                 num_heads: int = 4, num_layers: int = 1,
                 use_multihead_pool: bool = False):
        super().__init__()
        
        self.d_img = d_img
        self.d_txt = d_txt
        self.use_multihead_pool = use_multihead_pool
        
        # Attention pooling layers (TRAINABLE)
        # Use MultiHead for high-dimensional embeddings (balanced mode)
        if use_multihead_pool:
            # Determine num_heads based on dimension (must divide evenly)
            img_heads = min(num_heads, self._find_valid_heads(d_img, max_heads=8))
            txt_heads = min(num_heads, self._find_valid_heads(d_txt, max_heads=8))
            
            self.img_attn_pool = MultiHeadAttentionPooling(d_img, num_heads=img_heads, dropout=dropout)
            self.txt_attn_pool = MultiHeadAttentionPooling(d_txt, num_heads=txt_heads, dropout=dropout)
            logger.info(f"Using MultiHeadAttentionPooling: img_heads={img_heads}, txt_heads={txt_heads}")
        else:
            self.img_attn_pool = AttentionPooling(d_img)
            self.txt_attn_pool = AttentionPooling(d_txt)
            logger.info("Using simple AttentionPooling")
        
        # Gated Fusion classifier (TRAINABLE)
        self.fusion = GatedFusionClassifier(
            d_img=d_img,
            d_txt=d_txt,
            d_fused=d_fused,
            num_classes=NUM_CLASSES,
            dropout=dropout
        )
    
    def _find_valid_heads(self, dim: int, max_heads: int = 8) -> int:
        """Find largest valid num_heads that divides dim evenly."""
        for h in range(max_heads, 0, -1):
            if dim % h == 0:
                return h
        return 1
    
    def forward(self, img_embs: torch.Tensor, txt_embs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-computed embeddings.
        
        Args:
            img_embs: (batch, 16, d_img) - pre-computed frame embeddings
            txt_embs: (batch, 5, d_txt)  - pre-computed chunk embeddings
        
        Returns:
            logits: (batch, num_classes)
        """
        # Attention pool
        img_pooled = self.img_attn_pool(img_embs)  # (batch, d_img)
        txt_pooled = self.txt_attn_pool(txt_embs)  # (batch, d_txt)
        
        # Gated Fusion
        logits, _ = self.fusion(img_pooled, txt_pooled)
        return logits


# ============================================================================
# TRAINING
# ============================================================================
def collate_fn(batch):
    return {
        'sample_id': [b['sample_id'] for b in batch],
        'video_id': [b['video_id'] for b in batch],
        'img_embs': torch.stack([b['img_embs'] for b in batch]),  # (batch, 16, d_img)
        'txt_embs': torch.stack([b['txt_embs'] for b in batch]),  # (batch, 5, d_txt)
        'label': [b['label'] for b in batch],
        'y': torch.tensor([b['y'] for b in batch])
    }


def run_epoch(model, loader, optimizer, device, training: bool, 
              label_smoothing: float = 0.1, use_mixup: bool = False,
              mixup_alpha: float = 0.4) -> Metrics:
    """Run one training/validation epoch.
    
    Args:
        model: The model to train/evaluate
        loader: DataLoader
        optimizer: Optimizer (None for eval)
        device: Device to use
        training: Whether training or evaluation
        label_smoothing: Label smoothing factor (default 0.1, reduces overfitting)
        use_mixup: Whether to use Mixup augmentation (training only)
        mixup_alpha: Beta distribution parameter for Mixup
    """
    model.train() if training else model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    
    for batch in loader:
        img_embs = batch['img_embs'].to(device)  # Pre-computed!
        txt_embs = batch['txt_embs'].to(device)  # Pre-computed!
        y = batch['y'].to(device)
        
        if training:
            # Apply Mixup augmentation (embedding-level)
            if use_mixup and np.random.random() > 0.3:  # 70% chance to apply mixup
                mixed_img, mixed_txt, y_a, y_b, lam = mixup_embeddings(
                    img_embs, txt_embs, y, alpha=mixup_alpha
                )
                logits = model(mixed_img, mixed_txt)
                loss = mixup_criterion(logits, y_a, y_b, lam, label_smoothing)
            else:
                logits = model(img_embs, txt_embs)
                loss_dict = compute_classification_loss(
                    logits, y, 
                    use_focal_loss=True, 
                    focal_gamma=2.0,
                    label_smoothing=label_smoothing
                )
                loss = loss_dict['loss']
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(y)
        else:
            with torch.no_grad():
                logits = model(img_embs, txt_embs)
                loss_dict = compute_classification_loss(
                    logits, y, 
                    use_focal_loss=True, 
                    focal_gamma=2.0,
                    label_smoothing=label_smoothing
                )
                total_loss += loss_dict['loss'].item() * len(y)
        
        # Collect predictions (for both train and eval)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    
    return compute_metrics(all_labels, all_preds, total_loss / len(loader.dataset))


def collect_sample_predictions(model, loader, device) -> List[Dict]:
    """Collect predictions for each sample (for error analysis).
    
    Args:
        model: Trained model
        loader: DataLoader with samples
        device: Device
    
    Returns:
        List of dicts with sample_id, true_label, predicted_label, confidence, probabilities
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in loader:
            img_embs = batch['img_embs'].to(device)
            txt_embs = batch['txt_embs'].to(device)
            sample_ids = batch['sample_id']
            labels = batch['label']
            
            logits = model(img_embs, txt_embs)
            probs = torch.softmax(logits, dim=1)
            pred_indices = logits.argmax(1).cpu().tolist()
            confidences = probs.max(1).values.cpu().tolist()
            
            # Convert probabilities to dict
            for i, sample_id in enumerate(sample_ids):
                pred_idx = pred_indices[i]
                pred_label = IDX_TO_LABEL.get(pred_idx, 'Safe')
                
                # Get probabilities for each class
                p_final = {label: float(probs[i, idx].item()) 
                          for label, idx in LABEL_TO_IDX.items()}
                
                predictions.append({
                    'sample_id': sample_id,
                    'true_label': labels[i],
                    'predicted_label': pred_label,
                    'confidence': confidences[i],
                    'p_final': p_final
                })
    
    return predictions


def compute_gate_weights_analysis(model, loader, device) -> Dict:
    """
    Compute gate weights statistics for each class.
    
    Gate weights indicate how much the model trusts image vs text:
    - Weight close to 1.0 = trusts image more
    - Weight close to 0.0 = trusts text more
    
    Args:
        model: Trained VideoClassifierWithAttnPool model
        loader: DataLoader with samples
        device: Device
    
    Returns:
        Dict with per_class stats and overall stats
    """
    model.eval()
    
    # Check if model has gated fusion
    if not hasattr(model, 'fusion') or not hasattr(model.fusion, 'get_gate_weights'):
        logger.warning("Model does not have gated fusion - skipping gate weights analysis")
        return None
    
    gate_weights_by_label = {label: [] for label in LABELS}
    
    with torch.no_grad():
        for batch in loader:
            img_embs = batch['img_embs'].to(device)  # (batch, seq, d_img)
            txt_embs = batch['txt_embs'].to(device)  # (batch, seq, d_txt)
            labels = batch['label']  # List of label strings
            
            # Pool embeddings using attention pooling
            img_pooled = model.img_attn_pool(img_embs)  # (batch, d_img)
            txt_pooled = model.txt_attn_pool(txt_embs)  # (batch, d_txt)
            
            # Get gate weights from fusion module
            gate = model.fusion.get_gate_weights(img_pooled, txt_pooled)  # (batch, d_fused)
            gate_avg = gate.mean(dim=1).cpu().numpy()  # Average across dimensions to get (batch,)
            
            # Store by label
            for i, label in enumerate(labels):
                if label in gate_weights_by_label:
                    gate_weights_by_label[label].append(float(gate_avg[i]))
    
    # Compute statistics per class
    per_class_stats = {}
    all_weights = []
    
    for label in LABELS:
        weights = gate_weights_by_label[label]
        if weights:
            weights_array = np.array(weights)
            all_weights.extend(weights)
            
            per_class_stats[label] = {
                "count": len(weights),
                "mean": round(float(weights_array.mean()), 4),
                "std": round(float(weights_array.std()), 4),
                "min": round(float(weights_array.min()), 4),
                "max": round(float(weights_array.max()), 4),
                "median": round(float(np.median(weights_array)), 4),
                "q25": round(float(np.percentile(weights_array, 25)), 4),
                "q75": round(float(np.percentile(weights_array, 75)), 4),
                "img_weight": round(float(weights_array.mean()), 4),
                "txt_weight": round(1.0 - float(weights_array.mean()), 4),
                "dominant_modality": "image" if weights_array.mean() > 0.55 else ("text" if weights_array.mean() < 0.45 else "balanced")
            }
        else:
            per_class_stats[label] = {
                "count": 0, "mean": None, "std": None, "min": None, "max": None,
                "median": None, "q25": None, "q75": None,
                "img_weight": None, "txt_weight": None, "dominant_modality": None
            }
    
    # Overall statistics
    all_weights_array = np.array(all_weights) if all_weights else np.array([0.5])
    overall_stats = {
        "total_samples": len(all_weights),
        "mean": round(float(all_weights_array.mean()), 4),
        "std": round(float(all_weights_array.std()), 4),
        "img_weight": round(float(all_weights_array.mean()), 4),
        "txt_weight": round(1.0 - float(all_weights_array.mean()), 4),
    }
    
    # Generate insights
    insights = []
    for label, stats in per_class_stats.items():
        if stats['mean'] is not None:
            if stats['dominant_modality'] == 'image':
                insights.append({
                    "type": "info",
                    "label": label,
                    "message": f"{label}: Model trusts image features more ({stats['img_weight']:.0%} image vs {stats['txt_weight']:.0%} text)"
                })
            elif stats['dominant_modality'] == 'text':
                insights.append({
                    "type": "info",
                    "label": label,
                    "message": f"{label}: Model trusts text features more ({stats['txt_weight']:.0%} text vs {stats['img_weight']:.0%} image)"
                })
    
    return {
        "per_class": per_class_stats,
        "overall": overall_stats,
        "insights": insights,
        "explanation": {
            "gate_weight_meaning": "Gate weight indicates how much the model trusts each modality",
            "interpretation": {
                "1.0": "Fully trust image features",
                "0.5": "Equal trust between image and text",
                "0.0": "Fully trust text features"
            }
        }
    }


def save_model(model, mode: str, metrics: Metrics, epoch: int, total_epochs: int,
               training_config: Dict, all_epoch_metrics: List[tuple]) -> int:
    """Save model to MinIO and register with detailed metrics.
    
    Args:
        model: The trained model
        mode: Training mode
        metrics: Best validation metrics
        epoch: Best epoch number
        total_epochs: Total epochs trained
        training_config: Training configuration dict
        all_epoch_metrics: List of (epoch, train_metrics, val_metrics) tuples
    
    Returns:
        model_id: The registered model ID
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"v_gated_{ts}"
    
    # Save only trainable parts (attention pools + fusion)
    state_dict = {
        'img_attn_pool': model.img_attn_pool.state_dict(),
        'txt_attn_pool': model.txt_attn_pool.state_dict(),
        'fusion': model.fusion.state_dict(),
    }
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(state_dict, f.name)
        path = f"models/{mode}/fusion_gated_{version}.pt"
        storage.upload_file(path, f.name)
        os.unlink(f.name)
    
    # Register model and get model_id
    model_id = db.register_model(
        mode, 'fusion_gated', version, path, metrics.to_dict(), True,
        training_config=training_config,
        best_epoch=epoch,
        total_epochs=total_epochs,
        training_time=training_config.get('training_time_seconds')
    )
    
    # Save detailed metrics for each epoch
    if model_id and all_epoch_metrics:
        for ep, train_m, val_m in all_epoch_metrics:
            # Save training metrics
            db.save_training_metrics(
                model_id=model_id,
                epoch=ep,
                split='train',
                loss=train_m.loss,
                accuracy=train_m.accuracy,
                precision=train_m.precision,
                recall=train_m.recall,
                f1=train_m.f1,
                per_class_metrics=train_m.per_class,
                confusion_matrix=train_m.confusion_matrix.tolist() if train_m.confusion_matrix is not None else None,
                learning_rate=training_config.get('learning_rate'),
                batch_size=training_config.get('batch_size')
            )
            
            # Save validation metrics
            db.save_training_metrics(
                model_id=model_id,
                epoch=ep,
                split='val',
                loss=val_m.loss,
                accuracy=val_m.accuracy,
                precision=val_m.precision,
                recall=val_m.recall,
                f1=val_m.f1,
                per_class_metrics=val_m.per_class,
                confusion_matrix=val_m.confusion_matrix.tolist() if val_m.confusion_matrix is not None else None,
                learning_rate=training_config.get('learning_rate'),
                batch_size=training_config.get('batch_size')
            )
    
    logger.info(f"Saved: {path} (model_id={model_id})")
    return model_id


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train_with_advanced_fusion(
    mode: str = 'ultra_light',
    num_epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str = None,
    early_stopping: int = 7,
    dropout: float = 0.3,
    d_fused: int = 256,
    num_heads: int = 4,
    num_layers: int = 1,
    use_pregenerated_samples: bool = True,
    label_smoothing: float = None,  # Mode-specific default
    weight_decay: float = None,      # Mode-specific default
    **kwargs
):
    """
    Train video classifier with AttentionPooling + GatedFusion.
    
    Architecture:
        - Encoders: FROZEN
        - AttentionPooling: TRAINABLE (learns to weight frames/chunks)
        - GatedFusion: TRAINABLE (gated multimodal fusion)
        
    Mode-specific optimizations:
        - balanced: Higher d_fused (512), more regularization (dropout=0.4, label_smoothing=0.15)
        - ultra_light: Lower d_fused (256), moderate regularization
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    cfg = config.models.get(mode, {})
    d_img = cfg.get('image_embedding_dim', 1280)
    d_txt = cfg.get('text_embedding_dim', 768)
    
    # Mode-specific hyperparameter adjustments
    # Balanced mode needs more regularization to avoid overfitting
    if mode == 'balanced':
        if d_fused == 256:  # Only override if default
            d_fused = 512  # Larger fusion dim for higher capacity encoders
        if dropout == 0.3:  # Only override if default
            dropout = 0.4  # More dropout for larger model
        if label_smoothing is None:
            label_smoothing = 0.15  # More label smoothing
        if weight_decay is None:
            weight_decay = 0.05  # More weight decay
    else:  # ultra_light
        if label_smoothing is None:
            label_smoothing = 0.1
        if weight_decay is None:
            weight_decay = 0.01
    
    # Ensure d_fused is divisible by num_heads
    if d_fused % num_heads != 0:
        d_fused = (d_fused // num_heads + 1) * num_heads
        logger.info(f"Adjusted d_fused to {d_fused} (divisible by num_heads={num_heads})")
    
    print_config({
        'Mode': mode,
        'Device': device,
        'Epochs': num_epochs,
        'Batch Size': batch_size,
        'Learning Rate': learning_rate,
        'Label Smoothing': label_smoothing,
        'Weight Decay': weight_decay,
        'Early Stopping': early_stopping,
        'Dropout': dropout,
        'Fusion Dim': d_fused,
        'Attn Heads': num_heads,
        'Attn Layers': num_layers,
        'Image Dim': d_img,
        'Text Dim': d_txt,
    })
    
    # Load data
    print_header("LOADING DATA")
    
    if not use_pregenerated_samples:
        raise ValueError("Only pre-generated samples supported")
    
    train_samples = db.get_training_samples_v2(split='train')
    val_samples = db.get_training_samples_v2(split='val')
    
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples")
    print_class_distribution(train_samples, "Train")
    print_class_distribution(val_samples, "Val")
    
    if len(train_samples) < 10:
        raise ValueError(f"Not enough training samples: {len(train_samples)}")
    
    # Initialize encoders (for pre-computing embeddings)
    print_header("INITIALIZING ENCODERS")
    
    img_encoder = get_image_encoder(mode, num_classes=None, config=cfg)
    txt_encoder = get_text_encoder(mode, num_classes=None, config=cfg)
    
    img_encoder.eval()
    txt_encoder.eval()
    
    print(f"  Image: {cfg.get('image_encoder', 'efficientnet-b0')} ({d_img}d)")
    print(f"  Text:  {cfg.get('text_encoder', 'phobert')} ({d_txt}d)")
    
    # =========================================================================
    # PRE-COMPUTE EMBEDDINGS (saves to disk, memory efficient)
    # =========================================================================
    all_samples = train_samples + val_samples
    cache_dir = f"{CACHE_DIR}/{mode}"
    precompute_embeddings(all_samples, mode, img_encoder, txt_encoder, d_img, d_txt, cache_dir)
    
    # Free encoder memory immediately
    del img_encoder
    del txt_encoder
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("  ✅ Encoders freed from memory")
    
    # Create dataloaders (loads from disk on-demand)
    print_header("CREATING DATALOADERS")
    train_ds = PrecomputedDataset(cache_dir, train_samples, d_img, d_txt)
    val_ds = PrecomputedDataset(cache_dir, val_samples, d_img, d_txt)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    # Create model (only pooling + fusion, no encoders)
    print_header("CREATING MODEL")
    
    # Use MultiHead pooling for balanced mode (higher dimensions benefit more)
    use_multihead_pool = (mode == 'balanced')
    
    model = VideoClassifierWithAttnPool(
        d_img=d_img,
        d_txt=d_txt,
        d_fused=d_fused,
        dropout=dropout,
        num_heads=num_heads,
        num_layers=num_layers,
        use_multihead_pool=use_multihead_pool
    ).to(device)
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,} (~{trainable/1e6:.2f}M)")
    print(f"  Components: AttnPool(img) + AttnPool(txt) + GatedFusion")
    
    # Optimizer with mode-specific weight decay
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    # Training loop
    print_header("TRAINING")
    print(f"  Label smoothing: {label_smoothing}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Multihead pooling: {use_multihead_pool}")
    
    # Mixup for balanced mode (better regularization)
    use_mixup = (mode == 'balanced')
    mixup_alpha = 0.4 if mode == 'balanced' else 0.0
    print(f"  Mixup augmentation: {use_mixup} (alpha={mixup_alpha})")
    
    best_acc, best_metrics, patience_count = 0.0, None, 0
    best_epoch = 0
    all_epoch_metrics = []  # Store (epoch, train_metrics, val_metrics) for each epoch
    start_time = time.time()
    
    # Training configuration dict
    training_config = {
        'mode': mode,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'label_smoothing': label_smoothing,
        'weight_decay': weight_decay,
        'early_stopping': early_stopping,
        'dropout': dropout,
        'd_fused': d_fused,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'd_img': d_img,
        'd_txt': d_txt,
        'use_multihead_pool': use_multihead_pool,
        'use_mixup': use_mixup,
        'mixup_alpha': mixup_alpha,
    }
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'─'*70}\n EPOCH {epoch}/{num_epochs}\n{'─'*70}")
        epoch_start = time.time()
        
        train_m = run_epoch(model, train_loader, optimizer, device, training=True, 
                           label_smoothing=label_smoothing,
                           use_mixup=use_mixup, mixup_alpha=mixup_alpha)
        val_m = run_epoch(model, val_loader, None, device, training=False,
                         label_smoothing=label_smoothing,
                         use_mixup=False)  # Never use mixup for validation
        
        # Store metrics for this epoch
        all_epoch_metrics.append((epoch, train_m, val_m))
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        print_metrics_table(train_m, val_m)
        print_per_class(val_m)
        
        if val_m.accuracy > best_acc:
            best_acc, best_metrics, patience_count = val_m.accuracy, val_m, 0
            best_epoch = epoch
            print(f"\n  ✅ NEW BEST: {best_acc:.3f} | Time: {epoch_time:.1f}s")
        else:
            patience_count += 1
            print(f"\n  ⏳ No improvement ({patience_count}/{early_stopping}) | Time: {epoch_time:.1f}s")
        
        if patience_count >= early_stopping:
            print(f"\n  🛑 EARLY STOPPING at epoch {epoch}")
            break
    
    # Save training time
    training_config['training_time_seconds'] = time.time() - start_time
    
    # Save model with all metrics
    model_id = save_model(model, mode, best_metrics, best_epoch, epoch, training_config, all_epoch_metrics)
    
    # =========================================================================
    # SAVE SAMPLE PREDICTIONS FOR ERROR ANALYSIS
    # =========================================================================
    if model_id:
        print_header("SAVING SAMPLE PREDICTIONS")
        
        # Collect predictions for validation set
        print("  Collecting validation set predictions...")
        val_predictions = collect_sample_predictions(model, val_loader, device)
        for pred in val_predictions:
            pred['split'] = 'val'
        print(f"  ✅ Val predictions: {len(val_predictions)}")
        
        # Collect predictions for training set
        print("  Collecting training set predictions...")
        train_predictions = collect_sample_predictions(model, train_loader, device)
        for pred in train_predictions:
            pred['split'] = 'train'
        print(f"  ✅ Train predictions: {len(train_predictions)}")
        
        # Save to database
        all_predictions = val_predictions + train_predictions
        print(f"  Saving {len(all_predictions)} predictions to database...")
        db.save_sample_predictions_batch(model_id, all_predictions)
        print(f"  ✅ Sample predictions saved for model_id={model_id}")
        
        # =====================================================================
        # COMPUTE AND SAVE GATE WEIGHTS ANALYSIS
        # =====================================================================
        print_header("COMPUTING GATE WEIGHTS ANALYSIS")
        
        print("  Computing gate weights on validation set...")
        gate_weights_val = compute_gate_weights_analysis(model, val_loader, device)
        
        if gate_weights_val:
            print("  Per-class gate weights (validation):")
            for label, stats in gate_weights_val['per_class'].items():
                if stats['mean'] is not None:
                    print(f"    {label:<12}: img={stats['img_weight']:.2%}, txt={stats['txt_weight']:.2%} ({stats['dominant_modality']})")
            
            print(f"\n  Overall: img={gate_weights_val['overall']['img_weight']:.2%}, txt={gate_weights_val['overall']['txt_weight']:.2%}")
            
            # Store gate weights in model_registry (update metrics JSON)
            db.save_gate_weights(model_id, gate_weights_val, split='val')
            print(f"  ✅ Gate weights saved for model_id={model_id}")
        else:
            print("  ⚠️ Gate weights analysis not available (model may not use gated fusion)")
    
    # Final summary
    print_header("TRAINING COMPLETE")
    print(f"  Best Accuracy: {best_acc:.3f}")
    print(f"  Best F1:       {best_metrics.f1:.3f}")
    print(f"  Total Time:    {(time.time() - start_time)/60:.1f} minutes")
    print_confusion_matrix(best_metrics.confusion_matrix)
    
    # =========================================================================
    # EVALUATE ON TEST SET
    # =========================================================================
    print_header("EVALUATING ON TEST SET")
    
    test_samples = db.get_training_samples_v2(split='test')
    
    if len(test_samples) == 0:
        print("  ⚠️ No test samples found. Skipping test evaluation.")
    else:
        print(f"  Test samples: {len(test_samples)}")
        print_class_distribution(test_samples, "Test")
        
        # Pre-compute test embeddings if not cached
        # (reuse encoders if still in memory, otherwise skip - they should be cached)
        test_cache_needed = [s for s in test_samples if not os.path.exists(os.path.join(cache_dir, f"{s['sample_id']}.pt"))]
        
        if test_cache_needed:
            print(f"  ⚠️ {len(test_cache_needed)} test samples not cached. Loading encoders...")
            img_encoder = get_image_encoder(mode, num_classes=None, config=cfg)
            txt_encoder = get_text_encoder(mode, num_classes=None, config=cfg)
            img_encoder.eval()
            txt_encoder.eval()
            precompute_embeddings(test_cache_needed, mode, img_encoder, txt_encoder, d_img, d_txt, cache_dir)
            del img_encoder, txt_encoder
            gc.collect()
        
        # Create test dataloader
        test_ds = PrecomputedDataset(cache_dir, test_samples, d_img, d_txt)
        test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        
        # Evaluate (no mixup for testing)
        test_m = run_epoch(model, test_loader, None, device, training=False,
                          label_smoothing=label_smoothing, use_mixup=False)
        
        print_header("TEST RESULTS")
        print(f"\n┌{'─'*50}┐")
        print(f"│ {'Metric':<20} │ {'Test':>25} │")
        print(f"├{'─'*50}┤")
        print(f"│ {'Loss':<20} │ {test_m.loss:>25.4f} │")
        print(f"│ {'Accuracy':<20} │ {test_m.accuracy:>25.3f} │")
        print(f"│ {'Precision':<20} │ {test_m.precision:>25.3f} │")
        print(f"│ {'Recall':<20} │ {test_m.recall:>25.3f} │")
        print(f"│ {'F1 Score':<20} │ {test_m.f1:>25.3f} │")
        print(f"└{'─'*50}┘")
        
        print(f"\n  Test Per-Class:")
        print(f"  ┌{'─'*50}┐")
        print(f"  │ {'Class':<12} │ {'Prec':>8} │ {'Recall':>8} │ {'F1':>8} │")
        print(f"  ├{'─'*50}┤")
        for label, m in test_m.per_class.items():
            print(f"  │ {label:<12} │ {m['precision']:>8.3f} │ {m['recall']:>8.3f} │ {m['f1']:>8.3f} │")
        print(f"  └{'─'*50}┘")
        
        print_confusion_matrix(test_m.confusion_matrix)
        
        # Compare with validation
        print_header("VALIDATION vs TEST COMPARISON")
        val_test_gap = best_metrics.accuracy - test_m.accuracy
        print(f"  Val Accuracy:  {best_metrics.accuracy:.3f}")
        print(f"  Test Accuracy: {test_m.accuracy:.3f}")
        print(f"  Gap:           {val_test_gap:+.3f} {'⚠️ OVERFIT' if val_test_gap > 0.05 else '✅ OK'}")
        print(f"")
        print(f"  Val F1:  {best_metrics.f1:.3f}")
        print(f"  Test F1: {test_m.f1:.3f}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='ultra_light')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--early-stopping', type=int, default=7)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--d-fused', type=int, default=256)
    p.add_argument('--num-heads', type=int, default=4)
    p.add_argument('--num-layers', type=int, default=1)
    args = p.parse_args()
    
    train_with_advanced_fusion(
        mode=args.mode,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_stopping=args.early_stopping,
        dropout=args.dropout,
        d_fused=args.d_fused,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        use_pregenerated_samples=True,
    )

