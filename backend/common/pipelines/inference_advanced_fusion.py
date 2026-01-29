"""
Inference with Advanced Fusion models.

This module provides inference functions for models trained with
VideoClassifierWithAttnPool (AttentionPooling + GatedFusion).

Direct 4-class classification: Safe, Aggressive, Sexual, Superstition

IMPORTANT: Requires pre-processed samples from preprocessing_dag.
           No fallback - sample_id is REQUIRED.
"""

import os
import json
import torch
import torch.nn as nn
import logging
import tempfile
from typing import Dict, Optional, List
from PIL import Image
from io import BytesIO

from common.io import db, storage, config
from common.models.text_encoder import get_text_encoder
from common.models.image_encoder import get_image_encoder
from common.models.advanced_fusion import GatedFusionClassifier, AttentionFusionClassifier, LABELS

# Import pooling classes from training pipeline to avoid duplication
from common.pipelines.train_with_advanced_fusion import (
    AttentionPooling, MultiHeadAttentionPooling
)

logger = logging.getLogger(__name__)

# Constants matching preprocessing
NUM_FRAME_SEGMENTS = 16
NUM_TRANSCRIPT_CHUNKS = 5


class VideoClassifierWithAttnPool(nn.Module):
    """
    Video classifier with attention pooling and gated fusion.
    
    Architecture:
        Image: [16, d_img] → AttnPool → [d_img]
        Text:  [5, d_txt]  → AttnPool → [d_txt]
        Fusion: GatedFusion(img_emb, txt_emb) → logits
    
    Supports both simple AttentionPooling and MultiHeadAttentionPooling.
    """
    
    def __init__(self, d_img: int, d_txt: int, 
                 d_fused: int = 256, dropout: float = 0.0,
                 use_multihead_pool: bool = False, num_heads: int = 4):
        super().__init__()
        
        self.d_img = d_img
        self.d_txt = d_txt
        self.use_multihead_pool = use_multihead_pool
        
        # Attention pooling layers
        if use_multihead_pool:
            # Find valid head counts for each dimension
            img_heads = self._find_valid_heads(d_img, max_heads=num_heads)
            txt_heads = self._find_valid_heads(d_txt, max_heads=num_heads)
            self.img_attn_pool = MultiHeadAttentionPooling(d_img, num_heads=img_heads, dropout=dropout)
            self.txt_attn_pool = MultiHeadAttentionPooling(d_txt, num_heads=txt_heads, dropout=dropout)
        else:
            self.img_attn_pool = AttentionPooling(d_img)
            self.txt_attn_pool = AttentionPooling(d_txt)
        
        # Gated Fusion classifier
        self.fusion = GatedFusionClassifier(
            d_img=d_img,
            d_txt=d_txt,
            d_fused=d_fused,
            num_classes=4,
            dropout=dropout,
            use_layernorm=True
        )
    
    def _find_valid_heads(self, dim: int, max_heads: int = 8) -> int:
        """Find largest valid num_heads that divides dim evenly."""
        for h in range(max_heads, 0, -1):
            if dim % h == 0:
                return h
        return 1
    
    def forward(self, img_embs: torch.Tensor, txt_embs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with embeddings.
        
        Args:
            img_embs: (batch, 16, d_img) - frame embeddings
            txt_embs: (batch, 5, d_txt)  - chunk embeddings
        
        Returns:
            logits: (batch, num_classes)
        """
        # Attention pool
        img_pooled = self.img_attn_pool(img_embs)  # (batch, d_img)
        txt_pooled = self.txt_attn_pool(txt_embs)  # (batch, d_txt)
        
        # Gated Fusion
        logits, _ = self.fusion(img_pooled, txt_pooled)
        return logits
    
    def predict_single(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Prediction with already-pooled embeddings (for legacy compatibility).
        
        Args:
            img_emb: (1, d_img) - pooled image embedding
            txt_emb: (1, d_txt) - pooled text embedding
        
        Returns:
            logits: (1, num_classes)
        """
        logits, _ = self.fusion(img_emb, txt_emb)
        return logits


def load_advanced_fusion_models(mode='ultra_light'):
    """
    Load text encoder, image encoder, and fusion model.
    
    Args:
        mode: Model mode ('ultra_light' or 'balanced')
    
    Returns:
        Dict with:
            - text_encoder: Text encoder model
            - image_encoder: Image encoder model
            - fusion_model: VideoClassifierWithAttnPool (AttnPool + GatedFusion)
            - fusion_type: 'gated' or 'attention'
            - mode: Mode used
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Loading advanced fusion models for mode={mode}, device={device}")
    
    # Get active models from registry
    active_models = db.get_active_models(mode)
    
    # Find fusion model
    fusion_type = None
    fusion_artifact_path = None
    
    for model_type, model_info in active_models.items():
        if model_type.startswith('fusion_'):
            fusion_type = model_type.replace('fusion_', '')  # 'gated' or 'attention'
            fusion_artifact_path = model_info['artifact_path']
            break
    
    if not fusion_type or not fusion_artifact_path:
        logger.error("No active fusion model found. Use legacy inference or train fusion model first.")
        raise ValueError(f"No active fusion model found for mode {mode}")
    
    logger.info(f"Found fusion model: {fusion_type}")
    
    # Get model config
    model_config = config.models.get(mode, {})
    d_img = model_config.get('image_embedding_dim', 1280)
    d_txt = model_config.get('text_embedding_dim', 384)
    d_fused = model_config.get('hidden_dim', 256)  # Use hidden_dim as d_fused
    
    # Initialize encoders (without classifier heads)
    logger.info("Loading text encoder...")
    text_encoder = get_text_encoder(mode, num_classes=None, config=model_config)
    text_encoder = text_encoder.to(device).eval()
    
    logger.info("Loading image encoder...")
    image_encoder = get_image_encoder(mode, num_classes=None, config=model_config)
    image_encoder = image_encoder.to(device).eval()
    
    # Load fusion model weights from MinIO first to detect architecture
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        if not storage.download_file(fusion_artifact_path, tmp.name):
            raise ValueError(f"Failed to download fusion model from {fusion_artifact_path}")
        
        state_dict = torch.load(tmp.name, map_location=device)
        os.unlink(tmp.name)
    
    # Detect if MultiHeadAttentionPooling was used (check for 'queries' in img_attn_pool)
    use_multihead_pool = False
    if 'img_attn_pool' in state_dict:
        if 'queries' in state_dict['img_attn_pool']:
            use_multihead_pool = True
            logger.info("Detected MultiHeadAttentionPooling architecture")
        else:
            logger.info("Detected simple AttentionPooling architecture")
    
    # Initialize VideoClassifierWithAttnPool (matching training architecture)
    logger.info(f"Loading {fusion_type} fusion model...")
    
    fusion_model = VideoClassifierWithAttnPool(
        d_img=d_img,
        d_txt=d_txt,
        d_fused=d_fused,
        dropout=0.0,  # No dropout for inference
        use_multihead_pool=use_multihead_pool
    )
    
    # Load weights
    # The saved state_dict has structure:
    # {'img_attn_pool': {...}, 'txt_attn_pool': {...}, 'fusion': {...}}
    if 'img_attn_pool' in state_dict:
        fusion_model.img_attn_pool.load_state_dict(state_dict['img_attn_pool'])
        logger.info("  Loaded img_attn_pool")
    if 'txt_attn_pool' in state_dict:
        fusion_model.txt_attn_pool.load_state_dict(state_dict['txt_attn_pool'])
        logger.info("  Loaded txt_attn_pool")
    if 'fusion' in state_dict:
        fusion_model.fusion.load_state_dict(state_dict['fusion'])
        logger.info("  Loaded fusion (GatedFusionClassifier)")
    
    logger.info(f"✅ Loaded fusion model from {fusion_artifact_path}")
    
    fusion_model = fusion_model.to(device).eval()
    
    return {
        'text_encoder': text_encoder,
        'image_encoder': image_encoder,
        'fusion_model': fusion_model,
        'fusion_type': fusion_type,
        'mode': mode,
        'device': device,
        'd_img': d_img,
        'd_txt': d_txt
    }


def predict_advanced_fusion(video_id: str, models: Dict, sample_id: str = None) -> Optional[Dict]:
    """
    Direct 4-class prediction with advanced fusion.
    
    REQUIRES sample_id to load pre-generated frames and chunks from MinIO.
    No fallback - if sample_id is missing or data not found, returns None.
    
    Args:
        video_id: Video ID
        models: Dict from load_advanced_fusion_models()
        sample_id: REQUIRED - sample_id for pre-generated samples
    
    Returns:
        Dict with prediction results or None if error
    """
    try:
        text_encoder = models['text_encoder']
        image_encoder = models['image_encoder']
        fusion_model = models['fusion_model']
        device = models['device']
        mode = models['mode']
        d_img = models.get('d_img', 1280)
        d_txt = models.get('d_txt', 384)
        
        # sample_id is REQUIRED
        if not sample_id:
            logger.error(f"sample_id is required for {video_id}")
            return None
        
        logger.info(f"Predicting: video={video_id}, sample={sample_id}")
        
        # Load from pre-generated sample
        frames, chunks = load_sample_from_minio(sample_id, mode)
        
        if not frames:
            logger.error(f"No frames found for sample={sample_id}")
            return None
        
        if not chunks:
            logger.error(f"No chunks found for sample={sample_id}")
            return None
        
        logger.info(f"Loaded: {len(frames)} frames, {len(chunks)} chunks")
        
        # Encode and predict
        with torch.no_grad():
            # Encode frames - keep as sequence for attention pooling
            if frames:
                frame_embs = image_encoder.get_embeddings(frames)  # (N, D_img)
                # Pad to NUM_FRAME_SEGMENTS if needed
                if frame_embs.size(0) < NUM_FRAME_SEGMENTS:
                    pad = torch.zeros(NUM_FRAME_SEGMENTS - frame_embs.size(0), d_img)
                    frame_embs = torch.cat([frame_embs, pad], dim=0)
                elif frame_embs.size(0) > NUM_FRAME_SEGMENTS:
                    frame_embs = frame_embs[:NUM_FRAME_SEGMENTS]
                # Add batch dimension: (1, 16, D_img)
                img_embs = frame_embs.unsqueeze(0).to(device)
            else:
                img_embs = torch.zeros(1, NUM_FRAME_SEGMENTS, d_img).to(device)
            
            # Encode text chunks - keep as sequence for attention pooling
            if chunks:
                chunk_embs = text_encoder.get_embeddings(chunks)  # (N, D_txt)
                # Pad to NUM_TRANSCRIPT_CHUNKS if needed
                if chunk_embs.size(0) < NUM_TRANSCRIPT_CHUNKS:
                    pad = torch.zeros(NUM_TRANSCRIPT_CHUNKS - chunk_embs.size(0), d_txt)
                    chunk_embs = torch.cat([chunk_embs, pad], dim=0)
                elif chunk_embs.size(0) > NUM_TRANSCRIPT_CHUNKS:
                    chunk_embs = chunk_embs[:NUM_TRANSCRIPT_CHUNKS]
                # Add batch dimension: (1, 5, D_txt)
                txt_embs = chunk_embs.unsqueeze(0).to(device)
            else:
                txt_embs = torch.zeros(1, NUM_TRANSCRIPT_CHUNKS, d_txt).to(device)
            
            # VideoClassifierWithAttnPool forward: (batch, seq, dim) -> logits
            logits = fusion_model(img_embs, txt_embs)  # (1, num_classes)
            
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            
            class_probs = {label: probs[i].item() for i, label in enumerate(LABELS)}
            final_prediction = LABELS[pred_idx]
            
            logger.info(f"Prediction: {final_prediction} (conf: {confidence:.3f})")
            
            return {
                'prediction': final_prediction,
                'confidence': confidence,
                'is_harmful': final_prediction != 'Safe',
                'class_probs': class_probs,
                'fusion_type': models['fusion_type']
            }
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}", exc_info=True)
        return None


def load_sample_from_minio(sample_id: str, mode: str) -> tuple:
    """
    Load pre-generated sample from MinIO.
    
    Returns:
        Tuple of (frames: List[PIL.Image], chunks: List[str])
    """
    frames = []
    chunks = []
    
    # Load frames
    for i in range(NUM_FRAME_SEGMENTS):
        frame_path = f"samples/{sample_id}/frame_{i:02d}.jpg"
        try:
            data = storage.download_data(frame_path)
            if data:
                img = Image.open(BytesIO(data)).convert('RGB')
                frames.append(img)
        except Exception:
            pass
    
    # Load transcript chunks
    transcript_path = f"samples/{sample_id}/transcript.json"
    try:
        data = storage.download_data(transcript_path)
        if data:
            chunk_data = json.loads(data.decode('utf-8'))
            chunks = chunk_data.get('chunks', [])
    except Exception:
        pass
    
    logger.info(f"Loaded sample {sample_id}: {len(frames)} frames, {len(chunks)} chunks")
    return frames if frames else None, chunks if chunks else None


# Model cache (for reuse across multiple predictions)
_MODEL_CACHE = {}


def get_cached_models(mode='ultra_light'):
    """
    Get cached models or load if not in cache.
    
    Args:
        mode: Model mode
    
    Returns:
        Models dict from load_advanced_fusion_models()
    """
    if mode not in _MODEL_CACHE:
        logger.info(f"Loading models into cache for mode={mode}")
        _MODEL_CACHE[mode] = load_advanced_fusion_models(mode)
    else:
        logger.info(f"Using cached models for mode={mode}")
    
    return _MODEL_CACHE[mode]


def clear_model_cache():
    """Clear model cache (useful for testing or memory management)."""
    global _MODEL_CACHE
    _MODEL_CACHE = {}
    logger.info("Model cache cleared")

