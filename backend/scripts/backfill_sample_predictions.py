#!/usr/bin/env python3
"""
Backfill sample_predictions table from existing trained models.

This script:
1. Loads trained model from MinIO
2. Runs inference on all training/validation samples
3. Saves predictions to sample_predictions table

Usage:
    docker compose exec api python -m scripts.backfill_sample_predictions --model-id 40
"""

import os
import sys
import argparse
import tempfile
import torch
import gc

# Add project root to path
sys.path.insert(0, '/app')

from common.io.database import db
from common.io.storage import storage
from common.io.config import cfg
from common.models.image_encoder import get_image_encoder
from common.models.text_encoder import get_text_encoder
from common.pipelines.train_with_advanced_fusion import (
    VideoClassifierWithAttnPool,
    PrecomputedDataset,
    collate_fn,
    precompute_embeddings,
    collect_sample_predictions,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    NUM_FRAMES,
    NUM_CHUNKS,
    CACHE_DIR,
)
from torch.utils.data import DataLoader


def backfill_model_predictions(model_id: int, batch_size: int = 32):
    """Backfill predictions for a specific model."""
    
    print(f"\n{'='*60}")
    print(f" BACKFILLING SAMPLE PREDICTIONS FOR MODEL {model_id}")
    print(f"{'='*60}")
    
    # Get model info
    model_info = db.execute(
        "SELECT * FROM model_registry WHERE id = %s",
        (model_id,), fetch=True
    )
    
    if not model_info:
        print(f"ERROR: Model {model_id} not found")
        return False
    
    model_info = model_info[0]
    mode = model_info['mode']
    artifact_path = model_info['artifact_path']
    
    print(f"  Mode: {mode}")
    print(f"  Artifact: {artifact_path}")
    
    # Determine dimensions based on mode
    if mode == 'balanced':
        d_img = cfg.get('balanced_img_dim', 2048)
        d_txt = cfg.get('balanced_txt_dim', 768)
        d_fused = 512
    else:  # ultra_light
        d_img = cfg.get('ultra_light_img_dim', 512)
        d_txt = cfg.get('ultra_light_txt_dim', 384)
        d_fused = 256
    
    num_heads = 4
    num_layers = 1
    dropout = 0.3
    
    print(f"  Dimensions: d_img={d_img}, d_txt={d_txt}, d_fused={d_fused}")
    
    # Download model from MinIO
    print(f"\n  Downloading model from MinIO...")
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    storage.download_file(artifact_path, temp_path)
    state_dict = torch.load(temp_path, map_location='cpu', weights_only=True)
    os.unlink(temp_path)
    print(f"  ✅ Model downloaded")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    use_multihead_pool = (mode == 'balanced')
    model = VideoClassifierWithAttnPool(
        d_img=d_img,
        d_txt=d_txt,
        d_fused=d_fused,
        dropout=dropout,
        num_heads=num_heads,
        num_layers=num_layers,
        use_multihead_pool=use_multihead_pool
    )
    
    # Load weights
    model.img_attn_pool.load_state_dict(state_dict['img_attn_pool'])
    model.txt_attn_pool.load_state_dict(state_dict['txt_attn_pool'])
    model.fusion.load_state_dict(state_dict['fusion'])
    model.to(device)
    model.eval()
    print(f"  ✅ Model loaded")
    
    # Get samples
    train_samples = db.get_training_samples_v2(split='train')
    val_samples = db.get_training_samples_v2(split='val')
    test_samples = db.get_training_samples_v2(split='test')
    
    print(f"\n  Samples:")
    print(f"    Train: {len(train_samples)}")
    print(f"    Val:   {len(val_samples)}")
    print(f"    Test:  {len(test_samples)}")
    
    # Check/create cache
    cache_dir = f"{CACHE_DIR}/{mode}"
    os.makedirs(cache_dir, exist_ok=True)
    
    all_samples = train_samples + val_samples + test_samples
    
    # Check for missing embeddings
    missing = [s for s in all_samples if not os.path.exists(os.path.join(cache_dir, f"{s['sample_id']}.pt"))]
    
    if missing:
        print(f"\n  ⚠️ {len(missing)} samples missing embeddings. Computing...")
        img_encoder = get_image_encoder(mode, num_classes=None, config=cfg)
        txt_encoder = get_text_encoder(mode, num_classes=None, config=cfg)
        img_encoder.eval()
        txt_encoder.eval()
        precompute_embeddings(missing, mode, img_encoder, txt_encoder, d_img, d_txt, cache_dir)
        del img_encoder, txt_encoder
        gc.collect()
    
    # Collect predictions for each split
    all_predictions = []
    
    for split_name, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        if not samples:
            continue
            
        print(f"\n  Processing {split_name} set ({len(samples)} samples)...")
        ds = PrecomputedDataset(cache_dir, samples, d_img, d_txt)
        loader = DataLoader(ds, batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        
        preds = collect_sample_predictions(model, loader, device)
        for pred in preds:
            pred['split'] = split_name
        
        all_predictions.extend(preds)
        
        # Count correct/incorrect
        correct = sum(1 for p in preds if p['true_label'] == p['predicted_label'])
        print(f"    ✅ Correct: {correct}/{len(preds)} ({100*correct/len(preds):.1f}%)")
    
    # Save to database
    print(f"\n  Saving {len(all_predictions)} predictions to database...")
    db.save_sample_predictions_batch(model_id, all_predictions)
    print(f"  ✅ Done!")
    
    # Verify
    count = db.execute(
        "SELECT COUNT(*) as cnt FROM sample_predictions WHERE model_id = %s",
        (model_id,), fetch=True
    )[0]['cnt']
    print(f"\n  Verified: {count} records in sample_predictions for model_id={model_id}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Backfill sample predictions')
    parser.add_argument('--model-id', type=int, help='Specific model ID to backfill')
    parser.add_argument('--all-active', action='store_true', help='Backfill all active models')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()
    
    if args.model_id:
        backfill_model_predictions(args.model_id, args.batch_size)
    elif args.all_active:
        # Get all active models
        models = db.execute(
            "SELECT id FROM model_registry WHERE is_active = true ORDER BY id",
            fetch=True
        )
        for m in models:
            backfill_model_predictions(m['id'], args.batch_size)
    else:
        print("Please specify --model-id or --all-active")
        sys.exit(1)


if __name__ == '__main__':
    main()
