"""
DAG để backfill sample_predictions từ models đã train.

Chạy inference lại trên tất cả samples và lưu predictions vào bảng sample_predictions
để phục vụ cho Error Analysis feature.

Usage:
    Trigger DAG với config:
    {
        "model_id": 40,  // hoặc null để backfill tất cả active models
    }
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def backfill_sample_predictions(**context):
    """Backfill predictions for specified model or all active models."""
    import os
    import sys
    import tempfile
    import torch
    import gc
    
    sys.path.insert(0, '/opt/airflow')
    
    from common.io.database import db
    from common.io.storage import storage
    from common.io.config import config
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
    
    cfg = config  # Alias for compatibility
    
    conf = context.get('dag_run').conf or {}
    model_id = conf.get('model_id')
    batch_size = conf.get('batch_size', 32)
    
    # Get models to process
    if model_id:
        models = db.execute(
            "SELECT * FROM model_registry WHERE id = %s",
            (model_id,), fetch=True
        )
    else:
        models = db.execute(
            "SELECT * FROM model_registry WHERE is_active = true ORDER BY id",
            fetch=True
        )
    
    if not models:
        print("No models to process")
        return {"status": "no_models"}
    
    results = []
    
    for model_info in models:
        mid = model_info['id']
        mode = model_info['mode']
        artifact_path = model_info['artifact_path']
        
        print(f"\n{'='*60}")
        print(f" BACKFILLING MODEL {mid} ({mode})")
        print(f"{'='*60}")
        
        try:
            # Get dimensions from training_config (stored when model was trained)
            training_config = model_info.get('training_config') or {}
            
            # Use training config if available, otherwise fall back to defaults
            if training_config:
                d_img = training_config.get('d_img', 512)
                d_txt = training_config.get('d_txt', 384)
                d_fused = training_config.get('d_fused', 256)
                num_heads = training_config.get('num_heads', 4)
                num_layers = training_config.get('num_layers', 1)
                dropout = training_config.get('dropout', 0.3)
                use_multihead_pool = training_config.get('use_multihead_pool', False)
            else:
                # Fallback to mode-based defaults
                if mode == 'balanced':
                    d_img = cfg.get('balanced_img_dim', 2048)
                    d_txt = cfg.get('balanced_txt_dim', 768)
                    d_fused = 512
                    use_multihead_pool = True
                else:  # ultra_light
                    d_img = cfg.get('ultra_light_img_dim', 512)
                    d_txt = cfg.get('ultra_light_txt_dim', 384)
                    d_fused = 256
                    use_multihead_pool = False
                num_heads = 4
                num_layers = 1
                dropout = 0.3
            
            print(f"  Dimensions: d_img={d_img}, d_txt={d_txt}, d_fused={d_fused}")
            
            # Download model from MinIO
            print(f"  Downloading model from MinIO...")
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                temp_path = f.name
            
            storage.download_file(artifact_path, temp_path)
            state_dict = torch.load(temp_path, map_location='cpu', weights_only=True)
            os.unlink(temp_path)
            
            # Create model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"  Device: {device}")
            
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
            
            # Get samples
            train_samples = db.get_training_samples_v2(split='train')
            val_samples = db.get_training_samples_v2(split='val')
            test_samples = db.get_training_samples_v2(split='test')
            
            print(f"  Samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
            
            # Check/create cache
            cache_dir = f"{CACHE_DIR}/{mode}"
            os.makedirs(cache_dir, exist_ok=True)
            
            all_samples = train_samples + val_samples + test_samples
            
            # Check for missing embeddings
            missing = [s for s in all_samples if not os.path.exists(os.path.join(cache_dir, f"{s['sample_id']}.pt"))]
            
            if missing:
                print(f"  ⚠️ {len(missing)} samples missing embeddings. Computing...")
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
                    
                print(f"  Processing {split_name} ({len(samples)} samples)...")
                ds = PrecomputedDataset(cache_dir, samples, d_img, d_txt)
                loader = DataLoader(ds, batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
                
                preds = collect_sample_predictions(model, loader, device)
                for pred in preds:
                    pred['split'] = split_name
                
                all_predictions.extend(preds)
                
                correct = sum(1 for p in preds if p['true_label'] == p['predicted_label'])
                print(f"    Correct: {correct}/{len(preds)} ({100*correct/len(preds):.1f}%)")
            
            # Save to database
            print(f"  Saving {len(all_predictions)} predictions...")
            db.save_sample_predictions_batch(mid, all_predictions)
            
            # Cleanup
            del model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            results.append({
                "model_id": mid,
                "mode": mode,
                "predictions": len(all_predictions),
                "status": "success"
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "model_id": mid,
                "mode": mode,
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}


with DAG(
    dag_id='backfill_sample_predictions',
    default_args=default_args,
    description='Backfill sample_predictions table for error analysis',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['maintenance', 'error-analysis'],
) as dag:
    
    backfill_task = PythonOperator(
        task_id='backfill_predictions',
        python_callable=backfill_sample_predictions,
        provide_context=True,
    )
