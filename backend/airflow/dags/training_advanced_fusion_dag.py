"""
Training DAG with Advanced Fusion - Uses GatedFusion or AttentionFusion.

This is the NEW training approach that replaces the 4-model system with a unified
fusion model that learns adaptive weights from data.
"""

import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

sys.path.insert(0, '/opt/airflow')

from common.io import config, db
import logging

logger = logging.getLogger(__name__)

# Default DAG arguments
default_args = {
    'owner': 'video_classifier',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def check_training_requirements(**context):
    """Check if we have enough samples to train."""
    from collections import Counter
    
    # Get config
    conf = context.get('dag_run').conf or {}
    mode = conf.get('mode', os.environ.get('MODEL_MODE', config.system.get('mode', 'ultra_light')))
    use_pregenerated = conf.get('use_pregenerated_samples', None)
    
    # Auto-detect: try v2 first (default), fallback to v1
    if use_pregenerated is None:
        # Auto-detect based on data availability
        v2_samples = db.get_training_samples_v2(split='train')
        if len(v2_samples) > 0:
            use_pregenerated = True
        else:
            use_pregenerated = False
    
    # Use v2 table if pregenerated, else v1
    if use_pregenerated:
        train_samples = db.get_training_samples_v2(split='train')
        val_samples = db.get_training_samples_v2(split='val')
    else:
        train_samples = db.get_training_samples(split='train')
        val_samples = db.get_training_samples(split='val')
    
    train_count = len(train_samples)
    val_count = len(val_samples)
    
    # Check class distribution
    train_labels = Counter(s['label'] for s in train_samples)
    val_labels = Counter(s['label'] for s in val_samples)
    
    logger.info("=" * 60)
    logger.info(f"Training Data Statistics (use_pregenerated={use_pregenerated})")
    logger.info("=" * 60)
    logger.info(f"Total train samples: {train_count}")
    logger.info(f"Total val samples:   {val_count}")
    logger.info("")
    logger.info("Train class distribution:")
    for label, count in train_labels.items():
        logger.info(f"  {label:15s}: {count:4d} samples")
    logger.info("")
    logger.info("Val class distribution:")
    for label, count in val_labels.items():
        logger.info(f"  {label:15s}: {count:4d} samples")
    logger.info("=" * 60)
    
    # Check minimum samples per class
    min_train_per_class = min(train_labels.values()) if train_labels else 0
    min_val_per_class = min(val_labels.values()) if val_labels else 0
    
    min_required = 5  # At least 5 samples per class
    
    if min_train_per_class < min_required:
        raise ValueError(f"Not enough train samples. Minority class has only {min_train_per_class} samples, need {min_required}")
    
    if min_val_per_class < 2:
        raise ValueError(f"Not enough val samples. Minority class has only {min_val_per_class} samples, need 2")
    
    logger.info(f"✅ Requirements met:")
    logger.info(f"   Train: Min {min_train_per_class} samples/class")
    logger.info(f"   Val:   Min {min_val_per_class} samples/class")
    logger.info(f"   AFTER balancing: {min_train_per_class * len(train_labels)} train, {min_val_per_class * len(val_labels)} val")
    
    return {
        'train_count': train_count,
        'val_count': val_count,
        'balanced_train': min_train_per_class * len(train_labels),
        'balanced_val': min_val_per_class * len(val_labels)
    }


def decide_fusion_type(**context):
    """
    Decide which fusion type to use based on config or DAG params.
    
    Returns:
        'train_gated' or 'train_attention'
    """
    conf = context.get('dag_run').conf or {}
    fusion_type = conf.get('fusion_type', os.environ.get('FUSION_TYPE', 'gated'))
    
    logger.info(f"Fusion type selected: {fusion_type}")
    
    if fusion_type == 'attention':
        return 'train_attention_fusion'
    else:
        return 'train_gated_fusion'


def train_gated_fusion(**context):
    """Train with AttentionFusion (cross-attention between modalities)."""
    import torch
    
    # Import training function
    sys.path.insert(0, '/opt/airflow')
    from common.pipelines.train_with_advanced_fusion import train_with_advanced_fusion
    
    # Get config
    conf = context.get('dag_run').conf or {}
    mode = conf.get('mode', os.environ.get('MODEL_MODE', config.system.get('mode', 'ultra_light')))
    epochs = int(conf.get('epochs', os.environ.get('TRAIN_EPOCHS', 30)))
    batch_size = int(conf.get('batch_size', 16))
    learning_rate = float(conf.get('learning_rate', 1e-4))
    early_stopping = int(conf.get('early_stopping', 7))
    dropout = float(conf.get('dropout', 0.3))
    d_fused = int(conf.get('d_fused', 256))
    num_heads = int(conf.get('num_heads', 4))
    num_layers = int(conf.get('num_layers', 1))
    
    # Auto-detect pregenerated samples
    use_pregenerated = conf.get('use_pregenerated_samples', None)
    if use_pregenerated is None:
        v2_count = len(db.get_training_samples_v2(split='train'))
        use_pregenerated = v2_count > 0
        logger.info(f"Auto-detected use_pregenerated_samples={use_pregenerated} (v2 has {v2_count} samples)")
    
    logger.info("=" * 60)
    logger.info("VIDEO CLASSIFIER TRAINING (AttentionFusion)")
    logger.info("=" * 60)
    logger.info(f"Mode: {mode}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Early stopping: {early_stopping}")
    logger.info(f"Dropout: {dropout}")
    logger.info(f"Fusion dim: {d_fused}")
    logger.info(f"Attn heads: {num_heads}")
    logger.info(f"Attn layers: {num_layers}")
    logger.info(f"Pre-generated samples: {use_pregenerated}")
    logger.info("=" * 60)
    
    # Train
    try:
        train_with_advanced_fusion(
            mode=mode,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device='cpu',
            early_stopping=early_stopping,
            dropout=dropout,
            d_fused=d_fused,
            num_heads=num_heads,
            num_layers=num_layers,
            use_pregenerated_samples=use_pregenerated
        )
        
        logger.info("✅ Training completed!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise


def train_attention_fusion(**context):
    """Train with AttentionFusionClassifier (powerful, for GPU)."""
    import torch
    
    # Import training function
    sys.path.insert(0, '/opt/airflow')
    from common.pipelines.train_with_advanced_fusion import train_with_advanced_fusion
    
    # Get config
    conf = context.get('dag_run').conf or {}
    mode = conf.get('mode', os.environ.get('MODEL_MODE', config.system.get('mode', 'ultra_light')))
    epochs = int(conf.get('epochs', os.environ.get('TRAIN_EPOCHS', config.training.get('epochs', 20))))
    batch_size = int(conf.get('batch_size', 8))
    learning_rate = float(conf.get('learning_rate', 5e-5))
    
    # Auto-detect pregenerated samples
    use_pregenerated = conf.get('use_pregenerated_samples', None)
    if use_pregenerated is None:
        v2_count = len(db.get_training_samples_v2(split='train'))
        use_pregenerated = v2_count > 0
        logger.info(f"Auto-detected use_pregenerated_samples={use_pregenerated} (v2 has {v2_count} samples)")
    
    logger.info("=" * 80)
    logger.info("TRAINING WITH ATTENTION FUSION (Powerful)")
    logger.info("=" * 80)
    logger.info(f"Mode: {mode}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Pre-generated samples: {use_pregenerated}")
    logger.info("=" * 80)
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        logger.warning("⚠️ GPU not available, training on CPU (will be slow)")
    
    # Train
    try:
        train_with_advanced_fusion(
            mode=mode,
            fusion_type='attention',
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            use_pregenerated_samples=use_pregenerated
        )
        
        logger.info("✅ Attention fusion training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise


def validate_trained_models(**context):
    """
    Validate that models were trained and registered correctly.
    """
    conf = context.get('dag_run').conf or {}
    mode = conf.get('mode', os.environ.get('MODEL_MODE', config.system.get('mode', 'ultra_light')))
    
    logger.info("=" * 80)
    logger.info("VALIDATING TRAINED MODELS")
    logger.info("=" * 80)
    
    # Check if fusion model is registered
    active_models = db.get_active_models(mode)
    
    # Model name is 'fusion_gated' (simplified pipeline)
    model_type = 'fusion_gated'
    if model_type not in active_models:
        # Fallback: check for any fusion model
        fusion_models = [k for k in active_models.keys() if k.startswith('fusion')]
        if fusion_models:
            model_type = fusion_models[0]
            logger.info(f"Using fallback model: {model_type}")
        else:
            logger.error(f"❌ No fusion model found in registry!")
            raise ValueError(f"No fusion model was registered")
    
    model_info = active_models[model_type]
    logger.info(f"✅ Found active model: {model_type}")
    logger.info(f"  Version: {model_info['version']}")
    logger.info(f"  Path: {model_info['artifact_path']}")
    logger.info(f"  Metrics: {model_info.get('metrics', {})}")
    
    # Check if model file exists in MinIO
    from common.io import storage
    
    if not storage.object_exists(model_info['artifact_path']):
        logger.error(f"❌ Model file not found in MinIO: {model_info['artifact_path']}")
        raise ValueError("Model file not found in storage")
    
    logger.info(f"✅ Model file exists in MinIO")
    
    logger.info("=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)


def create_default_fusion_config(**context):
    """
    Create default fusion config (not used with advanced fusion, but kept for compatibility).
    
    Note: With advanced fusion, weights are learned, not manually set.
    But we keep this for backward compatibility with inference code.
    """
    conf = context.get('dag_run').conf or {}
    mode = conf.get('mode', os.environ.get('MODEL_MODE', config.system.get('mode', 'ultra_light')))
    
    # Check if config already exists
    existing_config = db.get_fusion_config(mode)
    
    if existing_config:
        logger.info(f"Fusion config already exists for mode: {mode}")
        return existing_config
    
    logger.info("Creating default fusion configuration...")
    
    # Default fusion weights (used as fallback, but fusion model learns better weights)
    default_weights = {
        'Superstition': {'text': 0.6, 'image': 0.4},  # Text-heavy
        'Aggressive': {'text': 0.5, 'image': 0.5},    # Image-heavy
        'Sexual': {'text': 0.2, 'image': 0.8},        # Image-heavy
        'Safe': {'text': 0.5, 'image': 0.5},          # Balanced
    }
    
    db.upsert_fusion_config(mode, default_weights)
    logger.info(f"✅ Created default fusion config for mode: {mode}")
    
    return default_weights


def log_training_summary(**context):
    """Log training summary and next steps."""
    conf = context.get('dag_run').conf or {}
    mode = conf.get('mode', os.environ.get('MODEL_MODE', 'ultra_light'))
    fusion_type = conf.get('fusion_type', 'gated')
    
    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Mode: {mode}")
    logger.info(f"Fusion type: {fusion_type}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Check model in database: GET /api/models")
    logger.info("2. Test inference: POST /api/upload")
    logger.info("3. Monitor predictions: GET /api/statistics")
    logger.info("4. Visualize fusion behavior (if using gated):")
    logger.info("   python backend/common/visualize_fusion.py")
    logger.info("=" * 80)


# Define the DAG
with DAG(
    'training_advanced_fusion',
    default_args=default_args,
    description='Training pipeline with Advanced Fusion (Gated or Attention)',
    schedule_interval=None,  # Manual trigger or triggered by preprocessing
    catchup=False,
    max_active_runs=1,
    tags=['training', 'ml', 'advanced_fusion'],
) as dag:
    
    # Step 1: Check requirements
    task_check = PythonOperator(
        task_id='check_requirements',
        python_callable=check_training_requirements,
    )
    
    # Step 2: Decide fusion type (branch)
    task_decide = BranchPythonOperator(
        task_id='decide_fusion_type',
        python_callable=decide_fusion_type,
    )
    
    # Step 3a: Train Gated Fusion (lightweight)
    task_train_gated = PythonOperator(
        task_id='train_gated_fusion',
        python_callable=train_gated_fusion,
    )
    
    # Step 3b: Train Attention Fusion (powerful)
    task_train_attention = PythonOperator(
        task_id='train_attention_fusion',
        python_callable=train_attention_fusion,
    )
    
    # Step 4: Join (empty operator)
    task_join = EmptyOperator(
        task_id='join_training',
        trigger_rule='none_failed_min_one_success',
    )
    
    # Step 5: Validate models
    task_validate = PythonOperator(
        task_id='validate_models',
        python_callable=validate_trained_models,
    )
    
    # Step 6: Create default fusion config (for compatibility)
    task_fusion_config = PythonOperator(
        task_id='create_fusion_config',
        python_callable=create_default_fusion_config,
    )
    
    # Step 7: Log summary
    task_summary = PythonOperator(
        task_id='log_summary',
        python_callable=log_training_summary,
    )
    
    # Define dependencies
    task_check >> task_decide
    task_decide >> [task_train_gated, task_train_attention]
    [task_train_gated, task_train_attention] >> task_join
    task_join >> task_validate >> task_fusion_config >> task_summary

