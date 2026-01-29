"""
Pipelines: Advanced fusion training & inference.

Exports:
    - train_with_advanced_fusion: Train GatedFusion or AttentionFusion models
    - load_advanced_fusion_models: Load models for inference
    - predict_advanced_fusion: Run prediction with fusion model
    - upload_videos_from_folder: Utility for bulk video upload
"""

# Advanced fusion (primary)
from common.pipelines.train_with_advanced_fusion import train_with_advanced_fusion  # noqa
from common.pipelines.inference_advanced_fusion import (  # noqa
    load_advanced_fusion_models,
    predict_advanced_fusion,
    get_cached_models,
    clear_model_cache,
)

# Utils
from common.pipelines.bulk_upload import upload_videos_from_folder  # noqa
