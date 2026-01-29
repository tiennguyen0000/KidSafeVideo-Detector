"""
PySpark utilities for Video Classification Pipeline.

This module provides Spark-based parallel processing for:
- Frame extraction
- Transcript processing  
- Batch inference
- Training data preparation

Design principles to avoid I/O explosion:
1. Coalesce partitions to reduce output files
2. Use broadcast variables for shared data
3. Cache intermediate DataFrames when reused
4. Prefer mapPartitions over map for batch operations
5. Limit shuffle operations
"""

from .session import SparkSessionManager, get_spark
from .preprocessing import SparkPreprocessor
from .batch_processor import SparkBatchProcessor

__all__ = [
    'SparkSessionManager',
    'get_spark', 
    'SparkPreprocessor',
    'SparkBatchProcessor',
]
