"""
Spark Session Management for Video Classification Pipeline.

Provides centralized Spark session creation with:
- Memory-optimized configuration for 16GB RAM
- Shuffle optimization to reduce I/O
- Checkpoint directory for fault tolerance
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SparkSessionManager:
    """
    Singleton manager for Spark sessions.
    
    Ensures only one Spark session exists per JVM and provides
    optimized configuration for video processing workloads.
    """
    
    _instance: Optional['SparkSessionManager'] = None
    _spark = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_or_create(
        cls,
        app_name: str = "VideoClassifier",
        master: str = None,
        memory: str = "4g",
        cores: int = None,
        shuffle_partitions: int = 8,
        checkpoint_dir: str = "/tmp/spark-checkpoints"
    ):
        """
        Get or create a Spark session with optimized settings.
        
        Args:
            app_name: Application name for Spark UI
            master: Spark master URL (default: local[*])
            memory: Driver/executor memory (default: 4g)
            cores: Number of cores (default: auto-detect)
            shuffle_partitions: Number of shuffle partitions (default: 8)
            checkpoint_dir: Directory for checkpoints
            
        Returns:
            SparkSession instance
        """
        if cls._spark is not None:
            return cls._spark
        
        try:
            from pyspark.sql import SparkSession
            
            # Auto-detect cores if not specified
            if cores is None:
                cores = os.cpu_count() or 4
            
            # Determine master URL
            if master is None:
                master = os.environ.get("SPARK_MASTER", f"local[{cores}]")
            
            logger.info(f"Creating Spark session: {app_name}")
            logger.info(f"  Master: {master}")
            logger.info(f"  Memory: {memory}")
            logger.info(f"  Cores: {cores}")
            logger.info(f"  Shuffle partitions: {shuffle_partitions}")
            
            builder = SparkSession.builder \
                .appName(app_name) \
                .master(master) \
                .config("spark.driver.memory", memory) \
                .config("spark.executor.memory", memory) \
                .config("spark.sql.shuffle.partitions", shuffle_partitions) \
                .config("spark.default.parallelism", cores * 2) \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .config("spark.driver.maxResultSize", "1g") \
                .config("spark.network.timeout", "600s") \
                .config("spark.sql.broadcastTimeout", "600") \
                .config("spark.rpc.askTimeout", "600s")
            
            cls._spark = builder.getOrCreate()
            
            # Set checkpoint directory
            os.makedirs(checkpoint_dir, exist_ok=True)
            cls._spark.sparkContext.setCheckpointDir(checkpoint_dir)
            
            # Set log level
            cls._spark.sparkContext.setLogLevel("WARN")
            
            logger.info(f"✅ Spark session created: {cls._spark.version}")
            return cls._spark
            
        except Exception as e:
            logger.error(f"❌ Failed to create Spark session: {e}")
            raise
    
    @classmethod
    def stop(cls):
        """Stop the Spark session."""
        if cls._spark is not None:
            cls._spark.stop()
            cls._spark = None
            logger.info("Spark session stopped")
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if PySpark is available."""
        try:
            import pyspark
            return True
        except ImportError:
            return False


def get_spark(
    app_name: str = "VideoClassifier",
    memory: str = "4g",
    cores: int = None,
    shuffle_partitions: int = 8
):
    """
    Convenience function to get a Spark session.
    
    Args:
        app_name: Application name
        memory: Memory allocation
        cores: Number of cores
        shuffle_partitions: Shuffle partition count
        
    Returns:
        SparkSession instance
    """
    return SparkSessionManager.get_or_create(
        app_name=app_name,
        memory=memory,
        cores=cores,
        shuffle_partitions=shuffle_partitions
    )
