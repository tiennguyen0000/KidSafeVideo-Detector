"""
Kafka client for video processing pipeline.

Topics:
    - video-preprocessing: Videos pending preprocessing
    - video-inference: Videos ready for inference
    - video-training: Training tasks
    - video-status: Status updates (for monitoring)

Usage:
    from common.io import kafka_producer, kafka_consumer
    
    # Producer
    kafka_producer.send_preprocessing_task(video_id, metadata)
    
    # Consumer
    for message in kafka_consumer.consume_preprocessing_tasks():
        process(message)
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Generator
from datetime import datetime

logger = logging.getLogger(__name__)

# Kafka topics
TOPIC_PREPROCESSING = 'video-preprocessing'
TOPIC_INFERENCE = 'video-inference'
TOPIC_TRAINING = 'video-training'
TOPIC_STATUS = 'video-status'


def get_kafka_config() -> Dict[str, Any]:
    """Get Kafka configuration from environment."""
    return {
        'bootstrap_servers': os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'),
    }


class KafkaProducer:
    """Kafka producer for video processing tasks."""
    
    def __init__(self):
        self._producer = None
        self._config = get_kafka_config()
    
    @property
    def producer(self):
        """Lazy initialization of Kafka producer."""
        if self._producer is None:
            try:
                from kafka import KafkaProducer as KP
                self._producer = KP(
                    bootstrap_servers=self._config['bootstrap_servers'],
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    acks='all',
                    retries=3,
                    retry_backoff_ms=500,
                )
                logger.info(f"Kafka producer connected to {self._config['bootstrap_servers']}")
            except Exception as e:
                logger.error(f"Failed to connect to Kafka: {e}")
                raise
        return self._producer
    
    def send_preprocessing_task(self, video_id: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Send video to preprocessing queue.
        
        Args:
            video_id: Video ID (URL or unique identifier)
            metadata: Additional metadata (label, title, transcript, etc.)
        
        Returns:
            True if successful
        """
        try:
            message = {
                'video_id': video_id,
                'timestamp': datetime.utcnow().isoformat(),
                'metadata': metadata or {},
            }
            
            future = self.producer.send(
                TOPIC_PREPROCESSING,
                key=video_id,
                value=message
            )
            # Wait for message to be sent (sync)
            future.get(timeout=10)
            
            logger.info(f"Sent preprocessing task: {video_id[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send preprocessing task: {e}")
            return False
    
    def send_inference_task(self, video_id: str, mode: str = 'ultra_light', 
                           sample_id: str = None) -> bool:
        """
        Send video to inference queue.
        
        Args:
            video_id: Video ID
            mode: Model mode (ultra_light or balanced)
            sample_id: Pre-computed sample ID
        
        Returns:
            True if successful
        """
        try:
            message = {
                'video_id': video_id,
                'mode': mode,
                'sample_id': sample_id,
                'timestamp': datetime.utcnow().isoformat(),
            }
            
            future = self.producer.send(
                TOPIC_INFERENCE,
                key=video_id,
                value=message
            )
            future.get(timeout=10)
            
            logger.info(f"Sent inference task: {video_id[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send inference task: {e}")
            return False
    
    def send_training_task(self, task_type: str, mode: str = 'ultra_light',
                          epochs: int = 30, config: Dict = None) -> bool:
        """
        Send training task to queue.
        
        Args:
            task_type: Type of training (e.g., 'train_fusion')
            mode: Model mode
            epochs: Number of epochs
            config: Additional training configuration
        
        Returns:
            True if successful
        """
        try:
            message = {
                'task_type': task_type,
                'mode': mode,
                'epochs': epochs,
                'config': config or {},
                'timestamp': datetime.utcnow().isoformat(),
            }
            
            future = self.producer.send(
                TOPIC_TRAINING,
                key=f"{mode}_{task_type}",
                value=message
            )
            future.get(timeout=10)
            
            logger.info(f"Sent training task: {task_type}, mode={mode}")
            return True
        except Exception as e:
            logger.error(f"Failed to send training task: {e}")
            return False
    
    def send_status_update(self, video_id: str, status: str, 
                          details: Dict = None) -> bool:
        """
        Send status update for monitoring.
        
        Args:
            video_id: Video ID
            status: New status (pending, processing, preprocessed, etc.)
            details: Additional details
        
        Returns:
            True if successful
        """
        try:
            message = {
                'video_id': video_id,
                'status': status,
                'details': details or {},
                'timestamp': datetime.utcnow().isoformat(),
            }
            
            future = self.producer.send(
                TOPIC_STATUS,
                key=video_id,
                value=message
            )
            future.get(timeout=10)
            
            logger.debug(f"Status update: {video_id[:30]}... -> {status}")
            return True
        except Exception as e:
            logger.error(f"Failed to send status update: {e}")
            return False
    
    def flush(self):
        """Flush pending messages."""
        if self._producer:
            self._producer.flush()
    
    def close(self):
        """Close producer connection."""
        if self._producer:
            self._producer.close()
            self._producer = None


class KafkaConsumer:
    """Kafka consumer for video processing tasks."""
    
    def __init__(self, group_id: str = 'video-classifier', auto_offset_reset: str = 'latest'):
        self._consumers = {}
        self._config = get_kafka_config()
        self._group_id = group_id
        self._auto_offset_reset = auto_offset_reset
    
    def _get_consumer(self, topic: str, auto_offset_reset: str = None):
        """Get or create consumer for a topic."""
        # Use instance-level setting if not specified
        offset_reset = auto_offset_reset or self._auto_offset_reset
        
        if topic not in self._consumers:
            try:
                from kafka import KafkaConsumer as KC
                self._consumers[topic] = KC(
                    topic,
                    bootstrap_servers=self._config['bootstrap_servers'],
                    group_id=f"{self._group_id}-{topic}",
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    key_deserializer=lambda k: k.decode('utf-8') if k else None,
                    auto_offset_reset=offset_reset,
                    enable_auto_commit=True,
                    auto_commit_interval_ms=5000,
                    max_poll_records=10,
                    session_timeout_ms=30000,
                    consumer_timeout_ms=1000,  # Timeout if no messages available
                )
                logger.info(f"Kafka consumer connected to {topic}")
            except Exception as e:
                logger.error(f"Failed to create consumer for {topic}: {e}")
                raise
        return self._consumers[topic]
    
    def consume_preprocessing_tasks(self, timeout_ms: int = 1000, 
                                    max_messages: int = None) -> Generator[Dict, None, None]:
        """
        Consume preprocessing tasks.
        
        Args:
            timeout_ms: Poll timeout in milliseconds
            max_messages: Maximum messages to consume (None = unlimited)
        
        Yields:
            Message dicts with video_id, metadata, timestamp
        """
        consumer = self._get_consumer(TOPIC_PREPROCESSING)
        count = 0
        
        while True:
            messages = consumer.poll(timeout_ms=timeout_ms)
            
            if not messages:
                # No messages, yield control
                if max_messages is not None and count >= max_messages:
                    break
                continue
            
            for topic_partition, records in messages.items():
                for record in records:
                    yield record.value
                    count += 1
                    
                    if max_messages is not None and count >= max_messages:
                        return
    
    def consume_inference_tasks(self, timeout_ms: int = 1000,
                               max_messages: int = None) -> Generator[Dict, None, None]:
        """
        Consume inference tasks.
        
        Args:
            timeout_ms: Poll timeout
            max_messages: Maximum messages to consume
        
        Yields:
            Message dicts with video_id, mode, sample_id, timestamp
        """
        consumer = self._get_consumer(TOPIC_INFERENCE)
        count = 0
        
        while True:
            messages = consumer.poll(timeout_ms=timeout_ms)
            
            if not messages:
                if max_messages is not None and count >= max_messages:
                    break
                continue
            
            for topic_partition, records in messages.items():
                for record in records:
                    yield record.value
                    count += 1
                    
                    if max_messages is not None and count >= max_messages:
                        return
    
    def get_pending_preprocessing_count(self) -> int:
        """Get approximate count of pending preprocessing tasks."""
        try:
            from kafka import KafkaAdminClient
            from kafka.admin import OffsetSpec
            
            admin = KafkaAdminClient(
                bootstrap_servers=self._config['bootstrap_servers']
            )
            
            # Get end offsets (latest messages)
            consumer = self._get_consumer(TOPIC_PREPROCESSING)
            partitions = consumer.partitions_for_topic(TOPIC_PREPROCESSING)
            
            if not partitions:
                return 0
            
            # This is approximate - full implementation would track consumer offsets
            return len(partitions) * 10  # Placeholder
        except Exception as e:
            logger.warning(f"Could not get pending count: {e}")
            return -1
    
    def close(self):
        """Close all consumer connections."""
        for topic, consumer in self._consumers.items():
            try:
                consumer.close()
            except Exception as e:
                logger.warning(f"Error closing consumer for {topic}: {e}")
        self._consumers = {}


# Singleton instances
_kafka_producer = None
_kafka_consumer = None


def get_kafka_producer() -> KafkaProducer:
    """Get singleton Kafka producer."""
    global _kafka_producer
    if _kafka_producer is None:
        _kafka_producer = KafkaProducer()
    return _kafka_producer


def get_kafka_consumer(group_id: str = 'video-classifier', 
                       auto_offset_reset: str = 'latest') -> KafkaConsumer:
    """
    Get Kafka consumer.
    
    Args:
        group_id: Consumer group ID
        auto_offset_reset: 'latest' to read from beginning, 'latest' for new messages only
    
    Returns:
        KafkaConsumer instance
    """
    # Create new consumer for unique group_ids (e.g., with timestamp)
    # This allows getting only latest messages when needed
    return KafkaConsumer(group_id, auto_offset_reset=auto_offset_reset)


# Module-level convenience functions
def send_preprocessing_task(video_id: str, metadata: Dict = None) -> bool:
    """Send video to preprocessing queue."""
    return get_kafka_producer().send_preprocessing_task(video_id, metadata)


def send_inference_task(video_id: str, mode: str = 'ultra_light', 
                       sample_id: str = None) -> bool:
    """Send video to inference queue."""
    return get_kafka_producer().send_inference_task(video_id, mode, sample_id)


def send_training_task(task_type: str, mode: str = 'ultra_light',
                      epochs: int = 30, config: Dict = None) -> bool:
    """Send training task."""
    return get_kafka_producer().send_training_task(task_type, mode, epochs, config)


def send_status_update(video_id: str, status: str, details: Dict = None) -> bool:
    """Send status update."""
    return get_kafka_producer().send_status_update(video_id, status, details)
