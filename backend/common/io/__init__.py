"""
IO & infra helpers (DB, storage, queue, config, kafka).

Re-exported for clearer imports:
    from common.io import db, storage, queue, config, kafka_producer
"""

from common.io.database import DatabaseClient, db  # noqa
from common.io.storage import StorageClient, storage  # noqa
from common.io.queue import QueueClient, queue  # noqa
from common.io.config import Config, config  # noqa

# Kafka (lazy import to avoid startup errors if Kafka is not available)
try:
    from common.io.kafka_client import (  # noqa
        get_kafka_producer, get_kafka_consumer,
        send_preprocessing_task, send_inference_task,
        send_training_task, send_status_update,
        KafkaProducer, KafkaConsumer,
        TOPIC_PREPROCESSING, TOPIC_INFERENCE, TOPIC_TRAINING, TOPIC_STATUS,
    )
    kafka_producer = get_kafka_producer
    kafka_consumer = get_kafka_consumer
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Kafka client not available: {e}")
    kafka_producer = None
    kafka_consumer = None

