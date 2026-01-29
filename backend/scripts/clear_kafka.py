"""
Script to clear all Kafka topics - remove all messages.

This script will:
1. Delete all existing topics
2. Recreate empty topics with same configuration

Usage:
    python -m backend.scripts.clear_kafka
"""

import os
import sys
import logging
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Kafka topics to clear
TOPICS = [
    'video-preprocessing',
    'video-inference',
    'video-training',
    'video-status',
]


def get_kafka_config():
    """Get Kafka configuration from environment."""
    return {
        'bootstrap_servers': os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'),
    }


def clear_kafka_topics():
    """Clear all Kafka topics by deleting and recreating them."""
    try:
        from kafka import KafkaAdminClient
        from kafka.admin import NewTopic
        from kafka.errors import TopicAlreadyExistsError
        
        config = get_kafka_config()
        logger.info(f"Connecting to Kafka at {config['bootstrap_servers']}")
        
        admin = KafkaAdminClient(
            bootstrap_servers=config['bootstrap_servers'],
            client_id='kafka-cleaner'
        )
        
        # Delete existing topics
        logger.info("Deleting existing topics...")
        try:
            admin.delete_topics(topics=TOPICS, timeout_ms=10000)
            logger.info(f"✅ Deleted topics: {', '.join(TOPICS)}")
        except Exception as e:
            logger.warning(f"Error deleting topics (may not exist): {e}")
        
        # Wait a bit for topics to be fully deleted
        import time
        time.sleep(2)
        
        # Recreate topics with same configuration as in docker-compose
        logger.info("Recreating topics...")
        new_topics = [
            NewTopic(
                name='video-preprocessing',
                num_partitions=3,
                replication_factor=1
            ),
            NewTopic(
                name='video-inference',
                num_partitions=3,
                replication_factor=1
            ),
            NewTopic(
                name='video-training',
                num_partitions=1,
                replication_factor=1
            ),
            NewTopic(
                name='video-status',
                num_partitions=3,
                replication_factor=1
            ),
        ]
        
        try:
            admin.create_topics(new_topics=new_topics, timeout_ms=10000)
            logger.info(f"✅ Created topics: {', '.join(TOPICS)}")
        except TopicAlreadyExistsError:
            logger.warning("Topics already exist (may have been recreated)")
        except Exception as e:
            logger.error(f"❌ Error creating topics: {e}")
            raise
        
        # Verify topics exist
        existing_topics = admin.list_topics()
        logger.info(f"✅ Existing topics: {sorted(existing_topics)}")
        
        admin.close()
        logger.info("=" * 80)
        logger.info("✅ Kafka topics cleared successfully!")
        logger.info(f"   Topics: {', '.join(TOPICS)}")
        logger.info("   All messages have been removed")
        logger.info("=" * 80)
        
    except ImportError:
        logger.error("❌ kafka-python not installed. Install with: pip install kafka-python")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error clearing Kafka topics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("Kafka Topics Cleaner")
    logger.info("=" * 80)
    logger.info(f"Topics to clear: {', '.join(TOPICS)}")
    logger.info("")
    
    # Skip confirmation if SKIP_CONFIRM env var is set
    if os.environ.get('SKIP_CONFIRM', '').lower() not in ['1', 'true', 'yes']:
        try:
            confirm = input("⚠️  This will DELETE ALL MESSAGES in Kafka topics. Continue? (yes/no): ")
            if confirm.lower() not in ['yes', 'y']:
                logger.info("Cancelled.")
                return
        except EOFError:
            # Running in non-interactive mode, proceed anyway
            logger.warning("Running in non-interactive mode, proceeding without confirmation...")
    
    clear_kafka_topics()


if __name__ == '__main__':
    main()

