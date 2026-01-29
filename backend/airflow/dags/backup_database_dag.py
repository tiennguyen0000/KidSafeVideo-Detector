"""
Database Backup DAG

Automatically backup database to JSON files:
- Videos
- Training samples
- Predictions
- Model registry
- Fusion config

Backup location: /opt/airflow/backups/db_backup_YYYYMMDD_HHMMSS/
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, '/opt/airflow')

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def backup_table(**context):
    """Backup a single table to JSON file."""
    from common.io import db
    
    table_name = context['params']['table_name']
    backup_dir = context['params']['backup_dir']
    
    try:
        query = f"SELECT * FROM {table_name}"
        rows = db.execute(query, fetch=True)
        
        if rows:
            backup_file = Path(backup_dir) / f"{table_name}.json"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(rows, f, indent=2, default=str)
            
            logger.info(f"✅ Backed up {table_name}: {len(rows)} records → {backup_file}")
            context['ti'].xcom_push(key=f'{table_name}_count', value=len(rows))
        else:
            logger.info(f"⚠️  {table_name} is empty, skipping backup")
            context['ti'].xcom_push(key=f'{table_name}_count', value=0)
        
        return {'table': table_name, 'records': len(rows) if rows else 0}
    except Exception as e:
        logger.error(f"❌ Error backing up {table_name}: {e}")
        raise


def create_backup_directory(**context):
    """Create timestamped backup directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f'/opt/airflow/backups/db_backup_{timestamp}'
    
    os.makedirs(backup_dir, exist_ok=True)
    logger.info(f"Created backup directory: {backup_dir}")
    
    # Push to XCom for downstream tasks
    context['ti'].xcom_push(key='backup_dir', value=backup_dir)
    context['ti'].xcom_push(key='backup_timestamp', value=timestamp)
    
    return backup_dir


def generate_backup_summary(**context):
    """Generate backup summary report."""
    ti = context['ti']
    
    backup_dir = ti.xcom_pull(key='backup_dir', task_ids='create_backup_dir')
    timestamp = ti.xcom_pull(key='backup_timestamp', task_ids='create_backup_dir')
    
    # Get record counts from each backup task
    tables = [
        'videos',
        'video_preprocess',
        'training_samples',
        'training_samples_v2',
        'predictions',
        'fusion_config',
        'model_registry'
    ]
    
    summary = {
        'backup_timestamp': timestamp,
        'backup_directory': backup_dir,
        'tables': {}
    }
    
    total_records = 0
    for table in tables:
        count = ti.xcom_pull(key=f'{table}_count', task_ids=f'backup_{table}') or 0
        summary['tables'][table] = count
        total_records += count
    
    summary['total_records'] = total_records
    
    # Save summary to JSON
    summary_file = Path(backup_dir) / 'backup_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 80)
    logger.info(f"BACKUP COMPLETED: {backup_dir}")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Total records: {total_records}")
    for table, count in summary['tables'].items():
        logger.info(f"  - {table}: {count} records")
    logger.info("=" * 80)
    
    return summary


def cleanup_old_backups(**context):
    """
    Remove backups older than 7 days to save disk space.
    Keeps only the most recent 5 backups regardless of age.
    """
    backup_root = Path('/opt/airflow/backups')
    
    if not backup_root.exists():
        logger.info("No backup directory found, skipping cleanup")
        return
    
    # Get all backup directories
    backup_dirs = []
    for item in backup_root.iterdir():
        if item.is_dir() and item.name.startswith('db_backup_'):
            try:
                # Extract timestamp from directory name
                timestamp_str = item.name.replace('db_backup_', '')
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                backup_dirs.append((timestamp, item))
            except ValueError:
                logger.warning(f"Invalid backup directory name: {item.name}")
    
    # Sort by timestamp (newest first)
    backup_dirs.sort(reverse=True, key=lambda x: x[0])
    
    # Keep only the most recent 5 backups
    if len(backup_dirs) > 5:
        logger.info(f"Found {len(backup_dirs)} backups, keeping only 5 most recent")
        
        for timestamp, backup_dir in backup_dirs[5:]:
            age_days = (datetime.now() - timestamp).days
            logger.info(f"Removing old backup: {backup_dir.name} (age: {age_days} days)")
            
            # Remove directory and all contents
            import shutil
            shutil.rmtree(backup_dir)
        
        logger.info(f"Cleanup completed: removed {len(backup_dirs) - 5} old backups")
    else:
        logger.info(f"Found {len(backup_dirs)} backups, no cleanup needed (keeping up to 5)")


with DAG(
    dag_id='database_backup',
    default_args=default_args,
    description='Backup database to JSON files (scheduled daily or manual trigger)',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['backup', 'database', 'maintenance'],
) as dag:
    
    # Task 1: Create backup directory
    task_create_dir = PythonOperator(
        task_id='create_backup_dir',
        python_callable=create_backup_directory,
    )
    
    # Task 2-8: Backup each table
    tables_to_backup = [
        'videos',
        'video_preprocess',
        'training_samples',
        'training_samples_v2',
        'predictions',
        'fusion_config',
        'model_registry'
    ]
    
    backup_tasks = []
    for table_name in tables_to_backup:
        task = PythonOperator(
            task_id=f'backup_{table_name}',
            python_callable=backup_table,
            params={
                'table_name': table_name,
                'backup_dir': '{{ ti.xcom_pull(key="backup_dir", task_ids="create_backup_dir") }}'
            }
        )
        backup_tasks.append(task)
    
    # Task 9: Generate summary
    task_summary = PythonOperator(
        task_id='generate_summary',
        python_callable=generate_backup_summary,
    )
    
    # Task 10: Cleanup old backups
    task_cleanup = PythonOperator(
        task_id='cleanup_old_backups',
        python_callable=cleanup_old_backups,
        trigger_rule='all_done',  # Run even if backup failed
    )
    
    # Define dependencies
    task_create_dir >> backup_tasks >> task_summary >> task_cleanup
