"""Bulk upload all videos from data/raw/videos to MinIO."""
import os
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, '/app')

from common.io import storage, db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_videos_from_folder(videos_dir: str = '/app/data/raw/videos'):
    """Upload all videos from subfolders to MinIO."""
    videos_path = Path(videos_dir)
    
    if not videos_path.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        return
    
    uploaded = 0
    skipped = 0
    failed = 0
    
    # Scan all subfolders
    for label_folder in videos_path.iterdir():
        if not label_folder.is_dir() or label_folder.name.startswith('.'):
            continue
        
        label = label_folder.name
        logger.info(f"Processing folder: {label}")
        
        # Upload all mp4 files
        for video_file in label_folder.glob('*.mp4'):
            filename = video_file.name
            
            # Generate video_id from filename (remove extension)
            video_id = filename.replace('.mp4', '')
            
            # Create safe storage path using MD5
            safe_id = hashlib.md5(video_id.encode()).hexdigest()
            storage_path = f"raw/{safe_id}.mp4"
            
            # Check if already in MinIO
            if storage.object_exists(storage_path):
                logger.debug(f"Already uploaded: {filename}")
                skipped += 1
                continue
            
            # Upload to MinIO
            try:
                success = storage.upload_file(storage_path, str(video_file), 'video/mp4')
                
                if success:
                    logger.info(f"✓ Uploaded: {filename} → {storage_path}")
                    uploaded += 1
                    
                    # Update DB if record exists
                    try:
                        # Try to find by filename
                        query = "SELECT video_id FROM videos WHERE filename = %s"
                        result = db.execute(query, (filename,), fetch=True)
                        
                        if result:
                            db_video_id = result[0][0]
                            update_query = """
                                UPDATE videos 
                                SET storage_path = %s 
                                WHERE video_id = %s AND storage_path IS NULL
                            """
                            db.execute(update_query, (storage_path, db_video_id))
                            logger.debug(f"  Updated DB: {db_video_id}")
                    except Exception as e:
                        logger.warning(f"  Could not update DB: {e}")
                else:
                    logger.error(f"✗ Failed: {filename}")
                    failed += 1
                    
            except Exception as e:
                logger.error(f"✗ Error uploading {filename}: {e}")
                failed += 1
    
    logger.info(f"""
╔══════════════════════════════════╗
║     BULK UPLOAD SUMMARY          ║
╠══════════════════════════════════╣
║  Uploaded:  {uploaded:6} videos     ║
║  Skipped:   {skipped:6} videos     ║
║  Failed:    {failed:6} videos     ║
╚══════════════════════════════════╝
    """)
    
    return {
        'uploaded': uploaded,
        'skipped': skipped,
        'failed': failed
    }


if __name__ == '__main__':
    logger.info("Starting bulk video upload...")
    results = upload_videos_from_folder()
    logger.info(f"Upload complete: {results}")
