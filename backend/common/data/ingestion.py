"""Data ingestion module for video classifier.

Supports two modes:
1. CSV with filename column: Upload from local files
2. CSV with only link + category_real: Download from YouTube/TikTok then upload
"""
import os
import csv
import logging
import hashlib
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

# Add common to path
sys.path.insert(0, '/app')

from common.io import db, storage, config
from common.data.transcript_cleaner import clean_transcript

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_video_platform(url: str) -> str:
    """Detect video platform from URL.
    
    Returns:
        'youtube', 'tiktok', or 'unknown'
    """
    url_lower = url.lower()
    if 'youtube.com' in url_lower or 'youtu.be' in url_lower:
        return 'youtube'
    elif 'tiktok.com' in url_lower:
        return 'tiktok'
    return 'unknown'


def download_video_from_url(video_url: str) -> Optional[Tuple[str, str]]:
    """
    Download video from URL (YouTube or TikTok).
    
    Args:
        video_url: Video URL
        
    Returns:
        Tuple of (video_id, local_path) or None if failed
    """
    platform = detect_video_platform(video_url)
    
    if platform == 'youtube':
        from common.data.youtube_downloader import YouTubeDownloader
        downloader = YouTubeDownloader()
        return downloader.download_video(video_url)
    
    elif platform == 'tiktok':
        from common.data.tiktok_downloader import TikTokDownloader
        downloader = TikTokDownloader()
        return downloader.download_video(video_url)
    
    else:
        logger.warning(f"Unknown video platform for URL: {video_url}")
        # Try yt-dlp as fallback
        try:
            import yt_dlp
            import tempfile
            
            video_id = hashlib.md5(video_url.encode()).hexdigest()[:12]
            output_dir = Path(tempfile.gettempdir()) / "video_downloads"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{video_id}.mp4"
            
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': str(output_path),
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            if output_path.exists():
                return (video_id, str(output_path))
                
        except Exception as e:
            logger.error(f"Failed to download from unknown platform: {e}")
    
    return None


class DataIngestion:
    """Handle data ingestion from CSV and local files to database and storage."""
    
    def __init__(self, csv_path: str = None, videos_dir: str = None):
        """
        Args:
            csv_path: path to labels.csv. Defaults:
                      - ENV RAW_LABELS_PATH
                      - else /opt/airflow/data/raw/labels.csv
                      - fallback /app/data/raw/labels.csv
            videos_dir: folder containing videos. Defaults:
                        - ENV RAW_VIDEOS_DIR
                        - else /opt/airflow/data/raw/videos
                        - fallback /app/data/raw/videos
        """
        default_csv = os.environ.get("RAW_LABELS_PATH", "/opt/airflow/data/raw/labels.csv")
        default_videos = os.environ.get("RAW_VIDEOS_DIR", "/opt/airflow/data/raw/videos")
        self.csv_path = Path(csv_path or default_csv or "/app/data/raw/labels.csv")
        self.videos_dir = Path(videos_dir or default_videos or "/app/data/raw/videos")
    
    def read_csv(self) -> Tuple[List[Dict], List[Dict]]:
        """Read and parse the labels CSV file.
        
        Returns:
            Tuple of (videos_with_file, videos_need_download)
            - videos_with_file: Videos that have local file (filename in CSV)
            - videos_need_download: Videos that need to be downloaded (no filename)
        """
        if not self.csv_path.exists():
            logger.error(f"CSV file not found: {self.csv_path}")
            return [], []
        
        videos_with_file = []
        videos_need_download = []
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip empty rows
                if not row.get('link'):
                    continue
                
                # Handle different CSV formats
                link = row['link'].strip()
                db.update_video_status(link, 'pending_preprocessing')
                
                # Filename in CSV format: "Safe/7315656508602256648.mp4"
                # Keep the full path including folder
                filename = row.get('filename', '').strip()
                
                # Handle label - prioritize 'category_real' column, fallback to 'label'
                label = row.get('category_real', '').strip()
                if not label:
                    label = row.get('label', '').strip()
                
                # If still no label, scan all columns for valid label values
                if not label or label.isdigit():
                    for key in row.keys():
                        val = row[key].strip() if row[key] else ''
                        if val in ['Aggressive', 'Sexual', 'Superstition', 'Safe']:
                            label = val
                            break
                
                # Label is required for training data
                if not label:
                    logger.warning(f"Skipping row without label: {link[:50]}...")
                    continue
                
                # Skip if video already exists in DB with storage_path
                existing = db.get_video(link)
                if existing and existing.get('storage_path'):
                    logger.debug(f"Video already exists with storage, skipping: {link}")
                    continue
                
                # Title - try multiple column names
                title = row.get('title', '').strip()
                if not title:
                    title = row.get('video_title', '').strip()
                
                # Transcript - try multiple column names
                transcript = row.get('speech2text', '').strip()
                if not transcript:
                    transcript = row.get('transcript', '').strip()
                
                video_data = {
                    'link': link,
                    'filename': filename,
                    'label': label,
                    'title': title,
                    'transcript': transcript
                }
                
                # Check if filename exists and file is present
                if filename:
                    video_path = self.videos_dir / filename
                    if video_path.exists():
                        videos_with_file.append(video_data)
                    else:
                        # Filename specified but file not found - need download
                        logger.info(f"File not found, will download: {link[:50]}...")
                        videos_need_download.append(video_data)
                else:
                    # No filename in CSV - need download
                    videos_need_download.append(video_data)
        
        logger.info(f"Read CSV: {len(videos_with_file)} with local file, {len(videos_need_download)} need download")
        return videos_with_file, videos_need_download
    
    def process_video(self, video_data: Dict, needs_download: bool = False) -> bool:
        """Process a single video: upload to MinIO and insert to database.
        
        Args:
            video_data: Video metadata dict
            needs_download: If True, download video from URL first
            
        Returns:
            True if successful, False otherwise
        """
        video_id = video_data['link']  # Use link as unique ID (may contain URL)
        filename = video_data.get('filename', '')  # May be empty if download needed
        label = video_data['label']
        title = video_data.get('title', '')
        
        # Clean transcript (remove CTAs, channel names, intro/outro)
        raw_transcript = video_data.get('transcript') if video_data.get('transcript') else None
        transcript = clean_transcript(raw_transcript) if raw_transcript else None
        
        if raw_transcript and not transcript:
            logger.warning(f"Transcript for {video_id} became empty after cleaning: '{raw_transcript[:100]}'")
        elif raw_transcript and transcript != raw_transcript:
            logger.info(f"Cleaned transcript for {video_id}: {len(raw_transcript)} → {len(transcript)} chars")
        
        # Sanitize storage path - MinIO doesn't allow /, :, @ in object names
        # Use MD5 hash of video_id for safe object key
        safe_video_id = hashlib.md5(video_id.encode()).hexdigest()
        storage_path = f"raw/{safe_video_id}.mp4"
        
        video_path = None
        temp_downloaded_path = None
        
        if needs_download:
            # Download video from URL
            logger.info(f"Downloading video: {video_id[:60]}...")
            result = download_video_from_url(video_id)
            
            if result:
                downloaded_id, local_path = result
                video_path = Path(local_path)
                temp_downloaded_path = local_path  # Remember for cleanup
                
                # Generate filename from downloaded video
                if not filename:
                    filename = f"{label}/{downloaded_id}.mp4"
                    
                logger.info(f"Downloaded: {video_id[:50]}... -> {local_path}")
            else:
                logger.error(f"Failed to download video: {video_id[:60]}...")
                return False
        else:
            # Build video path from CSV filename (format: "Safe/7315656508602256648.mp4")
            video_path = self.videos_dir / filename
            
            if not video_path.exists():
                logger.warning(f"Video file not found: {video_path}")
                return False
        
        logger.info(f"Video ID: {video_id} → Storage key: {storage_path}")
        
        # Upload to MinIO if file exists and not already in storage
        if video_path and video_path.exists():
            if not storage.object_exists(storage_path):
                logger.info(f"Uploading {filename} to MinIO...")
                success = storage.upload_file(storage_path, str(video_path), 'video/mp4')
                
                if success:
                    logger.info(f"Successfully uploaded {filename}")
                else:
                    logger.error(f"Failed to upload {filename}")
                    # Cleanup temp file
                    if temp_downloaded_path:
                        try:
                            os.unlink(temp_downloaded_path)
                        except:
                            pass
                    return False
            else:
                logger.info(f"Video {filename} already exists in MinIO, skipping upload")
            
            # Cleanup temp downloaded file
            if temp_downloaded_path:
                try:
                    os.unlink(temp_downloaded_path)
                except:
                    pass
        else:
            # Video not found
            if not storage.object_exists(storage_path):
                logger.warning(f"Video file not found locally and not in MinIO: {filename}")
                storage_path = None
        
        # Upsert to database (idempotent)
        try:
            db.upsert_video(
                video_id=video_id,
                filename=filename,
                label=label,
                title=title,
                transcript=transcript,
                storage_path=storage_path
            )
            
            logger.info(f"Upserted video to database: {video_id}")

            
            # Send to Kafka preprocessing queue (training data with labels)
            try:
                from common.io.kafka_client import send_preprocessing_task
                send_preprocessing_task(video_id, metadata={
                    'label': label,
                    'title': title,
                    'source': 'csv',
                    'storage_path': storage_path,
                    'has_transcript': transcript is not None,
                })
                logger.info(f"Sent to Kafka preprocessing queue: {video_id[:50]}...")
            except ImportError:
                logger.debug("Kafka not available, video will be picked up by polling")
            except Exception as kafka_error:
                logger.warning(f"Kafka send failed: {kafka_error}")
            
            return True
        except Exception as e:
            logger.error(f"Error upserting video {video_id} to database: {e}")
            return False
    
    def ingest_all(self) -> Dict[str, int]:
        """Ingest all videos from CSV.
        
        Handles two scenarios:
        1. Videos with local files (filename in CSV)
        2. Videos that need downloading (no filename or file not found)
        """
        videos_with_file, videos_need_download = self.read_csv()
        # for video in videos_need_download:
        #     logger.info(f"Turn on processing for download-needed video")
        #     db.update_video_status(video['link'], 'pending_download')
        
        # for video in videos_with_file:
        #     logger.info(f"Turn on processing for local-file video")
        #     db.update_video_status(video['video_id'], 'pending_preprocessing')

        total = len(videos_with_file) + len(videos_need_download)
        results = {
            'total': total,
            'success': 0,
            'failed': 0,
            'downloaded': 0,
            'local': 0,
        }
        
        # Process videos with local files
        logger.info(f"Processing {len(videos_with_file)} videos with local files...")
        for video_data in videos_with_file:
            if self.process_video(video_data, needs_download=False):
                results['success'] += 1
                results['local'] += 1
            else:
                results['failed'] += 1
        
        # Process videos that need downloading
        if videos_need_download:
            logger.info(f"Downloading and processing {len(videos_need_download)} videos...")
            for video_data in videos_need_download:
                if self.process_video(video_data, needs_download=True):
                    results['success'] += 1
                    results['downloaded'] += 1
                else:
                    results['failed'] += 1
        
        logger.info(f"Ingestion complete: {results}")
        return results

