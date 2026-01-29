"""
YouTube Video Downloader Module

Downloads videos from YouTube for inference pipeline.
Uses yt-dlp for reliable downloading with format selection.
"""

import os
import hashlib
import logging
import tempfile
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class YouTubeDownloader:
    """Download YouTube videos for inference."""
    
    def __init__(self, output_dir: str = None):
        """
        Args:
            output_dir: Directory to save downloaded videos.
                        Defaults to temp directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "yt_downloads"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download settings for efficiency
        self.format_options = {
            'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[ext=mp4]/best',
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'extract_audio': False,
            'merge_output_format': 'mp4',
            'postprocessor_args': [
                '-r', '6',  # 5-6 fps
            ],
        }
    
    def download_video(self, video_url: str) -> Optional[Tuple[str, str]]:
        """
        Download a single video from YouTube.
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Tuple of (video_id, local_path) or None if failed
        """
        try:
            import yt_dlp
        except ImportError:
            logger.error("yt-dlp not installed. Run: pip install yt-dlp")
            return None
        
        try:
            # Extract video ID from URL
            video_id = self._extract_video_id(video_url)
            if not video_id:
                logger.error(f"Could not extract video ID from: {video_url}")
                return None
            
            output_path = self.output_dir / f"{video_id}.mp4"
            
            # Skip if already downloaded
            if output_path.exists():
                logger.info(f"Video already downloaded: {video_id}")
                return (video_id, str(output_path))
            
            # Check if this is a TikTok URL
            is_tiktok = 'tiktok.com' in video_url
            
            # Download options - 240p resolution, 5-6 fps
            ydl_opts = {
                'format': '(bestvideo[height<=240]+bestaudio/best[height<=240]/best[height<=240])[ext=mp4]/best' if not is_tiktok else 'best[ext=mp4]/best',
                'outtmpl': str(output_path),
                'quiet': False,
                'no_warnings': False,
                'socket_timeout': 60 if is_tiktok else 30,  # Longer timeout for TikTok
                'retries': 2 if is_tiktok else 5,  # Fewer retries for TikTok (often fails anyway)
                'fragment_retries': 3 if is_tiktok else 10,
                'skip_unavailable_fragments': True,
                'ignoreerrors': False,
                'merge_output_format': 'mp4',
                'nocheckcertificate': True,
                'prefer_insecure': True,
                'geo_bypass': True,
                'extractor_retries': 1 if is_tiktok else 3,  # Minimal retries for TikTok
                # Add user-agent to avoid bot detection
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                },
            }
            
            # TikTok-specific options to handle impersonation and extraction issues
            if is_tiktok:
                # Suppress impersonation warnings (will still try to download but won't spam warnings)
                ydl_opts['warn_once'] = True
                # Note: TikTok often fails due to anti-bot protection - this is expected behavior
                # Videos will be skipped gracefully if download fails
            
            # Only add postprocessing for non-TikTok videos (TikTok videos are usually already in good format)
            if not is_tiktok:
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }]
                ydl_opts['postprocessor_args'] = [
                    '-c:v', 'libx264',  # Use h264 codec for re-encoding
                    '-c:a', 'aac',      # Use aac codec for audio
                    '-r', '6',          # 6 fps
                    '-vf', 'scale=-2:240',  # Ensure 240p height
                ]
            
            logger.info(f"Downloading video: {video_id} ({'TikTok' if is_tiktok else 'YouTube'})")
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                        
            except Exception as download_error:
                # Log specific error for TikTok
                error_msg = str(download_error)
                if is_tiktok:
                    # TikTok often fails due to anti-bot measures - log as warning and continue
                    logger.warning(f"TikTok download failed (TikTok anti-bot protection): {error_msg[:200]}")
                    logger.info(f"Skipping TikTok video download (video may still be accessible via thumbnail)")
                else:
                    logger.error(f"Download error: {error_msg}")
                # Cleanup partial download
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except:
                        pass
                return None
            
            # Check if file was downloaded successfully
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / 1024 / 1024
                logger.info(f"Downloaded successfully: {video_id} ({file_size_mb:.2f} MB)")
                return (video_id, str(output_path))
            else:
                logger.error(f"Download completed but file not found: {output_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading video {video_url}: {e}")
            return None
    
    def download_batch(self, videos: List[Dict], batch_size: int = 24) -> List[Dict]:
        """
        Download a batch of videos from search results.
        
        Args:
            videos: List of video dicts with 'videoUrl' field
            batch_size: Max videos to download (default 24)
            
        Returns:
            List of successfully downloaded video dicts with 'local_path' added
        """
        downloaded = []
        
        for i, video in enumerate(videos[:batch_size]):
            video_url = video.get('videoUrl')
            if not video_url:
                logger.warning(f"Video missing URL: {video.get('id')}")
                continue
            
            logger.info(f"[{i+1}/{min(len(videos), batch_size)}] Downloading: {video_url}")
            
            result = self.download_video(video_url)
            if result:
                video_id, local_path = result
                video_copy = video.copy()
                video_copy['local_path'] = local_path
                video_copy['youtube_id'] = video_id
                downloaded.append(video_copy)
            else:
                logger.warning(f"Failed to download: {video_url}")
        
        logger.info(f"Downloaded {len(downloaded)}/{min(len(videos), batch_size)} videos")
        return downloaded
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from URL (YouTube or TikTok)."""
        import re
        
        # TikTok URL format: https://www.tiktok.com/@username/video/1234567890
        tiktok_pattern = r'tiktok\.com/.*?/video/(\d+)'
        tiktok_match = re.search(tiktok_pattern, url)
        if tiktok_match:
            return tiktok_match.group(1)
        
        # YouTube URL patterns
        youtube_patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'([a-zA-Z0-9_-]{11})',  # Raw video ID
        ]
        
        for pattern in youtube_patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def cleanup(self, video_id: str = None):
        """
        Clean up downloaded files.
        
        Args:
            video_id: Specific video to clean up. If None, clean all.
        """
        try:
            if video_id:
                path = self.output_dir / f"{video_id}.mp4"
                if path.exists():
                    path.unlink()
                    logger.info(f"Cleaned up: {video_id}")
            else:
                for f in self.output_dir.glob("*.mp4"):
                    f.unlink()
                logger.info("Cleaned up all downloaded videos")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def download_and_ingest_videos(
    videos: List[Dict],
    db_client,
    storage_client,
    batch_size: int = 24
) -> Dict:
    """
    Download videos from YouTube and ingest them into the system.
    
    This is the main function called by the DAG.
    
    Args:
        videos: List of video dicts from YouTube search
        db_client: Database client instance
        storage_client: MinIO storage client instance
        batch_size: Max videos to process
        
    Returns:
        Dict with stats: {downloaded, ingested, skipped, failed}
    """
    stats = {
        'total': len(videos),
        'downloaded': 0,
        'ingested': 0,
        'skipped': 0,
        'failed': 0,
        'existing': 0,
    }
    
    downloader = YouTubeDownloader()
    
    videos_to_download = []
    
    for video in videos[:batch_size]:
        video_url = video.get('videoUrl', '')
        videos_to_download.append(video)
    
    
    if not videos_to_download:
        return stats
    
    # Download videos
    downloaded_videos = downloader.download_batch(videos_to_download, batch_size)
    stats['downloaded'] = len(downloaded_videos)
    
    # Ingest downloaded videos
    for video in downloaded_videos:
        try:
            video_url = video.get('videoUrl', '')
            video_id = video.get('youtube_id', '')
            local_path = video.get('local_path', '')
            
            if not local_path or not os.path.exists(local_path):
                logger.warning(f"Local file not found for {video_url}")
                stats['failed'] += 1
                continue
            
            # Generate safe storage key using MD5 hash
            safe_video_id = hashlib.md5(video_url.encode()).hexdigest()
            storage_path = f"raw/{safe_video_id}.mp4"
            
            # Upload to MinIO
            if not storage_client.object_exists(storage_path):
                success = storage_client.upload_file(storage_path, local_path, 'video/mp4')
                if not success:
                    logger.error(f"Failed to upload {video_url} to MinIO")
                    stats['failed'] += 1
                    continue
                logger.info(f"Uploaded to MinIO: {storage_path}")
            else:
                logger.info(f"Already in MinIO: {storage_path}")
            
            # Upsert to database (WITHOUT label for inference-only videos)
            # Use videoUrl as video_id for consistency with existing system
            db_client.upsert_video_for_inference(
                video_id=video_url,  # Use URL as ID for consistency
                filename=f"{video_id}.mp4",
                storage_path=storage_path,
                title=video.get('title', ''),
            )
            
            logger.info(f"Ingested video: {video_url}")
            stats['ingested'] += 1
            
            # Video status is already set to 'pending_preprocessing' by upsert_video_for_inference
            # Preprocessing DAG will scan PostgreSQL for videos with this status
            logger.info(f"Video marked as pending_preprocessing: {video_url[:50]}...")
            
            # Cleanup local file
            try:
                os.unlink(local_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error ingesting {video.get('videoUrl')}: {e}")
            stats['failed'] += 1
    
    # Cleanup
    downloader.cleanup()
    
    logger.info(f"Ingestion stats: {stats}")
    return stats
