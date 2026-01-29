"""
TikTok Video Downloader Module

Downloads videos from TikTok for inference/training pipeline.
Uses TikTokApi (tiktokvm) for reliable downloading with batch support.
"""

import os
import hashlib
import logging
import tempfile
import asyncio
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class TikTokDownloader:
    """Download TikTok videos for inference/training pipeline."""
    
    BATCH_SIZE = 50  # Download 50 videos at a time
    
    def __init__(self, output_dir: str = None):
        """
        Args:
            output_dir: Directory to save downloaded videos.
                        Defaults to temp directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "tiktok_downloads"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._api = None
    
    async def _get_api(self):
        """Initialize TikTokApi instance lazily."""
        if self._api is None:
            try:
                from TikTokApi import TikTokApi
                self._api = TikTokApi()
                # Create session for requests
                await self._api.create_sessions(
                    num_sessions=1,
                    headless=True,
                    suppress_resource_load_types=["image", "media", "font", "stylesheet"]
                )
            except ImportError:
                logger.error("TikTokApi not installed. Run: pip install TikTokApi")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize TikTokApi: {e}")
                return None
        return self._api
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from TikTok URL."""
        import re
        
        # TikTok URL formats:
        # https://www.tiktok.com/@username/video/1234567890
        # https://vm.tiktok.com/ZMxxxxxxxx/
        # https://vt.tiktok.com/ZMxxxxxxxx/
        
        patterns = [
            r'tiktok\.com/.*?/video/(\d+)',
            r'vm\.tiktok\.com/([A-Za-z0-9]+)',
            r'vt\.tiktok\.com/([A-Za-z0-9]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    async def download_video_async(self, video_url: str) -> Optional[Tuple[str, str]]:
        """
        Download a single video from TikTok asynchronously.
        
        Args:
            video_url: TikTok video URL
            
        Returns:
            Tuple of (video_id, local_path) or None if failed
        """
        try:
            video_id = self._extract_video_id(video_url)
            if not video_id:
                # Generate ID from URL hash
                video_id = hashlib.md5(video_url.encode()).hexdigest()[:12]
            
            output_path = self.output_dir / f"{video_id}.mp4"
            
            # Skip if already downloaded
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Video already downloaded: {video_id}")
                return (video_id, str(output_path))
            
            api = await self._get_api()
            if not api:
                # Fallback to yt-dlp
                return self._download_with_ytdlp(video_url, video_id, output_path)
            
            try:
                # Try TikTokApi first
                video = api.video(url=video_url)
                video_data = await video.bytes()
                
                with open(output_path, 'wb') as f:
                    f.write(video_data)
                
                file_size_mb = output_path.stat().st_size / 1024 / 1024
                logger.info(f"Downloaded via TikTokApi: {video_id} ({file_size_mb:.2f} MB)")
                return (video_id, str(output_path))
                
            except Exception as api_error:
                logger.warning(f"TikTokApi failed for {video_id}: {api_error}, falling back to yt-dlp")
                return self._download_with_ytdlp(video_url, video_id, output_path)
                
        except Exception as e:
            logger.error(f"Error downloading TikTok video {video_url}: {e}")
            return None
    
    def _download_with_ytdlp(self, video_url: str, video_id: str, output_path: Path) -> Optional[Tuple[str, str]]:
        """Fallback download using yt-dlp."""
        try:
            import yt_dlp
            
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': str(output_path),
                'quiet': True,
                'no_warnings': True,
                'socket_timeout': 60,
                'retries': 3,
                'fragment_retries': 5,
                'skip_unavailable_fragments': True,
                'merge_output_format': 'mp4',
                'nocheckcertificate': True,
                'geo_bypass': True,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                },
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            if output_path.exists() and output_path.stat().st_size > 0:
                file_size_mb = output_path.stat().st_size / 1024 / 1024
                logger.info(f"Downloaded via yt-dlp: {video_id} ({file_size_mb:.2f} MB)")
                return (video_id, str(output_path))
            
            return None
            
        except Exception as e:
            logger.error(f"yt-dlp download failed for {video_id}: {e}")
            return None
    
    def download_video(self, video_url: str) -> Optional[Tuple[str, str]]:
        """
        Synchronous wrapper for download_video_async.
        
        Args:
            video_url: TikTok video URL
            
        Returns:
            Tuple of (video_id, local_path) or None if failed
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.download_video_async(video_url))
    
    async def download_batch_async(self, videos: List[Dict], batch_size: int = None) -> List[Dict]:
        """
        Download a batch of TikTok videos asynchronously.
        
        Args:
            videos: List of video dicts with 'videoUrl' or 'link' field
            batch_size: Max videos to download (default: 50)
            
        Returns:
            List of successfully downloaded video dicts with 'local_path' added
        """
        batch_size = batch_size or self.BATCH_SIZE
        downloaded = []
        
        videos_to_process = videos[:batch_size]
        logger.info(f"Starting batch download of {len(videos_to_process)} TikTok videos")
        
        for i, video in enumerate(videos_to_process):
            video_url = video.get('videoUrl') or video.get('link', '')
            if not video_url:
                logger.warning(f"Video missing URL: {video.get('id')}")
                continue
            
            # Only process TikTok URLs
            if 'tiktok.com' not in video_url.lower():
                logger.debug(f"Skipping non-TikTok URL: {video_url}")
                continue
            
            logger.info(f"[{i+1}/{len(videos_to_process)}] Downloading: {video_url[:60]}...")
            
            result = await self.download_video_async(video_url)
            if result:
                video_id, local_path = result
                video_copy = video.copy()
                video_copy['local_path'] = local_path
                video_copy['tiktok_id'] = video_id
                downloaded.append(video_copy)
            else:
                logger.warning(f"Failed to download: {video_url[:60]}...")
        
        logger.info(f"Downloaded {len(downloaded)}/{len(videos_to_process)} TikTok videos")
        return downloaded
    
    def download_batch(self, videos: List[Dict], batch_size: int = None) -> List[Dict]:
        """
        Synchronous wrapper for download_batch_async.
        
        Args:
            videos: List of video dicts with 'videoUrl' or 'link' field
            batch_size: Max videos to download (default: 50)
            
        Returns:
            List of successfully downloaded video dicts with 'local_path' added
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.download_batch_async(videos, batch_size))
    
    async def close(self):
        """Close TikTokApi sessions."""
        if self._api:
            try:
                await self._api.close_sessions()
            except:
                pass
            self._api = None
    
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
                logger.info("Cleaned up all downloaded TikTok videos")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def download_and_ingest_tiktok_videos(
    videos: List[Dict],
    db_client,
    storage_client,
    batch_size: int = 50
) -> Dict:
    """
    Download TikTok videos and ingest them into the system.
    
    This is the main function called by the DAG for TikTok videos.
    
    Args:
        videos: List of video dicts with 'link' or 'videoUrl' field
        db_client: Database client instance
        storage_client: MinIO storage client instance
        batch_size: Max videos to process (default: 50)
        
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
    
    # Filter only TikTok videos
    tiktok_videos = [v for v in videos if 'tiktok.com' in (v.get('link') or v.get('videoUrl', '')).lower()]
    
    if not tiktok_videos:
        logger.info("No TikTok videos to process")
        return stats
    
    logger.info(f"Processing {len(tiktok_videos)} TikTok videos")
    
    downloader = TikTokDownloader()
    
    # Download videos
    downloaded_videos = downloader.download_batch(tiktok_videos, batch_size)
    stats['downloaded'] = len(downloaded_videos)
    
    # Ingest downloaded videos
    for video in downloaded_videos:
        try:
            video_url = video.get('link') or video.get('videoUrl', '')
            video_id = video.get('tiktok_id', '')
            local_path = video.get('local_path', '')
            label = video.get('category_real') or video.get('label')
            
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
            
            # Determine filename (from video_id or URL hash)
            filename = f"{video_id}.mp4" if video_id else f"{safe_video_id}.mp4"
            
            # Upsert to database
            if label:
                # Training data with label
                db_client.upsert_video(
                    video_id=video_url,
                    filename=filename,
                    label=label,
                    title=video.get('title', ''),
                    transcript=video.get('speech2text') or video.get('transcript'),
                    storage_path=storage_path
                )
            else:
                # Inference-only video
                db_client.upsert_video_for_inference(
                    video_id=video_url,
                    filename=filename,
                    storage_path=storage_path,
                    title=video.get('title', ''),
                )
            
            logger.info(f"Ingested TikTok video: {video_url[:50]}...")
            stats['ingested'] += 1
            
            # Cleanup local file
            try:
                os.unlink(local_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error ingesting {video.get('link') or video.get('videoUrl')}: {e}")
            stats['failed'] += 1
    
    # Cleanup
    downloader.cleanup()
    
    logger.info(f"TikTok ingestion stats: {stats}")
    return stats
