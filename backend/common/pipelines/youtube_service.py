"""
YouTube API Service
Tìm kiếm video từ YouTube bằng keyword
"""

import os
from typing import List, Dict, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging

logger = logging.getLogger(__name__)

# Lấy API key từ environment variable
# Tạo API key tại: https://console.cloud.google.com/apis/credentials
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE")


class YouTubeService:
    """Service để tương tác với YouTube Data API v3"""
    
    def __init__(self, api_key: str = None):
        self.api_key = YOUTUBE_API_KEY
        if self.api_key == "YOUR_API_KEY_HERE":
            logger.warning("YouTube API key chưa được set. Vui lòng set YOUTUBE_API_KEY trong environment.")
        
        try:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        except Exception as e:
            logger.error(f"Không thể khởi tạo YouTube API client: {e}")
            self.youtube = None
    
    def search_videos(
        self, 
        keyword: str, 
        video_type: str = "regular",
        max_results: int = 24,
        language: str = "vi"
    ) -> List[Dict]:
        """
        Tìm kiếm video trên YouTube bằng keyword
        
        Args:
            keyword: Từ khóa tìm kiếm
            video_type: 'regular' hoặc 'short'
            max_results: Số lượng video tối đa (mặc định 24)
            language: Ngôn ngữ (vi, en, ...)
        
        Returns:
            List[Dict]: Danh sách video với metadata
        """
        if not self.youtube:
            logger.error("YouTube API client chưa được khởi tạo")
            return []
        
        try:
            # Tham số tìm kiếm
            search_params = {
                'q': keyword,
                'part': 'id,snippet',
                'maxResults': max_results,
                'type': 'video',
                'relevanceLanguage': language,
                'order': 'relevance',  # Sắp xếp theo độ liên quan
                'safeSearch': 'none',  # Không filter (vì ta muốn detect harmful content)
            }
            
            # Nếu là short video, tìm video ngắn (< 60s)
            if video_type == "short":
                search_params['videoDuration'] = 'short'  # < 4 phút
                search_params['q'] = f"{keyword} shorts"
            
            # Gọi API search
            search_response = self.youtube.search().list(**search_params).execute()
            
            # Lấy video IDs
            video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
            
            if not video_ids:
                logger.warning(f"Không tìm thấy video nào với keyword: {keyword}")
                return []
            
            # Lấy thông tin chi tiết của video (statistics, contentDetails)
            videos_response = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=','.join(video_ids)
            ).execute()
            
            # Parse kết quả và filter theo duration
            videos = []
            filtered_count = 0
            for item in videos_response.get('items', []):
                # Check duration first (only videos <= 5 minutes)
                duration_str = item.get('contentDetails', {}).get('duration', '')
                duration_seconds = self._parse_duration_to_seconds(duration_str)
                
                # Skip videos longer than 5 minutes (300 seconds)
                if duration_seconds > 300:
                    filtered_count += 1
                    continue
                
                video_data = self._parse_video_item(item, video_type)
                videos.append(video_data)
            
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} videos longer than 5 minutes")
            logger.info(f"Tìm thấy {len(videos)} video với keyword '{keyword}'")
            return videos
            
        except HttpError as e:
            logger.error(f"YouTube API HTTP Error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching YouTube videos: {e}")
            return []
    
    def _parse_video_item(self, item: Dict, video_type: str) -> Dict:
        """Parse YouTube API response thành format cho frontend"""
        video_id = item['id']
        snippet = item['snippet']
        statistics = item.get('statistics', {})
        
        # Xác định platform
        duration = item.get('contentDetails', {}).get('duration', '')
        is_short = self._is_short_video(duration) or video_type == "short"
        
        platform = 'youtube-shorts' if is_short else 'youtube'
        
        return {
            'id': f"yt_{video_id}",
            'video_id': video_id,
            'title': snippet.get('title', 'Untitled'),
            'description': snippet.get('description', ''),
            'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
            'channel_title': snippet.get('channelName', 'Unknown'),
            'published_at': snippet.get('publishedAt', ''),
            'likes': int(statistics.get('likeCount', 0)),
            'comments': int(statistics.get('commentCount', 0)),
            'views': int(statistics.get('viewCount', 0)),
            'videoUrl': f"https://www.youtube.com/watch?v={video_id}",
            'embedUrl': f"https://www.youtube.com/embed/{video_id}?autoplay=1",
            'platform': platform,
            'type': video_type,
            'status': 'pending',  # Sẽ được inference sau
            'category': 'Unknown',
            'inferenceProgress': 0
        }
    
    def _parse_duration_to_seconds(self, duration: str) -> int:
        """
        Parse YouTube duration format to seconds.
        Duration format: PT1H2M30S (1 hour 2 minutes 30 seconds)
        """
        try:
            import re
            match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
            if match:
                hours = int(match.group(1) or 0)
                minutes = int(match.group(2) or 0)
                seconds = int(match.group(3) or 0)
                return hours * 3600 + minutes * 60 + seconds
        except:
            pass
        return 0
    
    def _is_short_video(self, duration: str) -> bool:
        """
        Kiểm tra xem video có phải là short không (< 60s)
        Duration format: PT1M30S (1 phút 30 giây)
        """
        return self._parse_duration_to_seconds(duration) <= 60
    
    def get_video_details(self, video_id: str) -> Optional[Dict]:
        """
        Lấy thông tin chi tiết của 1 video
        
        Args:
            video_id: YouTube video ID
        
        Returns:
            Dict: Video metadata
        """
        if not self.youtube:
            return None
        
        try:
            response = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            ).execute()
            
            items = response.get('items', [])
            if items:
                return self._parse_video_item(items[0], 'regular')
            return None
            
        except Exception as e:
            logger.error(f"Error getting video details: {e}")
            return None


# Singleton instance
_youtube_service = None

def get_youtube_service() -> YouTubeService:
    """Get or create YouTube service instance"""
    global _youtube_service
    if _youtube_service is None:
        _youtube_service = YouTubeService()
    return _youtube_service
