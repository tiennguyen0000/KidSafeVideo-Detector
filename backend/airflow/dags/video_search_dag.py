"""
Video Search DAG
Search videos from YouTube and cache results
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "video_classifier",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def search_videos(**context):
    """Search videos from YouTube only."""
    from common.pipelines.youtube_service import YouTubeService
    from common.io import queue
    import logging
    
    logger = logging.getLogger(__name__)
    
    conf = context["dag_run"].conf or {}
    keyword = conf.get("keyword", "")
    max_results = conf.get("max_results", 24)
    dag_run_id = context["dag_run"].run_id
    
    if not keyword:
        logger.warning("No keyword provided for video search")
        return {"status": "error", "error": "No keyword provided"}
    
    all_videos = []
    
    # =========================================================================
    # Search YouTube only (25 videos: 13 regular + 12 shorts)
    # =========================================================================
    logger.info(f"Searching YouTube for: {keyword}")
    try:
        youtube = YouTubeService()

        # Search 25 videos: 13 regular + 12 shorts
        # 1) Search video thường (13 videos)
        youtube_videos = youtube.search_videos(
            keyword=keyword,
            video_type="video",
            max_results=13
        )

        # 2) Search Shorts (12 videos)
        youtube_shorts = youtube.search_videos(
            keyword=keyword,
            video_type="short",
            max_results=12
        )

        # Add platform info + merge + dedup
        merged = {}
        for v in (youtube_videos + youtube_shorts):
            v["source"] = "youtube"
            vid = v.get("video_id") or v.get("id") or v.get("videoId")
            if vid:
                merged[vid] = v
            else:
                merged[id(v)] = v  # fallback

        all_videos.extend(list(merged.values()))
        logger.info(f"  YouTube: found {len(merged)} videos (videos={len(youtube_videos)}, shorts={len(youtube_shorts)})")

    except Exception as e:
        logger.error(f"  YouTube search error: {e}")

    
    # =========================================================================
    # Limit to 25 videos total (random sampling if exceeds)
    # =========================================================================
    logger.info(f"Total: {len(all_videos)} videos from YouTube (before limit)")
    
    # Limit to 25 videos total - random sample if exceeds
    max_videos = 25
    if len(all_videos) > max_videos:
        import random
        random.shuffle(all_videos)  # Shuffle for random selection
        all_videos = all_videos[:max_videos]
        logger.info(f"Limited to {max_videos} videos (randomly selected)")
    
    # Cache results in database (cross-container via PostgreSQL)
    queue.cache_search_results(dag_run_id, all_videos)
    
    youtube_count = len([v for v in all_videos if v.get("source") == "youtube"])
    
    logger.info(f"Final result: {len(all_videos)} videos (YouTube: {youtube_count})")
    
    return {
        "status": "completed", 
        "count": len(all_videos), 
        "source": "youtube",
        "youtube_count": youtube_count,
    }


with DAG(
    dag_id="video_search",
    default_args=default_args,
    description="Search videos from YouTube",
    schedule_interval=None,  # Triggered manually via API
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["video", "search", "youtube"],
    max_active_runs=5,
) as dag:
    
    search_task = PythonOperator(
        task_id="search_videos",
        python_callable=search_videos,
        provide_context=True,
    )
