import './VideoCard.css';

/**
 * Get category badge class
 */
function getCategoryClass(category) {
    const map = {
        'Safe': 'badge-safe',
        'Sexual': 'badge-sexual',
        'Aggressive': 'badge-aggressive',
        'Superstition': 'badge-superstition',
    };
    return map[category] || 'badge-safe';
}

/**
 * Format view count
 */
function formatCount(count) {
    if (!count) return '0';
    if (count >= 1000000) return `${(count / 1000000).toFixed(1)}M`;
    if (count >= 1000) return `${(count / 1000).toFixed(1)}K`;
    return count.toString();
}

/**
 * Detect video type from video object (regular/short/tiktok)
 * Uses platform, source, and URL to determine type
 */
function detectVideoType(video) {
    const videoUrl = video.videoUrl || video.url || '';
    const platform = video.platform || '';
    const source = video.source || '';
    
    // Check TikTok first (most specific)
    if (platform === 'tiktok' || source === 'tiktok' || videoUrl.includes('tiktok.com')) {
        return 'tiktok';
    }
    
    // Check YouTube Shorts
    if (platform === 'youtube-shorts' || videoUrl.includes('youtube.com/shorts/') || videoUrl.includes('shorts/')) {
        return 'short';
    }
    
    // Check if video has 'short' in type or if duration suggests it's a short
    if (video.type === 'short' || video.duration <= 60) {
        return 'short';
    }
    
    // Default to regular
    return 'regular';
}

/**
 * Get thumbnail URL for video
 * Handles YouTube (with/without prefix) and TikTok videos
 */
function getThumbnailUrl(video) {
    // If thumbnail is provided, use it (backend should always provide this)
    if (video.thumbnail) {
        return video.thumbnail;
    }
    
    // Extract real video ID (remove platform prefix like yt_)
    let videoId = video.video_id || video.id;
    
    if (!videoId) {
        return ''; // No video ID available
    }
    
    // Remove prefix if exists (e.g., "yt_abc123" -> "abc123")
    if (videoId.startsWith('yt_')) {
        videoId = videoId.substring(3);
    } else if (videoId.includes('_')) {
        // For other prefixes like "tiktok_123"
        videoId = videoId.split('_').slice(1).join('_');
    }
    
    // Check platform
    const platform = video.platform || (video.source === 'tiktok' ? 'tiktok' : null);
    
    // For YouTube/YouTube Shorts, generate thumbnail URL as fallback
    if (platform === 'youtube' || platform === 'youtube-shorts' || !platform) {
        return `https://img.youtube.com/vi/${videoId}/hqdefault.jpg`;
    }
    
    // For TikTok without thumbnail (shouldn't happen, but return empty to avoid broken image)
    return '';
}

function VideoCard({
    video,
    videoType, // User-selected type (for filtering)
    inference,
    isSafe,
    isDone,
    isForceShown,
    onClick,
    onForceShow
}) {
    const isLoading = inference?.status === 'loading';
    const isHarmful = isDone && inference?.result?.is_harmful;
    const category = inference?.result?.prediction || inference?.result?.category || 'Unknown';
    const confidence = inference?.result?.confidence;

    // Detect actual video type from video object
    const detectedType = detectVideoType(video);
    const isShort = detectedType === 'short' || detectedType === 'tiktok';
    const isTikTok = detectedType === 'tiktok';

    // Should blur: loading OR (harmful AND not force shown)
    const shouldBlur = isLoading || (isHarmful && !isForceShown);

    // Can click: done AND (safe OR force shown)
    const canClick = isDone && (isSafe || isForceShown);

    return (
        <div
            className={`video-card ${isShort ? 'short' : 'regular'} ${isTikTok ? 'tiktok' : ''} ${canClick ? 'clickable' : ''}`}
            onClick={canClick ? onClick : undefined}
        >
            {/* Thumbnail */}
            <div className="video-thumbnail">
                <img
                    src={getThumbnailUrl(video)}
                    alt={video.title}
                    className={shouldBlur ? 'blurred' : ''}
                    onError={(e) => {
                        // Fallback to placeholder if image fails to load
                        e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIwIiBoZWlnaHQ9IjE4MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzMzIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtc2l6ZT0iMTgiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5ObyB0aHVtYm5haWw8L3RleHQ+PC9zdmc+';
                    }}
                />

                {/* Loading Overlay */}
                {isLoading && (
                    <div className="thumbnail-overlay loading-overlay">
                        <div className="loading-spinner-small"></div>
                        <span>Đang phân tích...</span>
                    </div>
                )}

                {/* Harmful Overlay */}
                {isHarmful && !isForceShown && (
                    <div className={`thumbnail-overlay harmful-overlay ${category.toLowerCase()}`}>
                        <span className="harmful-text">Nội dung {category}</span>
                        <button
                            className="btn btn-secondary btn-sm"
                            onClick={(e) => {
                                e.stopPropagation();
                                onForceShow();
                            }}
                        >
                            Vẫn xem
                        </button>
                    </div>
                )}

                {/* Safe Badge */}
                {isSafe && (
                    <div className="safe-badge">
                        <span>Safe</span>
                    </div>
                )}

                {/* Duration */}
                {video.duration && (
                    <span className="video-duration">{video.duration}</span>
                )}
            </div>

            {/* Info */}
            <div className="video-info">
                <h3 className="video-title" title={video.title}>
                    {video.title}
                </h3>

                <div className="video-meta">
                    {(video.channel || video.channel_title || video.author) && (
                        <span className="video-channel">
                            {video.channel || video.channel_title || video.author}
                        </span>
                    )}

                    <div className="video-stats">
                        {(video.views !== undefined || video.play_count !== undefined) && (
                            <span className="stat">
                                👁 {formatCount(video.views || video.play_count)}
                            </span>
                        )}
                        {video.likes !== undefined && (
                            <span className="stat">
                                👍 {formatCount(video.likes)}
                            </span>
                        )}
                        {video.comments !== undefined && (
                            <span className="stat">
                                💬 {formatCount(video.comments)}
                            </span>
                        )}
                    </div>
                </div>

                {/* Inference Result */}
                {isDone && (
                    <div className="video-inference">
                        <span className={`badge ${getCategoryClass(category)}`}>
                            {category}
                        </span>
                        {confidence && (
                            <span className="confidence">
                                {(confidence * 100).toFixed(1)}%
                            </span>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export default VideoCard;
