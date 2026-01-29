import { useEffect, useCallback } from 'react';
import './VideoPlayer.css';

/**
 * Get embed URL for different platforms
 */
function getEmbedUrl(video, videoType) {
    // Extract real video ID (remove platform prefix like yt_)
    let videoId = video.video_id || video.id;
    
    // Remove prefix if exists (e.g., "yt_abc123" -> "abc123")
    if (videoId.startsWith('yt_')) {
        videoId = videoId.substring(3);
    } else if (videoId.includes('_')) {
        // For other prefixes like "tiktok_123"
        videoId = videoId.split('_').slice(1).join('_');
    }

    // YouTube
    if (video.platform === 'youtube' || video.platform === 'youtube-shorts' || !video.platform) {
        if (videoType === 'short' || video.platform === 'youtube-shorts') {
            return `https://www.youtube.com/embed/${videoId}?autoplay=1&loop=1`;
        }
        return `https://www.youtube.com/embed/${videoId}?autoplay=1`;
    }

    // TikTok
    if (video.platform === 'tiktok') {
        return `https://www.tiktok.com/player/v1/${videoId}?autoplay=1`;
    }

    // Default to YouTube
    return `https://www.youtube.com/embed/${videoId}?autoplay=1`;
}

function VideoPlayer({ video, videoType, onClose }) {
    // Close on Escape key
    const handleKeyDown = useCallback((e) => {
        if (e.key === 'Escape') {
            onClose();
        }
    }, [onClose]);

    useEffect(() => {
        document.addEventListener('keydown', handleKeyDown);
        document.body.style.overflow = 'hidden';

        return () => {
            document.removeEventListener('keydown', handleKeyDown);
            document.body.style.overflow = '';
        };
    }, [handleKeyDown]);

    const embedUrl = getEmbedUrl(video, videoType);
    const isShort = videoType === 'short';

    return (
        <div className="video-player-wrapper">
            {/* Backdrop */}
            <div className="player-backdrop" onClick={onClose} />

            {/* Player Container */}
            <div className={`player-container ${isShort ? 'short' : ''}`}>
                {/* Header */}
                <div className="player-header">
                    <h2 className="player-title">{video.title}</h2>
                    <button className="player-close" onClick={onClose} aria-label="Close">
                        ✕
                    </button>
                </div>

                {/* Video Frame */}
                <div className="player-frame">
                    <iframe
                        src={embedUrl}
                        title={video.title}
                        frameBorder="0"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowFullScreen
                    />
                </div>

                {/* Video Info */}
                <div className="player-info">
                    {video.channel && (
                        <p className="player-channel">{video.channel}</p>
                    )}
                    {video.description && (
                        <p className="player-description">{video.description}</p>
                    )}
                </div>
            </div>
        </div>
    );
}

export default VideoPlayer;
