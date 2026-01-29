import { useState, useCallback } from 'react';
import useInference from '../hooks/useInference';
import VideoCard from '../components/VideoCard';
import VideoPlayer from '../components/VideoPlayer';
import SearchBar from '../components/SearchBar';
import './InferencePage.css';

/**
 * Get consistent video ID from video object
 * Handles both YouTube (with id field) and TikTok (with videoId/video_id field)
 */
function getVideoId(video) {
    return video.id || video.videoId || video.video_id || video.videoUrl || '';
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
 * Filter videos based on selected video type
 */
function filterVideosByType(videos, selectedType) {
    if (selectedType === 'short') {
        // Show both shorts and tiktok when "short" tab is selected
        return videos.filter(video => {
            const detectedType = detectVideoType(video);
            return detectedType === 'short' || detectedType === 'tiktok';
        });
    } else {
        // Show only regular videos
        return videos.filter(video => {
            const detectedType = detectVideoType(video);
            return detectedType === 'regular';
        });
    }
}

function InferencePage() {
    const [videoType, setVideoType] = useState('regular'); // 'regular' or 'short'
    const [selectedVideo, setSelectedVideo] = useState(null);
    const [forceShowVideo, setForceShowVideo] = useState({}); // { videoId: true }

    const {
        videos,
        loading,
        error,
        search,
        getInferenceResult,
        isVideoSafe,
        isInferenceDone,
    } = useInference();

    // Handle video click
    const handleVideoClick = useCallback((video) => {
        const videoId = getVideoId(video);
        const inference = getInferenceResult(videoId);

        // If still loading inference, do nothing
        if (!isInferenceDone(videoId)) {
            return;
        }

        // If safe or user forced show, open player
        if (isVideoSafe(videoId) || forceShowVideo[videoId]) {
            setSelectedVideo(video);
        }
    }, [getInferenceResult, isInferenceDone, isVideoSafe, forceShowVideo]);

    // Handle force show harmful video
    const handleForceShow = useCallback((videoId) => {
        setForceShowVideo(prev => ({ ...prev, [videoId]: true }));
    }, []);

    // Close video player
    const handleClosePlayer = useCallback(() => {
        setSelectedVideo(null);
    }, []);

    return (
        <div className={`inference-page ${selectedVideo ? 'player-open' : ''}`}>
            {/* Header */}
            <div className="page-header">
                <h1 className="page-title">Video Inference</h1>
                <p className="page-subtitle">Tìm kiếm và phân loại video độc hại bằng AI</p>
            </div>

            {/* Controls */}
            <div className="inference-controls">
                {/* Video Type Toggle */}
                <div className="toggle-group video-type-toggle">
                    <button
                        className={`toggle-option ${videoType === 'regular' ? 'active' : ''}`}
                        onClick={() => setVideoType('regular')}
                    >
                        Video thường
                    </button>
                    <button
                        className={`toggle-option ${videoType === 'short' ? 'active' : ''}`}
                        onClick={() => setVideoType('short')}
                    >
                        Shorts / TikTok
                    </button>
                </div>

                {/* Search Bar */}
                <SearchBar
                    onSearch={(keyword) => search(keyword, videoType)}
                    loading={loading}
                    placeholder={videoType === 'short' ? 'Tìm kiếm shorts...' : 'Tìm kiếm video...'}
                />
            </div>

            {/* Error Message */}
            {error && (
                <div className="error-message">
                    {error}
                </div>
            )}

            {/* Video Grid */}
            <div className={`video-grid ${videoType === 'short' ? 'shorts-grid' : ''}`}>
                {filterVideosByType(videos, videoType).map((video) => {
                    const videoId = getVideoId(video);
                    return (
                        <VideoCard
                            key={videoId}
                            video={video}
                            videoType={videoType}
                            inference={getInferenceResult(videoId)}
                            isSafe={isVideoSafe(videoId)}
                            isDone={isInferenceDone(videoId)}
                            isForceShown={forceShowVideo[videoId]}
                            onClick={() => handleVideoClick(video)}
                            onForceShow={() => handleForceShow(videoId)}
                        />
                    );
                })}
            </div>

            {/* Empty State */}
            {!loading && videos.length === 0 && !error && (
                <div className="empty-state">
                    <h3>Bắt đầu tìm kiếm video</h3>
                    <p>Nhập từ khóa để tìm kiếm và phân loại video</p>
                </div>
            )}

            {/* Loading State */}
            {loading && (
                <div className="loading-state">
                    <div className="loading-spinner"></div>
                    <p>Đang tìm kiếm video...</p>
                </div>
            )}

            {/* Video Player Modal */}
            {selectedVideo && (
                <VideoPlayer
                    video={selectedVideo}
                    videoType={videoType}
                    onClose={handleClosePlayer}
                />
            )}
        </div>
    );
}

export default InferencePage;
