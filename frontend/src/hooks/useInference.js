import { useState, useCallback } from 'react';
import { searchVideos, getSearchResults, runInference, runBatchInference, getPrediction } from '../api';

/**
 * Get consistent video ID from video object
 * Handles both YouTube (with id field) and TikTok (with videoId/video_id field)
 */
function getVideoId(video) {
    return video.id || video.videoId || video.video_id || video.videoUrl || '';
}

/**
 * Custom hook for inference logic
 * Manages video search, inference status, and results
 */
export function useInference() {
    const [videos, setVideos] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [inferenceStatus, setInferenceStatus] = useState({}); // { videoId: { status, result } }

    /**
     * Search for videos
     */
    const search = useCallback(async (keyword, videoType = 'regular') => {
        if (!keyword.trim()) return;

        setLoading(true);
        setError(null);
        setInferenceStatus({});
        setVideos([]);

        try {
            // Start search (returns dag_run_id)
            const response = await searchVideos(keyword, videoType);
            
            if (response.status === 'error') {
                throw new Error(response.message || 'Search failed');
            }

            const dagRunId = response.dag_run_id;
            let foundVideos = false;
            
            // Poll for results every 2 seconds
            const pollInterval = setInterval(async () => {
                try {
                    const result = await getSearchResults(dagRunId);
                    
                    if (result.status === 'completed' && result.videos) {
                        clearInterval(pollInterval);
                        foundVideos = true;
                        setVideos(result.videos);
                        setLoading(false);

                        // Auto-start batch inference for all videos
                        runBatchInferenceForVideos(result.videos, 'local');
                    } else if (result.status === 'error') {
                        clearInterval(pollInterval);
                        foundVideos = true;
                        setError(result.error || 'Search failed');
                        setLoading(false);
                    }
                    // If status === 'pending', keep polling
                } catch (err) {
                    clearInterval(pollInterval);
                    foundVideos = true;
                    setError(err.message);
                    setLoading(false);
                }
            }, 2000);

            // Timeout after 30 seconds
            setTimeout(() => {
                clearInterval(pollInterval);
                if (!foundVideos) {
                    setError('Search timeout - please try again');
                    setLoading(false);
                }
            }, 30000);

        } catch (err) {
            setError(err.message);
            setVideos([]);
            setLoading(false);
        }
    }, []);
    
    /**
     * Run batch inference for all videos
     * This uses the new batch endpoint that:
     * - Returns cached predictions immediately for existing videos
     * - Triggers download + preprocessing + inference for new videos
     */
    const runBatchInferenceForVideos = useCallback(async (videoList, pipeline = 'local') => {
        // Set all videos to loading initially
        const initialStatus = {};
        videoList.forEach(video => {
            const videoId = getVideoId(video);
            initialStatus[videoId] = { status: 'loading', result: null };
        });
        setInferenceStatus(initialStatus);
        
        try {
            const response = await runBatchInference(videoList, pipeline);
            
            // Update status for videos with existing predictions (from feature store or cache)
            if (response.existing_predictions) {
                response.existing_predictions.forEach(pred => {
                    setInferenceStatus(prev => ({
                        ...prev,
                        [pred.video_id]: {
                            status: 'done',
                            result: {
                                prediction: pred.prediction,
                                confidence: pred.confidence,
                                source: pred.source,
                                is_harmful: pred.prediction !== 'Safe'
                            }
                        }
                    }));
                });
            }
            
            // For queued videos, poll for results
            if (response.queued_videos && response.queued_videos.length > 0) {
                // These videos are being processed, poll for results
                pollQueuedVideos(response.queued_videos, pipeline);
            }
            
        } catch (err) {
            console.error('Batch inference error:', err);
            // Set all videos to error state
            videoList.forEach(video => {
                const videoId = getVideoId(video);
                setInferenceStatus(prev => ({
                    ...prev,
                    [videoId]: { status: 'error', result: null, error: err.message }
                }));
            });
        }
    }, []);
    
    /**
     * Poll for inference results for queued videos
     */
    const pollQueuedVideos = useCallback((queuedVideoIds, pipeline) => {
        const pollInterval = setInterval(async () => {
            let allDone = true;
            
            for (const videoId of queuedVideoIds) {
                const currentStatus = inferenceStatus[videoId];
                if (currentStatus?.status === 'done' || currentStatus?.status === 'error') {
                    continue;
                }
                
                try {
                    // Poll individual video result using GET endpoint (better for polling)
                    const result = await getPrediction(videoId);
                    
                    if (result.status === 'completed') {
                        setInferenceStatus(prev => ({
                            ...prev,
                            [videoId]: {
                                status: 'done',
                                result: {
                                    prediction: result.prediction,
                                    confidence: result.confidence,
                                    source: result.source || 'inference',
                                    is_harmful: result.prediction !== 'Safe'
                                }
                            }
                        }));
                    } else if (result.status === 'pending' || 
                               result.status === 'queued' || 
                               result.status === 'pending_preprocessing' ||
                               result.status === 'pending_inference' ||
                               result.status === 'processing') {
                        // Still processing, continue polling
                        allDone = false;
                    } else if (result.status === 'error') {
                        // Error status
                        setInferenceStatus(prev => ({
                            ...prev,
                            [videoId]: { status: 'error', result: null, error: result.message || 'Inference failed' }
                        }));
                    } else {
                        // Unknown status, continue polling
                        allDone = false;
                    }
                } catch (err) {
                    console.error(`Error polling inference for ${videoId}:`, err);
                    allDone = false;
                }
            }
            
            if (allDone) {
                clearInterval(pollInterval);
            }
        }, 5000); // Poll every 5 seconds
        
        // Timeout after 5 minutes
        setTimeout(() => {
            clearInterval(pollInterval);
        }, 300000);
    }, [inferenceStatus]);

    /**
     * Run inference for a single video
     */
    const runInferenceForVideo = useCallback(async (videoId, pipeline = 'local') => {
        setInferenceStatus(prev => ({
            ...prev,
            [videoId]: { status: 'loading', result: null }
        }));

        try {
            const result = await runInference(videoId, pipeline);
            setInferenceStatus(prev => ({
                ...prev,
                [videoId]: {
                    status: 'done',
                    result: {
                        prediction: result.prediction,
                        confidence: result.confidence,
                        source: result.source || 'inference',
                        is_harmful: result.prediction !== 'Safe'
                    }
                }
            }));
        } catch (err) {
            setInferenceStatus(prev => ({
                ...prev,
                [videoId]: { status: 'error', result: null, error: err.message }
            }));
        }
    }, []);

    /**
     * Get inference result for a video
     */
    const getInferenceResult = useCallback((videoId) => {
        return inferenceStatus[videoId] || { status: 'pending', result: null };
    }, [inferenceStatus]);

    /**
     * Check if a video is safe
     */
    const isVideoSafe = useCallback((videoId) => {
        const inference = inferenceStatus[videoId];
        return inference?.status === 'done' && !inference?.result?.is_harmful;
    }, [inferenceStatus]);

    /**
     * Check if inference is done for a video
     */
    const isInferenceDone = useCallback((videoId) => {
        const inference = inferenceStatus[videoId];
        return inference?.status === 'done' || inference?.status === 'error';
    }, [inferenceStatus]);

    return {
        videos,
        loading,
        error,
        search,
        runInferenceForVideo,
        getInferenceResult,
        isVideoSafe,
        isInferenceDone,
    };
}

export default useInference;
