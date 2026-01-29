/**
 * API Service Layer
 * Centralized API calls to backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

/**
 * Helper function for API requests
 */
async function request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;

    // Don't set Content-Type for FormData - let browser handle it
    const isFormData = options.body instanceof FormData;

    const config = {
        ...options,
        headers: isFormData
            ? { ...options.headers }
            : {
                'Content-Type': 'application/json',
                ...options.headers,
            },
    };

    try {
        const response = await fetch(url, config);

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error(`API Error [${endpoint}]:`, error);
        throw error;
    }
}

/**
 * Health check
 */
export async function checkHealth() {
    return request('/health');
}

/**
 * Search videos from YouTube
 * @param {string} keyword - Search term
 * @param {string} videoType - 'regular' or 'short'
 * @param {number} maxResults - Max videos to return
 */
export async function searchVideos(keyword, videoType = 'regular', maxResults = 24) {
    return request('/api/search-videos', {
        method: 'POST',
        body: JSON.stringify({ keyword, video_type: videoType, max_results: maxResults }),
    });
}

/**
 * Get search results by dag_run_id
 * @param {string} dagRunId - DAG run ID from search response
 */
export async function getSearchResults(dagRunId) {
    return request(`/api/search-results/${dagRunId}`);
}

/**
 * Run inference on a video
 * @param {string} videoId - YouTube video ID
 * @param {string} pipeline - 'local' or 'colab'
 */
export async function runInference(videoId, pipeline = 'local') {
    return request('/api/inference', {
        method: 'POST',
        body: JSON.stringify({ video_id: videoId, pipeline }),
    });
}

/**
 * Run batch inference on multiple videos from search results
 * @param {Array} videos - List of video objects from search
 * @param {string} pipeline - 'local' or 'colab'
 */
export async function runBatchInference(videos, pipeline = 'local') {
    return request('/api/inference/batch', {
        method: 'POST',
        body: JSON.stringify({ videos, pipeline }),
    });
}

/**
 * Get prediction for a video
 * @param {string} videoId - Video ID
 */
export async function getPrediction(videoId) {
    return request(`/api/inference/${videoId}`);
}

/**
 * List videos with optional filters
 * @param {string} status - Filter by status
 * @param {number} limit - Max results
 */
export async function listVideos(status = null, limit = 100) {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    params.append('limit', limit.toString());

    return request(`/api/videos?${params.toString()}`);
}

/**
 * Get system statistics
 */
export async function getStatistics() {
    return request('/api/statistics');
}

/**
 * Get list of available CSV files for ingestion
 */
export async function listCsvFiles() {
    return request('/api/csv-files');
}

/**
 * Trigger data ingestion from CSV
 * @param {boolean} autoTrain - Auto trigger training
 * @param {string} csvPath - Optional custom CSV file path
 */
export async function triggerIngestion(autoTrain = true, csvPath = null) {
    const params = new URLSearchParams();
    params.append('auto_train', autoTrain.toString());
    if (csvPath) {
        params.append('csv_path', csvPath);
    }
    
    return request(`/api/ingest?${params.toString()}`, {
        method: 'POST',
    });
}

/**
 * Trigger model training
 * @param {string} mode - 'ultra_light' or 'balanced'
 * @param {number} epochs - Number of epochs
 * @param {string} fusionType - 'gated' or 'attention'
 */
export async function triggerTraining(mode = 'ultra_light', epochs = 20, fusionType = 'gated') {
    const formData = new FormData();
    formData.append('mode', mode);
    formData.append('epochs', epochs.toString());
    formData.append('fusion_type', fusionType);

    return request('/api/train', {
        method: 'POST',
        body: formData,
    });
}

/**
 * List registered models
 */
export async function listModels() {
    return request('/api/models');
}

/**
 * Get fusion configuration
 */
export async function getFusionConfig() {
    return request('/api/fusion-config');
}

/**
 * Trigger preprocessing pipeline
 */
export async function triggerPreprocessing() {
    return request('/api/trigger-preprocessing', {
        method: 'POST',
    });
}

export default {
    checkHealth,
    searchVideos,
    getSearchResults,
    runInference,
    getPrediction,
    listVideos,
    getStatistics,
    listCsvFiles,
    triggerIngestion,
    triggerTraining,
    listModels,
    getFusionConfig,
    triggerPreprocessing,
};
