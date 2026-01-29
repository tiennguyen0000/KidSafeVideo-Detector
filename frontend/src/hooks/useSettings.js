import { useState, useEffect, useCallback } from 'react';
import { checkHealth, getStatistics } from '../api';

/**
 * Custom hook for settings and system status
 * Handles pipeline preference, health check, and statistics
 */
export function useSettings() {
    // User preferences (persisted to localStorage)
    const [defaultPipeline, setDefaultPipelineState] = useState(() => {
        if (typeof window !== 'undefined') {
            return localStorage.getItem('defaultPipeline') || 'local';
        }
        return 'local';
    });

    // System status
    const [health, setHealth] = useState(null);
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    /**
     * Persist default pipeline to localStorage
     */
    useEffect(() => {
        localStorage.setItem('defaultPipeline', defaultPipeline);
    }, [defaultPipeline]);

    /**
     * Set default pipeline for inference
     */
    const setDefaultPipeline = useCallback((pipeline) => {
        if (pipeline === 'local' || pipeline === 'colab') {
            setDefaultPipelineState(pipeline);
        }
    }, []);

    /**
     * Load system health and statistics
     */
    const loadSystemStatus = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            const [healthRes, statsRes] = await Promise.all([
                checkHealth().catch(err => ({ status: 'error', error: err.message })),
                getStatistics().catch(() => null),
            ]);

            setHealth(healthRes);
            setStats(statsRes);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, []);

    // Load on mount
    useEffect(() => {
        loadSystemStatus();
    }, [loadSystemStatus]);

    /**
     * Check if backend is connected
     */
    const isBackendConnected = health?.status === 'healthy';
    const isDatabaseConnected = health?.database === 'connected';

    /**
     * Get current model mode from backend
     */
    const currentMode = health?.mode || 'ultra_light';

    return {
        // Preferences
        defaultPipeline,
        setDefaultPipeline,

        // System status
        health,
        stats,
        loading,
        error,
        loadSystemStatus,

        // Computed
        isBackendConnected,
        isDatabaseConnected,
        currentMode,
    };
}

export default useSettings;
