import { useState, useEffect, useCallback } from 'react';
import { triggerTraining, listModels, getFusionConfig } from '../api';

/**
 * Pipeline configurations
 */
export const PIPELINES = {
    local: {
        id: 'local',
        name: 'Local (Ultra-Light)',
        mode: 'ultra_light',
        fusionType: 'gated',
        models: ['EfficientNet-B0', 'SentenceBERT Tiny'],
        fusion: 'GMU (Gated Multimodal Unit)',
        description: 'Chạy trên CPU, nhanh hơn, phù hợp máy 16GB RAM',
    },
    colab: {
        id: 'colab',
        name: 'Colab (Balanced)',
        mode: 'balanced',
        fusionType: 'attention',
        models: ['ResNet50', 'PhoBERT'],
        fusion: 'Attention Fusion',
        description: 'Chính xác hơn, cần GPU',
    },
};

/**
 * Custom hook for training functionality
 * Handles pipeline selection, training trigger, model history
 */
export function useTrain() {
    const [pipeline, setPipeline] = useState('local');
    const [epochs, setEpochs] = useState(20);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);
    const [status, setStatus] = useState('idle'); // idle, training, done, error

    // Model history
    const [models, setModels] = useState([]);
    const [modelsLoading, setModelsLoading] = useState(false);

    // Fusion config
    const [fusionConfig, setFusionConfig] = useState({});

    /**
     * Load models and fusion config
     */
    const loadData = useCallback(async () => {
        setModelsLoading(true);
        try {
            const [modelsRes, fusionRes] = await Promise.all([
                listModels().catch(() => ({ models: [] })),
                getFusionConfig().catch(() => ({})),
            ]);
            setModels(modelsRes.models || []);
            setFusionConfig(fusionRes || {});
        } finally {
            setModelsLoading(false);
        }
    }, []);

    // Load on mount
    useEffect(() => {
        loadData();
    }, [loadData]);

    /**
     * Start training with current pipeline settings
     */
    const startTraining = useCallback(async () => {
        const pipelineConfig = PIPELINES[pipeline];
        if (!pipelineConfig) {
            setError('Invalid pipeline selected');
            return;
        }

        setLoading(true);
        setError(null);
        setStatus('training');

        try {
            const response = await triggerTraining(
                pipelineConfig.mode,
                epochs,
                pipelineConfig.fusionType
            );
            setResult(response);
            setStatus('done');

            // Reload models after training
            await loadData();

            return response;
        } catch (err) {
            setError(err.message);
            setStatus('error');
            throw err;
        } finally {
            setLoading(false);
        }
    }, [pipeline, epochs, loadData]);

    /**
     * Get models for a specific pipeline
     */
    const getModelsByPipeline = useCallback((pipelineId) => {
        const mode = PIPELINES[pipelineId]?.mode;
        if (!mode) return [];
        return models.filter(m => m.mode === mode);
    }, [models]);

    /**
     * Get active model for a pipeline
     */
    const getActiveModel = useCallback((pipelineId, modelType) => {
        const pipelineModels = getModelsByPipeline(pipelineId);
        return pipelineModels.find(m => m.model_type === modelType && m.is_active);
    }, [getModelsByPipeline]);

    /**
     * Get fusion config for a pipeline
     */
    const getFusionConfigByPipeline = useCallback((pipelineId) => {
        const mode = PIPELINES[pipelineId]?.mode;
        return fusionConfig[mode] || {};
    }, [fusionConfig]);

    /**
     * Reset training state
     */
    const reset = useCallback(() => {
        setLoading(false);
        setError(null);
        setResult(null);
        setStatus('idle');
    }, []);

    return {
        // Pipeline settings
        pipeline,
        setPipeline,
        epochs,
        setEpochs,
        pipelineConfig: PIPELINES[pipeline],

        // Training state
        loading,
        error,
        result,
        status,
        startTraining,
        reset,

        // Models
        models,
        modelsLoading,
        loadData,
        getModelsByPipeline,
        getActiveModel,

        // Fusion config
        fusionConfig,
        getFusionConfigByPipeline,
    };
}

export default useTrain;
