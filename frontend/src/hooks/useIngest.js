import { useState, useCallback, useEffect } from 'react';
import { triggerIngestion, triggerPreprocessing, listCsvFiles } from '../api';

/**
 * Custom hook for data ingestion functionality
 * Handles CSV upload, preprocessing, and auto-training
 */
export function useIngest() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);
    const [status, setStatus] = useState('idle'); // idle, uploading, processing, done, error
    const [csvFiles, setCsvFiles] = useState([]);
    const [csvFilesLoading, setCsvFilesLoading] = useState(false);

    /**
     * Load available CSV files
     */
    const loadCsvFiles = useCallback(async () => {
        setCsvFilesLoading(true);
        try {
            const response = await listCsvFiles();
            setCsvFiles(response.files || []);
        } catch (err) {
            console.error('Failed to load CSV files:', err);
            setCsvFiles([]);
        } finally {
            setCsvFilesLoading(false);
        }
    }, []);

    // Load CSV files on mount
    useEffect(() => {
        loadCsvFiles();
    }, [loadCsvFiles]);

    /**
     * Trigger ingestion from labels.csv in data/raw/
     * @param {boolean} autoTrain - Automatically trigger training after preprocessing
     * @param {string} csvPath - Optional custom CSV file path
     */
    const ingest = useCallback(async (autoTrain = true, csvPath = null) => {
        setLoading(true);
        setError(null);
        setStatus('uploading');

        try {
            const response = await triggerIngestion(autoTrain, csvPath);
            setResult(response);
            setStatus('done');
            return response;
        } catch (err) {
            setError(err.message);
            setStatus('error');
            throw err;
        } finally {
            setLoading(false);
        }
    }, []);

    /**
     * Trigger preprocessing pipeline manually
     */
    const triggerPreprocess = useCallback(async () => {
        setLoading(true);
        setError(null);
        setStatus('processing');

        try {
            const response = await triggerPreprocessing();
            setResult(response);
            setStatus('done');
            return response;
        } catch (err) {
            setError(err.message);
            setStatus('error');
            throw err;
        } finally {
            setLoading(false);
        }
    }, []);

    /**
     * Reset state
     */
    const reset = useCallback(() => {
        setLoading(false);
        setError(null);
        setResult(null);
        setStatus('idle');
    }, []);

    return {
        loading,
        error,
        result,
        status,
        ingest,
        triggerPreprocess,
        reset,
        csvFiles,
        csvFilesLoading,
        loadCsvFiles,
    };
}

export default useIngest;
