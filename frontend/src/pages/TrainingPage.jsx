import { useState, useEffect, useCallback } from 'react';
import { useIngest } from '../hooks/useIngest';
import { useTrain, PIPELINES } from '../hooks/useTrain';
import MetricsModal from '../components/MetricsModal';
import './TrainingPage.css';

function TrainingPage() {
    // Use custom hooks
    const {
        loading: ingestLoading,
        error: ingestError,
        result: ingestResult,
        status: ingestStatus,
        ingest,
        reset: resetIngest,
        csvFiles,
        csvFilesLoading,
        loadCsvFiles,
    } = useIngest();

    const {
        pipeline,
        setPipeline,
        epochs,
        setEpochs,
        pipelineConfig,
        loading: trainLoading,
        error: trainError,
        result: trainResult,
        status: trainStatus,
        startTraining,
        reset: resetTraining,
        models,
        modelsLoading,
        loadData,
        getModelsByPipeline,
        fusionConfig,
        getFusionConfigByPipeline,
    } = useTrain();

    const [autoTrain, setAutoTrain] = useState(true);
    const [selectedCsv, setSelectedCsv] = useState('labels.csv');
    const [message, setMessage] = useState(null);
    const [selectedModelId, setSelectedModelId] = useState(null);

    // Debug selected model ID
    useEffect(() => {
        console.log('selectedModelId changed:', selectedModelId);
    }, [selectedModelId]);

    // Handle ingest result
    useEffect(() => {
        if (ingestResult) {
            setMessage({
                type: 'success',
                text: ingestResult.message || 'Ingestion triggered successfully!',
            });
        }
    }, [ingestResult]);

    // Handle ingest error
    useEffect(() => {
        if (ingestError) {
            setMessage({
                type: 'error',
                text: ingestError,
            });
        }
    }, [ingestError]);

    // Handle train result
    useEffect(() => {
        if (trainResult) {
            setMessage({
                type: 'success',
                text: trainResult.message || 'Training triggered successfully!',
            });
        }
    }, [trainResult]);

    // Handle train error
    useEffect(() => {
        if (trainError) {
            setMessage({
                type: 'error',
                text: trainError,
            });
        }
    }, [trainError]);

    // Handle ingest click
    const handleIngest = useCallback(async () => {
        setMessage(null);
        try {
            // Pass selected CSV if not default
            const csvPath = selectedCsv !== 'labels.csv' ? selectedCsv : null;
            await ingest(autoTrain, csvPath);
        } catch (err) {
            // Error handled by hook
        }
    }, [ingest, autoTrain, selectedCsv]);

    // Handle train click
    const handleTrain = useCallback(async () => {
        setMessage(null);
        try {
            await startTraining();
        } catch (err) {
            // Error handled by hook
        }
    }, [startTraining]);

    // Get models for current pipeline
    const currentModels = getModelsByPipeline(pipeline);
    const currentFusionConfig = getFusionConfigByPipeline(pipeline);

    return (
        <div className="training-page">
            {/* Header */}
            <div className="page-header">
                <h1 className="page-title">Ingest & Training</h1>
                <p className="page-subtitle">Upload data và train models phân loại video</p>
            </div>

            {/* Message */}
            {message && (
                <div className={`message message-${message.type}`}>
                    {message.text}
                </div>
            )}

            {/* Main Grid */}
            <div className="training-grid">
                {/* Ingest Section */}
                <div className="card ingest-card">
                    <div className="card-header">
                        <h2 className="card-title">Data Ingestion</h2>
                    </div>

                    <div className="card-body">
                        <p className="card-description">
                            Chọn file CSV từ thư mục <code>data/raw/</code> để ingest vào hệ thống.
                            CSV cần có ít nhất 2 cột: <code>link</code> và <code>category_real</code>.
                        </p>

                        {/* CSV File Selection */}
                        <div className="csv-select-group">
                            <label htmlFor="csv-select">Chọn file CSV:</label>
                            <select
                                id="csv-select"
                                className="select"
                                value={selectedCsv}
                                onChange={(e) => setSelectedCsv(e.target.value)}
                                disabled={csvFilesLoading || ingestLoading}
                            >
                                {csvFilesLoading ? (
                                    <option>Đang tải...</option>
                                ) : csvFiles.length === 0 ? (
                                    <option value="labels.csv">labels.csv (mặc định)</option>
                                ) : (
                                    csvFiles.map((file) => (
                                        <option key={file.name} value={file.name}>
                                            {file.name} ({(file.size / 1024).toFixed(1)} KB)
                                        </option>
                                    ))
                                )}
                            </select>
                            <button
                                className="btn btn-sm btn-secondary"
                                onClick={loadCsvFiles}
                                disabled={csvFilesLoading}
                                title="Refresh danh sách file CSV"
                            >
                                ↻
                            </button>
                        </div>

                        <div className="ingest-options">
                            <label className="checkbox-label">
                                <input
                                    type="checkbox"
                                    checked={autoTrain}
                                    onChange={(e) => setAutoTrain(e.target.checked)}
                                />
                                <span className="checkbox-text">Auto-train sau khi preprocessing xong</span>
                            </label>
                        </div>

                        <button
                            className="btn btn-primary btn-lg"
                            onClick={handleIngest}
                            disabled={ingestLoading}
                        >
                            {ingestLoading ? (
                                ingestStatus === 'uploading' ? 'Đang upload...' : 'Đang xử lý...'
                            ) : (
                                `Trigger Ingestion (${selectedCsv})`
                            )}
                        </button>
                    </div>
                </div>

                {/* Pipeline Section */}
                <div className="card pipeline-card">
                    <div className="card-header">
                        <h2 className="card-title">Training Pipeline</h2>
                    </div>

                    <div className="card-body">
                        {/* Pipeline Toggle */}
                        <div className="toggle-group pipeline-toggle">
                            <button
                                className={`toggle-option ${pipeline === 'local' ? 'active' : ''}`}
                                onClick={() => setPipeline('local')}
                            >
                                Local
                            </button>
                            <button
                                className={`toggle-option ${pipeline === 'colab' ? 'active' : ''}`}
                                onClick={() => setPipeline('colab')}
                            >
                                Colab
                            </button>
                        </div>

                        {/* Pipeline Info */}
                        <div className={`pipeline-info pipeline-${pipeline === 'local' ? 'success' : 'accent'}`}>
                            <h3>{pipelineConfig?.name}</h3>
                            <div className="pipeline-models">
                                <strong>Models:</strong>
                                <ul>
                                    {pipelineConfig?.models.map((model, idx) => (
                                        <li key={idx}>{model}</li>
                                    ))}
                                </ul>
                            </div>
                            <div className="pipeline-fusion">
                                <strong>Fusion:</strong> {pipelineConfig?.fusion}
                            </div>
                            <p style={{ marginTop: '8px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                                {pipelineConfig?.description}
                            </p>
                        </div>

                        {/* Epochs Input */}
                        <div className="epochs-input">
                            <label htmlFor="epochs">Epochs:</label>
                            <input
                                type="number"
                                id="epochs"
                                className="input"
                                value={epochs}
                                onChange={(e) => setEpochs(Math.max(1, parseInt(e.target.value) || 1))}
                                min="1"
                                max="100"
                                style={{ width: '100px', marginLeft: '8px' }}
                            />
                        </div>

                        <button
                            className="btn btn-primary btn-lg"
                            onClick={handleTrain}
                            disabled={trainLoading || trainStatus === 'training'}
                        >
                            {trainStatus === 'training' ? 'Training...' : 'Start Training'}
                        </button>
                    </div>
                </div>
            </div>

            {/* Model History */}
            <div className="card models-card">
                <div className="card-header">
                    <h2 className="card-title">Model History ({pipeline === 'local' ? 'Ultra-Light' : 'Balanced'})</h2>
                    <button className="btn btn-secondary" onClick={loadData} disabled={modelsLoading}>
                        {modelsLoading ? 'Loading...' : 'Refresh'}
                    </button>
                </div>

                <div className="card-body">
                    {currentModels.length === 0 ? (
                        <div className="empty-models">
                            <p>Chưa có model nào được train cho pipeline này.</p>
                        </div>
                    ) : (
                        <div className="models-table-wrapper">
                            <table className="models-table">
                                <thead>
                                    <tr>
                                        <th>Type</th>
                                        <th>Version</th>
                                        <th>Status</th>
                                        <th>Accuracy</th>
                                        <th>Created</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {currentModels.map((model) => (
                                        <tr key={model.id}>
                                            <td>{model.model_type}</td>
                                            <td>v{model.version}</td>
                                            <td>
                                                {model.is_active ? (
                                                    <span className="status-active">Active</span>
                                                ) : (
                                                    <span className="status-inactive">Inactive</span>
                                                )}
                                            </td>
                                            <td>
                                                <span className="metrics">
                                                    {model.metrics?.accuracy
                                                        ? `${(model.metrics.accuracy * 100).toFixed(1)}%`
                                                        : '-'}
                                                </span>
                                            </td>
                                            <td>
                                                {model.created_at
                                                    ? new Date(model.created_at).toLocaleDateString('vi-VN')
                                                    : '-'}
                                            </td>
                                            <td>
                                                <button
                                                    className="btn btn-sm btn-secondary"
                                                    onClick={() => {
                                                        console.log('Button clicked, model.id:', model.id);
                                                        setSelectedModelId(model.id);
                                                    }}
                                                    title="View detailed metrics"
                                                >
                                                    View Metrics
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            </div>

            {/* Fusion Config */}
            {Object.keys(currentFusionConfig).length > 0 && (
                <div className="card fusion-card">
                    <div className="card-header">
                        <h2 className="card-title">Fusion Weights</h2>
                    </div>

                    <div className="card-body">
                        <div className="fusion-grid">
                            {Object.entries(currentFusionConfig).map(([className, weights]) => (
                                <div key={className} className="fusion-mode">
                                    <h4>{className}</h4>
                                    <pre>
                                        Text: {weights.w_text?.toFixed(2) || '0.50'}{'\n'}
                                        Image: {weights.w_img?.toFixed(2) || '0.50'}
                                    </pre>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Metrics Modal */}
            {selectedModelId && (
                <MetricsModal
                    modelId={selectedModelId}
                    onClose={() => setSelectedModelId(null)}
                />
            )}
        </div>
    );
}

export default TrainingPage;
