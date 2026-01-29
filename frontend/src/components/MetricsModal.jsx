import { useState, useEffect } from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend,
    ArcElement,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import './MetricsModal.css';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend,
    ArcElement
);

const CLASS_NAMES = ['Safe', 'Aggressive', 'Sexual', 'Superstition'];
const CLASS_COLORS = {
    'Safe': { bg: 'rgba(34, 197, 94, 0.6)', border: 'rgb(34, 197, 94)' },
    'Aggressive': { bg: 'rgba(239, 68, 68, 0.6)', border: 'rgb(239, 68, 68)' },
    'Sexual': { bg: 'rgba(168, 85, 247, 0.6)', border: 'rgb(168, 85, 247)' },
    'Superstition': { bg: 'rgba(59, 130, 246, 0.6)', border: 'rgb(59, 130, 246)' }
};

function MetricsModal({ modelId, onClose }) {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [data, setData] = useState(null);
    
    // Error Analysis State
    const [selectedCell, setSelectedCell] = useState(null); // {trueLabel, predLabel}
    const [cellSamples, setCellSamples] = useState(null);
    const [loadingSamples, setLoadingSamples] = useState(false);
    const [errorAnalysis, setErrorAnalysis] = useState(null);
    const [loadingAnalysis, setLoadingAnalysis] = useState(false);
    
    // Sample Details State
    const [selectedSample, setSelectedSample] = useState(null);
    const [sampleDetails, setSampleDetails] = useState(null);
    const [loadingSampleDetails, setLoadingSampleDetails] = useState(false);
    
    // Gate Weights State
    const [gateWeights, setGateWeights] = useState(null);
    const [loadingGateWeights, setLoadingGateWeights] = useState(false);

    // Fetch metrics on mount
    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                setLoading(true);
                const response = await fetch(`/api/models/${modelId}/metrics`);
                if (!response.ok) {
                    throw new Error('Failed to fetch metrics');
                }
                const result = await response.json();
                setData(result);
            } catch (err) {
                console.error('Error fetching metrics:', err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };
        fetchMetrics();
    }, [modelId]);
    
    // Fetch error analysis on mount
    useEffect(() => {
        const fetchErrorAnalysis = async () => {
            try {
                setLoadingAnalysis(true);
                const response = await fetch(`/api/models/${modelId}/error-analysis?split=val`);
                if (response.ok) {
                    const result = await response.json();
                    setErrorAnalysis(result);
                }
            } catch (err) {
                console.error('Error fetching error analysis:', err);
            } finally {
                setLoadingAnalysis(false);
            }
        };
        fetchErrorAnalysis();
    }, [modelId]);
    
    // Fetch gate weights analysis on mount
    useEffect(() => {
        const fetchGateWeights = async () => {
            try {
                setLoadingGateWeights(true);
                const response = await fetch(`/api/models/${modelId}/gate-weights?split=val`);
                if (response.ok) {
                    const result = await response.json();
                    setGateWeights(result);
                }
            } catch (err) {
                console.error('Error fetching gate weights:', err);
            } finally {
                setLoadingGateWeights(false);
            }
        };
        fetchGateWeights();
    }, [modelId]);
    
    // Fetch samples when cell is clicked
    useEffect(() => {
        if (!selectedCell) {
            setCellSamples(null);
            return;
        }
        
        const fetchSamples = async () => {
            try {
                setLoadingSamples(true);
                const { trueLabel, predLabel } = selectedCell;
                const response = await fetch(
                    `/api/models/${modelId}/confusion-samples?true_label=${trueLabel}&pred_label=${predLabel}&split=val&limit=30`
                );
                if (response.ok) {
                    const result = await response.json();
                    setCellSamples(result);
                }
            } catch (err) {
                console.error('Error fetching samples:', err);
            } finally {
                setLoadingSamples(false);
            }
        };
        fetchSamples();
    }, [selectedCell, modelId]);
    
    // Fetch sample details when sample is selected
    useEffect(() => {
        if (!selectedSample) {
            setSampleDetails(null);
            return;
        }
        
        const fetchDetails = async () => {
            try {
                setLoadingSampleDetails(true);
                const response = await fetch(`/api/samples/${selectedSample}/details`);
                if (response.ok) {
                    const result = await response.json();
                    setSampleDetails(result);
                }
            } catch (err) {
                console.error('Error fetching sample details:', err);
            } finally {
                setLoadingSampleDetails(false);
            }
        };
        fetchDetails();
    }, [selectedSample]);
    
    const handleCellClick = (trueLabel, predLabel) => {
        setSelectedCell({ trueLabel, predLabel });
        setSelectedSample(null);
        setSampleDetails(null);
    };
    
    const handleSampleClick = (sampleId) => {
        setSelectedSample(sampleId);
    };

    if (loading) {
        return (
            <div className="modal-overlay" onClick={onClose}>
                <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                    <div className="loading">Loading metrics...</div>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="modal-overlay" onClick={onClose}>
                <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                    <div className="error">Error: {error}</div>
                    <button className="btn btn-secondary" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        );
    }

    if (!data) return null;

    const { model, epochs, summary } = data;

    // Prepare chart data
    const epochNumbers = epochs.map((e) => e.epoch);
    const trainLoss = epochs.map((e) => e.train?.loss);
    const valLoss = epochs.map((e) => e.val?.loss);
    const trainAcc = epochs.map((e) => e.train?.accuracy);
    const valAcc = epochs.map((e) => e.val?.accuracy);
    const trainF1 = epochs.map((e) => e.train?.f1);
    const valF1 = epochs.map((e) => e.val?.f1);

    // Loss chart
    const lossChartData = {
        labels: epochNumbers,
        datasets: [
            {
                label: 'Train Loss',
                data: trainLoss,
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                tension: 0.3,
            },
            {
                label: 'Val Loss',
                data: valLoss,
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                tension: 0.3,
            },
        ],
    };

    // Accuracy chart
    const accuracyChartData = {
        labels: epochNumbers,
        datasets: [
            {
                label: 'Train Accuracy',
                data: trainAcc,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.3,
            },
            {
                label: 'Val Accuracy',
                data: valAcc,
                borderColor: 'rgb(153, 102, 255)',
                backgroundColor: 'rgba(153, 102, 255, 0.1)',
                tension: 0.3,
            },
        ],
    };

    // F1 Score chart
    const f1ChartData = {
        labels: epochNumbers,
        datasets: [
            {
                label: 'Train F1',
                data: trainF1,
                borderColor: 'rgb(255, 159, 64)',
                backgroundColor: 'rgba(255, 159, 64, 0.1)',
                tension: 0.3,
            },
            {
                label: 'Val F1',
                data: valF1,
                borderColor: 'rgb(255, 205, 86)',
                backgroundColor: 'rgba(255, 205, 86, 0.1)',
                tension: 0.3,
            },
        ],
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
            },
        },
        scales: {
            y: {
                beginAtZero: true,
            },
        },
    };

    // Get best epoch metrics
    const bestEpochData = epochs.find((e) => e.epoch === summary.best_epoch);
    const bestValMetrics = bestEpochData?.val;

    // Confusion matrix for best epoch
    const confusionMatrix = bestValMetrics?.confusion_matrix;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content modal-xlarge" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <h2>Training Metrics - {model.version}</h2>
                    <button className="btn-close" onClick={onClose}>
                        ×
                    </button>
                </div>

                <div className="modal-body">
                    {/* Summary Section */}
                    <div className="metrics-summary">
                        <div className="summary-card">
                            <div className="summary-label">Mode</div>
                            <div className="summary-value">{model.mode}</div>
                        </div>
                        <div className="summary-card">
                            <div className="summary-label">Best Epoch</div>
                            <div className="summary-value">{summary.best_epoch}/{summary.total_epochs}</div>
                        </div>
                        <div className="summary-card">
                            <div className="summary-label">Best Accuracy</div>
                            <div className="summary-value">
                                {summary.best_accuracy ? `${(summary.best_accuracy * 100).toFixed(1)}%` : 'N/A'}
                            </div>
                        </div>
                        <div className="summary-card">
                            <div className="summary-label">Best F1</div>
                            <div className="summary-value">
                                {summary.best_f1 ? summary.best_f1.toFixed(3) : 'N/A'}
                            </div>
                        </div>
                        <div className="summary-card">
                            <div className="summary-label">Training Time</div>
                            <div className="summary-value">
                                {summary.training_time_seconds
                                    ? `${(summary.training_time_seconds / 60).toFixed(1)} min`
                                    : 'N/A'}
                            </div>
                        </div>
                    </div>

                    {/* Charts Section */}
                    <div className="charts-grid">
                        <div className="chart-container">
                            <h3>Loss Over Epochs</h3>
                            <div className="chart-wrapper">
                                <Line data={lossChartData} options={chartOptions} />
                            </div>
                        </div>

                        <div className="chart-container">
                            <h3>Accuracy Over Epochs</h3>
                            <div className="chart-wrapper">
                                <Line data={accuracyChartData} options={chartOptions} />
                            </div>
                        </div>

                        <div className="chart-container">
                            <h3>F1 Score Over Epochs</h3>
                            <div className="chart-wrapper">
                                <Line data={f1ChartData} options={chartOptions} />
                            </div>
                        </div>
                    </div>

                    {/* Best Epoch Metrics */}
                    {bestValMetrics && (
                        <div className="best-epoch-section">
                            <h3>Best Epoch ({summary.best_epoch}) - Validation Metrics</h3>

                            <div className="per-class-metrics">
                                <table className="metrics-table">
                                    <thead>
                                        <tr>
                                            <th>Class</th>
                                            <th>Precision</th>
                                            <th>Recall</th>
                                            <th>F1 Score</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {bestValMetrics.per_class_metrics &&
                                            Object.entries(bestValMetrics.per_class_metrics).map(
                                                ([className, metrics]) => (
                                                    <tr key={className}>
                                                        <td>{className}</td>
                                                        <td>{metrics.precision.toFixed(3)}</td>
                                                        <td>{metrics.recall.toFixed(3)}</td>
                                                        <td>{metrics.f1.toFixed(3)}</td>
                                                    </tr>
                                                )
                                            )}
                                    </tbody>
                                </table>
                            </div>

                            {/* Confusion Matrix with Error Analysis Panel */}
                            {confusionMatrix && (
                                <div className="confusion-analysis-container">
                                    <h4>Confusion Matrix & Error Analysis</h4>
                                    <p className="cm-instruction">Click on any cell to view samples</p>
                                    
                                    <div className="confusion-analysis-split">
                                        {/* Left: Confusion Matrix */}
                                        <div className="cm-panel">
                                            <div className="cm-header">
                                                <span className="cm-label-pred">Predicted →</span>
                                            </div>
                                            <div className="cm-with-label">
                                                <span className="cm-label-true">Actual ↓</span>
                                                <table className="cm-table cm-interactive">
                                                    <thead>
                                                        <tr>
                                                            <th></th>
                                                            {CLASS_NAMES.map((name) => (
                                                                <th key={name}>{name.substring(0, 4)}</th>
                                                            ))}
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {confusionMatrix.map((row, i) => (
                                                            <tr key={i}>
                                                                <th>{CLASS_NAMES[i].substring(0, 4)}</th>
                                                                {row.map((val, j) => (
                                                                    <td 
                                                                        key={j} 
                                                                        className={`cm-cell ${i === j ? 'diagonal' : 'off-diagonal'} ${
                                                                            selectedCell?.trueLabel === CLASS_NAMES[i] && 
                                                                            selectedCell?.predLabel === CLASS_NAMES[j] ? 'selected' : ''
                                                                        }`}
                                                                        onClick={() => handleCellClick(CLASS_NAMES[i], CLASS_NAMES[j])}
                                                                        title={`True: ${CLASS_NAMES[i]}, Pred: ${CLASS_NAMES[j]}`}
                                                                    >
                                                                        {val}
                                                                    </td>
                                                                ))}
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                        
                                        {/* Right: Samples Panel */}
                                        <div className="samples-panel">
                                            {!selectedCell && !errorAnalysis && (
                                                <div className="samples-placeholder">
                                                    <p>Click on a confusion matrix cell to view samples</p>
                                                </div>
                                            )}
                                            
                                            {!selectedCell && errorAnalysis && (
                                                <ErrorAnalysisPanel analysis={errorAnalysis} loading={loadingAnalysis} />
                                            )}
                                            
                                            {selectedCell && (
                                                <SamplesPanel 
                                                    selectedCell={selectedCell}
                                                    cellSamples={cellSamples}
                                                    loadingSamples={loadingSamples}
                                                    selectedSample={selectedSample}
                                                    sampleDetails={sampleDetails}
                                                    loadingSampleDetails={loadingSampleDetails}
                                                    onSampleClick={handleSampleClick}
                                                    onBack={() => {
                                                        setSelectedCell(null);
                                                        setSelectedSample(null);
                                                    }}
                                                />
                                            )}
                                        </div>
                                    </div>
                                </div>
                            )}
                            
                            {/* Gate Weights Analysis Section */}
                            <GateWeightsSection 
                                gateWeights={gateWeights} 
                                loading={loadingGateWeights} 
                            />
                        </div>
                    )}
                </div>

                <div className="modal-footer">
                    <button className="btn btn-secondary" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
}

// Gate Weights Analysis Section Component
function GateWeightsSection({ gateWeights, loading }) {
    if (loading) {
        return (
            <div className="gate-weights-section">
                <h3>Gate Weights Analysis</h3>
                <div className="loading-small">Analyzing gate weights...</div>
            </div>
        );
    }
    
    if (!gateWeights || gateWeights.error) {
        return (
            <div className="gate-weights-section">
                <h3>Gate Weights Analysis</h3>
                <div className="gate-weights-placeholder">
                    <p>{gateWeights?.error || 'Gate weights analysis not available'}</p>
                    <small>This analysis shows how the model balances between image and text modalities for each class.</small>
                </div>
            </div>
        );
    }
    
    const { per_class, overall, insights, samples_processed } = gateWeights;
    
    // Prepare chart data
    const chartData = {
        labels: CLASS_NAMES,
        datasets: [
            {
                label: 'Image Weight',
                data: CLASS_NAMES.map(cls => (per_class[cls]?.img_weight || 0) * 100),
                backgroundColor: 'rgba(59, 130, 246, 0.7)',
                borderColor: 'rgb(59, 130, 246)',
                borderWidth: 1,
            },
            {
                label: 'Text Weight',
                data: CLASS_NAMES.map(cls => (per_class[cls]?.txt_weight || 0) * 100),
                backgroundColor: 'rgba(168, 85, 247, 0.7)',
                borderColor: 'rgb(168, 85, 247)',
                borderWidth: 1,
            },
        ],
    };
    
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        scales: {
            x: {
                stacked: true,
                max: 100,
                title: {
                    display: true,
                    text: 'Modality Weight (%)',
                },
                ticks: {
                    callback: (value) => `${value}%`,
                },
            },
            y: {
                stacked: true,
            },
        },
        plugins: {
            tooltip: {
                callbacks: {
                    label: (context) => {
                        return `${context.dataset.label}: ${context.raw.toFixed(1)}%`;
                    },
                },
            },
            legend: {
                position: 'top',
            },
        },
    };
    
    return (
        <div className="gate-weights-section">
            <h3>Gate Weights Analysis</h3>
            <p className="section-subtitle">
                Shows how the model trusts image vs text features for each class 
                <span className="samples-count">({samples_processed} samples analyzed)</span>
            </p>
            
            {/* Visualization */}
            <div className="gate-weights-content">
                {/* Chart */}
                <div className="gate-weights-chart">
                    <div className="chart-wrapper" style={{ height: '200px' }}>
                        <Bar data={chartData} options={chartOptions} />
                    </div>
                </div>
                
                {/* Stats Table */}
                <div className="gate-weights-table-wrapper">
                    <table className="gate-weights-table">
                        <thead>
                            <tr>
                                <th>Class</th>
                                <th>Image %</th>
                                <th>Text %</th>
                                <th>Dominant</th>
                                <th>Samples</th>
                            </tr>
                        </thead>
                        <tbody>
                            {CLASS_NAMES.map((cls) => {
                                const stats = per_class[cls];
                                if (!stats || stats.count === 0) return null;
                                
                                return (
                                    <tr key={cls}>
                                        <td>
                                            <span 
                                                className="class-indicator" 
                                                style={{ backgroundColor: CLASS_COLORS[cls]?.border }}
                                            />
                                            {cls}
                                        </td>
                                        <td className="weight-cell">
                                            <div className="weight-bar">
                                                <div 
                                                    className="weight-fill img" 
                                                    style={{ width: `${stats.img_weight * 100}%` }}
                                                />
                                            </div>
                                            <span>{(stats.img_weight * 100).toFixed(1)}%</span>
                                        </td>
                                        <td className="weight-cell">
                                            <div className="weight-bar">
                                                <div 
                                                    className="weight-fill txt" 
                                                    style={{ width: `${stats.txt_weight * 100}%` }}
                                                />
                                            </div>
                                            <span>{(stats.txt_weight * 100).toFixed(1)}%</span>
                                        </td>
                                        <td>
                                            <span className={`dominant-badge ${stats.dominant_modality}`}>
                                                {stats.dominant_modality === 'image' ? '🖼️ Image' : 
                                                 stats.dominant_modality === 'text' ? '📝 Text' : '⚖️ Balanced'}
                                            </span>
                                        </td>
                                        <td>{stats.count}</td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
                
                {/* Overall Stats */}
                <div className="gate-weights-overall">
                    <h4>Overall Model Behavior</h4>
                    <div className="overall-stats">
                        <div className="overall-stat">
                            <span className="stat-icon">🖼️</span>
                            <div className="stat-info">
                                <span className="stat-value">{(overall.img_weight * 100).toFixed(1)}%</span>
                                <span className="stat-label">Image Weight</span>
                            </div>
                        </div>
                        <div className="overall-stat">
                            <span className="stat-icon">📝</span>
                            <div className="stat-info">
                                <span className="stat-value">{(overall.txt_weight * 100).toFixed(1)}%</span>
                                <span className="stat-label">Text Weight</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                {/* Insights */}
                {insights && insights.length > 0 && (
                    <div className="gate-weights-insights">
                        <h4>Insights</h4>
                        <div className="insights-list">
                            {insights.map((insight, i) => (
                                <div key={i} className={`insight-item insight-${insight.type}`}>
                                    <span className="insight-label">{insight.label}</span>
                                    <span className="insight-message">{insight.message}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
            
            {/* Explanation */}
            <div className="gate-weights-explanation">
                <details>
                    <summary>How to interpret gate weights?</summary>
                    <div className="explanation-content">
                        <p>Gate weights show how the model decides to trust each modality:</p>
                        <ul>
                            <li><strong>Image Weight close to 100%:</strong> Model relies heavily on visual features (frames)</li>
                            <li><strong>Text Weight close to 100%:</strong> Model relies heavily on text features (transcript/title)</li>
                            <li><strong>Balanced (~50%):</strong> Model uses both modalities equally</li>
                        </ul>
                        <p><strong>Expected patterns:</strong></p>
                        <ul>
                            <li><strong>Sexual:</strong> Higher image weight - visual content is more indicative</li>
                            <li><strong>Aggressive:</strong> Higher text weight - verbal aggression often in speech</li>
                            <li><strong>Safe:</strong> Balanced - uses both modalities</li>
                            <li><strong>Superstition:</strong> Higher text weight - content often described verbally</li>
                        </ul>
                    </div>
                </details>
            </div>
        </div>
    );
}

// Error Analysis Panel Component
function ErrorAnalysisPanel({ analysis, loading }) {
    if (loading) {
        return <div className="loading-small">Loading analysis...</div>;
    }
    
    if (!analysis) {
        return <div className="samples-placeholder">No analysis available</div>;
    }
    
    return (
        <div className="error-analysis-panel">
            <h5>Error Analysis Summary</h5>
            
            {/* Data Quality */}
            <div className="analysis-section">
                <h6>Data Quality</h6>
                <div className="stat-grid">
                    <div className="stat-item">
                        <span className="stat-value">{analysis.data_quality.total_videos}</span>
                        <span className="stat-label">Total Videos</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-value">{analysis.data_quality.empty_transcript_count}</span>
                        <span className="stat-label">Empty Transcript</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-value">{analysis.data_quality.empty_title_count}</span>
                        <span className="stat-label">Empty Title</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-value">{(analysis.data_quality.empty_transcript_ratio * 100).toFixed(1)}%</span>
                        <span className="stat-label">No Transcript %</span>
                    </div>
                </div>
            </div>
            
            {/* Per-class stats */}
            <div className="analysis-section">
                <h6>Per-Class Statistics</h6>
                <table className="mini-table">
                    <thead>
                        <tr>
                            <th>Class</th>
                            <th>Videos</th>
                            <th>Samples</th>
                            <th>Avg Len</th>
                        </tr>
                    </thead>
                    <tbody>
                        {analysis.per_class_stats.map(stat => (
                            <tr key={stat.label}>
                                <td>{stat.label}</td>
                                <td>{stat.video_count}</td>
                                <td>{stat.sample_count}</td>
                                <td>{stat.avg_transcript_length}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            
            {/* Misclassification patterns */}
            {analysis.misclassification_patterns.length > 0 && (
                <div className="analysis-section">
                    <h6>Top Misclassifications</h6>
                    <div className="misclass-list">
                        {analysis.misclassification_patterns.slice(0, 5).map((m, i) => (
                            <div key={i} className="misclass-item">
                                <span className="misclass-labels">
                                    {m.true_label} → {m.pred_label}
                                </span>
                                <span className="misclass-count">{m.count} cases</span>
                                <span className="misclass-conf">{(m.avg_confidence * 100).toFixed(0)}% conf</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
            
            {/* Insights */}
            {analysis.insights && analysis.insights.length > 0 && (
                <div className="analysis-section">
                    <h6>Insights</h6>
                    <div className="insights-list">
                        {analysis.insights.map((insight, i) => (
                            <div key={i} className={`insight-item insight-${insight.type}`}>
                                <span className="insight-text">{insight.message}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
            
            {/* Accuracy by transcript quality */}
            {analysis.error_by_transcript_quality.length > 0 && (
                <div className="analysis-section">
                    <h6>Accuracy by Transcript Quality</h6>
                    <div className="quality-bars">
                        {analysis.error_by_transcript_quality.map((q, i) => (
                            <div key={i} className="quality-bar-item">
                                <span className="quality-label">{q.quality.replace('_', ' ')}</span>
                                <div className="quality-bar">
                                    <div 
                                        className="quality-bar-fill" 
                                        style={{ width: `${q.accuracy * 100}%` }}
                                    />
                                </div>
                                <span className="quality-value">{(q.accuracy * 100).toFixed(0)}%</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

// Samples Panel Component
function SamplesPanel({ 
    selectedCell, 
    cellSamples, 
    loadingSamples, 
    selectedSample,
    sampleDetails,
    loadingSampleDetails,
    onSampleClick,
    onBack 
}) {
    const isError = selectedCell.trueLabel !== selectedCell.predLabel;
    
    return (
        <div className="samples-content">
            <div className="samples-header">
                <button className="btn-back" onClick={onBack}>← Back</button>
                <div className="samples-title">
                    <span className={`cell-type ${isError ? 'error' : 'correct'}`}>
                        {isError ? 'Errors' : 'Correct'}
                    </span>
                    <span className="cell-labels">
                        {selectedCell.trueLabel} → {selectedCell.predLabel}
                    </span>
                </div>
            </div>
            
            {loadingSamples && <div className="loading-small">Loading samples...</div>}
            
            {!loadingSamples && cellSamples && (
                <>
                    <div className="samples-count">
                        Found {cellSamples.count} samples
                    </div>
                    
                    <div className="samples-list-container">
                        {/* Sample List */}
                        <div className="samples-list">
                            {cellSamples.samples.map((sample) => (
                                <div 
                                    key={sample.sample_id}
                                    className={`sample-item ${selectedSample === sample.sample_id ? 'selected' : ''}`}
                                    onClick={() => onSampleClick(sample.sample_id)}
                                >
                                    <div className="sample-title">
                                        {sample.title || 'No title'}
                                    </div>
                                    <div className="sample-meta">
                                        <span className={`confidence ${sample.confidence > 0.8 ? 'high' : sample.confidence > 0.5 ? 'medium' : 'low'}`}>
                                            {sample.confidence ? `${(sample.confidence * 100).toFixed(0)}%` : 'N/A'}
                                        </span>
                                        <span className={`transcript-status ${sample.has_transcript ? 'has' : 'no'}`}>
                                            {sample.has_transcript ? 'Has' : 'No'}
                                        </span>
                                        {sample.augment_idx > 0 && (
                                            <span className="augment-tag">aug{sample.augment_idx}</span>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                        
                        {/* Sample Details */}
                        {selectedSample && (
                            <div className="sample-details-panel">
                                {loadingSampleDetails && <div className="loading-small">Loading...</div>}
                                
                                {!loadingSampleDetails && sampleDetails && (
                                    <SampleDetailsView details={sampleDetails} />
                                )}
                            </div>
                        )}
                    </div>
                </>
            )}
        </div>
    );
}

// Sample Details View Component
function SampleDetailsView({ details }) {
    const [activeTab, setActiveTab] = useState('frames');
    
    return (
        <div className="sample-details">
            <div className="details-header">
                <h6>{details.title || 'Untitled'}</h6>
                <span className="label-badge">{details.label}</span>
            </div>
            
            <div className="details-tabs">
                <button 
                    className={`tab ${activeTab === 'frames' ? 'active' : ''}`}
                    onClick={() => setActiveTab('frames')}
                >
                    Frames ({details.frame_count})
                </button>
                <button 
                    className={`tab ${activeTab === 'transcript' ? 'active' : ''}`}
                    onClick={() => setActiveTab('transcript')}
                >
                    Transcript
                </button>
            </div>
            
            <div className="details-content">
                {activeTab === 'frames' && (
                    <div className="frames-grid">
                        {details.frames.length === 0 && (
                            <div className="no-data">No frames available</div>
                        )}
                        {details.frames.map((frame) => (
                            <div key={frame.index} className="frame-item">
                                <img 
                                    src={frame.url} 
                                    alt={`Frame ${frame.index}`}
                                    loading="lazy"
                                    onError={(e) => {
                                        e.target.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect fill="%23333" width="100" height="100"/><text fill="%23666" x="50%" y="50%" text-anchor="middle" dy=".3em">No Image</text></svg>';
                                    }}
                                />
                                <span className="frame-index">#{frame.index}</span>
                            </div>
                        ))}
                    </div>
                )}
                
                {activeTab === 'transcript' && (
                    <div className="transcript-content">
                        {details.raw_transcript ? (
                            <>
                                <div className="transcript-section">
                                    <h6>Original Transcript</h6>
                                    <p className="transcript-text">{details.raw_transcript}</p>
                                </div>
                                
                                {details.transcript_chunks.length > 0 && (
                                    <div className="transcript-section">
                                        <h6>Selected Chunks</h6>
                                        <div className="chunks-list">
                                            {details.transcript_chunks.map((chunk) => (
                                                <div key={chunk.index} className="chunk-item">
                                                    <span className="chunk-index">#{chunk.index}</span>
                                                    <span className="chunk-text">{chunk.text}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </>
                        ) : (
                            <div className="no-data">No transcript available for this video</div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export default MetricsModal;
