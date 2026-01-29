import { useTheme } from '../hooks/useTheme';
import { useSettings } from '../hooks/useSettings';
import './SettingsPage.css';

function SettingsPage() {
    // Use custom hooks
    const { theme, setTheme, toggleTheme, isDark } = useTheme();
    const {
        defaultPipeline,
        setDefaultPipeline,
        health,
        stats,
        loading,
        loadSystemStatus,
        isBackendConnected,
        isDatabaseConnected,
    } = useSettings();

    return (
        <div className="settings-page">
            {/* Header */}
            <div className="page-header">
                <h1 className="page-title">Settings</h1>
                <p className="page-subtitle">Cấu hình giao diện và hệ thống</p>
            </div>

            <div className="settings-grid">
                {/* Appearance */}
                <div className="card settings-card">
                    <div className="card-header">
                        <h2 className="card-title">Appearance</h2>
                    </div>

                    <div className="card-body">
                        <div className="setting-item">
                            <div className="setting-info">
                                <h3>Theme</h3>
                                <p>Chọn giao diện sáng hoặc tối</p>
                            </div>
                            <div className="toggle-group theme-toggle">
                                <button
                                    className={`toggle-option ${isDark ? 'active' : ''}`}
                                    onClick={() => setTheme('dark')}
                                >
                                    Dark
                                </button>
                                <button
                                    className={`toggle-option ${!isDark ? 'active' : ''}`}
                                    onClick={() => setTheme('light')}
                                >
                                    Light
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Inference Settings */}
                <div className="card settings-card">
                    <div className="card-header">
                        <h2 className="card-title">Inference</h2>
                    </div>

                    <div className="card-body">
                        <div className="setting-item">
                            <div className="setting-info">
                                <h3>Default Pipeline</h3>
                                <p>Pipeline mặc định khi chạy inference</p>
                            </div>
                            <div className="toggle-group">
                                <button
                                    className={`toggle-option ${defaultPipeline === 'local' ? 'active' : ''}`}
                                    onClick={() => setDefaultPipeline('local')}
                                >
                                    Local
                                </button>
                                <button
                                    className={`toggle-option ${defaultPipeline === 'colab' ? 'active' : ''}`}
                                    onClick={() => setDefaultPipeline('colab')}
                                >
                                    Colab
                                </button>
                            </div>
                        </div>

                        <div className="pipeline-description">
                            {defaultPipeline === 'local' ? (
                                <div className="pipeline-detail local">
                                    <strong>Local (Ultra-Light Mode)</strong>
                                    <ul>
                                        <li>EfficientNet-B0 + SentenceBERT Tiny</li>
                                        <li>GMU Fusion</li>
                                        <li>Chạy trên CPU, nhanh hơn</li>
                                    </ul>
                                </div>
                            ) : (
                                <div className="pipeline-detail colab">
                                    <strong>Colab (Balanced Mode)</strong>
                                    <ul>
                                        <li>ResNet50 + PhoBERT</li>
                                        <li>Attention Fusion</li>
                                        <li>Chính xác hơn, cần GPU</li>
                                    </ul>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* System Status */}
                <div className="card settings-card status-card">
                    <div className="card-header">
                        <h2 className="card-title">System Status</h2>
                        <button className="btn btn-secondary" onClick={loadSystemStatus} disabled={loading}>
                            {loading ? 'Loading...' : 'Refresh'}
                        </button>
                    </div>

                    <div className="card-body">
                        {loading ? (
                            <div className="loading-inline">
                                Loading...
                            </div>
                        ) : (
                            <>
                                {/* Health */}
                                <div className="status-section">
                                    <h3>Backend Health</h3>
                                    <div className={`status-row ${isBackendConnected ? 'success' : 'error'}`}>
                                        <span>{isBackendConnected ? 'Connected' : 'Disconnected'}</span>
                                        {health?.mode && (
                                            <span className="status-mode">Mode: {health.mode}</span>
                                        )}
                                    </div>
                                </div>

                                {/* Database */}
                                <div className="status-section">
                                    <h3>Database</h3>
                                    <div className={`status-row ${isDatabaseConnected ? 'success' : 'error'}`}>
                                        <span>{isDatabaseConnected ? 'Connected' : 'Disconnected'}</span>
                                    </div>
                                </div>

                                {/* Statistics */}
                                {stats && (
                                    <div className="status-section">
                                        <h3>Statistics</h3>
                                        <div className="stats-grid">
                                            <div className="stat-item">
                                                <span className="stat-value">{stats.total_videos || 0}</span>
                                                <span className="stat-label">Videos</span>
                                            </div>
                                            <div className="stat-item">
                                                <span className="stat-value">{stats.total_predictions || 0}</span>
                                                <span className="stat-label">Predictions</span>
                                            </div>
                                            <div className="stat-item">
                                                <span className="stat-value">{stats.inference_queue_length || 0}</span>
                                                <span className="stat-label">Queue</span>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>

                {/* About */}
                <div className="card settings-card about-card">
                    <div className="card-header">
                        <h2 className="card-title">About</h2>
                    </div>

                    <div className="card-body">
                        <div className="about-content">
                            <h3>Video Classifier</h3>
                            <p>Hệ thống phân loại video độc hại sử dụng Multimodal AI</p>

                            <div className="about-details">
                                <div className="about-row">
                                    <span>Version</span>
                                    <span>1.0.0</span>
                                </div>
                                <div className="about-row">
                                    <span>Frontend</span>
                                    <span>React + Vite</span>
                                </div>
                                <div className="about-row">
                                    <span>Backend</span>
                                    <span>FastAPI + Airflow</span>
                                </div>
                                <div className="about-row">
                                    <span>ML Stack</span>
                                    <span>PyTorch</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default SettingsPage;
