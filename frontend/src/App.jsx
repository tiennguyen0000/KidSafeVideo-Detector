import { Routes, Route, NavLink } from 'react-router-dom'
import InferencePage from './pages/InferencePage'
import TrainingPage from './pages/TrainingPage'
import SettingsPage from './pages/SettingsPage'
import './App.css'

function App() {
    return (
        <div className="app">
            {/* Navigation */}
            <nav className="navbar">
                <div className="nav-brand">
                    <span className="brand-text">Video Classifier</span>
                </div>
                <div className="nav-links">
                    <NavLink to="/" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
                        Inference
                    </NavLink>
                    <NavLink to="/training" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
                        Training
                    </NavLink>
                    <NavLink to="/settings" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
                        Settings
                    </NavLink>
                </div>
            </nav>

            {/* Main Content */}
            <main className="main-content">
                <Routes>
                    <Route path="/" element={<InferencePage />} />
                    <Route path="/training" element={<TrainingPage />} />
                    <Route path="/settings" element={<SettingsPage />} />
                </Routes>
            </main>
        </div>
    )
}

export default App
