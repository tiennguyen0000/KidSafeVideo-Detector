import { useState } from 'react';
import './SearchBar.css';

function SearchBar({ onSearch, loading, placeholder = 'Tìm kiếm video...' }) {
    const [query, setQuery] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (query.trim()) {
            onSearch(query.trim());
        }
    };

    return (
        <form className="search-bar" onSubmit={handleSubmit}>
            <div className="search-input-wrapper">
                <input
                    type="text"
                    className="search-input"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder={placeholder}
                    disabled={loading}
                />
                {query && (
                    <button
                        type="button"
                        className="clear-btn"
                        onClick={() => setQuery('')}
                        aria-label="Clear search"
                    >
                        ✕
                    </button>
                )}
            </div>
            <button
                type="submit"
                className="btn btn-primary search-btn"
                disabled={loading || !query.trim()}
            >
                {loading ? 'Đang tìm...' : 'Tìm kiếm'}
            </button>
        </form>
    );
}

export default SearchBar;
