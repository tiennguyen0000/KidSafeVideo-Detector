import { useState, useEffect, useCallback } from 'react';

/**
 * Custom hook for theme management
 * Handles dark/light theme with localStorage persistence
 */
export function useTheme() {
    const [theme, setThemeState] = useState(() => {
        // Get from localStorage or default to 'dark'
        if (typeof window !== 'undefined') {
            return localStorage.getItem('theme') || 'dark';
        }
        return 'dark';
    });

    /**
     * Apply theme to document
     */
    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
    }, [theme]);

    /**
     * Set theme
     */
    const setTheme = useCallback((newTheme) => {
        if (newTheme === 'dark' || newTheme === 'light') {
            setThemeState(newTheme);
        }
    }, []);

    /**
     * Toggle between dark and light
     */
    const toggleTheme = useCallback(() => {
        setThemeState(prev => prev === 'dark' ? 'light' : 'dark');
    }, []);

    /**
     * Check if current theme is dark
     */
    const isDark = theme === 'dark';
    const isLight = theme === 'light';

    return {
        theme,
        setTheme,
        toggleTheme,
        isDark,
        isLight,
    };
}

export default useTheme;
