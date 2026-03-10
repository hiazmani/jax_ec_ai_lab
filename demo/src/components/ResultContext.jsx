import React, { createContext, useState, useEffect, useContext } from "react";

/**
 * results = null  → no run yet
 * results = {...} → JSON returned by backend
 */
export const ResultsContext = createContext({
    results: null,
    /* eslint-disable no-unused-vars */
    setResults: (_json) => {},
    clearResults: () => {},
    /* eslint-enable no-unused-vars */
});

/**
 * <ResultsProvider> must wrap the entire slide-deck
 * (e.g. in src/App.jsx or src/main.jsx).
 */
export const ResultsProvider = ({ children }) => {
    const [results, setResults] = useState(null);

    // 1.  Hydrate from localStorage (on reload / back button)
    useEffect(() => {
        if (!results) {
            const cached = localStorage.getItem("lastRun");
            if (cached) setResults(JSON.parse(cached));
        }
    }, [results]);

    // 2.  Helper to wipe results (e.g. when user starts a new scenario)
    const clearResults = () => {
        setResults(null);
        localStorage.removeItem("lastRun");
    };

    return (
        <ResultsContext.Provider value={{ results, setResults, clearResults }}>
            {children}
        </ResultsContext.Provider>
    );
};

/** Small hook to consume context */
export const useResults = () => useContext(ResultsContext);
