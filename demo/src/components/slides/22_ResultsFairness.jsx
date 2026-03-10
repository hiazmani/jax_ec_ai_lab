import { useState, useEffect } from 'react';
import PixiSplash from '../PixiSplash.jsx';
import { useResults } from '../ResultContext';

const ResultsFairness = ({ goToNext }) => {
  const { results } = useResults();
  if (!results) return handleMissingData();

  const gini = results.metrics.fairness_gini;
  // Take the absolute value of the Gini index
  const absoluteGini = Math.abs(gini);

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Background rings */}
      <PixiSplash numRings={4} className="absolute inset-0 z-0" />

      {/* Centered content */}
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-4 py-6 text-center">
        <h1 className="slide-title mb-4">Fairness in the Community 🤝</h1>
        <p className="slide-content mb-6 max-w-2xl">
          Savings are great, but it’s important to share them fairly. A community should ensure benefits
          are distributed, so no one is left much worse off. We use the <strong>Gini index</strong> to measure
          equality in cost savings:<br/>0 = perfect equality, 1 = maximum inequality.
        </p>

        {/* Gini gauge */}
        <div className="w-full max-w-md mb-2">
          <div className="bg-gray-300 h-4 rounded overflow-hidden">
            <div
              className="bg-green-600 h-full"
              style={{ width: `${(absoluteGini * 100).toFixed(1)}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-700 mt-1">
            <span>0.0 (equal)</span>
            <span>1.0 (unequal)</span>
          </div>
        </div>
        <div className="mb-6 text-sm text-gray-800">
          <strong>Gini index:</strong> {absoluteGini.toFixed(3)}
        </div>

        <button
          onClick={goToNext}
          className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
        >
          Did your community become more self-sufficient?
        </button>
      </div>
    </div>
  );
};

export default ResultsFairness;
