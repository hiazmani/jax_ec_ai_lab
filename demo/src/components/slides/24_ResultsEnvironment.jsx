import React from 'react';
import PixiSplash from '../PixiSplash.jsx';
import BarChart from '../BarChart';
import { useResults } from '../ResultContext';

const ResultsEnvironment = ({ goToNext }) => {
  const { results } = useResults();
  if (!results) return handleMissingData();

  // Metrics
  const baselineEm = results.baseline.emissions;
  const tradingEm = results.metrics.emissions;
  const reductionPct = ((baselineEm - tradingEm) / baselineEm) * 100;

  // Chart data
  const emData = [
    { label: 'No Trading', value: baselineEm, color: '#EF4444' },
    { label: 'With Trading', value: tradingEm, color: '#10B981' }
  ];

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Background rings */}
      <PixiSplash numRings={4} className="absolute inset-0 z-0" />

      {/* Centered content */}
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-4 py-6 text-center">
        <h1 className="slide-title mb-4">Results: Environmental Benefits 🌍</h1>
        <p className="slide-content mb-6 max-w-2xl">
          By sharing and consuming locally, your community reduces dependency on the external grid and
          avoids CO₂ emissions from conventional power plants. Below, compare total yearly emissions with
          and without energy trading, and see the percentage reduction achieved.
        </p>

        {/* Emissions bar chart */}
        <div className="bg-opacity-80 p-6 rounded-lg shadow-md mb-4 flex justify-center">
          <BarChart data={emData} width={500} height={300} roundValues />
        </div>

        {/* Reduction gauge */}
        <div className="w-full max-w-md mb-6">
          <br/>
          <div className="bg-gray-300 h-4 rounded overflow-hidden">
            <div
              className="bg-green-600 h-full"
              style={{ width: `${reductionPct.toFixed(1)}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-700 mt-1">
            <span>0%</span>
            <span>100%</span>
          </div>
          <p className="text-sm text-gray-800 mt-1">
            <strong>Emissions Reduced:</strong> {reductionPct.toFixed(1)}%
          </p>
        </div>

        <button
          onClick={goToNext}
          className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
        >
          Any other benefits?
        </button>
      </div>
    </div>
  );
};

export default ResultsEnvironment;
