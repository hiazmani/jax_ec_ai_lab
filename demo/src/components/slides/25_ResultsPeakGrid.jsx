import React from 'react';
import PixiSplash from '../PixiSplash.jsx';
import BarChart from '../BarChart';
import { useResults } from '../ResultContext';

const ResultsPeak = ({ goToNext }) => {
  const { results } = useResults();
  if (!results) return handleMissingData();

  // Peak imports
  const baselinePeak = results.baseline.peak_grid_import;
  const tradingPeak = results.metrics.peak_grid_import;
  const reductionKw = baselinePeak - tradingPeak;
  const reductionPct = (reductionKw / baselinePeak) * 100;

  // Chart data
  const peakData = [
    { label: 'No Trading', value: baselinePeak, color: '#EF4444' },
    { label: 'With Trading', value: tradingPeak, color: '#10B981' }
  ];

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Background rings */}
      <PixiSplash numRings={4} className="absolute inset-0 z-0" />

      {/* Foreground content */}
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-4 py-6 text-center">
        <h1 className="slide-title mb-4">Peak Grid Import Reduction 📉</h1>
        <p className="slide-content mb-6 max-w-2xl">
          Lowering the highest instant demand on the grid prevents expensive upgrades and improves stability.
          Compare the community’s annual peak grid import with and without local energy trading below.
        </p>

        {/* Peak bar chart */}
        <div className="bg-opacity-80 p-6 rounded-lg shadow-md mb-4 flex justify-center">
          <BarChart data={peakData} width={500} height={300} roundValues />
        </div>

        {/* Reduction gauge */}
        <div className="w-full max-w-md mb-6">
          <br />
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
            <strong>Peak Reduction:</strong> {reductionKw.toLocaleString()} kW ({reductionPct.toFixed(1)}%)
          </p>
        </div>

        <button
          onClick={goToNext}
          className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
        >
          Summary & What's Next?
        </button>
      </div>
    </div>
  );
};

export default ResultsPeak;
