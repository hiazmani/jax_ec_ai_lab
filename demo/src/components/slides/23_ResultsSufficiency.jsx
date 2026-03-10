import { useState } from 'react';
import PixiSplash from '../PixiSplash.jsx';
import BarChart from '../BarChart';
import { useResults } from '../ResultContext';

const ResultsSufficiency = ({ goToNext }) => {
  const { results } = useResults();
  if (!results) return handleMissingData();

  // Metrics
  const baselineSelf = results.baseline.self_sustainability_ratio * 100;
  const tradingSelf = results.metrics.self_sustainability_ratio * 100;
  const baselineSpill = results.baseline.renewable_spill;
  const tradingSpill = results.metrics.renewable_spill;
  const spillImprovement = ((baselineSpill - tradingSpill) / baselineSpill) * 100;

  // Chart data
  const selfData = [
    { label: 'No Trading', value: baselineSelf, color: '#EF4444' },
    { label: 'With Trading', value: tradingSelf, color: '#10B981' }
  ];
  const spillData = [
    { label: 'No Trading', value: baselineSpill, color: '#EF4444' },
    { label: 'With Trading', value: tradingSpill, color: '#10B981' }
  ];

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Background decoration */}
      <PixiSplash numRings={4} className="absolute inset-0 z-0" />

      {/* Foreground content */}
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-4 py-6 text-center">
        <h1 className="slide-title mb-4">Energy Self‑Sufficiency 🔋</h1>
        <p className="slide-content mb-6 max-w-2xl">
          Local trading doesn’t just save money, it helps your community rely on its own clean energy more.
          Below you can compare how much of your needs were met internally (self‑sufficiency) and how
          much renewable energy was unused (spillage).
        </p>

        <div className="flex flex-col md:flex-row gap-6 mb-6">
          {/* Self-Sufficiency Chart */}
          <div className="bg-opacity-80 p-4 rounded-lg shadow-md">
            <h2 className="text-sm font-medium mb-2">Self‑Sufficiency (%)</h2>
            <BarChart data={selfData} width={300} height={200} roundValues />
          </div>

          {/* Renewable Spillage Chart */}
          <div className="bg-opacity-80 p-4 rounded-lg shadow-md">
            <h2 className="text-sm font-medium mb-2">Renewable Spillage (kWh)</h2>
            <BarChart data={spillData} width={300} height={200} roundValues />
          </div>
        </div>

        {/* Summary bullet points */}
        <span>
          <br/>
          <ul className="list-disc list-inside mb-6 text-sm text-left max-w-lg">
            <li>
              <strong>Self‑Sufficiency</strong>: No Trading <span className="text-red-600">{baselineSelf.toFixed(1)}%</span>,
              With Trading <span className="text-green-600">{tradingSelf.toFixed(1)}%</span>
            </li>
            <li>
              <strong>Renewable Spillage</strong>:
                <ul className="list-disc list-inside mb-6 text-sm text-left max-w-lg">
                  <li className="ml-4">No Trading <span className="text-red-600">{baselineSpill.toLocaleString()} kWh</span></li>
                  <li className="ml-4">With Trading <span className="text-green-600">{tradingSpill.toLocaleString()} kWh</span></li>
                </ul>
            </li>
            <li>
              <strong>Spillage Improvement</strong>: <span className="text-green-600">{spillImprovement.toFixed(1)}%</span> reduction
            </li>
          </ul>
        </span>

        <button
          onClick={goToNext}
          className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
        >
          What about the environment?
        </button>
      </div>
    </div>
  );
};

export default ResultsSufficiency;
