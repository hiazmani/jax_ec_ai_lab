import { useState } from 'react';
import PixiSplash from '../PixiSplash.jsx';
import BarChart from '../BarChart';
import MultiBarChart from '../MultiBarChart';
import { useSimulationConfig } from '../SimulationConfigContext';
import { useResults } from '../ResultContext';

const ResultsCost = ({ goToNext, handleMissingData }) => {
  const { agents } = useSimulationConfig();
  const { results } = useResults();
  if (!results) return handleMissingData();
  console.log('Results:', results);

  const baselineCom = results.baseline.community_cost;
  const tradingCom = results.metrics.community_cost;
  const savingsCom = baselineCom - tradingCom;

  // Per-agent arrays (aligned with agents order)
  const baselinePer = results.baseline.per_agent_cost;
  const tradingPer = results.metrics.per_agent_cost;
  const agentLabels = agents.map(a => `${a.id} (${a.decision_making === 'RL' ? 'AI' : 'Rule'})`);

  const [tab, setTab] = useState('community'); // 'community' or 'per_agent'

  // Data for community chart
  const communityData = [
    { label: 'No Trading', value: baselineCom, color: '#EF4444' },
    { label: 'With Trading', value: tradingCom, color: '#10B981' }
  ];

  // DataSets for multi-bar
  const multiDataSets = [
    { label: 'No Trading', values: baselinePer, color: '#EF4444' },
    { label: 'With Trading', values: tradingPer, color: '#10B981' }
  ];

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Background */}
      <PixiSplash numRings={4} className="absolute inset-0 z-0" />

      {/* Content */}
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-4 py-6">
        <h1 className="slide-title mb-2 text-center">Results: Energy Cost Savings 💰</h1>
        <p className="slide-content mb-4 text-center max-w-2xl">
          Local energy trading can lower your community’s overall electricity bills by sharing surplus solar energy.
          Use the tabs below to view total savings or dive into how each household fared.
        </p>

        {/* Tabs */}
        <div className="flex space-x-4 mb-4">
          <button
            onClick={() => setTab('community')}
            className={`px-4 py-2 rounded ${tab === 'community' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
          >Community</button>
          <button
            onClick={() => setTab('per_agent')}
            className={`px-4 py-2 rounded ${tab === 'per_agent' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
          >By Agent</button>
        </div>
        <br/>

        {/* Chart Panel */}
        <div className="bg-opacity-80 p-6 rounded-lg shadow-md mb-4 flex justify-center max-h-[50vh] overflow-auto">
          {tab === 'community' ? (
            <BarChart data={communityData} width={500} height={300} />
          ) : (
            <MultiBarChart groups={agentLabels} dataSets={multiDataSets} width={700} height={400} />
          )}
        </div>

        {/* Summary */}
        {tab === 'community' ? (
          <span>
            <br/>
            <ul className="list-disc list-inside mb-6 text-sm text-center">
              <li><strong>No Trading:</strong> €{baselineCom.toLocaleString()}</li>
              <li><strong>With Trading:</strong> €{tradingCom.toLocaleString()}</li>
              <li><strong>Total Savings:</strong> €{savingsCom.toLocaleString()}</li>
            </ul>
          </span>
        ) : (
          <p className="text-sm text-center mb-6">
            Compare each household’s annual electricity cost with (green) and without (red) energy trading.
          </p>
        )}

        <button
          onClick={goToNext}
          className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
        >
          Next: Fairness Analysis
        </button>
      </div>
    </div>
  );
};

export default ResultsCost;
