import { useState } from 'react';
import PixiSplash from '../PixiSplash.jsx';
import { useSimulationConfig } from '../SimulationConfigContext';
import { useResults } from '../ResultContext';
import SimulationSummary from '..//SimulationSummary';

const RunSimulation = ({ goToNext }) => {
  const [isLoading, setIsLoading] = useState(false);
  const { agents, exchangeMechanism, communityName, simulationMode, selectedScenario } = useSimulationConfig();
  const { setResults } = useResults();

  const startSimulation = async () => {
    setIsLoading(true);
    const url = simulationMode === 'predefined'
      ? `http://127.0.0.1:5001/api/predefined?scenario=${selectedScenario}`
      : 'http://127.0.0.1:5001/api/start_simulation';

    const payload = simulationMode === 'predefined'
      ? null
      : JSON.stringify({
          agents: agents.map(a => ({
            id: a.id,
            profile: a.profile,
            decision_making: a.decision_making === 'RL' ? 'ppo' : 'rbc',
            solar: { enabled: a.solar, number_of_pvs: a.solar ? a.number_of_pvs : 0 },
            battery: { enabled: a.battery, capacity: a.battery ? a.battery_capacity : 0 },
          })),
          exchange_mechanism: exchangeMechanism || 'none',
          community_name: communityName,
        });

    try {
      const res = await fetch(url, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
      });
      if (res.ok) {
        const data = await res.json();
        setResults(data);
        goToNext();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Background Decoration */}
      <PixiSplash numRings={2} className="absolute inset-0 z-0" />

      {/* Foreground Content */}
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-4 py-6">
        <h1 className="slide-title mb-4 text-center">Run the Simulation!</h1>
        <p className="slide-content mb-6 text-center max-w-2xl">
          It’s showtime—simulate a typical day (24h) hour by hour. Watch how each participant uses energy,
          solar generation, battery charge/discharge, and grid interactions.
        </p>

        <div className="bg-white bg-opacity-80 p-6 rounded-lg shadow-md w-full max-w-5xl max-h-[40vh] overflow-y-auto mb-6">
          <SimulationSummary showEquipment showDecision showExchange />
        </div>

        <button
          onClick={startSimulation}
          disabled={isLoading}
          className="w-full max-w-sm bg-blue-600 hover:bg-blue-700 text-white py-2 rounded shadow focus:outline-none disabled:opacity-50"
        >
          {isLoading ? 'Running Simulation...' : 'Start Simulation'}
        </button>

        {isLoading && <p className="mt-4 text-gray-600">Simulation is running, please wait...</p>}
      </div>
    </div>
  );
};

export default RunSimulation;
