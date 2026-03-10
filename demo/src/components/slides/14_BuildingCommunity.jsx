import { useEffect, useState } from 'react';
import { useSimulationConfig } from '../SimulationConfigContext';
import PixiSplash from '../PixiSplash.jsx';

const BuildingCommunity = ({ goToScenarioCreation, goToAgentSelection }) => {
  const [scenarios, setScenarios] = useState({});
  const {
    communityName,
    setCommunityName,
    setSimulationMode,
    selectedScenario,
    setSelectedScenario,
    resetConfig,
    addAgent
  } = useSimulationConfig();

  const handleSimulateScenario = () => {
    if (selectedScenario) {
      resetConfig();
      const info = scenarios[selectedScenario];
      for (let i = 0; i < info.num_agents; i++) {
        addAgent({ id: `Agent_${i+1}` });
      }
      setSimulationMode('predefined');
      goToAgentSelection();
    } else {
      alert('Please select a scenario first.');
    }
  };

  useEffect(() => {
    fetch('http://127.0.0.1:5001/api/scenarios')
      .then(res => res.json())
      .then(data => setScenarios(data.scenarios))
      .catch(console.error);
  }, []);

  const handleCreateCommunity = () => {
    // Start fresh when entering “Create Your Own” mode
    resetConfig();
    if (communityName) {
      setSimulationMode('custom');
      // Clear any previously selected scenario
      setSelectedScenario('');
      goToScenarioCreation();
    } else {
      alert('Please enter a community name.');
    }
  };

  const current = scenarios[selectedScenario];

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Background Decoration */}
      <PixiSplash numRings={2} className="absolute inset-0 z-0" />

      {/* Centered Content */}
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-4 py-6">
        <h1 className="slide-title mb-4 text-center">Explore Energy Communities! 🏠🏡🏘️</h1>
        <p className="slide-content mb-6 text-center max-w-2xl">
          Now that we've learned the basics, it's time to <strong>try it yourself</strong>. Use real or custom data to see how energy communities
          work and help us use energy more efficiently.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-6xl">
          {/* Option A */}
          <div className=" bg-opacity-80 p-6 rounded-lg shadow-md flex flex-col">
            <h2 className="text-lg font-semibold mb-2">Option A: Predefined Scenarios</h2>
            <p className="text-sm text-gray-700 flex-grow">
              Explore real-world datasets from existing energy communities used by researchers and companies.
              Quickly see how different sharing strategies perform under realistic conditions.
            </p>
            <label className="text-sm font-medium mt-4">Choose a Scenario:</label>
            <select
              value={selectedScenario || ''}
              onChange={e => setSelectedScenario(e.target.value)}
              className="mt-1 px-2 py-2 border rounded w-full"
            >
              <option value="" disabled>Select scenario</option>
              {Object.keys(scenarios).map(name => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
            {current && (
              <div className="mt-3 text-xs bg-blue-50 p-2 rounded border border-blue-200">
                <div><strong>Agents:</strong> {current.num_agents}</div>
                <div><strong>Steps:</strong> {current.num_steps}</div>
              </div>
            )}
            <button
              onClick={handleSimulateScenario}
              disabled={!selectedScenario}
              className="mt-4 bg-blue-600 text-white py-2 rounded hover:bg-blue-700 disabled:bg-gray-400"
            >Simulate Scenario</button>
          </div>

          {/* Option B */}
          <div className="bg-opacity-80 p-6 rounded-lg shadow-md flex flex-col">
            <h2 className="text-lg font-semibold mb-2">Option B: Create Your Own Community</h2>
            <p className="text-sm text-gray-700 flex-grow">
              Build a custom community: add typical profiles, solar panels, and batteries. Experiment with rules
              and see energy flows in a sandbox environment.
            </p>
            <label className="text-sm font-medium mt-4">Community Name:</label>
            <input
              type="text"
              value={communityName}
              onChange={e => setCommunityName(e.target.value)}
              placeholder="My Community"
              className="mt-1 px-3 py-2 border rounded w-full"
            />
            <button
              onClick={handleCreateCommunity}
              className="mt-4 bg-green-600 text-white py-2 rounded hover:bg-green-700"
            >Start Building</button>
          </div>

          {/* Option C */}
          <div className="bg-opacity-80 p-6 rounded-lg shadow-md flex flex-col">
            <h2 className="text-lg font-semibold mb-2">Option C: Upload Your Own Data</h2>
            <p className="text-sm text-gray-700 flex-grow">
              Have your own community's energy data? Upload a file to simulate your real-life community and
              explore new sharing strategies tailored to your dataset.
            </p>
            <input type="file" className="mt-2" />
            <button
              onClick={() => alert('Data upload feature coming soon!')}
              className="mt-4 bg-yellow-500 text-white py-2 rounded hover:bg-yellow-600"
            >Explore your data!</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BuildingCommunity;
