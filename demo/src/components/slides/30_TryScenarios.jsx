import React from 'react';
import PixiSplash from '../PixiSplash.jsx';
import { useSimulationConfig } from '../SimulationConfigContext';

const TryScenarios = ({ goToNext }) => {
  const { resetConfig } = useSimulationConfig();

  const handleReset = () => {
    if (window.confirm('Are you sure you want to reset your community and start fresh?')) {
      resetConfig();
      goToNext();
    }
  };

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Background rings */}
      <PixiSplash numRings={4} className="absolute inset-0 z-0" />

      {/* Foreground content */}
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-6 py-4 text-center">
        <h1 className="slide-title mb-4">Try Different Scenarios</h1>
        <p className="slide-content mb-6 max-w-3xl">
          Congratulations on running your first simulation! Now, make this sandbox your own by experimenting with different setups:
        </p>

        <ul className="list-disc list-inside mb-8 text-left max-w-3xl space-y-3">
          <li><strong>Add or Modify Participants:</strong> Mix homes, businesses, schools – see how diversity changes outcomes.</li>
          <li><strong>Swap Equipment:</strong> Try everyone with solar, or share one big battery. How does that shift energy flows?</li>
          <li><strong>Switch Decision Strategies:</strong> Compare rule-based vs AI-managed agents to see which intelligently adapt best.</li>
          <li><strong>Explore Trading Mechanisms:</strong> From no trading, to fixed prices, to double auctions – each market design tells a new story.</li>
        </ul>

        <div className="flex space-x-4 mb-6">
          <button
            onClick={goToNext}
            className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
          >
            Enough for now, let's finish up!
          </button>
          <button
            onClick={handleReset}
            className="bg-gray-200 text-gray-800 px-6 py-2 rounded hover:bg-gray-300"
          >
            Reset & Start Over
          </button>
        </div>

        <p className="text-sm text-gray-700">
          <br/>
          This is your energy sandbox—explore freely and discover the most sustainable, fair, and efficient community setups!
        </p>
      </div>
    </div>
  );
};

export default TryScenarios;
