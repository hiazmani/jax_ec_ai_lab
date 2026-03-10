import { useState } from 'react';
import PixiSplash from '../PixiSplash.jsx';
import { useSimulationConfig } from '../SimulationConfigContext';
import SimulationSummary from '../SimulationSummary';

const ExchangeMechanismSlide = ({ goToNext }) => {
  const { exchangeMechanism, setExchangeMechanism } = useSimulationConfig();
  const options = [
    { key: 'none', label: 'No Trading', style: 'bg-gray-200 text-gray-800' },
    { key: 'midpoint', label: 'Fixed Pricing (Midpoint)', style: 'bg-gray-200 text-gray-800' },
    { key: 'double_auction', label: 'Double Auction Marketplace', style: 'bg-gray-200 text-gray-800' },
    { key: 'agent_pricing', label: 'AI-Based Pricing (Dynamic)', style: 'bg-gray-200 text-gray-800' }
  ];

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Background */}
      <PixiSplash numRings={2} className="absolute inset-0 z-0" />

      {/* Foreground Content */}
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-4 py-6">
        <h1 className="slide-title mb-6 text-center">Choose an Exchange Mechanism</h1>
        <span className="max-w-5xl text-left bg-opacity-80 p-6 rounded-lg">
          Finally, decide <strong>how the participants will exchange energy</strong> (the market
          mechanism):
          <br/><br/>
          <ul className="list-disc list-inside space-y-3 text-sm text-left">
              <li><strong>No Trading:</strong> Each home uses its own solar, its battery, and if it’s
                  short, buys from the external grid. Houses without solar or battery are entirely
                  dependent on the grid.
              </li>
              <li><strong>Fixed Pricing:</strong> A fixed price for buying and selling energy. For
                  example, if you
                  have excess solar, you sell it to your neighbor at a fixed price of $0.10/kWh. This
                  is like a
                  simple contract.
              </li>
              <li><strong>Double Auction Marketplace:</strong> An eBay for energy every hour. Those
                  who have extra
                  will offer to sell, those who need will bid to buy, and a price is determined that
                  matches
                  supply and demand.
              </li>
              <li><strong>AI-Based Pricing:</strong> A "Community Manager" AI monitors the whole community
                  and dynamically sets an internal price to encourage sharing and minimize total grid costs.
              </li>
          </ul>

          <br/>

          Choose one of these methods for your community. You can start with “No Trading” to see what
          happens if
          everyone is isolated, and later try “Double Auction” or others to see the difference.
      </span>

        {/* Buttons */}
        <div className="flex flex-wrap gap-4 mb-8">
          {options.map(opt => (
            <button
              key={opt.key}
              onClick={() => setExchangeMechanism(opt.key)}
              className={`px-4 py-2 rounded ${opt.style} ${exchangeMechanism === opt.key ? 'ring-2 ring-offset-2 ring-green-500' : ''}`}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <br/><br/>

        {/* Summary */}
        <div className="bg-opacity-80 p-6 rounded-lg w-full max-w-5xl max-h-[40vh] overflow-y-auto">
          <SimulationSummary showEquipment={true} showDecision={true} showExchange={true} />
        </div>

        <button
          onClick={goToNext}
          className="mt-6 bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
        >
          Next: Run the Simulation
        </button>
      </div>
    </div>
  );
};

export default ExchangeMechanismSlide;
