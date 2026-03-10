import { useSimulationConfig } from '../../components/SimulationConfigContext';
import SimulationSummary from '../../components/SimulationSummary';
import PixiSplash from '../PixiSplash.jsx';

const DecisionMakingParticipants = ({ goToNext }) => {
  const { agents, updateAgent } = useSimulationConfig();

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Background splash could be added here, e.g. <PixiSplash /> */}
      <PixiSplash numRings={2} className="absolute inset-0 z-0" />

      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-6 py-4">
        <h1 className="slide-title mb-4">Set Decision Making: Rules or Smart Agents</h1>
        <div className="grid gap-8 w-full max-w-5xl">
          {/* First row: description and controls */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Description */}
            <div className="bg-opacity-80 p-6 rounded shadow-md">
              <p className="slide-content text-left">
                How will each participant decide when to use, store, or share energy? In this demo, you can choose between:
              </p>
              <ul className="list-disc list-inside mt-4 text-left">
                <li>
                  <strong>Rule-Based:</strong> Straightforward if-then rules. For example, "sell any solar power not
                  needed now" or "use battery power before buying from the grid." Simple and predictable.
                </li>
                <li className="mt-2">
                  <strong>AI Agent (Learning):</strong> A smarter agent that learns over time (reinforcement learning).
                  It experiments to minimize costs or maximize rewards, improving its strategy by trial and error.
                </li>
              </ul>
              <p className="italic mt-4 text-left">
                Assign either mode to each participant. Mix and match to see if smarter control makes a difference!
              </p>
            </div>

            {/* Decision controls */}
            <div className="bg-opacity-80 p-6 rounded shadow-md max-h-[60vh] overflow-y-auto">
              <h2 className="text-lg font-semibold italic mb-4 text-center">
                Choose each participant's decision mode
              </h2>
              <div className="space-y-3">
                {agents.map((agent) => (
                  <div
                    key={agent.id}
                    className="flex justify-between items-center p-3 rounded border border-stone-300"
                  >
                    <span className="font-mono text-gray-800">{agent.id}</span>
                    <select
                      value={agent.decision_making || 'RBC'}
                      onChange={(e) => updateAgent(agent.id, { decision_making: e.target.value })}
                      className="border border-gray-300 px-3 py-1 rounded"
                    >
                      <option value="RBC">Rule-Based</option>
                      <option value="Q_Learning">Simple AI Agent (Q-Learning)</option>
                      <option value="PPO">Advanced AI Agent (PPO)</option>
                    </select>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Summary below controls */}
          <div className="bg-white bg-opacity-80 p-6 rounded shadow-md w-full max-h-[40vh] overflow-y-auto">
            <SimulationSummary showEquipment={true} showDecision={true} />
          </div>

          <button
            className="mt-4 bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
            onClick={goToNext}
          >
            Finally, let’s choose how energy is exchanged!
          </button>
        </div>
      </div>
    </div>
  );
};

export default DecisionMakingParticipants;
