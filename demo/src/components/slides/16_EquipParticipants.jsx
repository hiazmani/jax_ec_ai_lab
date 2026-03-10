import { useSimulationConfig } from '../../components/SimulationConfigContext';
import SimulationSummary from '../../components/SimulationSummary';
import PixiSplash from '../PixiSplash.jsx';

const profileIcons = {
    '9-to-5 Worker': '🏠',
    'Retired Individual': '👵🏻',
    Student: '🧑‍🏫',
    'Elementary School': '🏫',
    'Vacation Home': '🏖️',
    'Health Center': '🏥',
};

const EquipParticipants = ({ goToNext }) => {
    const { agents, updateAgent } = useSimulationConfig();

    return (
        <div className="relative w-full h-screen overflow-hidden">
            {/* Background splash */}
            <PixiSplash numRings={2} className="absolute inset-0 z-0" />

            {/* Centered content */}
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-6 py-4">
                <h1 className="slide-title mb-4 text-center">Equip Participants with Solar & Batteries</h1>

                {/* Top panels */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 w-full max-w-6xl mb-6">
                    {/* Description */}
                    <div className="bg-opacity-80 p-6 rounded shadow-md text-left">
                        <p className="slide-content mb-4">
                            Decide which participants have renewable energy or storage:
                        </p>
                        <ul className="list-disc list-inside space-y-2">
                            <li><strong>Solar Panels:</strong> Check for homes with solar. They generate power by day. No solar → rely on community or grid.</li>
                            <li><strong>Battery Storage:</strong> Check for homes with batteries. Store excess energy to use later. No battery → excess solar must be shared or sent to grid.</li>
                        </ul>
                        <p className="italic mt-4">
                            Mix and match: e.g., School gets solar only, Family Home gets both, Home Office only solar.
                        </p>
                    </div>

                    {/* Equipment selection */}
                    <div className="bg-opacity-80 p-6 rounded shadow-md max-h-[40vh] overflow-y-auto">
                        <h2 className="text-lg font-semibold mb-4">Equip your participants</h2>
                        <div className="space-y-6">
                            {agents.map((agent) => (
                                <div key={agent.id} className="border-b border-gray-300 pb-4">
                                    <h3 className="font-medium mb-2">{agent.id}</h3>

                                    {/* Solar */}
                                    <label className="flex items-center mb-2">
                                        <input
                                            type="checkbox"
                                            checked={agent.solar}
                                            onChange={() => updateAgent(agent.id, { solar: !agent.solar })}
                                            className="mr-2"
                                        />
                                        ☀️ Solar Panels
                                    </label>
                                    <div className={`flex items-center space-x-3 mb-4 ${!agent.solar ? 'opacity-40 pointer-events-none' : ''}`}>
                                        <input
                                            type="range"
                                            min="1"
                                            max="40"
                                            value={agent.number_of_pvs}
                                            onChange={(e) => updateAgent(agent.id, { number_of_pvs: +e.target.value })}
                                            className="flex-1"
                                        />
                                        <span>{agent.number_of_pvs} panels</span>
                                        <span className="text-gray-500">(~€{agent.solar_price.toLocaleString()})</span>
                                    </div>

                                    {/* Battery */}
                                    <label className="flex items-center mb-2">
                                        <input
                                            type="checkbox"
                                            checked={agent.battery}
                                            onChange={() => updateAgent(agent.id, { battery: !agent.battery })}
                                            className="mr-2"
                                        />
                                        🔋 Battery
                                    </label>
                                    <div className={`flex items-center space-x-3 ${!agent.battery ? 'opacity-40 pointer-events-none' : ''}`}>
                                        <input
                                            type="range"
                                            min="3"
                                            max="20"
                                            step="0.5"
                                            value={agent.battery_capacity}
                                            onChange={(e) => updateAgent(agent.id, { battery_capacity: +e.target.value })}
                                            className="flex-1"
                                        />
                                        <span>{agent.battery_capacity} kWh</span>
                                        <span className="text-gray-500">(~€{agent.battery_price.toLocaleString()})</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Bottom Summary */}
                <div className="w-full max-w-6xl bg-white bg-opacity-80 p-6 rounded shadow-md max-h-[30vh] overflow-y-auto">
                    <SimulationSummary showEquipment={true} />
                </div>

                <button
                    className="mt-6 bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
                    onClick={goToNext}
                >
                    Let’s set how they decide!
                </button>
            </div>
        </div>
    );
};

export default EquipParticipants;
