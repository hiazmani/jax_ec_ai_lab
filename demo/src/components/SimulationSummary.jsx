import { useSimulationConfig } from './SimulationConfigContext';

const profileIcons = {
    "9-to-5 Worker": "🏠",
    "Retired Individual": "👵🏻",
    "Student": "🧑‍🏫",
    "Elementary School": "🏫",
    "Vacation Home": "🏖️",
    "Health Center": "🏥",
};

const SimulationSummary = ({ showMechanism = false, showEquipment = false, showDecision = false, showDownload = false }) => {
    const { agents, exchangeMechanism, removeAgent, communityName } = useSimulationConfig();

    const handleDownload = () => {
        const data = {
            agents: agents.map(agent => ({
                id: agent.id,
                profile: agent.profile,
                decision_making: agent.decision_making,
                solar: {
                    enabled: agent.solar,
                    number_of_pvs: agent.solar ? (agent.number_of_pvs) : 0,
                },
                battery: {
                    enabled: agent.battery,
                    capacity: agent.battery ? (agent.battery_capacity) : 0,
                },
            })),
            exchange_mechanism: exchangeMechanism || 'none',
            community_name: communityName
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'simulation_config.json';
        link.click();
        URL.revokeObjectURL(url);
    };

    if (agents.length === 0) {
        return (
            <div className="border p-4 mb-6 rounded-lg border-brown-600 bg-gray-200">
                <h2 className="font-semibold mb-2 text-lg">Your Energy Community - Summary</h2>
                <p className="text-gray-500">No participants added yet.</p>
            </div>
        );
    }

    return (
        <div className="bg-background p-4 rounded shadow text-sm overflow-x-auto">
            <div className="flex items-center justify-between mb-2">
                <h2 className="font-semibold text-lg">Your Energy Community - Summary</h2>
                {showDownload && (
                    <button
                        onClick={handleDownload}
                        className="text-sm text-blue-700 border border-blue-300 rounded px-3 py-1 hover:bg-blue-100"
                    >
                        ⬇️ Download Community
                    </button>
                )}
            </div>

            <br/>

            <p className="text-gray-500 mb-2 text-left">Here’s a summary of your community: <i><span className="font-bold">{communityName}</span></i></p>

            {showMechanism && (
                <div className="mb-4 text-left">
                    <strong>Exchange Mechanism:</strong>
                    <p className="text-gray-500">You have chosen: <span className="font-bold">{exchangeMechanism || 'None selected'}</span></p>
                </div>
            )}

            <br/>

            <table className="table-auto w-full text-center border-collapse">
                <thead>
                <tr className="text-xs text-gray-700 border-b">
                    <th className="py-1 pr-4">Name</th>
                    <th className="py-1 pr-4">Consumption Profile</th>
                    {showEquipment && <th className="py-1 pr-4">Solar</th>}
                    {showEquipment && <th className="py-1 pr-4">Battery</th>}
                    {showEquipment && <th className="py-1 pr-4">Solar Panels</th>}
                    {showEquipment && <th className="py-1 pr-4">Battery Capacity</th>}
                    {showDecision && <th className="py-1 pr-4">Decision</th>}
                    <th className="py-1 pr-4">Remove</th>
                </tr>
                </thead>
                <tbody>
                {agents.map((agent) => (
                    <tr key={agent.id} className="border-t">
                        <td className="py-1 pr-4 font-mono text-sm border-r border-gray-200">{agent.id}</td>
                        <td className="py-1 pr-4 border-r border-gray-200">
                            {agent.consumption ? '📊 Uploaded Data' : (profileIcons[agent.profile] ? `${profileIcons[agent.profile]} ${agent.profile}` : agent.profile)}
                        </td>
                        {showEquipment && <td className="py-1 pr-4 border-r border-gray-200">{agent.solar ? '✅' : '❌'}</td>}
                        {showEquipment && <td className="py-1 pr-4 border-r border-gray-200">{agent.battery ? '✅' : '❌'}</td>}
                        {showEquipment && <td className="py-1 pr-4 border-r border-gray-200">{agent.solar ? agent.number_of_pvs : '❌'}</td>}
                        {showEquipment && <td className="py-1 pr-4 border-r border-gray-200">{agent.battery ? agent.battery_capacity : '❌'}</td>}
                        {showDecision && <td className="py-1 pr-4 border-r border-gray-200">
                            {agent.decision_making === 'RL' ? 'AI Agent (Learning)' : 'Rule-Based'}
                        </td>}
                        <td className="py-1 pr-4">
                            <button
                                onClick={() => removeAgent(agent.id)}
                                className="text-red-600 hover:text-red-800 font-bold align-top"
                            >
                                ✕
                            </button>
                        </td>
                    </tr>
                ))}
                </tbody>
            </table>
        </div>
    );
};

export default SimulationSummary;
