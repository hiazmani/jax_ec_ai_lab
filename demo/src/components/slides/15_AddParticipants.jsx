import { useState, useEffect } from 'react';
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

const AddParticipant = ({ goToNext }) => {
    const [newAgentId, setNewAgentId] = useState('');
    const [profile, setProfile] = useState('');
    const [profileOptions, setProfileOptions] = useState([]);
    const [nameError, setNameError] = useState('');
    const [fetchError, setFetchError] = useState('');
    const { agents, addAgent } = useSimulationConfig();

    useEffect(() => {
        fetch('http://127.0.0.1:5001/api/profiles')
            .then((res) => res.json())
            .then((data) => {
                setProfileOptions(data.profiles);
                if (data.profiles.length) setProfile(data.profiles[0].label);
            })
            .catch(() => setFetchError('Cannot connect to server.'));
    }, []);

    useEffect(() => {
        if (newAgentId.trim() && agents.some((a) => a.id === newAgentId.trim())) {
            setNameError('Name already taken.');
        } else {
            setNameError('');
        }
    }, [newAgentId, agents]);

    const handleAdd = () => {
        const id = newAgentId.trim();
        if (!id || nameError) return;
        addAgent({ id, profile: profile, solar: false, battery: false });
        setNewAgentId('');
    };

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const data = JSON.parse(event.target.result);
                if (data.agents && Array.isArray(data.agents)) {
                    data.agents.forEach(agent => {
                        // Support full agent objects
                        addAgent({
                            id: agent.id || `Agent_${Math.random().toString(36).substr(2, 5)}`,
                            profile: agent.profile || 'Uploaded Data',
                            consumption: agent.consumption,
                            production: agent.production,
                            solar: agent.solar?.enabled || false,
                            battery: agent.battery?.enabled || false,
                            decision_making: agent.decision_making || 'RBC'
                        });
                    });
                }
            } catch (err) {
                console.error("Failed to parse JSON:", err);
                alert("Invalid JSON file");
            }
        };
        reader.readAsText(file);
    };

    return (
        <div className="relative w-full h-screen overflow-hidden">
            {/* Background splash */}
            <PixiSplash numRings={3} className="absolute inset-0 z-0" />

            {/* Centered content */}
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-6 py-4">
                <h1 className="slide-title mb-4">Add Participants</h1>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 w-full max-w-4xl">
                    {/* Instructions & Form */}
                    <div className="bg-opacity-80 p-6 rounded shadow-md overflow-y-auto max-h-[70vh]">
                        <div className="slide-content mb-4 text-left">
                            <p>First, let’s add some members to your energy community.</p>
                            <br/>
                            <div className="border-t border-b py-4 mb-4">
                                <label className="block text-sm font-bold mb-2">Option A: Quick Add</label>
                                <div className="space-y-4">
                                    <div>
                                        <label className="block text-xs font-medium">Name:</label>
                                        <input
                                            type="text"
                                            placeholder="e.g., home_1"
                                            value={newAgentId}
                                            onChange={(e) => setNewAgentId(e.target.value)}
                                            className="w-full px-3 py-1 border rounded text-sm"
                                        />
                                        {nameError && <p className="text-red-600 text-xs mt-1">{nameError}</p>}
                                    </div>
                                    <div>
                                        <label className="block text-xs font-medium">Profile:</label>
                                        <select
                                            value={profile}
                                            onChange={(e) => setProfile(e.target.value)}
                                            className="w-full px-3 py-1 border rounded text-sm"
                                        >
                                            {profileOptions.map((p) => (
                                                <option key={p.label} value={p.label}>
                                                    {profileIcons[p.label]} {p.label}
                                                </option>
                                            ))}
                                        </select>
                                    </div>
                                    <button
                                        onClick={handleAdd}
                                        className="w-full bg-green-600 text-white py-1 rounded hover:bg-green-700 text-sm font-bold"
                                    >
                                        + Add Agent
                                    </button>
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm font-bold mb-2">Option B: Bulk Upload (.json)</label>
                                <input 
                                    type="file" 
                                    accept=".json" 
                                    onChange={handleFileUpload}
                                    className="text-xs w-full mb-2"
                                />
                                <p className="text-[10px] text-gray-500 italic">Upload a JSON with an "agents" array.</p>
                            </div>
                        </div>
                        {fetchError && <p className="text-red-600 mb-2">{fetchError}</p>}
                    </div>

                    {/* Summary with scrollable table */}
                    <div className="bg-opacity-80 p-6 rounded shadow-md w-full max-h-[70vh] overflow-y-auto">
                        <SimulationSummary />
                    </div>
                </div>

                <button
                    className="mt-6 bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
                    onClick={goToNext}
                >
                    Let’s equip the participants!
                </button>
            </div>
        </div>
    );
};

export default AddParticipant;
