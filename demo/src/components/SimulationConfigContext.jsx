import React, { createContext, useContext, useEffect, useState } from 'react';

const SimulationConfigContext = createContext();
export const useSimulationConfig = () => useContext(SimulationConfigContext);

// Helpers for localStorage
const STORAGE_KEY = 'simulationConfig';

const loadFromStorage = () => {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        return raw ? JSON.parse(raw) : { agents: [], exchangeMechanism: null, communityName: 'My Amazing Community', solarPanels: 1, batteryCapacity: 10, simulationMode: 'predefined', selectedScenario: '' };
    } catch {
        return { agents: [], exchangeMechanism: null, communityName: 'My Amazing Community', solarPanels: 1, batteryCapacity: 10, simulationMode: 'predefined', selectedScenario: '' };
    }
};

const saveToStorage = (data) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
};

const calculateBatteryPrice = (capacity) => {
    return capacity * 600;
};

const calculateSolarPanelPrice = (numberOfPanels) => {
    return numberOfPanels * 4000;
};

export const SimulationConfigProvider = ({ children }) => {
    const [agents, setAgents] = useState(() => loadFromStorage().agents);
    const [exchangeMechanism, setExchangeMechanism] = useState(() => loadFromStorage().exchangeMechanism);
    const [communityName, setCommunityName] = useState(() => loadFromStorage().communityName);
    const [solarPanels, setSolarPanels] = useState(() => loadFromStorage().solarPanels);
    const [batteryCapacity, setBatteryCapacity] = useState(() => loadFromStorage().batteryCapacity);
    const [simulationMode, setSimulationMode] = useState(() => loadFromStorage().simulationMode);
    const [selectedScenario, setSelectedScenario] = useState(() => loadFromStorage().selectedScenario);

    useEffect(() => {
        saveToStorage({ agents, exchangeMechanism, communityName, solarPanels, batteryCapacity, simulationMode, selectedScenario });
    }, [agents, exchangeMechanism, communityName, solarPanels, batteryCapacity, simulationMode, selectedScenario]);

    const addAgent = (agent) => {
        setAgents(prev => [...prev, agent]);
        // Set default values for battery and solar if not provided
        agent.batteryCapacity = 10;
        agent.number_of_pvs = 1;
        agent.battery_price = calculateBatteryPrice(agent.batteryCapacity);
        agent.solar_price = calculateSolarPanelPrice(agent.number_of_pvs);
    };

    const updateAgent = (id, updates) => {
        setAgents(prev =>
            prev.map(agent => {
                if (agent.id === id) {
                    const updatedAgent = { ...agent, ...updates };
                    if (updates.battery_capacity !== undefined) {
                        updatedAgent.battery_price = calculateBatteryPrice(updates.battery_capacity);
                    } else if (agent.battery && agent.battery_capacity === undefined) {
                        updatedAgent.battery_capacity = 10;
                        updatedAgent.battery_price = calculateBatteryPrice(10);
                    }
                    if (updates.number_of_pvs !== undefined) {
                        updatedAgent.solar_price = calculateSolarPanelPrice(updates.number_of_pvs);
                    } else if (agent.solar && agent.number_of_pvs === undefined) {
                        updatedAgent.number_of_pvs = 1;
                        updatedAgent.solar_price = calculateSolarPanelPrice(1);
                    }
                    return updatedAgent;
                }
                return agent;
            })
        );
    };

    const removeAgent = (id) => {
        setAgents(prev => prev.filter(agent => agent.id !== id));
    };

    const resetConfig = () => {
        setAgents([]);
        setExchangeMechanism(null);
        setSolarPanels(0);
        setBatteryCapacity(0);
    };

    return (
        <SimulationConfigContext.Provider value={{
            agents,
            exchangeMechanism,
            communityName,
            solarPanels,
            batteryCapacity,
            simulationMode,
            selectedScenario,
            setCommunityName,
            setExchangeMechanism,
            setSolarPanels,
            setBatteryCapacity,
            setSimulationMode,
            setSelectedScenario,
            addAgent,
            updateAgent,
            removeAgent,
            resetConfig,
        }}>
            {children}
        </SimulationConfigContext.Provider>
    );
};
