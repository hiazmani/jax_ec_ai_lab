import { useResults } from "./ResultContext.jsx";

export default function StartSimulationButton({ use_predefined, payload, goToNext }) {
    // payload is the JSON you build from the form/slide inputs
    const { setResults } = useResults();

    async function loadPredefined(scenarioName) {
        console.log("Trying to load in with the following scenario name: ", scenarioName);
        try {
            const res = await fetch(`/api/predefined/${scenarioName}`);
            const json = await res.json();
            setResults(json);                       // from ResultsContext
            localStorage.setItem("lastRun", JSON.stringify(json));
            goToNext();
        }

        catch (err) {
            console.error("Failed to load predefined scenario:", err);
            // Handle error: show a message to the user or log it
            alert("Oops, something went wrong. Could not load the scenario. Try again.");
        }

    }

    return (
        <button
            className="px-4 py-2 bg-blue-600 text-white rounded"
            onClick={() => {
                if (use_predefined) {
                    loadPredefined(payload)
                }
            }}
        >
            Run Simulation
        </button>
    );
}
