"""Flask server for handling energy community simulations."""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from config_preconfigured import PREDEFINED_SCENARIOS
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS, cross_origin

from jax_ec.data.load import load_simulation_data
from jax_ec.sim_runner import run_full_simulation

RESULT_DIR = os.environ.get("JAX_EC_RESULTS_DIR", "results")

app = Flask(__name__)
# Allow restricting origins via env var (e.g. "http://localhost:5173")
allowed_origins = os.environ.get("JAX_EC_ALLOWED_ORIGINS", "*")
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
app.config["CORS_HEADERS"] = "Content-Type"

# Directory for energy profiles
PROFILE_DIR = os.environ.get("JAX_EC_PROFILE_DIR", "server/energy_profiles")


@app.route("/api/profiles", methods=["GET"])
def list_profiles():
    """List available synthetic energy consumption profiles.

    The profiles are read from a JSON file stored locally and returned as a JSON.

    Args:
        None

    Returns:
        JSON response containing the profiles.
    """
    try:
        file_name = "synthetic_energy_profiles.json"
        with open(
            os.path.join(PROFILE_DIR, file_name), "r"
        ) as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        print(f"Error loading synthetic_energy_profiles.json: {e}")
        return jsonify({"error": "Failed to load profiles"}), 500


def get_predefined_profiles():
    """Load and cache predefined energy profiles."""
    try:
        with open(os.path.join(PROFILE_DIR, "synthetic_energy_profiles.json"), "r") as f:
            return json.load(f)["profiles"]
    except Exception as e:
        print(f"Error loading predefined profiles: {e}")
        return []


@app.route("/api/start_simulation", methods=["PUT"])
@cross_origin()
def start_simulation():
    """Start a new simulation with the provided configuration."""
    print("Starting simulation...")
    try:
        simulation_data = request.get_json()
        if not simulation_data or "agents" not in simulation_data:
            return jsonify({"error": "Invalid request: 'agents' list is required"}), 400

        profiles = get_predefined_profiles()

        # Pre-process agents: Inject profile data if missing
        for household in simulation_data["agents"]:
            if "consumption" not in household:
                # Need to look up profile
                profile_name = household.get("profile")
                if not profile_name:
                    return jsonify({"error": f"Agent {household.get('id')} must provide 'consumption' or a 'profile' name"}), 400
                
                # Find profile
                for p in profiles:
                    if p["label"] == profile_name:
                        household["profile_data"] = p
                        break
                else:
                    return jsonify({"error": f"Profile '{profile_name}' not found for agent {household.get('id')}"}), 404

        # Create JAX-EC compatible JSON
        try:
            jax_ec_data = create_jax_ec_json(simulation_data)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400

        # Load into JAX-EC dataset object
        dataset = load_simulation_data(jax_ec_data)

        # Map exchange mechanism
        exchange_map = {
            "none": "no_exchange",
            "midpoint": "midpoint",
            "agent_pricing": "agent_pricing",
            "double_auction": "double_auction"
        }
        raw_exchange = simulation_data.get("exchange_mechanism", "no_exchange")
        exchange = exchange_map.get(raw_exchange, raw_exchange)

        # Map battery config
        any_battery = any(a.get("battery", {}).get("enabled", False) for a in simulation_data["agents"])
        batteries = "default" if any_battery else "disabled"

        # Extract decision making types
        decision_making = [a.get("decision_making", "rbc") for a in simulation_data["agents"]]

        sim_cfg = {
            "dataset": dataset,
            "scenario_name": simulation_data.get("community_name", "custom_simulation"),
            "exchange": exchange,
            "batteries": batteries,
            "decision_making": decision_making,
        }

        # Run simulation
        results = run_full_simulation(sim_cfg)
        return jsonify(results)

    except Exception as e:
        print(f"Error processing simulation data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error during simulation"}), 500


def create_jax_ec_json(simulation_data):
    """Create a JAX-EC compatible JSON with validation."""
    agents = simulation_data["agents"]
    if not agents:
        raise ValueError("No agents provided in simulation data")

    # 1. Determine baseline time steps and timestamps
    # Priority: First agent's data (custom or profile)
    first_agent = agents[0]
    if "consumption" in first_agent:
        time_steps = len(first_agent["consumption"])
        # Custom agents should ideally provide timestamps in the root
        timestamps = simulation_data.get("timestamps")
        if not timestamps:
            # Generate dummy timestamps if missing
            timestamps = [(datetime(2025, 1, 1) + pd.Timedelta(hours=i)).isoformat() for i in range(time_steps)]
    else:
        # Profile-based
        pd_data = first_agent.get("profile_data")
        if not pd_data:
            raise ValueError(f"Agent {first_agent.get('id')} missing data")
        time_steps = len(pd_data["consumption"])
        timestamps = pd_data["timestamps"]

    if len(timestamps) != time_steps:
        raise ValueError(f"Timestamp count ({len(timestamps)}) does not match consumption data length ({time_steps})")

    # 2. Extract pricing
    custom_pricing = simulation_data.get("pricing", {})
    tou = custom_pricing.get("time_of_use_tariff")
    fit = custom_pricing.get("feed_in_tariff")

    if tou is None:
        tou = generate_tou(time_steps)
    if fit is None:
        fit = [t * 0.1 for t in tou]

    if len(tou) != time_steps or len(fit) != time_steps:
        raise ValueError(f"Pricing array lengths do not match time steps ({time_steps})")

    # 3. Temporal data
    hour_of_day = [datetime.fromisoformat(ts).hour for ts in timestamps]
    day_of_week = [datetime.fromisoformat(ts).weekday() for ts in timestamps]

    jax_ec_data = {
        "metadata": {
            "name": simulation_data.get("community_name", "User Community"),
            "description": "User-created community via API",
            "number_of_agents": len(agents),
            "time_resolution": "hourly",
            "time_period": {"start_date": timestamps[0], "end_date": timestamps[-1]},
        },
        "temporal": {
            "timestamps": timestamps,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "month_of_year": [1] * time_steps,
            "year": [2025] * time_steps,
        },
        "pricing": {
            "feed_in_tariff": fit,
            "time_of_use_tariff": tou,
        },
        "households": [],
    }

    # 4. Process each agent
    for agent in agents:
        # Consumption
        if "consumption" in agent:
            cons = agent["consumption"]
        else:
            cons = agent["profile_data"]["consumption"]
        
        if len(cons) != time_steps:
            raise ValueError(f"Agent {agent['id']} consumption length mismatch")

        # Production
        if "production" in agent:
            prod = agent["production"]
        else:
            # Fallback to solar generation logic
            solar_config = agent.get("solar", {})
            if solar_config.get("enabled", False):
                num_pvs = solar_config.get("number_of_pvs", 1)
                prod = generate_solar_production(timestamps, num_pvs)
            else:
                prod = [0.0] * time_steps

        if len(prod) != time_steps:
            raise ValueError(f"Agent {agent['id']} production length mismatch")

        jax_ec_data["households"].append({
            "id": agent["id"],
            "consumption": cons,
            "production": prod,
        })

    return jax_ec_data


def generate_tou(time_steps):
    """Generate a constant time-of-use tariff for the given number of time steps."""
    return [0.1] * time_steps


def generate_solar_production(timestamps, number_of_pvs):
    """Generate solar production data using PVLib for given timestamps and PV panels.

    Uses default parameters for a European location.

    Args:
        timestamps: List of ISO format timestamp strings.
        number_of_pvs: Number of PV panels.

    Returns:
        List of solar production values (kWh) for each timestamp.
    """
    # Convert timestamps to pandas DatetimeIndex
    times = pd.to_datetime(timestamps)
    # Create a Location object for a default European location
    location = pvlib.location.Location(
        latitude=50,
        longitude=8,
        tz="Europe/Berlin",
        altitude=100,
        name="Default European location",
    )
    # Calculate clear-sky irradiance using the Ineichen model
    cs = location.get_clearsky(times)  # DataFrame with 'ghi', 'dni', 'dhi'
    # Assume panel parameters
    panel_area = 1.7  # in m^2
    efficiency = 0.15  # 15% efficiency
    # Calculate production in kWh per panel: (GHI * area * efficiency) / 1000
    production_per_panel = (cs["ghi"] * panel_area * efficiency) / 1000.0
    # Total production is production per panel times number of panels
    total_production = production_per_panel * number_of_pvs
    return total_production.tolist()


@app.route("/api/visualize", methods=["GET"])
def visualize_data():
    """Visualize the energy cons. and prod. data from the JAX-EC JSON file."""
    try:
        with open("jax_ec_data.json", "r") as f:
            jax_ec_data = json.load(f)

        households = jax_ec_data["households"]
        timestamps = jax_ec_data["temporal"]["timestamps"]

        num_households = len(households)
        fig, axs = plt.subplots(num_households, 1, figsize=(15, 5 * num_households))

        if num_households == 1:
            axs = [axs]

        for idx, household in enumerate(households):
            axs[idx].plot(timestamps, household["consumption"], label="Consumption")
            axs[idx].plot(timestamps, household["production"], label="Production")
            axs[idx].set_title(f"Household {household['id']}")
            axs[idx].set_xlabel("Timestamp")
            axs[idx].set_ylabel("Energy (kWh)")
            axs[idx].legend()

        plt.tight_layout()
        plt.savefig("visualization.png")
    except Exception as e:
        print(f"Error visualizing data: {e}")
        return jsonify({"error": "Failed to visualize data"}), 500


@app.route("/api/retrieve_results", methods=["GET"])
def retrieve_results():
    """Retrieve the results of the last simulation."""
    try:
        # Assuming the results are saved in a file named 'results.json'
        with open("midpoint_trade_metrics.json", "r") as f:
            results = json.load(f)
        return jsonify(results)
    except Exception as e:
        print(f"Error retrieving results: {e}")
        return jsonify({"error": "Failed to retrieve results"}), 500


@app.route("/api/predefined", methods=["PUT"])
@cross_origin()  # Allow cross-origin requests for local dev
def get_predefined():
    """Get results for a predefined scenario.

    Retrieves the scenario name from the request URL, checks if it exists in the
    predefined scenarios, runs the simulation if results are not cached, and
    returns the results as a JSON response.

    Example: '/api/predefined?scenario=example'

    Args:
        None (scenario name is passed as a query parameter in the URL)

    Returns:
        JSON response containing the simulation results or error message
    """
    # Retrieve the scenario name from the request in the URL
    scenario = request.args.get("scenario")
    print(f"Fetching predefined scenario: {scenario}")
    if scenario not in PREDEFINED_SCENARIOS:
        print(f"Scenario {scenario} not found in predefined scenarios.")
        return jsonify({"error": "unknown scenario"}), 404
    # check if results are cached
    result_path = os.path.join(RESULT_DIR, f"{scenario}.json")
    print(f"Looking for results in {result_path}")
    if not os.path.isfile(result_path):
        # first call – compute & cache
        res = run_full_simulation(PREDEFINED_SCENARIOS[scenario])
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(result_path, "w") as fp:
            json.dump(res, fp)
    # send file (fast & sets correct mimetype)
    response = send_file(result_path, mimetype="application/json")
    return response


@app.route("/api/scenarios", methods=["GET"])
@cross_origin()
def list_scenarios():
    """List available predefined scenarios with details about agents and PV production.

    Args:
        None

    Returns:
        JSON response containing the list of scenarios with details.
    """
    scenarios_list = {}
    for name, scenario in PREDEFINED_SCENARIOS.items():
        # Extract the path of the scenario from the configuration
        result_path = os.path.join(RESULT_DIR, f"{name}.json")
        with open(result_path, "r") as f:
            scenario_data = json.load(f)

            dataset_path = scenario["dataset"]
            dataset = load_simulation_data(dataset_path)
            print(f"Loaded dataset for {name}: {dataset}")

            agent_has_pv = {}
            # Check if agent has PV production
            for a_id, a_idx in dataset.households.id_to_index.items():
                # Check if the agent has a solar panel
                if dataset.households.production[a_idx].sum() > 0:
                    h_prod = dataset.households.production[a_idx]
                    print(f"Agent {a_id} has a solar panel with production: {h_prod}")
                    agent_has_pv[a_id] = True
                else:
                    print(f"Agent {a_id} does not have a solar panel.")
                    agent_has_pv[a_id] = False

            _scenario_cfg = {
                "name": name,
                "num_agents": scenario_data.get("num_agents", 0),
                "num_timesteps": scenario_data.get("num_timesteps", ""),
                "agent_has_pv": agent_has_pv,
            }
            scenarios_list[name] = _scenario_cfg
    print(f"Final scenarios list: {scenarios_list}")
    return jsonify(scenarios=scenarios_list)


if __name__ == "__main__":
    os.makedirs(PROFILE_DIR, exist_ok=True)
    app.run(debug=True, port=5001, host="0.0.0.0")

    # # Debugging the informatio being sent to the frontend
    # test_dataset_name = "citylearn2020"
    # test_dataset_path = PREDEFINED_SCENARIOS[test_dataset_name]["dataset"]
    # # Create the dataset using the dataset name
    # dataset = load_simulation_data(test_dataset_path)
    # print(f"dataset: {dataset}")

    # # We want to know for each agent whether or not it has a solar panel
    # # and if so, how many
    # for a_id, a_idx in dataset.households.id_to_index.items():
    #     # Check if the agent has a solar panel
    #     if dataset.households.production[a_idx].sum() > 0:
    #         h_prod = dataset.households.production[a_idx]
    #         print(f"Agent {a_id} has a solar panel with production: {h_prod}")
    #     else:
    #         print(f"Agent {a_id} does not have a solar panel.")
