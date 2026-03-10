from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr

from jax_ec.agents.api import init_agent_state
from jax_ec.data.load import load_simulation_data
from jax_ec.environment.battery import (
    get_default_battery_cfg,
    get_disabled_battery_cfg,
)
from jax_ec.environment.env import JAXECEnvironment
from jax_ec.simulate import EMISSION_PER_KWH, compute_metrics, simulate_vectorized


# ---------------------------------------------------------------------
# agent helper --------------------------------------------------------
# ---------------------------------------------------------------------
def _get_agent_types(cfg: dict, num_agents: int) -> list[str]:
    """Extract agent types from config.

    Supports:
    - agents: "rbc" (string, applies to all)
    - decision_making: ["rbc", "ppo", ...] (list of types)
    """
    if "decision_making" in cfg and isinstance(cfg["decision_making"], list):
        return cfg["decision_making"]

    # Fallback to 'agents' string
    base_type = cfg.get("agents", "rbc")
    # map common names
    mapping = {
        "pv_rbc": "RBC",
        "rbc": "RBC",
        "ppo": "PPO",
        "q_learning": "Q_LEARNING",
    }
    t = mapping.get(base_type.lower(), "RBC")
    return [t] * num_agents


# ---------------------------------------------------------------------
# battery cfg registry -------------------------------------------------
# ---------------------------------------------------------------------
def build_battery_cfg_map(dataset, mode: str):
    """Returns a dict mapping agent IDs to battery config dicts.

    Args:
        dataset: The dataset containing household information.
        mode (str): The mode for battery configuration. Can be "disabled" or "default".

    Returns:
        dict: A dictionary mapping agent IDs to battery configuration dictionaries.
    """
    if mode == "disabled":
        cfg_builder = get_disabled_battery_cfg
    elif mode == "default":
        cfg_builder = get_default_battery_cfg
    else:
        raise ValueError(f"Unknown battery mode: {mode}")
    return {aid: cfg_builder() for aid in dataset.households.id_to_index}


# ---------------------------------------------------------------------
# main entry -----------------------------------------------------------
# ---------------------------------------------------------------------
def run_full_simulation(cfg: dict, compute_baseline: bool = True) -> dict:
    """Returns metrics dict compatible with the slide deck.

    Args:
        cfg (dict): Configuration dictionary with the following keys:
            - dataset (str or JAXECDataset): Path to the dataset JSON file or the dataset object itself.
            - exchange (str): Exchange mechanism: 'midpoint_price'/'no_exchange'.
            - batteries (str): Battery configuration mode: 'disabled'/'default'.
            - agents (str): Type of agents to use, e.g., 'pv_rbc'.
            - decision_making (list[str]): Optional list of agent types for each building.
        compute_baseline (bool): Compute a baseline simulation with no exchange.

    Returns:
        dict: A dictionary containing simulation results and metrics.
    """
    # 1. Load dataset ---------------------------------------------------
    if isinstance(cfg["dataset"], str):
        ds = load_simulation_data(cfg["dataset"])
        scenario_name = Path(cfg["dataset"]).stem
    else:
        ds = cfg["dataset"]
        scenario_name = cfg.get("scenario_name", "custom_simulation")

    # fallback flat tariffs if missing
    if ds.pricing.time_of_use_tariff.max() == 0:
        ds.pricing.time_of_use_tariff = jnp.full_like(
            ds.pricing.time_of_use_tariff, 0.25
        )
    if ds.pricing.feed_in_tariff.max() == 0:
        ds.pricing.feed_in_tariff = jnp.full_like(ds.pricing.feed_in_tariff, 0.05)

    steps = ds.households.consumption.shape[1]
    num_agents = len(ds.households.id_to_index)

    # 2. Batteries ------------------------------------------------------
    bat_cfg = build_battery_cfg_map(ds, cfg["batteries"])

    # 3. Agents ---------------------------------------------------------
    agent_types = _get_agent_types(cfg, num_agents)
    rng = jr.PRNGKey(cfg.get("seed", 42))

    # 4. Simulation for chosen mechanism --------------------------------
    # Pricing agent logic is only active for agent_pricing mechanism
    has_pricing_agent = (cfg["exchange"] == "agent_pricing")
    
    env_mech = JAXECEnvironment(
        ds, bat_cfg, cfg["exchange"], 
        allow_grid_charging=False,
        has_pricing_agent=has_pricing_agent
    )
    # initialize agent state with real obs_dim from env
    # observation_spaces is {aid: (low, high, shape)}
    obs_dim = next(iter(env_mech.observation_spaces.values()))[2][0]
    
    # Pre-trained weight paths
    WEIGHT_DIR = Path(__file__).parent / "agents" / "weights"
    build_weights = WEIGHT_DIR / "building_ppo_v1.pkl"
    man_weights = WEIGHT_DIR / "manager_ppo_v1.pkl"
    
    w_path = str(build_weights) if build_weights.exists() else None
    m_path = str(man_weights) if man_weights.exists() else None

    astate = init_agent_state(
        rng, agent_types, obs_dim, 
        has_pricing_agent=has_pricing_agent,
        weight_path=w_path,
        manager_weight_path=m_path
    )

    res_mech_raw = simulate_vectorized(env_mech, astate, rng, steps)
    res_mech = compute_metrics(env_mech, res_mech_raw)

    # 5. Optional baseline (no exchange) --------------------------------
    res_base = None
    if compute_baseline and cfg["exchange"] != "no_exchange":
        env_base = JAXECEnvironment(
            ds, bat_cfg, "no_exchange",
            has_pricing_agent=has_pricing_agent
        )
        res_base_raw = simulate_vectorized(env_base, astate, rng, steps)
        res_base = compute_metrics(env_base, res_base_raw)

    # 6. Assemble return dict ------------------------------------------
    out = {
        "scenario": scenario_name,
        "num_agents": num_agents,
        "num_timesteps": steps,
        "mechanism": cfg["exchange"],
        "metrics": res_mech,
    }

    if res_base:
        out["baseline"] = res_base
        # Compute deltas for a few common keys
        out["delta"] = {
            "community_cost": res_base["community_cost"] - res_mech["community_cost"],
            "grid_import_kWh": res_base["grid_import_kWh"]
            - res_mech["grid_import_kWh"],
            "emissions_kg": (res_base["emissions"] - res_mech["emissions"]),
        }

    return _to_python(out)


# ---------------------------------------------------------------------
# utils ----------------------------------------------------------------
# ---------------------------------------------------------------------
def _to_python(obj):
    """Recursively convert JAX / numpy scalars to Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj
