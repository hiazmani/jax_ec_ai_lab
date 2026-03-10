import json  # added import

import jax
import jax.numpy as jnp

from jax_ec.data.load import load_simulation_data
from jax_ec.environment.battery import get_disabled_battery_cfg
from jax_ec.environment.env import JAXECEnvironment
from jax_ec.utils.fairness import gini

# 38 gram CO2/kWh.
EMISSION_PER_KWH = 38.0 / 1000.0  # kg CO2/kWh


class PVTradeAgent:
    """Battery-less RBC for midpoint exchange.

    * If the agent has PV surplus (net_load < 0) → offer all surplus:
        a_batt = 0, a_qty = +1
    * If the agent has deficit (net_load > 0)     → request all deficit:
        a_batt = 0, a_qty = -1
    * a_price = 0  (midpoint)
    """

    def __init__(self, obs_map):
        self.idx_cons = obs_map.CONSUMPTION
        self.idx_prod = obs_map.PRODUCTION

    def act(self, obs):
        """Act based on the current observation.

        The agent will either offer its surplus or request energy based on its net load.

        Args:
            obs: The current observation of the agent.

        Returns:
            A numpy array containing the action to be taken.
        """
        net = obs[self.idx_cons] - obs[self.idx_prod]
        a_qty = jnp.where(net > 0, -1.0, jnp.where(net < 0, +1.0, 0.0))
        # There is no battery, so we set a_batt to 0, and a_price to 0 (midpoint)
        return jnp.array([0.0, a_qty, 0.0], dtype=jnp.float32)


def simulate(env, agent_factory, num_steps):
    """Simulate the environment with the given agent factory.

    This function runs the simulation for a specified number of steps and collects the
    total community cost and grid import.

    Args:
        env: The environment to simulate.
        agent_factory: A factory function to create agents.
        num_steps: The number of steps to simulate.

    Returns:
        A dictionary containing the total community cost and grid import in kWh.
    """
    obs, state = env.reset(jax.random.PRNGKey(0))
    agents = {aid: agent_factory(env.observation_mapping) for aid in env.agents}

    total_reward = 0.0
    per_agent_total_reward = jnp.zeros(len(env.agents))
    total_grid = jnp.zeros(len(env.agents))
    peak_grid_import = 0.0
    total_renewable_spill = 0.0

    key = jax.random.PRNGKey(42)
    for _ in range(num_steps):
        acts = {aid: agents[aid].act(obs[env.id_to_index(aid)]) for aid in env.agents}
        key, sub = jax.random.split(key)
        obs, state, rew, _, info = env.step_env(sub, state, acts)

        # Update peak grid import
        peak_grid_import = jnp.maximum(peak_grid_import, info["net_grid_energy"].max())

        total_reward += rew.sum()
        per_agent_total_reward += rew
        total_grid += info["net_grid_energy"]

        total_renewable_spill += jnp.sum(info["renewable_spill"])

    # Compute the Gini index of the total reward
    community_gini = gini(per_agent_total_reward)
    # Compute the self-sustainability ratio of the community
    self_sustainability_ratio = jnp.sum(total_grid) / jnp.sum(
        env.dataset.households.consumption
    )
    # Compute the CO2 emissions
    emissions = jnp.sum(total_grid) * EMISSION_PER_KWH

    return {
        # Economics
        "community_cost": -total_reward,  # € (positive cost)
        "per_agent_cost": -per_agent_total_reward,  # € (positive cost)
        "community_gini": community_gini,
        # Energy Balance
        "grid_import_kWh": total_grid.clip(min=0).sum(),
        "grid_export_kWh": total_grid.clip(max=0).sum(),
        "peak_grid_import": peak_grid_import,
        "self_sustainability_ratio": self_sustainability_ratio,
        # Fairness
        "fairness_gini": community_gini,
        # Environment
        "emissions": emissions,
        "renewable_spill": total_renewable_spill,
    }


def run_midpoint_vs_none(dataset_path):
    """Run the simulation comparing the midpoint and no-trade mechanisms.

    This function loads the dataset, sets up the environment, and runs the simulation
    for both the midpoint and no-trade mechanisms. It then prints the results.

    Args:
        dataset_path: The path to the dataset file.

    Returns:
        None
    """
    ds = load_simulation_data(dataset_path)  # citylearn JSON etc.
    if ds.pricing.time_of_use_tariff.max() == 0:
        print("[WARNING] No ToU prices in dataset. Using default 0.25 €/kWh.")
        ds.pricing.time_of_use_tariff = jnp.full(
            ds.pricing.time_of_use_tariff.shape, 0.25
        )
    if ds.pricing.feed_in_tariff.max() == 0:
        print("[WARNING] No FiT prices in dataset. Using default 0.05 €/kWh.")
        ds.pricing.feed_in_tariff = jnp.full(ds.pricing.feed_in_tariff.shape, 0.05)

    num_steps = ds.households.consumption.shape[1]

    cfg_no_bat = {aid: get_disabled_battery_cfg() for aid in ds.households.id_to_index}

    env_no = JAXECEnvironment(ds, cfg_no_bat, "no_exchange")
    env_mid = JAXECEnvironment(ds, cfg_no_bat, "midpoint")

    res_no = simulate(env_no, PVTradeAgent, num_steps)
    res_mid = simulate(env_mid, PVTradeAgent, num_steps)

    print("--- community metrics over", num_steps, "steps ---")
    # print(res_no)
    for k in res_no:
        if not k == "per_agent_cost":
            print(f"{k:<18}  No-trade: {res_no[k]:8.2f}   Midpoint: {res_mid[k]:8.2f}")

    Δ_cost = res_no["community_cost"] - res_mid["community_cost"]
    Δ_grid = res_no["grid_import_kWh"] - res_mid["grid_import_kWh"]
    print(f"\nΔ cost  (No − Mid)  = {Δ_cost:8.2f} €")
    print(f"Δ grid kWh          = {Δ_grid:8.2f}")

    # Show the per-agent bill
    for i, aid in enumerate(env_no.agents):
        print(f"Agent {i} id: {aid}")
        print(f"\t* no_trade cost: {res_no['per_agent_cost'][i]:8.2f} €")
        print(f"\t* midpoint  cost: {res_mid['per_agent_cost'][i]:8.2f} €")

    # Save the results in a JSON file
    metrics = {
        "no_trade": {
            k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res_no.items()
        },
        "midpoint": {
            k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res_mid.items()
        },
        "delta": {
            "cost": Δ_cost.tolist(),
            "grid_kWh": Δ_grid.tolist(),
        },
    }
    with open("midpoint_trade_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "jax_ec/data/datasets/citylearn_challenge_2020_climate_zone_1.json"
    run_midpoint_vs_none(dataset_path)
