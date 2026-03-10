import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jax_ec.data.load import load_simulation_data
from jax_ec.environment.battery import get_default_battery_cfg, get_disabled_battery_cfg
from jax_ec.environment.env import JAXECEnvironment


class SimpleSurplusAgent:
    """Charges when surplus (prod > cons), discharges when deficit (prod < cons)."""

    def __init__(self, obs_mapping):
        self.idx_prod = obs_mapping.PRODUCTION
        self.idx_cons = obs_mapping.CONSUMPTION
        self.idx_soc = obs_mapping.BATTERY_SOC
        self.idx_cap = obs_mapping.BATTERY_CAP

    def act(self, obs):
        """Act based on the current observation.

        The agent will charge the battery when there is surplus production and discharge
        the battery when there is a deficit.

        Args:
            obs: The current observation of the agent.

        Returns:
            A numpy array containing the action to be taken.
        """
        prod = obs[self.idx_prod]
        cons = obs[self.idx_cons]
        soc = obs[self.idx_soc]
        cap = obs[self.idx_cap]

        net = prod - cons

        charge = jnp.clip(net, 0, cap * (1 - soc)) / cap if cap > 0 else 0.0
        discharge = -jnp.clip(-net, 0, cap * soc) / cap if cap > 0 else 0.0

        a_batt = jnp.where(net >= 0, -charge, -discharge)
        return jnp.array([a_batt, 0.0, 0.0], dtype=jnp.float32)


def run_simulation(env, agent_factory, num_steps):
    """Run a simulation with the given environment and agent factory.

    Args:
        env: The environment to simulate.
        agent_factory: A factory function to create agents.
        num_steps: The number of steps to simulate.

    Returns:
        rewards: A dictionary mapping agent IDs to their rewards over time.
    """
    obs, state = env.reset(jax.random.PRNGKey(0))
    agent_ids = list(env.dataset.households.id_to_index.keys())
    agents = {
        aid: agent_factory(obs[env.dataset.households.id_to_index[aid]])
        for aid in agent_ids
    }

    rewards = {aid: np.zeros(num_steps) for aid in agent_ids}
    residuals = {aid: np.zeros(num_steps) for aid in agent_ids}
    consumptions = {aid: np.zeros(num_steps) for aid in agent_ids}
    productions = {aid: np.zeros(num_steps) for aid in agent_ids}

    key = jax.random.PRNGKey(42)
    for t in range(num_steps):
        actions = {
            aid: agents[aid].act(obs[env.dataset.households.id_to_index[aid]])
            for aid in agent_ids
        }
        key, sub = jax.random.split(key)
        obs, state, rew, dones, info = env.step(sub, state, actions)

        for aid in agent_ids:
            idx = env.dataset.households.id_to_index[aid]
            rewards[aid][t] = rew[idx]
            residuals[aid][t] = info["residual"][idx]
            consumptions[aid][t] = obs[idx][env.observation_mapping.CONSUMPTION]
            productions[aid][t] = obs[idx][env.observation_mapping.PRODUCTION]

    return rewards, residuals, consumptions, productions


def test_hypothesis_battery_vs_no_battery():
    """Test the hypothesis that batteries reduce costs and residuals.

    Hypothesis:
    1. Agents w/ batteries will have lower total costs than agents without batteries.
    2. Agents w/ batteries will have lower residuals than agents without batteries.
    3. The diff in costs/residuals will be more pronounced in scenarios with high PV.
    This function runs two simulations: one with/without batteries

    Args:
        None

    Returns:
        None
    """
    # dataset_name = "jax_ec/data/datasets/citylearn_challenge_2020_climate_zone_1"
    dataset_name = "jax_ec/data/datasets/citylearn_challenge_2023_phase_1.json"
    dataset = load_simulation_data(dataset_name)
    # Set the FiT to 0.0 for the whole dataset
    # dataset.pricing.feed_in_tariff = jnp.zeros_like(dataset.pricing.feed_in_tariff)
    # num_steps = dataset.households.consumption.shape[1]
    num_steps = 24  # One week of data
    agent_ids = list(dataset.households.id_to_index.keys())

    # Config with battery for all agents
    battery_cfgs = {aid: get_default_battery_cfg() for aid in agent_ids}
    env_batt = JAXECEnvironment(
        dataset=dataset,
        battery_configs=battery_cfgs,
    )

    # Config with no batteries
    disabled_cfgs = {aid: get_disabled_battery_cfg() for aid in agent_ids}
    env_no_batt = JAXECEnvironment(
        dataset=dataset,
        battery_configs=disabled_cfgs,
    )

    # Run both simulations
    rewards_batt, residuals_batt, cons_batt, prod_batt = run_simulation(
        env_batt,
        lambda obs: SimpleSurplusAgent(env_batt.observation_mapping),
        num_steps,
    )
    rewards_nobatt, residuals_nobatt, cons_nobatt, prod_nobatt = run_simulation(
        env_no_batt,
        lambda obs: SimpleSurplusAgent(env_no_batt.observation_mapping),
        num_steps,
    )
    print("[SUMMARY] Simulation completed.")
    print("[SUMMARY] Rewards collected.")

    print(" * Rewards of agents with battery: ", rewards_batt)
    print(" * Rewards of agents without battery: ", rewards_nobatt)

    # Plotting
    fig, axs = plt.subplots(len(agent_ids), 1, figsize=(6, len(agent_ids) * 2.5))
    if len(agent_ids) == 1:
        axs = [axs]

    for i, aid in enumerate(agent_ids):
        axs[i].plot(rewards_nobatt[aid], label="No Battery", linestyle="--")
        axs[i].plot(rewards_batt[aid], label="Battery")
        axs[i].set_title(f"Agent {aid} Reward Over Time")
        axs[i].set_ylabel("Reward")
        axs[i].legend()
        print(f"[SUMMARY] Plotting rewards for Agent {aid}.")
        print(f"\t* No Battery total reward: {rewards_nobatt[aid].sum():.2f}")
        print(f"\t* Battery    total reward: {rewards_batt[aid].sum():.2f}")

    plt.xlabel("Timestep")
    plt.tight_layout()
    plt.savefig("battery_vs_no_battery.png")

    fig, axs = plt.subplots(len(agent_ids), 1, figsize=(6, len(agent_ids) * 2.5))
    if len(agent_ids) == 1:
        axs = [axs]
    for i, aid in enumerate(agent_ids):
        axs[i].plot(residuals_nobatt[aid], label="No Battery", linestyle="--")
        axs[i].plot(residuals_batt[aid], label="Battery")
        axs[i].plot(cons_batt[aid], label="Consumption", color="orange", linestyle=":")
        axs[i].plot(prod_batt[aid], label="Production", color="green", linestyle=":")
        axs[i].set_title(f"Agent {aid} Residual Over Time")
        axs[i].set_ylabel("Residual")
        axs[i].legend()

        print(f"[SUMMARY] Plotting residuals for Agent {aid}.")
        print(f"\t* No Battery total residual: {residuals_nobatt[aid].sum():.2f}")
        print(f"\t* Battery    total residual: {residuals_batt[aid].sum():.2f}")

    plt.xlabel("Timestep")
    plt.tight_layout()
    plt.savefig("battery_vs_no_battery_residuals.png")


if __name__ == "__main__":
    test_hypothesis_battery_vs_no_battery()
