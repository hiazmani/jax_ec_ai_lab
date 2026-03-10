"""Try to run a simulation from the demo data and check if it runs without errors."""

import chex
import jax
import jax.numpy as jnp
import jax.random as jr

from jax_ec.data.load import load_simulation_data
from jax_ec.environment.env import JAXECEnvironment


def evaluate(
    eval_key: chex.PRNGKey,
    eval_env: JAXECEnvironment,
    num_steps: int = 168,
) -> None:
    """Evaluate the agents in the environment.

    Args:
        eval_key: A PRNGKey for random number generation.
        eval_env: The JAXECEnvironment to evaluate.
        num_steps: Number of steps to evaluate for. Default is 168 (1 week)

    Returns:
        A tuple containing:
            - net_energy: Array of net energy for each agent at each timestep.
            - rewards: Array of rewards for each agent at each timestep.
            - info: Additional info from the environment at each timestep.
    """
    eval_key, reset_key = jr.split(eval_key)
    obs, state = env.reset(reset_key)

    timestep_keys = jr.split(eval_key, num_steps)

    def single_eval_step(eval_state, step_key):
        step, obs, state = eval_state

        actions = jnp.zeros(env.num_agents)
        next_obs, next_state, rewards, dones, info = eval_env.step(
            step_key, state, actions
        )

        prod = eval_env.dataset.households.production[:, step]
        cons = eval_env.dataset.households.consumption[:, step]
        battery_action = actions * state.batteries.nominal_power
        net_energy = (cons - prod) - battery_action

        return (step + 1, next_obs, next_state), (net_energy, rewards, info)

    _, eval_outputs = jax.lax.scan(single_eval_step, (0, obs, state), timestep_keys)
    return eval_outputs


if __name__ == "__main__":
    key = jr.PRNGKey(0)
    # Split the key for reproducibility
    key, _key = jr.split(key, 2)

    # Load the data
    demo_file_path = "tests/data/jax_ec_data.json"
    data = load_simulation_data(demo_file_path)
    env = JAXECEnvironment(data)
    obs_mapping = (
        env.observation_mapping
    )  # Observation mapping tells us the order of the observations

    # Create the environment
    env = JAXECEnvironment(data)
    res = evaluate(eval_key=key, eval_env=env, num_steps=168)  # 1 week of timesteps
    print(res)
