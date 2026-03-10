"""Contains a simple simulation function.

Runs a given scenario for one week (168 hours) and returns the results.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flax import struct
from omegaconf import DictConfig
from rich.pretty import pprint

from jax_ec.agents.api import select_actions_generic
from jax_ec.agents.types import AgentState
from jax_ec.environment import setup_environment
from jax_ec.environment.env import JAXECEnvironment, ObsIdx
from jax_ec.utils.fairness import gini

# 38 gram CO2/kWh.
EMISSION_PER_KWH = 38.0 / 1000.0  # kg CO2/kWh


@struct.dataclass
class SimulationResult:
    """Results of a vectorized simulation."""

    obs: jnp.ndarray  # (T, N, obs_dim)
    actions: jnp.ndarray  # (T, N, 3)
    rewards: jnp.ndarray  # (T, N)
    dones: jnp.ndarray  # (T, N)
    net_grid_energy: jnp.ndarray  # (T, N)
    renewable_spill: jnp.ndarray  # (T, N)


def simulate_vectorized(
    env: JAXECEnvironment,
    agent_state: AgentState,
    rng: jax.Array,
    num_steps: int = None,
) -> SimulationResult:
    """Run a vectorized simulation using jax.lax.scan.

    Args:
        env: The environment.
        agent_state: The initial agent state (batch-first).
        rng: Random key.
        num_steps: Number of steps (defaults to env.num_timesteps - 1).

    Returns:
        SimulationResult: History of the simulation.
    """
    if num_steps is None:
        num_steps = env.num_timesteps - 1

    obs_mapping = env.observation_mapping
    k_reset, k_scan = jr.split(rng)
    obs0, state0 = env.reset(k_reset)

    def step_fn(carry, t_key):
        obs, state = carry
        # Select actions (generic handles RBC/Q/PPO)
        actions_mat, _, _ = select_actions_generic(t_key, obs, agent_state, obs_mapping)

        # Step environment
        next_obs, next_state, rewards, dones, info = env.step_env_matrix(
            t_key, state, actions_mat
        )

        step_data = SimulationResult(
            obs=obs,
            actions=actions_mat,
            rewards=rewards,
            dones=dones,
            net_grid_energy=info["net_grid_energy"],
            renewable_spill=info["renewable_spill"],
        )
        return (next_obs, next_state), step_data

    step_keys = jr.split(k_scan, num_steps)
    _, history = jax.lax.scan(step_fn, (obs0, state0), step_keys)

    return history


def compute_metrics(
    env: JAXECEnvironment,
    res: SimulationResult,
) -> dict:
    """Compute summary metrics from simulation history.

    Args:
        env: The environment.
        res: The simulation result (history).

    Returns:
        dict: A dictionary of metrics.
    """
    # Number of REAL building agents
    num_buildings = env.num_agents
    
    # res.rewards is (T, N_total)
    # We only care about the building agents for per-agent metrics
    per_agent_total_reward = jnp.sum(res.rewards, axis=0)[:num_buildings]
    total_reward = jnp.sum(per_agent_total_reward)

    # Grid stats (also sliced to buildings if necessary, but grid energy is usually (N,))
    # Wait, JAXECEnvironment info["net_grid_energy"] is already (N,) in _step_core.
    # But if has_pricing_agent is True, it might have been padded? 
    # No, env.step_env_matrix calls _step_core which returns (N,) rewards/dones/info.
    # Wait, I added padding to rewards in _step_core.
    # Let's check net_grid_energy padding too.
    
    net_grid = res.net_grid_energy[:, :num_buildings]
    total_grid_per_agent = jnp.sum(net_grid, axis=0)
    
    peak_grid_import = jnp.max(net_grid)
    total_renewable_spill = jnp.sum(res.renewable_spill[:, :num_buildings])

    community_gini = gini(per_agent_total_reward)

    # self-sustainability ratio = (total energy import) / (total energy consumption)
    total_consumption = jnp.sum(env.dataset.households.consumption)
    
    # net_grid is net, so positive is import, negative is export.
    grid_import_kWh = jnp.sum(jnp.clip(net_grid, min=0))
    grid_export_kWh = jnp.sum(jnp.clip(-net_grid, min=0)) # negate to get positive export value

    self_sustainability_ratio = grid_import_kWh / (total_consumption + 1e-8)
    emissions = grid_import_kWh * EMISSION_PER_KWH

    return {
        # Economics
        "community_cost": -total_reward,  # € (positive cost)
        "per_agent_cost": -per_agent_total_reward,  # € (positive cost)
        "community_gini": community_gini,
        # Energy Balance
        "grid_import_kWh": grid_import_kWh,
        "grid_export_kWh": grid_export_kWh,
        "peak_grid_import": peak_grid_import,
        "self_sustainability_ratio": self_sustainability_ratio,
        # Fairness
        "fairness_gini": community_gini,
        # Environment
        "emissions": emissions,
        "renewable_spill": total_renewable_spill,
    }
