"""Q-learning agent.

Tabular Q-learning agent.
"""

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class QAgentState:
    """Dataclass for the state of a Q-learning agent."""

    q_table: jnp.ndarray  # shape (24, num_action_bins)
    action_map: jnp.ndarray  # shape (num_action_bins,)
    epsilon: float
    alpha: float
    gamma: float


def create_q_agent_state(
    idx, num_action_bins=10, epsilon=0.1, alpha=0.1, gamma=0.95
) -> QAgentState:
    """Create a Q-learning agent state."""
    action_map = jnp.linspace(-1.0, 1.0, num_action_bins)
    q_table = jnp.zeros((24, num_action_bins))
    return QAgentState(
        q_table=q_table,
        action_map=action_map,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
    )


# Action selection function
def q_act(key, q_state: QAgentState, observation, obs_mapping) -> float:
    """Select action using epsilon-greedy policy."""
    hour_idx = obs_mapping.HOUR_OF_DAY
    hour = observation[hour_idx].astype(int)
    random_val = jax.random.uniform(key)
    hour_table = jax.lax.dynamic_index_in_dim(q_state.q_table, hour)
    greedy_action_idx = jnp.argmax(hour_table)
    random_action_idx = jax.random.randint(key, (), 0, len(q_state.action_map))
    action_idx = jnp.where(
        random_val < q_state.epsilon, random_action_idx, greedy_action_idx
    )
    action = q_state.action_map[action_idx]
    return action


# Q-table update function
def q_update(
    q_state: QAgentState, observation, action, reward, next_observation, obs_mapping
) -> QAgentState:
    """Update Q-table using the Q-learning update rule."""
    hour_idx = obs_mapping.HOUR_OF_DAY
    hour = observation[hour_idx].astype(int)
    next_hour = next_observation[hour_idx].astype(int)

    action_idx = jnp.argmin(jnp.abs(q_state.action_map - action))

    best_next_action_value = jnp.max(q_state.q_table[next_hour])
    td_target = reward + q_state.gamma * best_next_action_value
    td_error = td_target - q_state.q_table[hour, action_idx]

    q_table_updated = q_state.q_table.at[hour, action_idx].set(
        q_state.q_table[hour, action_idx] + q_state.alpha * td_error
    )

    return q_state.replace(q_table=q_table_updated)
