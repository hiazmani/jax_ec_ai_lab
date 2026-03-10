"""Types for agent states."""

import enum

import jax.numpy as jnp
from flax import struct

from jax_ec.agents.ppo import PPOAgentState

N_AGENT_KINDS = 3  # RBC, Q, PPO


class AgentKind(enum.IntEnum):
    """Enum for agent kinds.

    Tells the update and action selection functions which AgentState field to use.
    """

    RBC = 0
    Q = 1
    PPO = 2
    PRICING_MANAGER = 3


@struct.dataclass
class RBCState:
    """Dataclass for the state of a rule-based controller (RBC) agent."""

    # placeholder to keep tree shape stable; can add params later
    dummy: jnp.ndarray  # shape (N,) zeros


@struct.dataclass
class QAgentState:
    """Dataclass for the state of a Q-learning agent."""

    q_table: jnp.ndarray  # shape (24, num_action_bins)
    action_map: jnp.ndarray  # shape (num_action_bins,)
    epsilon: float
    alpha: float
    gamma: float


@struct.dataclass
class AgentState:
    """Unified per-agent state PyTree (batch-first everywhere)."""

    kind: jnp.ndarray  # (N,) int32; values in AgentKind
    rbc: RBCState  # fields with leading (N, ...)
    q: QAgentState
    ppo: PPOAgentState
    pricing_ppo: PPOAgentState # For the manager (N+1-th entry)


## -- Initializers -- ##
def init_rbc_state(N: int) -> RBCState:
    """Initialize RBCState with dummy zeros."""
    return RBCState(dummy=jnp.zeros((N,), jnp.float32))


def init_q_agent_state(
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
