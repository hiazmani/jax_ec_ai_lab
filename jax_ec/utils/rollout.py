# jax_ec/utils/rollout.py

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RolloutInfo:
    """Generic bag of per-step tensors; all time-major (T, N, …)."""

    obs: jnp.ndarray
    actions: jnp.ndarray
    logp: jnp.ndarray
    values: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    # You can add fields later (e.g., prices) without breaking signature.


def init_rollout(T, N, obs_dim, act_dim, dtype=jnp.float32) -> RolloutInfo:
    """Initialize empty rollout buffer with zeros."""
    return RolloutInfo(
        obs=jnp.zeros((T, N, obs_dim)),
        actions=jnp.zeros((T, N, act_dim)),
        logp=jnp.zeros((T, N)),
        values=jnp.zeros((T, N)),
        rewards=jnp.zeros((T, N)),
        dones=jnp.zeros((T, N)),
    )


def write_step(buf: RolloutInfo, t: int, obs, act, logp, val, rew, done) -> RolloutInfo:
    """Write single step into the RolloutBuffer."""
    return buf.replace(
        obs=buf.obs.at[t].set(obs),
        actions=buf.actions.at[t].set(act),
        logp=buf.logp.at[t].set(logp),
        values=buf.values.at[t].set(val),
        rewards=buf.rewards.at[t].set(rew),
        dones=buf.dones.at[t].set(done),
    )
