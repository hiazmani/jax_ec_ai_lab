"""Mixed policy for Q and RBC agents."""

import jax
import jax.numpy as jnp
import jax.random as jr

from jax_ec.agents.rbc import rbc_batt_action


def select_actions(
    step_key: jax.Array,
    obs: jnp.ndarray,  # (N, obs_dim)
    q_states,  # PyTree of QAgentState with leading dim N
    is_q_mask: jnp.ndarray,  # (N,) bool
    obs_mapping,
    q_act_fn,  # function (key, q_state, obs, obs_mapping) -> a_batt
):
    """Mixed policy action selection.

    Q agents use epsilon-greedy; RBC agents use rule.
    Returns (actions_mat(N,3), a_batt(N,))
    """
    N = obs.shape[0]
    keys = jr.split(step_key, N)

    # Q battery actions
    q_a_batt = jax.vmap(lambda k, qs, ob: q_act_fn(k, qs, ob, obs_mapping))(
        keys, q_states, obs
    )

    # RBC battery actions
    rbc_a_batt = jax.vmap(lambda ob: rbc_batt_action(ob, obs_mapping))(obs)

    # choose per-agent
    a_batt = jnp.where(is_q_mask, q_a_batt, rbc_a_batt)
    a_qty = jnp.zeros_like(a_batt)  # internal market placeholders (off in this demo)
    a_price = jnp.zeros_like(a_batt)

    actions_mat = jnp.stack([a_batt, a_qty, a_price], axis=1)
    return actions_mat, a_batt
