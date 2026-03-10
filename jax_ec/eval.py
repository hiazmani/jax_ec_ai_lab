"""Evaluation functions for JAX-EC agents (agent-agnostic)."""

import jax
import jax.numpy as jnp
import jax.random as jr

from jax_ec.agents.api import select_actions_generic
from jax_ec.agents.types import AgentKind, AgentState


def _set_q_epsilon_zero(agent_state: AgentState) -> AgentState:
    """Return a copy with epsilon=0 for Q agents only (RBC/PPO unchanged)."""
    is_q = agent_state.kind == int(AgentKind.Q)
    new_eps = jnp.where(is_q, 0.0, agent_state.q.epsilon)
    return agent_state.replace(q=agent_state.q.replace(epsilon=new_eps))


def eval_rollout(
    env,
    agent_state: AgentState,
    key,
    steps: int | None = None,
):
    """Run an evaluation rollout with current agent_state (ε=0 for Q).

    Args:
        env: JAXECEnvironment
        agent_state: AgentState with leading (N, …) everywhere
        key: PRNGKey
        steps: optional max steps to run (default: full episode)

    Returns:
        eval_ret: (N,) per-agent total return
        rew_hist: (T, N) per-step rewards
        grid_hist: (T, N) per-step net-grid energy
    """
    obs_mapping = env.observation_mapping
    N = len(env.agents)
    T = env.num_timesteps - 1 if steps is None else min(steps, env.num_timesteps - 1)

    # Don’t explore during evaluation for Q agents
    agent_state_eval = _set_q_epsilon_zero(agent_state)

    @jax.jit
    def run_eval(astate, rng):
        k_reset, k_scan = jax.random.split(rng)
        obs0, state0 = env.reset(k_reset)

        def step_fn(carry, t_key):
            obs, state, ret = carry
            actions_mat, _, _ = select_actions_generic(t_key, obs, astate, obs_mapping)
            next_obs, next_state, rewards, dones, info = env.step_env_matrix(
                t_key, state, actions_mat
            )
            return (next_obs, next_state, ret + rewards), (
                rewards,
                info["net_grid_energy"],
            )

        step_keys = jax.random.split(k_scan, steps)
        (_, _, total_ret), (rew_hist, grid_hist) = jax.lax.scan(
            step_fn, (obs0, state0, jnp.zeros((obs0.shape[0],), jnp.float32)), step_keys
        )
        return total_ret, rew_hist, grid_hist


    return run_eval(agent_state_eval, key)
