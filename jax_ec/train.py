"""Training loop for multi-agent RL with JAX-EC."""

import jax
import jax.numpy as jnp
import jax.random as jr

from jax_ec.agents.api import (
    ppo_post_rollout_update,
    select_actions_generic,
    update_generic,
)
from jax_ec.agents.ppo import PPOConfig, PPORollout
from jax_ec.agents.types import AgentState
from jax_ec.environment.env import JAXECEnvironment

K = 64  # rollout horizon for PPO


def build_run_epoch(env: JAXECEnvironment, ppo_cfg: PPOConfig, manager_ppo_cfg: PPOConfig = None):
    """Builds a function that runs one training epoch (K steps + PPO update)."""
    obs_mapping = env.observation_mapping
    N = env.num_agents
    T_total = env.num_timesteps - 1
    K_cfg = int(ppo_cfg.rollout_len)
    T_static = int(min(K_cfg, T_total))
    
    # Ensure config is not None for tracing
    m_cfg = manager_ppo_cfg if manager_ppo_cfg is not None else PPOConfig()

    @jax.jit
    def run_epoch(agent_state: AgentState, epoch_key: jax.Array):
        k0, kscan = jr.split(epoch_key)
        obs0, state0 = env.reset(k0)

        def step_fn(carry, t_key):
            obs, state, astate, cum_rew = carry
            actions_mat, a_batt, aux = select_actions_generic(
                t_key, obs, astate, obs_mapping
            )
            next_obs, next_state, rewards, dones, info = env.step_env_matrix(
                t_key, state, actions_mat
            )

            # per-step Q update
            astate2 = update_generic(
                astate, obs, a_batt, rewards, next_obs, obs_mapping
            )

            # Building PPO step data
            ppo_step = (
                obs,
                actions_mat[:, : ppo_cfg.act_dim],
                aux["ppo_logp"],
                aux["ppo_value"],
                rewards,
                jnp.zeros_like(rewards),
            )
            
            # Manager PPO step data (index N or N+1 row)
            # We want to keep shapes (N_total,) so rollout pieces are (T, N_total)
            m_rew = jnp.sum(rewards)
            # Reward for manager is community total, broadcasted to all agents (masked later)
            m_rew_vec = jnp.full_like(rewards, m_rew)
            
            manager_ppo_step = (
                obs, # whole obs (T, N_total, obs_dim)
                actions_mat[:, 2:3], # (N_total, 1)
                aux["manager_logp"], # (N_total,)
                aux["manager_value"], # (N_total,)
                m_rew_vec, # (N_total,)
                jnp.zeros_like(rewards), # (N_total,)
            )

            cum_rew = cum_rew + rewards
            return (next_obs, next_state, astate2, cum_rew), (ppo_step, manager_ppo_step)

        step_keys = jr.split(kscan, T_static)
        (obsT, stateT, astateT, cum_rew), (ppo_hist, manager_hist) = jax.lax.scan(
            step_fn,
            (obs0, state0, agent_state, jnp.zeros((obs0.shape[0],), jnp.float32)),
            step_keys,
        )

        # Building Rollout
        rollout = PPORollout(
            obs=ppo_hist[0],  # (T,N_total,obs_dim)
            act=ppo_hist[1],  # (T,N_total,act_dim)
            logp=ppo_hist[2],
            val=ppo_hist[3],
            rew=ppo_hist[4],
            done=ppo_hist[5],
        )
        
        # Manager Rollout
        m_rollout = PPORollout(
            obs=manager_hist[0],
            act=manager_hist[1],
            logp=manager_hist[2],
            val=manager_hist[3],
            rew=manager_hist[4],
            done=manager_hist[5],
        )

        astate_new, ppo_metrics = ppo_post_rollout_update(
            astateT, rollout, obsT, ppo_cfg, epoch_key,
            manager_rollout=m_rollout, manager_ppo_cfg=m_cfg
        )

        return astate_new, cum_rew, ppo_metrics

    return run_epoch
