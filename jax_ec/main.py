# Unified demo: train a mixed population (RBC + Q-learning) in JAXECEnvironment.

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
from omegaconf import DictConfig, OmegaConf

from jax_ec.agents.ppo import PPOConfig, ppo_init, save_ppo_weights, PPOAgentState
from jax_ec.agents.q_agent import create_q_agent_state
from jax_ec.agents.types import AgentKind, AgentState, init_rbc_state
from jax_ec.data.load import load_simulation_data
from jax_ec.environment.env import JAXECEnvironment
from jax_ec.eval import eval_rollout
from jax_ec.train import build_run_epoch
from jax_ec.utils.battery import make_battery_cfgs


def _kinds_to_ints(names, N):
    mapping = {
        "RBC": int(AgentKind.RBC),
        "Q": int(AgentKind.Q),
        "PPO": int(AgentKind.PPO),
    }
    arr = jnp.array([mapping[n] for n in names], dtype=jnp.int32)
    if arr.size != N:
        # if user provides fewer names than agents, repeat last; or truncate if too many
        if arr.size < N:
            pad = jnp.full((N - arr.size,), arr[-1], dtype=jnp.int32)
            arr = jnp.concatenate([arr, pad], axis=0)
        else:
            arr = arr[:N]
    return arr


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        None
    """
    print("Config:\n", OmegaConf.to_yaml(cfg, resolve=True))

    # 1) Load env
    dataset = load_simulation_data(cfg.dataset_json)
    agent_ids = list(dataset.households.id_to_index.keys())
    battery_cfgs = make_battery_cfgs(agent_ids, cfg.battery)
    has_pricing_agent = cfg.get("has_pricing_agent", False)
    env = JAXECEnvironment(
        dataset=dataset,
        battery_configs=battery_cfgs,
        exchange_name=cfg.exchange_name,
        exchange_kwargs=cfg.exchange_kwargs if "exchange_kwargs" in cfg else None,
        allow_grid_charging=cfg.allow_grid_charging,
        has_pricing_agent=has_pricing_agent,
    )
    N = env.num_agents
    print(f"Env ready: {N} agents, {env.num_timesteps} steps.")

    # 2) Initialize AgentState using factory
    obs_dim = next(iter(env.observation_spaces.values()))[2][0]
    ppo_cfg = PPOConfig(
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_eps=cfg.ppo.clip_eps,
        vf_coef=cfg.ppo.vf_coef,
        ent_coef=cfg.ppo.ent_coef,
        lr=cfg.ppo.lr,
        max_grad_norm=cfg.ppo.max_grad_norm,
        update_epochs=cfg.ppo.update_epochs,
        num_minibatches=cfg.ppo.num_minibatches,
        act_dim=cfg.ppo.act_dim,
        hidden=cfg.ppo.hidden,
        activation=cfg.ppo.activation,
        rollout_len=cfg.ppo.rollout_len,
    )
    
    from jax_ec.agents.api import init_agent_state
    
    # Ensure kinds list matches number of building agents N
    kinds_raw = list(cfg.population.kinds)
    if len(kinds_raw) < N:
        kinds_raw = (kinds_raw * (N // len(kinds_raw) + 1))[:N]
    elif len(kinds_raw) > N:
        kinds_raw = kinds_raw[:N]

    agent_state = init_agent_state(
        rng=jr.PRNGKey(cfg.seed),
        agent_types=kinds_raw,
        obs_dim=obs_dim,
        ppo_cfg=ppo_cfg,
        q_bins=cfg.q_agent.num_bins,
        has_pricing_agent=has_pricing_agent,
        weight_path=cfg.get("weight_path"),
        manager_weight_path=cfg.get("manager_weight_path")
    )

    # 3) Build epoch runner
    run_epoch = build_run_epoch(env, ppo_cfg)

    # 7) Train
    key = jr.PRNGKey(cfg.seed)
    for epoch in range(cfg.num_epochs):
        key, ekey = jr.split(key)
        agent_state, cum_rew, ppo_metrics = run_epoch(agent_state, ekey)
        print(
            f"[epoch {epoch+1:02d}] mean_return={float(jnp.mean(cum_rew)):.4f}  "
            f"PPO(loss={float(ppo_metrics.get('ppo_loss', 0.0)):.4f}, "
            f"ent={float(ppo_metrics.get('ppo_entropy', 0.0)):.4f})"
        )
    # 6) Eval
    key, ekey = jr.split(key)
    eval_ret, rew_hist, grid_hist = eval_rollout(
        env, agent_state, ekey, steps=cfg.eval_steps
    )
    print(f"[eval] per-agent total return: {eval_ret}")

    # 7) Save Weights
    if cfg.get("save_path"):
        save_ppo_weights(PPOAgentState(train_state=agent_state.ppo.train_state), cfg.save_path)
    if cfg.get("save_manager_path"):
        save_ppo_weights(PPOAgentState(train_state=agent_state.pricing_ppo.train_state), cfg.save_manager_path)

    td = min(10, rew_hist.shape[0])
    print("First 10 step rewards (per agent):")
    print(jnp.round(rew_hist[:td], 4))
    print("First 10 step net-grid (per agent):")
    print(jnp.round(grid_hist[:td], 4))


if __name__ == "__main__":
    main()
