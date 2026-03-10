"""Generic agent API for mixed-agent training/evaluation."""

# jax_ec/agents/api.py
import jax
import jax.numpy as jnp
import jax.random as jr

from jax_ec.agents.ppo import (
    PPOConfig,
    PPORollout,
    ppo_init,
    ppo_select_action,
    ppo_update,
)
from jax_ec.agents.q_agent import create_q_agent_state, q_act, q_update
from jax_ec.agents.rbc import rbc_batt_action
from jax_ec.agents.types import (
    AgentKind,
    AgentState,
    init_rbc_state,
)


def init_agent_state(
    rng: jax.Array,
    agent_types: list[str],
    obs_dim: int,
    ppo_cfg: PPOConfig = PPOConfig(),
    q_bins: int = 10,
    has_pricing_agent: bool = False,
    pricing_ppo_cfg: PPOConfig = PPOConfig(),
    pricing_obs_dim: int = 6,
    weight_path: str = None,
    manager_weight_path: str = None,
) -> AgentState:
    """Initialize a mixed AgentState from a list of type strings.

    Supported types: "RBC", "Q_LEARNING", "PPO" (case-insensitive).
    """
    building_types = agent_types
    N_build = len(building_types)
    N_total = N_build + 1 if has_pricing_agent else N_build
    
    type_map = {
        "RBC": AgentKind.RBC,
        "Q": AgentKind.Q,
        "Q_LEARNING": AgentKind.Q,
        "PPO": AgentKind.PPO,
    }
    
    # Map building types to kinds
    kinds_list = []
    for i in range(N_build):
        t = building_types[i % len(building_types)]
        kinds_list.append(type_map[t.upper()])

    if has_pricing_agent:
        kinds_list.append(AgentKind.PRICING_MANAGER)
    kinds = jnp.array(kinds_list, dtype=jnp.int32)

    # 1. RBC
    rbc = init_rbc_state(N_total)

    # 2. Q-Learning
    def _make_q(i):
        return create_q_agent_state(i, num_action_bins=q_bins)
    q = jax.vmap(_make_q)(jnp.arange(N_total))

    # 3. PPO for Buildings
    rng, k_build, k_price = jr.split(rng, 3)
    ppo = ppo_init(k_build, obs_dim, ppo_cfg, weight_path=weight_path)
    
    # 4. PPO for Pricing Agent (Manager)
    # Even if none is present, we need a dummy one to keep the PyTree stable
    pricing_ppo = ppo_init(k_price, pricing_obs_dim, pricing_ppo_cfg, weight_path=manager_weight_path)

    return AgentState(kind=kinds, rbc=rbc, q=q, ppo=ppo, pricing_ppo=pricing_ppo)


def select_actions_generic(
    step_key: jax.Array,
    obs: jnp.ndarray,  # (N or N+1, obs_dim)
    agent_state: AgentState,  # batch-first subtrees
    obs_mapping,
):
    """Returns (actions_mat (N or N+1,3), a_batt (N or N+1,), aux)."""
    N_total_dyn = obs.shape[0]
    kinds = agent_state.kind
    has_manager = jnp.any(kinds == int(AgentKind.PRICING_MANAGER))
    
    keys = jr.split(step_key, N_total_dyn)
    
    # -- Stage 1: Pricing Agent Action -- #
    # It observes the last row of 'obs' (if manager exists)
    manager_obs = obs[-1, :6] # only 6 dims for manager
    # We call ppo_select_action but only on the manager's ppo state
    # Wait, ppo_select_action currently vmaps over the entire population.
    # We need a ppo_select_action that can be used on a single row.
    
    # For now, let's keep things simple:
    # 1. Select manager price
    def _get_price(_):
        res = ppo_select_action(step_key, agent_state.pricing_ppo, manager_obs[None, :])
        return res.actions[0, 2], res.logp[0], res.value[0] # index 2 is trade_price

    price, p_logp, p_val = jax.lax.cond(
        has_manager,
        _get_price,
        lambda _: (0.0, 0.0, 0.0),
        operand=None
    )

    # building_mask: 1 for buildings, 0 for manager
    # kinds is (N_total_dyn,)
    building_mask = (kinds != int(AgentKind.PRICING_MANAGER))
    
    # Pad obs to a static MAX_OBS_DIM (e.g. 10) to avoid dynamic shape issues later
    MAX_OBS = 10
    obs_padded = jnp.pad(obs, ((0, 0), (0, MAX_OBS - obs.shape[1])))

    def _inject_price_all(obs_mat, p):
        # obs_mat is (N_total, 10)
        # We only want to set index 6 for rows where building_mask is True
        new_col = jnp.where(building_mask, p, obs_mat[:, 6])
        return obs_mat.at[:, 6].set(new_col)

    obs_buildings = jax.lax.cond(
        has_manager,
        lambda _: _inject_price_all(obs_padded, price),
        lambda _: obs_padded,
        operand=None
    )

    # RBC actions
    a_batt_rbc = jax.vmap(lambda ob: rbc_batt_action(ob, obs_mapping))(obs_buildings)
    # Correct RBC trading: if net_load (cons-prod) > 0, we need to BUY (-1.0)
    # obs_buildings indices: 0 is CONS, 1 is PROD
    a_qty_rbc = jnp.where((obs_buildings[:, 0] - obs_buildings[:, 1]) > 0, -1.0, 1.0)

    # Q actions
    a_batt_q = jax.vmap(lambda k, qs, ob: q_act(k, qs, ob, obs_mapping))(
        keys, agent_state.q, obs_buildings
    )
    a_qty_q = jnp.where((obs_buildings[:, 0] - obs_buildings[:, 1]) > 0, -1.0, 1.0)

    # PPO (Building policy)
    ppo_out = ppo_select_action(step_key, agent_state.ppo, obs_buildings)
    act_dim = ppo_out.actions.shape[-1]
    a_batt_ppo = ppo_out.actions[:, 0]
    a_qty_ppo = ppo_out.actions[:, 1] if act_dim == 3 else jnp.zeros((N_total_dyn,))
    a_price_ppo = ppo_out.actions[:, 2] if act_dim == 3 else jnp.zeros((N_total_dyn,))

    # Stitch battery actions
    a_batt = (
        jnp.where(kinds == int(AgentKind.RBC), a_batt_rbc, 0.0)
        + jnp.where(kinds == int(AgentKind.Q), a_batt_q, 0.0)
        + jnp.where(kinds == int(AgentKind.PPO), a_batt_ppo, 0.0)
    )

    # Stitch trading quantity
    # If act_dim=3, PPO uses its own quantity. Otherwise, everyone defaults to 1.0/ -1.0 (trade everything).
    a_qty = (
        jnp.where(kinds == int(AgentKind.RBC), a_qty_rbc, 0.0)
        + jnp.where(kinds == int(AgentKind.Q), a_qty_q, 0.0)
        + jnp.where(kinds == int(AgentKind.PPO), a_qty_ppo if act_dim == 3 else jnp.where((obs_buildings[:, 0] - obs_buildings[:, 1]) > 0, -1.0, 1.0), 0.0)
    )

    # Stitch trading price
    a_price = jnp.where(
        (kinds == int(AgentKind.PPO)) & (act_dim == 3),
        a_price_ppo,
        0.0, # Default price action (midpoint)
    )
    
    # If manager exists, override its price in the final matrix
    # The matrix should be (N_total, 3)
    # Building rows are 0..N-1, Manager row is N
    actions_mat = jnp.stack([a_batt, a_qty, a_price], axis=1)
    actions_mat = jax.lax.cond(
        has_manager,
        lambda m: m.at[-1, 2].set(price),
        lambda m: m,
        operand=actions_mat
    )
    
    aux = dict(
        ppo_logp=ppo_out.logp, 
        ppo_value=ppo_out.value,
        manager_logp=jnp.full((N_total_dyn,), p_logp),
        manager_value=jnp.full((N_total_dyn,), p_val)
    )

    return actions_mat, a_batt, aux


def update_generic(
    agent_state: AgentState,
    obs: jnp.ndarray,
    a_batt: jnp.ndarray,
    reward: jnp.ndarray,
    next_obs: jnp.ndarray,
    obs_mapping,
) -> AgentState:
    """Per-step update: Q-only; PPO updated via post-rollout function."""
    kind = agent_state.kind
    is_q = kind == int(AgentKind.Q)

    q_new = jax.vmap(
        lambda qs, m, ob, ab, rw, nob: jax.lax.cond(
            m,
            lambda _: q_update(qs, ob, ab, rw, nob, obs_mapping),
            lambda _: qs,
            operand=None,
        ),
        in_axes=(0, 0, 0, 0, 0, 0),
    )(agent_state.q, is_q, obs, a_batt, reward, next_obs)

    return agent_state.replace(q=q_new)


def ppo_post_rollout_update(
    agent_state: AgentState,
    rollout: PPORollout,
    last_obs: jnp.ndarray,  # (N_total, obs_dim)
    ppo_cfg,
    rng: jax.Array,
    manager_rollout: PPORollout = None,
    manager_ppo_cfg = None,
) -> tuple[AgentState, dict]:
    """Apply PPO update for masked PPO agents AND Pricing Manager if present."""
    kinds = agent_state.kind
    is_ppo = kinds == int(AgentKind.PPO)
    any_ppo = jnp.any(is_ppo)
    
    is_manager = kinds == int(AgentKind.PRICING_MANAGER)
    has_manager = jnp.any(is_manager)

    # -- 1. Building PPO Update -- #
    def _do_ppo(astate):
        new_ppo, m = ppo_update(
            astate.ppo,
            rollout,
            last_obs,
            ppo_cfg,
            rng,
            agent_mask=is_ppo.astype(jnp.float32),
        )
        return astate.replace(ppo=new_ppo), m

    def _skip_ppo(astate):
        return astate, {
            "loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0
        }

    astate, m_ppo = jax.lax.cond(any_ppo, _do_ppo, _skip_ppo, agent_state)
    metrics = {f"ppo_{k}": v for k, v in m_ppo.items()}

    # -- 2. Manager PPO Update -- #
    def _do_manager(astate):
        new_pricing_ppo, m = ppo_update(
            astate.pricing_ppo,
            manager_rollout,
            last_obs,
            manager_ppo_cfg,
            rng,
            agent_mask=is_manager.astype(jnp.float32),
        )
        return astate.replace(pricing_ppo=new_pricing_ppo), m

    def _skip_manager(astate):
        return astate, {
            "loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0
        }

    # We only run manager update if has_manager AND manager_rollout is provided
    # However, for JIT, manager_rollout cannot be None in the 'True' branch.
    # We use a dummy check or assume it's provided if has_manager is true.
    can_update_manager = has_manager & (manager_rollout is not None)

    astate, m_man = jax.lax.cond(can_update_manager, _do_manager, _skip_manager, astate)
    metrics.update({f"manager_{k}": v for k, v in m_man.items()})

    return astate, metrics
