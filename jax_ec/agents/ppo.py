# jax_ec/agents/ppo.py
from typing import NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState


def _safe_entropy(pi):
    """Entropy that works for both discrete and squashed continuous policies.

    - For Transformed distributions (e.g., Tanh(Normal)), use base entropy
      (ignores the tanh Jacobian term — common PPO approximation).
    - Otherwise, use the distribution's own entropy().
    """
    # Python isinstance runs at trace time and is static for a fixed policy head
    if isinstance(pi, distrax.Transformed):
        return pi.distribution.entropy()
    return pi.entropy()


# ---------- Config & State --------------------------------------------------
@struct.dataclass
class PPOConfig:
    """Hyperparameters for PPO."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    rollout_len: int = 128  # per-agent rollout length
    # Policy shape
    act_dim: int = 1  # start with battery-only; set to 3 for [batt, qty, price]
    hidden: int = 64
    activation: str = "tanh"  # "relu" or "tanh"


class PPOTrainState(TrainState):
    """Train state for PPO, with optax optimizer."""

    # TrainState is already a PyTree; we keep it as-is.
    pass


@struct.dataclass
class PPOAgentState:
    """Whole-population PPO params (shared policy across PPO agents by default)."""

    train_state: PPOTrainState


# ---------- Network ---------------------------------------------------------
class ActorCritic(nn.Module):
    """Combined actor-critic network for PPO."""

    act_dim: int
    hidden: int = 64
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        """Returns (pi, v) given observation x."""
        Act = nn.relu if self.activation == "relu" else nn.tanh

        # Actor
        a = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        a = Act(a)
        a = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(a)
        a = Act(a)
        mean = nn.Dense(self.act_dim, kernel_init=nn.initializers.orthogonal(0.01))(a)

        # trainable log_std (per-action)
        log_std = self.param("log_std", nn.initializers.zeros, (self.act_dim,))
        base = distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        pi = distrax.Transformed(base, distrax.Block(distrax.Tanh(), self.act_dim))

        # Critic
        c = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        c = Act(c)
        c = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(c)
        c = Act(c)
        v = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(c)
        v = jnp.squeeze(v, axis=-1)

        return pi, v


import pickle
from pathlib import Path

# ... (inside ActorCritic or after)

def save_ppo_weights(ppo_state: PPOAgentState, path: str):
    """Save PPO parameters to a pickle file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(ppo_state.train_state.params, f)
    print(f"Saved PPO weights to {path}")


def load_ppo_weights(path: str):
    """Load PPO parameters from a pickle file."""
    with open(path, "rb") as f:
        params = pickle.load(f)
    return params

def ppo_init(
    rng: jax.Array,
    obs_dim: int,
    cfg: PPOConfig,
    weight_path: str = None,
) -> PPOAgentState:
    """Create PPOAgentState with fresh or loaded params."""
    dummy_obs_dim = obs_dim
    loaded_params = None

    if weight_path and Path(weight_path).exists():
        loaded_params = load_ppo_weights(weight_path)
        # Infer obs_dim from loaded weights: Dense_0 kernel shape is (obs_dim, hidden)
        dummy_obs_dim = loaded_params["params"]["Dense_0"]["kernel"].shape[0]
        print(f"Loaded PPO weights from {weight_path} (obs_dim={dummy_obs_dim})")

    net = ActorCritic(act_dim=cfg.act_dim, hidden=cfg.hidden, activation=cfg.activation)
    rng, key = jax.random.split(rng)
    dummy = jnp.zeros((dummy_obs_dim,), dtype=jnp.float32)

    if loaded_params is not None:
        params = loaded_params
    else:
        params = net.init(key, dummy)

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr, eps=1e-5),
    )
    ts = PPOTrainState.create(apply_fn=net.apply, params=params, tx=tx)
    return PPOAgentState(train_state=ts)


class PPOSelectOut(NamedTuple):
    """Output of ppo_select_action()."""

    actions: jnp.ndarray  # (N, act_dim) in [-1,1]
    logp: jnp.ndarray  # (N,)
    value: jnp.ndarray  # (N,)


def ppo_select_action(
    key: jax.Array,
    ppo_state: PPOAgentState,
    obs: jnp.ndarray,  # (N, obs_dim)
) -> PPOSelectOut:
    """Vectorized action selection for PPO (shared policy)."""
    ts = ppo_state.train_state
    
    def _one(k, ob):
        # determine expected obs_dim from params (static)
        expected_dim = ts.params["params"]["Dense_0"]["kernel"].shape[0]
        
        # Static way to handle this without jnp.pad (which requires static width)
        # We assume ob is always at least expected_dim or we can slice safely.
        # If we need to support smaller obs, we'd need to know the MAX possible obs_dim at compile time.
        # Let's assume MAX_OBS = 10 for now and pad all inputs to 10 upfront in api.py if needed.
        # BUT for now, let's just slice. If it's too small, it will crash here,
        # which is better than a Tracer error.
        ob_correct = ob[:expected_dim]
        
        pi, v = ts.apply_fn(ts.params, ob_correct)
        a = pi.sample(seed=k)  # (-1, 1) after Tanh
        logp = pi.log_prob(a)
        return a, logp, v

    keys = jax.random.split(key, obs.shape[0])
    a, logp, v = jax.vmap(_one)(keys, obs)
    return PPOSelectOut(actions=a, logp=logp, value=v)


# ---------- Rollout → GAE ---------------------------------------------------


def _compute_gae(rew, val, done, last_val, gamma, lam):
    """Compute GAE advantages & returns.

    Args:
        rew:  (T, N)
        val:  (T, N)
        done: (T, N)  (0.0/1.0 or bool)
        last_val: (N,) value at t = T
        gamma: discount factor
        lam: GAE lambda

    Returns:
        adv: (T, N)
        ret: (T, N)  where ret = adv + val
    """
    # ensure dtypes line up
    rew = rew.astype(jnp.float32)
    val = val.astype(jnp.float32)
    done = done.astype(jnp.float32)
    last_val = last_val.astype(jnp.float32)

    # append bootstrap value at t=T
    v_ext = jnp.concatenate([val, last_val[None, :]], axis=0)  # (T+1, N)

    # δ_t = r_t + γ * V_{t+1} * (1 - done_t) - V_t
    deltas = rew + gamma * v_ext[1:] * (1.0 - done) - v_ext[:-1]  # (T, N)

    def scan_fn(gae, inputs):
        delta_t, done_t = inputs
        gae = delta_t + gamma * lam * (1.0 - done_t) * gae
        return gae, gae

    # reverse-time scan
    _, adv_rev = jax.lax.scan(
        scan_fn,
        jnp.zeros_like(last_val),  # (N,)
        (deltas[::-1], done[::-1]),  # each (T, N)
    )
    adv = adv_rev[::-1]  # (T, N)
    ret = adv + val  # (T, N)
    return adv, ret


# ---------- PPO update ------------------------------------------------------


class PPORollout(struct.PyTreeNode):
    """Time-major rollout for PPO, per-population (shared policy).

    Shapes: all (T, N, …)
    """

    obs: jnp.ndarray  # (T, N, obs_dim)
    act: jnp.ndarray  # (T, N, act_dim)
    logp: jnp.ndarray  # (T, N)
    val: jnp.ndarray  # (T, N)
    rew: jnp.ndarray  # (T, N)
    done: jnp.ndarray  # (T, N)


def ppo_update(
    ppo_state: PPOAgentState,
    rollout: PPORollout,
    last_obs: jnp.ndarray,  # (N, obs_dim)
    cfg: PPOConfig,
    rng: jax.Array,
    agent_mask: jnp.ndarray,  # (N,) bool or {0,1} marking PPO agents
):
    """One PPO update over a collected rollout (time-major).

    Only agents with mask==1 contribute to the loss.
    """
    ts = ppo_state.train_state
    net = ActorCritic(act_dim=cfg.act_dim, hidden=cfg.hidden, activation=cfg.activation)
    T, N = rollout.rew.shape
    B = T * N

    # determine expected obs_dim from params
    expected_dim = ts.params["params"]["Dense_0"]["kernel"].shape[0]
    
    # compute bootstrap value V(s_T)
    # Ensure last_obs matches expected_dim (e.g. 6 for manager vs 7 for buildings)
    last_obs_correct = last_obs[:, :expected_dim]
    _, last_val = ts.apply_fn(ts.params, last_obs_correct)

    # advantages/returns
    adv, ret = _compute_gae(
        rollout.rew, rollout.val, rollout.done, last_val, cfg.gamma, cfg.gae_lambda
    )

    # flatten
    def flat(x):
        return x.reshape((B,) + x.shape[2:])

    f_obs, f_act, f_logp_old, f_val, f_adv, f_ret = map(
        flat, (rollout.obs, rollout.act, rollout.logp, rollout.val, adv, ret)
    )
    f_mask = (
        jnp.repeat(agent_mask[None, :], T, axis=0).reshape((B,)).astype(f_adv.dtype)
    )

    # normalize advantages only over masked entries
    denom = jnp.maximum(jnp.sum(f_mask), 1.0)
    mean_adv = jnp.sum(f_adv * f_mask) / denom
    var_adv = jnp.sum(((f_adv - mean_adv) ** 2) * f_mask) / denom
    f_adv = (f_adv - mean_adv) / jnp.sqrt(var_adv + 1e-8)

    # minibatching
    min_batch = B // cfg.num_minibatches
    perm = jax.random.permutation(rng, B)

    def shard(x):
        return jnp.take(x, perm, axis=0).reshape(
            (cfg.num_minibatches, min_batch) + x.shape[1:]
        )

    s_obs, s_act, s_logp_old, s_val, s_adv, s_ret, s_mask = map(
        shard, (f_obs, f_act, f_logp_old, f_val, f_adv, f_ret, f_mask)
    )

    def loss_and_grads(params, batch):
        obs_b, act_b, logp_old_b, val_b, adv_b, ret_b, mask_b = batch
        # Ensure observations match expected_dim
        obs_b_correct = obs_b[:, :expected_dim]
        pi, v = net.apply(params, obs_b_correct)
        logp = pi.log_prob(act_b)

        # actor
        ratio = jnp.exp(logp - logp_old_b)
        unclipped = ratio * adv_b
        clipped = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_b
        actor_loss = -jnp.sum(jnp.minimum(unclipped, clipped) * mask_b) / (
            jnp.sum(mask_b) + 1e-8
        )

        # critic (clipped)
        v_clipped = val_b + jnp.clip(v - val_b, -cfg.clip_eps, cfg.clip_eps)
        vf_raw = (v - ret_b) ** 2
        vf_clip = (v_clipped - ret_b) ** 2
        vf_loss = (
            0.5
            * jnp.sum(jnp.maximum(vf_raw, vf_clip) * mask_b)
            / (jnp.sum(mask_b) + 1e-8)
        )

        # entropy
        ent = jnp.sum(_safe_entropy(pi) * mask_b) / (jnp.sum(mask_b) + 1e-8)

        total = actor_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * ent
        return total, (actor_loss, vf_loss, ent)

    @jax.jit
    def one_epoch(ts: PPOTrainState, rng_epoch):
        def step(ts, mb_idx):
            batch = (
                s_obs[mb_idx],
                s_act[mb_idx],
                s_logp_old[mb_idx],
                s_val[mb_idx],
                s_adv[mb_idx],
                s_ret[mb_idx],
                s_mask[mb_idx],
            )
            (loss, aux), grads = jax.value_and_grad(loss_and_grads, has_aux=True)(
                ts.params, batch
            )
            ts = ts.apply_gradients(grads=grads)
            return ts, (loss, *aux)

        return jax.lax.scan(step, ts, jnp.arange(cfg.num_minibatches))

    def run_epochs(ts, rng0):
        def body(carry, _):
            ts, rng_e = carry
            rng_e, _ = jax.random.split(rng_e)
            ts, (loss, actor, critic, ent) = one_epoch(ts, rng_e)
            return (ts, rng_e), (loss, actor, critic, ent)

        (ts, _), hist = jax.lax.scan(body, (ts, rng), jnp.arange(cfg.update_epochs))
        metrics = dict(
            loss=jnp.mean(hist[0]),
            actor_loss=jnp.mean(hist[1]),
            critic_loss=jnp.mean(hist[2]),
            entropy=jnp.mean(hist[3]),
        )
        return ts, metrics

    new_ts, metrics = run_epochs(ts, rng)
    return ppo_state.replace(train_state=new_ts), metrics
