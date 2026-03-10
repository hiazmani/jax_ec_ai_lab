from enum import IntEnum
from functools import partial
from typing import Dict

import chex
import jax
import jax.numpy as jnp
from flax import struct
from omegaconf import DictConfig, OmegaConf

from jax_ec.data.types import JAXECDataset
from jax_ec.environment.battery import (
    charge_battery,
    create_battery,
)
from jax_ec.environment.exchange_mechanisms import get_mechanism
from jax_ec.environment.exchange_mechanisms.base import (
    ActionSpec,
    BaseExchangeMechanism,
    ExchangeInput,
)
from jax_ec.environment.jaxmarl import MultiAgentEnv, State
from jax_ec.environment.spaces import Box


@struct.dataclass
class EnergyState(State):
    """Dataclass for environment state."""

    step: int
    batteries: jnp.ndarray
    internal_price: float = 0.0  # Current internal price set by Pricing Agent


class ObsIdx(IntEnum):
    """Indices for the observation vector."""

    CONSUMPTION = 0
    PRODUCTION = 1
    HOUR_OF_DAY = 2
    DAY_OF_WEEK = 3
    BATTERY_SOC = 4
    BATTERY_CAP = 5


class JAXECEnvironment(MultiAgentEnv):
    """JAX-EC Environment with batteries and exchange mechanisms."""

    def __init__(
        self,
        dataset: JAXECDataset,
        battery_configs: DictConfig,
        exchange_name: str = "no_exchange",
        exchange_kwargs: Dict | None = None,
        allow_grid_charging: bool = False,
        has_pricing_agent: bool = False,
    ):
        super().__init__(dataset.metadata.number_of_agents)
        self.dataset = dataset
        self.agents = list(dataset.households.id_to_index.keys())
        self.num_timesteps = len(dataset.temporal.timestamps)
        self.allow_grid_charging = allow_grid_charging  # Control flag
        self.has_pricing_agent = has_pricing_agent

        # -- Observation Space -- #
        obs_dim = 7 if has_pricing_agent else 6
        self.observation_spaces = {a: (-jnp.inf, jnp.inf, (obs_dim,)) for a in self.agents}
        if has_pricing_agent:
            self.observation_spaces["PricingAgent"] = (-jnp.inf, jnp.inf, (6,))

        # -- Battery Configuration -- #
        self.battery_configs = OmegaConf.create(battery_configs)
        cfg_list = [self.battery_configs[aid] for aid in self.agents]
        self.batt_cap = jnp.array([c["capacity"] for c in cfg_list])
        self.batt_loss_coef = jnp.array([c["loss_coefficient"] for c in cfg_list])
        self.batt_nominal_power = jnp.array([c["nominal_power"] for c in cfg_list])
        self.batt_round_trip_eff = jnp.array(
            [c["round_trip_efficiency"] for c in cfg_list]
        )
        self.batt_init_soc = jnp.array([c["soc"] for c in cfg_list])
        self.batt_depth_of_discharge = jnp.array(
            [c["depth_of_discharge"] for c in cfg_list]
        )

        # -- Exchange Mechanism -- #
        self.exchange: BaseExchangeMechanism = (
            get_mechanism(exchange_name, **exchange_kwargs)
            if exchange_kwargs
            else get_mechanism(exchange_name)
        )

        # -- Action Space -- #
        self.action_spaces = {
            a: {
                Box(
                    low=ActionSpec.low,
                    high=ActionSpec.high,
                    shape=ActionSpec.shape,
                    dtype=jnp.float32,
                ),
            }
            for a in self.agents
        }

    @property
    def observation_mapping(self):
        """Mapping of observation indices to their meanings."""
        return ObsIdx

    def id_to_index(self, id_):
        """Map agent ID to index."""
        return self.dataset.households.id_to_index[id_]

    def index_to_id(self, index):
        """Map agent index to ID."""
        return self.dataset.households.index_to_id[index]

    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey):
        """Vectorised environment reset."""

        def _make_batt(cap, loss_coef, p_nom, rte, soc0, dod):
            return create_battery(
                capacity=cap,
                loss_coefficient=loss_coef,
                nominal_power=p_nom,
                round_trip_efficiency=rte,
                soc=soc0,
                depth_of_discharge=dod,
            )

        batteries = jax.vmap(_make_batt)(
            self.batt_cap,
            self.batt_loss_coef,
            self.batt_nominal_power,
            self.batt_round_trip_eff,
            self.batt_init_soc,
            self.batt_depth_of_discharge,
        )

        state = EnergyState(done=False, step=0, batteries=batteries)
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def _step_core(
        self,
        key: chex.PRNGKey,
        state: EnergyState,
        a_batt: jnp.ndarray,  # (N,)
        a_qty: jnp.ndarray,  # (N,)
        a_price: jnp.ndarray,  # (N,)
    ):
        """Vectorised core of one environment step.

        This contains the exact same logic you already have; we just feed arrays.

        Args:
            key (chex.PRNGKey): JAX random key.
            state (EnergyState): Current environment state.
            a_batt (jnp.ndarray): Battery action for each agent, shape (N,).
            a_qty (jnp.ndarray): Trade quantity action for each agent, shape (N,).
            a_price (jnp.ndarray): Trade price action for each agent, shape (N,).

        Returns:
            observations (chex.Array): Observations after the step, shape (N, 6).
            next_state (EnergyState): Next environment state.
            rewards (jnp.ndarray): Rewards for each agent, shape (N,).
            dones (Dict[str, bool]): Done flags for each agent and "__all__".
            info (Dict): Additional information from the step.
        """
        # ---- time bookkeeping -------------------------------------------------
        next_step = state.step + 1
        done_flag = next_step >= self.num_timesteps - 1
        dones = {aid: done_flag for aid in self.agents}
        dones["__all__"] = done_flag

        # ---- net load without battery ----------------------------------------
        prod = self.dataset.households.production[:, next_step]
        cons = self.dataset.households.consumption[:, next_step]
        net_load = cons - prod  # +deficit / -surplus

        # ---- battery charge/discharge ----------------------------------------
        req_kwh = a_batt * self.batt_nominal_power
        # If grid-charging is disabled, clip positive net import for charging
        req_kwh = jnp.where(
            (self.allow_grid_charging | (req_kwh >= 0.0)),
            req_kwh,
            jnp.maximum(-jnp.clip(-net_load, 0.0), req_kwh),
        )
        new_batts, batt_energy = charge_battery(state.batteries, req_kwh)
        residual = net_load - batt_energy

        # ---- exchange mechanism ----------------------------------------------
        tou = self.dataset.pricing.time_of_use_tariff[next_step]
        fit = self.dataset.pricing.feed_in_tariff[next_step]

        # scale quantity by absolute residual; map normalized price to [FiT, ToU]
        trade_q = a_qty * jnp.abs(residual)
        p_mid = 0.5 * (tou + fit)
        p_rng = 0.5 * (tou - fit)
        trade_p = p_mid + a_price * p_rng

        exch_input = ExchangeInput(
            net_load=residual,
            trade_quantity=trade_q,
            trade_price=trade_p,
            tou_price=tou,
            fit_price=fit,
        )
        exch_output = self.exchange.settle(exch_input)

        # ---- costs, rewards, metrics -----------------------------------------
        exch_net_grid = exch_output.net_grid_energy
        price_grid = jnp.where(exch_net_grid > 0, tou, fit)
        grid_cost = exch_net_grid * price_grid

        price_internal = exch_output.prices
        # internal_transfers: +sold to community, -bought from community
        # internal_cost should be: -(internal_transfers * price_internal)
        # so if I sold (positive), my cost is negative (revenue).
        internal_cost = -(exch_output.internal_transfers * price_internal)

        rewards = -(grid_cost + internal_cost)

        renewable_spill = jnp.where(exch_net_grid < 0, -exch_net_grid, 0.0)

        # Pad rewards to match observation matrix size (N or N+1)
        # We also need to pad other fields in info if they are used in vectorized history
        if self.has_pricing_agent:
            # rewards: (N,) -> (N+1,)
            rewards = jnp.pad(rewards, (0, 1))
            net_grid_out = jnp.pad(exch_net_grid, (0, 1))
            spill_out = jnp.pad(renewable_spill, (0, 1))
            price_out = jnp.pad(exch_output.prices, (0, 1))
            trade_out = jnp.pad(exch_output.internal_transfers, (0, 1))
        else:
            net_grid_out = exch_net_grid
            spill_out = renewable_spill
            price_out = exch_output.prices
            trade_out = exch_output.internal_transfers

        info = {
            "residual": residual,
            "a_batt": a_batt,
            "a_qty": a_qty,
            "a_price": a_price,
            "tou_price": tou,
            "fit_price": fit,
            "requested_energy": req_kwh,
            "effective_batt": batt_energy,
            "net_grid_energy": net_grid_out,
            # Pricing Information
            "exch_price": price_out,
            "grid_cost": grid_cost,
            "internal_cost": internal_cost,
            "internal_trade": trade_out,
            # Additional Metrics
            "renewable_spill": spill_out,
        }

        next_state = EnergyState(done=done_flag, step=next_step, batteries=new_batts)
        return self.get_obs(next_state), next_state, rewards, dones, info

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: chex.PRNGKey, state: EnergyState, actions: Dict[str, chex.Array]
    ):
        """Current API (dict of per-agent 3-vectors) — unchanged behavior.

        Packs the dict → (N,3) and delegates to the core.
        """
        # actions[aid] must be length-3 arrays [a_batt, a_qty, a_price]
        actions_mat = jnp.stack([actions[aid] for aid in self.agents])  # (N,3)
        a_batt, a_qty, a_price = actions_mat.T
        return self._step_core(key, state, a_batt, a_qty, a_price)

    @partial(jax.jit, static_argnums=(0,))
    def step_env_matrix(
        self, key: chex.PRNGKey, state: EnergyState, actions_mat: jnp.ndarray
    ):
        """New API for JIT/scan friendly training loops.

        actions_mat shape: (N or N+1, 3) with columns [a_batt, a_qty, a_price].
        If N+1, the last agent's trade_price sets the internal_price.
        """
        if self.has_pricing_agent:
            # Stage 1: Pricing Agent sets the price (from its action index 2)
            global_price_action = actions_mat[-1, 2]
            state = state.replace(internal_price=global_price_action)
            
            # Stage 2: Buildings act based on that price
            building_actions = actions_mat[:-1]
            a_batt, a_qty, a_price = building_actions.T
        else:
            a_batt, a_qty, a_price = actions_mat.T
            
        return self._step_core(key, state, a_batt, a_qty, a_price)

    def get_obs(self, state: EnergyState) -> chex.Array:
        """Vectorised observation function.

        Observation vector for each building agent:
        [0] consumption (kWh)
        [1] production (kWh)
        [2] hour of day (0-23)
        [3] day of week (0-6)
        [4] battery state of charge (kWh)
        [5] battery capacity (kWh)
        [6] internal price (€/kWh) - Optional (if has_pricing_agent)

        Observation for Pricing Agent (if exists):
        [0] total consumption (kWh)
        [1] total production (kWh)
        [2] hour of day
        [3] day of week
        [4] grid ToU
        [5] grid FiT

        Returns:
            observations (chex.Array): Array (N or N+1, ObsDim)
        """

        def _create_building_obs(_idx: chex.Array) -> chex.Array:
            # Basic 6 dims
            obs = jnp.array([
                self.dataset.households.consumption[_idx, state.step],
                self.dataset.households.production[_idx, state.step],
                self.dataset.temporal.hour_of_day[state.step],
                self.dataset.temporal.day_of_week[state.step],
                state.batteries.soc[_idx],
                self.batt_cap[_idx],
            ])
            if self.has_pricing_agent:
                obs = jnp.append(obs, state.internal_price)
            return obs

        building_obs = jax.vmap(_create_building_obs)(
            jnp.array(list(self.dataset.households.id_to_index.values()))
        )
        
        if not self.has_pricing_agent:
            return building_obs
            
        # Pricing Agent Obs (6 dims)
        pricing_obs = jnp.array([
            jnp.sum(self.dataset.households.consumption[:, state.step]),
            jnp.sum(self.dataset.households.production[:, state.step]),
            self.dataset.temporal.hour_of_day[state.step],
            self.dataset.temporal.day_of_week[state.step],
            self.dataset.pricing.time_of_use_tariff[state.step],
            self.dataset.pricing.feed_in_tariff[state.step],
        ])
        
        # Pad pricing_obs to match building_obs width (7)
        pricing_obs_padded = jnp.pad(pricing_obs, (0, building_obs.shape[1] - pricing_obs.size))
        
        return jnp.concatenate([building_obs, pricing_obs_padded[None, :]], axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: EnergyState) -> Dict[str, chex.Array]:
        """Returns the available actions for each agent.

        For simplicity, all actions are always available in this environment.

        Args:
            state (EnergyState): The current state of the environment.

        Returns:
            Dict[str, chex.Array]: A dictionary mapping agent IDs to available actions.
        """
        return {
            agent: jnp.array([1.0]) for agent in self.dataset.households.id_to_index
        }

    @property
    def agent_classes(self):
        """Returns a dictionary with agent classes."""
        return {"agent": list(self.dataset.households.id_to_index.keys())}
