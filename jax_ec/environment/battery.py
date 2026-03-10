"""Implementation of electrical battery systems for households in the energy community.

The battery implementation is a simplified version of the battery model in the CityLearn
environment [1].

Sign convention for charging and discharging:
    * +ve action -> discharge (energy sent to AC bus)
    * −ve action -> charge    (energy drawn from AC bus to battery)

References:
    [1] "CityLearn: Standardizing Research in Multi-Agent Reinforcement Learning for
         Demand Response and Urban Energy Management", (Vàsquez-Canteli et al., 2020)
"""

from typing import Tuple

import jax.numpy as jnp
from chex import dataclass


@dataclass
class BatteryState:
    """Dataclass for battery state and parameters."""

    capacity: float  # kWh   (current usable capacity)
    soc: float  # 0‥1  (state‑of‑charge)
    nominal_power: float  # kW   (max charge / discharge per step)
    round_trip_efficiency: float  # 0‥1
    capacity_loss_coefficient: float  # ∆capacity per kWh throughput
    loss_coefficient: float  # daily or hourly self‑discharge
    depth_of_discharge: float  # 0‥1  (max usable fraction below 100 %)


def _clip_request(state: BatteryState, req_e: float) -> float:
    """Clips the requested energy to the battery's limits.

    Args:
        state: BatteryState object containing battery parameters.
        req_e: Requested energy (kWh) to be charged or discharged.

    Returns:
        Clipped requested energy (kWh) based on battery limits.
    """
    # nominal power limit  (assuming 1 h timestep)
    req_e = jnp.clip(req_e, -state.nominal_power, state.nominal_power)

    # SoC limits
    min_soc = 1.0 - state.depth_of_discharge
    max_charge = -(1.0 - state.soc) * state.capacity
    max_discharge = (state.soc - min_soc) * state.capacity

    return jnp.clip(req_e, max_charge, max_discharge)


# -------------------------------------------------------------------------
def charge_battery(
    battery: BatteryState,
    requested_energy: float,  # kWh (+ discharge, – charge)
) -> Tuple[BatteryState, float]:
    """Charges the battery.

    Returns (new_state, effective_energy),

    where effective_energy is what actually flowed **to / from the AC bus**:
        +ve  → discharged to meet load
        –ve  → drawn from grid to charge
    """
    # 1) clip request to feasible envelope
    eff_e = _clip_request(battery, requested_energy)  # kWh

    # 2) translate to change in stored energy considering efficiency
    eta = battery.round_trip_efficiency
    delta_store = jnp.where(
        battery.capacity > 0,
        jnp.where(
            eff_e >= 0,
            -eff_e / eta,  # discharge: battery loses more than grid gains
            -eff_e * eta,  # charge   : battery gains less than grid draws
        ),
        0.0,
    )

    # 3) update SoC
    new_soc = jnp.clip(
        battery.soc + delta_store / battery.capacity,
        1.0 - battery.depth_of_discharge,
        1.0,
    )

    # 4) capacity fade (optional, simple linear model)
    throughput = jnp.abs(delta_store)
    new_capacity = jnp.maximum(
        battery.capacity - battery.capacity_loss_coefficient * throughput,
        0.0,
    )

    # 5) self‑discharge / standing losses
    # new_soc *= 1.0 - battery.loss_coefficient
    new_soc = jnp.where(
        battery.capacity > 0.0,
        jnp.clip(
            battery.soc + delta_store / battery.capacity,
            1.0 - battery.depth_of_discharge,
            1.0,
        ),
        0.0,  # disabled battery: no change
    )

    new_state = battery.__class__(  # works with dataclass & Flax struct
        capacity=new_capacity,
        soc=new_soc,
        nominal_power=battery.nominal_power,
        round_trip_efficiency=battery.round_trip_efficiency,
        capacity_loss_coefficient=battery.capacity_loss_coefficient,
        loss_coefficient=battery.loss_coefficient,
        depth_of_discharge=battery.depth_of_discharge,
    )

    # eff_e is what the env should subtract from net_load
    return new_state, eff_e


def create_battery(
    capacity: float = 10.0,
    nominal_power: float = 5.0,
    soc: float = 0.5,
    round_trip_efficiency: float = 0.95,
    capacity_loss_coefficient: float = 0.01,
    loss_coefficient: float = 0.001,
    depth_of_discharge: float = 1.0,
) -> BatteryState:
    """Creates a BatteryState object with the given parameters.

    Args:
        capacity: Usable capacity in kWh.
        nominal_power: Max charge/discharge power in kW.
        soc: Initial state of charge (0‥1).
        round_trip_efficiency: Round-trip efficiency (0‥1).
        capacity_loss_coefficient: Capacity loss per kWh throughput.
        loss_coefficient: Self-discharge rate (daily or hourly).
        depth_of_discharge: Depth of discharge (0‥1).

    Returns:
        BatteryState: A BatteryState object with the specified parameters.
    """
    return BatteryState(
        capacity=capacity,
        soc=soc,
        nominal_power=nominal_power,
        round_trip_efficiency=round_trip_efficiency,
        capacity_loss_coefficient=capacity_loss_coefficient,
        loss_coefficient=loss_coefficient,
        depth_of_discharge=depth_of_discharge,
    )


def get_default_battery_cfg():
    """Returns a default battery configuration dictionary.

    Returns:
        dict: A dictionary containing default battery parameters.
    """
    return {
        "capacity": 10.0,
        "nominal_power": 3.32,
        "soc": 0.5,
        "round_trip_efficiency": 0.99,
        "capacity_loss_coefficient": 1e-05,
        "loss_coefficient": 0.0001,
        "depth_of_discharge": 1.0,
    }


def get_disabled_battery_cfg():
    """Returns a disabled battery configuration dictionary.

    Returns:
        dict: A dictionary containing parameters for a disabled battery.
    """
    return {
        "capacity": 0.0,
        "nominal_power": 0.0,
        "soc": 0.0,
        "round_trip_efficiency": 1.0,
        "capacity_loss_coefficient": 0.0,
        "loss_coefficient": 0.0,
        "depth_of_discharge": 1.0,
    }
