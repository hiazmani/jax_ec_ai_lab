"""Rule-based Agent.

Rules:
1. Charge the battery when there is excess production.
2. Discharge the battery when there is excess consumption.
3. Charge only during specified charge hours.
4. Discharge only during specified discharge hours.
"""

import jax.numpy as jnp


def rbc_batt_action(
    obs: jnp.ndarray,
    obs_mapping,
    charge_hours=(10, 15),  # inclusive [start, end]
    discharge_hours=((6, 9), (17, 22)),  # two windows: [start, end) & [start, end]
    aggressiveness: float = 1.0,
) -> jnp.ndarray:
    """Time-window RBC (previous strategy).

    * Charge in the charge window using surplus, limited by remaining SoC headroom.
    * Discharge in the discharge windows to cover deficit, limited by stored energy.
    * Return a normalized action in [-1, 1] (fraction of nominal power requested).
      (Env multiplies by nominal power and clips against net surplus/deficit.)
    """
    CONS = obs_mapping.CONSUMPTION
    PROD = obs_mapping.PRODUCTION
    HOD = obs_mapping.HOUR_OF_DAY
    SOC = obs_mapping.BATTERY_SOC
    CAP = obs_mapping.BATTERY_CAP

    cons = obs[CONS]
    prod = obs[PROD]
    hour = obs[HOD]
    soc = obs[SOC]
    cap = obs[CAP]

    # Net energy this step (kWh): +surplus means charge, -deficit means discharge.
    net = prod - cons

    # Energy-limited charge/discharge potentials (kWh), respecting SoC and capacity.
    # (We use max(cap, 1e-9) later to avoid division-by-zero.)
    charge_possible_kwh = jnp.clip(net, 0.0, cap * (1.0 - soc))
    discharge_possible_kwh = jnp.clip(
        -net, 0.0, cap * (1.0 - soc + soc) * 0.0 + cap * soc
    )
    # (equivalently: jnp.clip(-net, 0.0, cap * soc))

    # Time windows
    in_charge = (hour >= charge_hours[0]) & (hour <= charge_hours[1])
    in_discharge = (
        (hour >= discharge_hours[0][0]) & (hour < discharge_hours[0][1])
    ) | ((hour >= discharge_hours[1][0]) & (hour <= discharge_hours[1][1]))

    # Convert kWh to action fraction by dividing by capacity
    cap_safe = jnp.maximum(cap, 1e-9)
    a_charge = jnp.where(
        in_charge, (charge_possible_kwh / cap_safe) * aggressiveness, 0.0
    )
    # Discharge is negative action
    a_discharge = jnp.where(
        in_discharge, -(discharge_possible_kwh / cap_safe) * aggressiveness, 0.0
    )

    a = a_charge + a_discharge
    return jnp.clip(a, -1.0, 1.0)


def rbc_trade_action(
    obs: jnp.ndarray,
    obs_mapping,
) -> jnp.ndarray:
    """Returns a trade quantity action for RBC.

    Standard strategy: Try to trade 100% of your remaining surplus/deficit.
    Action 1.0 means 'sell everything surplus', -1.0 means 'buy everything deficit'.
    """
    # Simply return 1.0. 
    # In JAXECEnvironment, trade_q = a_qty * abs(residual).
    # Since a_qty=1.0, trade_q = abs(residual).
    # The environment logic further determines if it's buy or sell based on the sign of residual.
    # Actually, JAXECEnvironment._step_core uses: trade_q = a_qty * jnp.abs(residual).
    # It assumes a_qty is signed or the mechanism handles it?
    # Let's check environment again.
    return 1.0
