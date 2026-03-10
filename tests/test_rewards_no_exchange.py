"""This file contains tests for the JAXECEnvironment class.

It focuses specifically on testing the environment with no local exchange.
The tests cover scenarios with grid charging enabled and disabled.
It also verifies that battery state of charge (SoC) updates correctly based on
available surplus energy and that trading actions remain consistent regardless of
the grid-charging setting.
"""

import jax
import jax.numpy as jnp

from jax_ec.data.types import (
    HouseholdData,
    JAXECDataset,
    Metadata,
    PricingData,
    TemporalData,
)
from jax_ec.environment.battery import get_default_battery_cfg
from jax_ec.environment.env import JAXECEnvironment


def make_static_dataset(net_load, num_steps=1):
    """Create a static dataset with specified net load for testing.

    Args:
        net_load: Array of net load values for each agent.
        num_steps: Number of time steps in the dataset. Default is 1.

    Returns:
        A JAXECDataset instance with the specified net load.
    """
    n_agents = len(net_load)
    cons = jnp.array([net_load + 1.0] * n_agents).reshape(n_agents, num_steps)
    prod = jnp.array([1.0] * n_agents * num_steps).reshape(n_agents, num_steps)

    return JAXECDataset(
        metadata=Metadata(
            name="test",
            description="test",
            number_of_agents=n_agents,
            pricing_data_included=True,
        ),
        temporal=TemporalData(
            timestamps=["2025-01-01T00:00:00"] * num_steps,
            hour_of_day=jnp.array([0] * num_steps),
            day_of_week=jnp.array([0] * num_steps),
            month_of_year=jnp.array([1] * num_steps),
            year=jnp.array([2025] * num_steps),
        ),
        pricing=PricingData(
            feed_in_tariff=jnp.array([0.05] * num_steps),
            time_of_use_tariff=jnp.array([0.30] * num_steps),
        ),
        households=HouseholdData(
            id_to_index={"agent_0": 0},
            index_to_id={0: "agent_0"},
            consumption=cons,
            production=prod,
        ),
    )


def make_env(grid_enabled: bool):
    """Create a JAXECEnvironment with specified grid charging setting.

    Args:
        grid_enabled: Whether grid charging is enabled.

    Returns:
        Instance of JAXECEnvironment configured with specified grid charging setting.
    """
    dataset = make_static_dataset(net_load=jnp.array([1.0]))
    _bat_cfg = get_default_battery_cfg()
    # Set the round-trip efficiency to 1.0 for testing
    _bat_cfg["round_trip_efficiency"] = 1.0
    # Set the loss coefficient to 0.0 for testing
    _bat_cfg["loss_coefficient"] = 0.0
    return JAXECEnvironment(
        dataset=dataset,
        battery_configs={"agent_0": _bat_cfg},
        exchange_name="no_exchange",
        allow_grid_charging=grid_enabled,
    )


def test_net_load_with_grid_charging_enabled():
    """Test net load calculation when grid charging is enabled.

    When grid charging is enabled, the battery should be able to charge from the grid
    if there is a deficit. This test verifies that the net grid energy reflects
    the battery charging behavior correctly.
    """
    env = make_env(grid_enabled=True)
    _, state = env.reset(jax.random.PRNGKey(0))

    actions = {"agent_0": jnp.array([-1.0, 0.0, 0.0])}  # Request 1 unit battery charge
    _, _, _, _, info = env.step_env(jax.random.PRNGKey(1), state, actions)

    assert info["effective_batt"][0] < 0.0  # Energy was drawn to charge
    assert jnp.isclose(info["net_grid_energy"][0], 1.0 - info["effective_batt"][0])


def test_net_load_with_grid_charging_disabled():
    """Test net load calculation when grid charging is disabled.

    When grid charging is disabled, the battery should not charge from the grid.
    This test verifies that the net grid energy reflects the battery charging behavior
    correctly. The battery should not charge when there is no surplus energy.
    """
    env = make_env(grid_enabled=False)
    _, state = env.reset(jax.random.PRNGKey(0))

    actions = {"agent_0": jnp.array([-1.0, 0.0, 0.0])}  # Request 1 unit battery charge
    _, _, _, _, info = env.step_env(jax.random.PRNGKey(1), state, actions)

    # Battery charging should be clipped to available surplus = 0
    assert jnp.isclose(info["effective_batt"][0], 0.0)
    assert jnp.isclose(info["net_grid_energy"][0], 1.0)  # Only base load


def test_discharge_behavior_grid_mode():
    """Test battery discharge behavior when grid charging is enabled.

    When grid charging is enabled, the battery should be able to discharge to
    supply the load. This test verifies that the net grid energy reflects the
    battery discharging behavior correctly.
    """
    env = make_env(grid_enabled=True)
    _, state = env.reset(jax.random.PRNGKey(0))

    # Set SoC to full before discharging
    state = state.replace(batteries=state.batteries.replace(soc=jnp.array([1.0])))
    actions = {"agent_0": jnp.array([1.0, 0.0, 0.0])}  # Discharge full rate
    _, _, _, _, info = env.step_env(jax.random.PRNGKey(2), state, actions)

    assert info["effective_batt"][0] > 0.0
    assert jnp.isclose(info["net_grid_energy"][0], 1.0 - info["effective_batt"][0])


def test_batt_soc_does_not_rise_without_surplus():
    """Ensure battery SoC does not increase when there is no surplus energy.

    This test verifies that when grid charging is disabled and there is no surplus
    energy (net load is positive), the battery state of charge (SoC) remains unchanged
    even if a charge action is requested.
    """
    env = make_env(grid_enabled=False)
    obs, state = env.reset(jax.random.PRNGKey(0))

    soc_idx = env.observation_mapping.BATTERY_SOC
    soc_before = obs[0, soc_idx]

    # request charge when there is *no* surplus (net_load = +1 kWh)
    actions = {"agent_0": jnp.array([-1.0, 0.0, 0.0])}
    obs2, _, *_ = env.step_env(jax.random.PRNGKey(1), state, actions)

    soc_after = obs2[0, soc_idx]
    assert jnp.isclose(soc_after, soc_before)  # SoC unchanged


def test_trade_unchanged_by_grid_setting():
    """Ensure exchange logic remains consistent regardless of grid-charging flag.

    This test verifies that the trading actions and resulting quantities remain
    consistent whether grid charging is enabled or disabled. It checks that the
    traded quantity ('a_qty') is the same in both scenarios.
    """
    # Use same setup with trade actions
    env1 = make_env(grid_enabled=True)
    env2 = make_env(grid_enabled=False)

    _, state1 = env1.reset(jax.random.PRNGKey(0))
    _, state2 = env2.reset(jax.random.PRNGKey(0))

    action = jnp.array([0.0, 1.0, 0.0])
    actions = {"agent_0": action}

    _, _, _, _, info1 = env1.step_env(jax.random.PRNGKey(1), state1, actions)
    _, _, _, _, info2 = env2.step_env(jax.random.PRNGKey(1), state2, actions)

    assert jnp.isclose(info1["a_qty"], info2["a_qty"])
