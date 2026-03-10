"""Contains unit tests for the reset function of the environment."""

import jax
import jax.numpy as jnp
import pytest

from jax_ec.data.types import (
    HouseholdData,
    JAXECDataset,
    Metadata,
    PricingData,
    TemporalData,
)
from jax_ec.environment.battery import get_default_battery_cfg, get_disabled_battery_cfg
from jax_ec.environment.env import JAXECEnvironment


def make_dummy_dataset(num_steps=24):
    """Create a simple 2-agent dataset for testing.

    Args:
        num_steps: Number of time steps in the dataset. Default is 24 (1 day).

    Returns:
        A JAXECDataset instance with constant consumption and production.
    """
    cons = jnp.ones((2, num_steps)) * 2.0
    prod = jnp.ones((2, num_steps)) * 1.0

    return JAXECDataset(
        metadata=Metadata(
            name="reset-test",
            description="simple test dataset",
            number_of_agents=2,
            pricing_data_included=True,
        ),
        temporal=TemporalData(
            timestamps=[f"2025-01-01T{h:02d}:00:00" for h in range(num_steps)],
            hour_of_day=jnp.arange(num_steps),
            day_of_week=jnp.zeros(num_steps),
            month_of_year=jnp.ones(num_steps),
            year=2025 * jnp.ones(num_steps),
        ),
        pricing=PricingData(
            feed_in_tariff=0.05 * jnp.ones(num_steps),
            time_of_use_tariff=0.30 * jnp.ones(num_steps),
        ),
        households=HouseholdData(
            id_to_index={"agent_0": 0, "agent_1": 1},
            index_to_id={0: "agent_0", 1: "agent_1"},
            consumption=cons,
            production=prod,
        ),
    )


@pytest.fixture
def simple_env():
    """Fixture to create a simple environment with two agents.

    Agent 0 has a default battery config with initial SoC = 0.5.
    Agent 1 has a disabled battery config (SoC = 0).

    Returns:
        A JAXECEnvironment instance.
    """
    data = make_dummy_dataset()
    battery_cfg = {
        "agent_0": get_default_battery_cfg(),
        "agent_1": get_disabled_battery_cfg(),
    }

    _env = JAXECEnvironment(data, battery_configs=battery_cfg)
    return _env


def test_reset_initial_soc_correct(simple_env):
    """Test that the initial state of charge (SoC) is set correctly for each agent."""
    obs, state = simple_env.reset(jax.random.PRNGKey(0))
    soc_idx = simple_env.observation_mapping.BATTERY_SOC

    socs = obs[:, soc_idx]
    # Agent 0 has initial SoC = 0.5, agent 1 is disabled -> SoC = 0
    assert jnp.isclose(socs[0], 0.5)
    assert jnp.isclose(socs[1], 0.0)


def test_reset_after_steps(simple_env):
    """Test that the state of charge (SoC) is reset correctly after a few steps."""
    obs, state = simple_env.reset(jax.random.PRNGKey(1))
    init_capacity = state.batteries.capacity
    actions = {
        "agent_0": jnp.array([1.0, 0.0, 0.0]),
        "agent_1": jnp.array([0.0, 0.0, 0.0]),
    }
    for _ in range(5):
        obs, state, *_ = simple_env.step(jax.random.PRNGKey(2), state, actions)

    # Now reset and verify SoC is back to original
    obs_reset, state_reset = simple_env.reset(jax.random.PRNGKey(3))
    soc_idx = simple_env.observation_mapping.BATTERY_SOC
    assert jnp.isclose(obs_reset[0, soc_idx], 0.5)
    assert jnp.isclose(obs_reset[1, soc_idx], 0.0)
    # Check that the capacity is the same (i.e. we have a new battery)
    assert jnp.all(state_reset.batteries.capacity == init_capacity)


def test_observation_shape_and_keys(simple_env):
    """Test that the observation has the correct shape and includes key features."""
    obs, _ = simple_env.reset(jax.random.PRNGKey(4))
    assert obs.shape[0] == 2  # two agents
    assert obs.shape[1] >= 5  # should include hour, prod, cons, soc, cap


def test_battery_state_created_properly(simple_env):
    """Test that the battery state is created properly in the environment state."""
    _, state = simple_env.reset(jax.random.PRNGKey(5))
    batts = state.batteries
    print(f"Battery states: {batts}")
    assert jnp.isclose(batts.soc[0], 0.5)
    assert batts.capacity[0] > 0.0
    assert batts.capacity[1] == 0.0
