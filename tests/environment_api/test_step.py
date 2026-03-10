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
    # Create constant consumption and production arrays
    cons = jnp.ones((2, num_steps)) * 2.0
    prod = jnp.ones((2, num_steps)) * 1.0

    # Create and return the dataset
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
    """Fixture to create a simple environment with two agents."""
    data = make_dummy_dataset()
    battery_cfg = {
        "agent_0": get_default_battery_cfg(),
        "agent_1": get_disabled_battery_cfg(),
    }

    _env = JAXECEnvironment(data, battery_configs=battery_cfg)
    return _env


def test_battery_soc_update(simple_env):
    """Test that battery SoC updates correctly for an agent with a battery."""
    obs, state = simple_env.reset(jax.random.PRNGKey(0))
    soc_idx = simple_env.observation_mapping.BATTERY_SOC
    soc_before = obs[0, soc_idx]

    actions = {
        "agent_0": jnp.array([-1.0, 0.0, 0.0]),  # charge
        "agent_1": jnp.array([-1.0, 0.0, 0.0]),  # no battery, should not change
    }

    obs2, state, *_ = simple_env.step(jax.random.PRNGKey(1), state, actions)
    soc_after = obs2[0, soc_idx]
    soc_disabled = obs2[1, soc_idx]

    assert soc_after > soc_before
    assert soc_disabled == 0.0


def test_residual_matches_net_load(simple_env):
    """Test that the net grid energy matches the net load when no exchange is used."""
    obs, state = simple_env.reset(jax.random.PRNGKey(0))
    net_0 = (
        obs[0, simple_env.observation_mapping.CONSUMPTION]
        - obs[0, simple_env.observation_mapping.PRODUCTION]
    )
    net_1 = (
        obs[1, simple_env.observation_mapping.CONSUMPTION]
        - obs[1, simple_env.observation_mapping.PRODUCTION]
    )

    actions = {
        "agent_0": jnp.array([0.0, 0.0, 0.0]),
        "agent_1": jnp.array([0.0, 0.0, 0.0]),
    }

    _, state, _, _, info = simple_env.step(jax.random.PRNGKey(2), state, actions)
    # Should match because NoExchange does nothing
    assert jnp.isclose(info["net_grid_energy"][0], net_0)
    assert jnp.isclose(info["net_grid_energy"][1], net_1)  # agent_1's cons - prod

    # Reset the environment again
    obs, state = simple_env.reset(jax.random.PRNGKey(0))
    net_0 = (
        obs[0, simple_env.observation_mapping.CONSUMPTION]
        - obs[0, simple_env.observation_mapping.PRODUCTION]
    )
    net_1 = (
        obs[1, simple_env.observation_mapping.CONSUMPTION]
        - obs[1, simple_env.observation_mapping.PRODUCTION]
    )
    soc_0 = obs[0, simple_env.observation_mapping.BATTERY_SOC]
    soc_1 = obs[1, simple_env.observation_mapping.BATTERY_SOC]

    # Now charge both agent's batteries at 50%
    actions = {
        "agent_0": jnp.array([-0.5, 0.0, 0.0]),
        "agent_1": jnp.array([-0.5, 0.0, 0.0]),
    }

    _, state, _, _, info = simple_env.step(jax.random.PRNGKey(3), state, actions)
    # Should match with net_1 for agent 1 (no battery), but not for agent 0
    assert not jnp.isclose(info["net_grid_energy"][0], net_0)
    assert jnp.isclose(info["net_grid_energy"][1], net_1)
    # Check the battery SoC
    assert state.batteries.soc[0] > soc_0  # should increase
    assert jnp.isclose(state.batteries.soc[1], soc_1)  # should not change


def test_reward_matches_energy_cost(simple_env):
    """Test that the reward matches the energy cost for each agent."""
    obs, state = simple_env.reset(jax.random.PRNGKey(0))
    tou = 0.30
    fit = 0.05

    # Discharge agent_0 to increase grid export
    actions = {
        "agent_0": jnp.array([-1.0, 0.0, 0.0]),
        "agent_1": jnp.array([0.0, 0.0, 0.0]),
    }

    _, state, reward, _, info = simple_env.step(jax.random.PRNGKey(3), state, actions)
    # net_grid_energy positive → buying, negative → selling
    for i in range(2):
        ng = info["net_grid_energy"][i]
        price = tou if ng > 0 else fit
        expected = -ng * price
        assert jnp.isclose(reward[i], expected, atol=1e-4)


def test_next_obs_matches_dataset(simple_env):
    """Test that the next observation matches the dataset values."""
    dataset = simple_env.dataset
    obs, state = simple_env.reset(jax.random.PRNGKey(0))

    actions = {
        "agent_0": jnp.array([0.0, 0.0, 0.0]),
        "agent_1": jnp.array([0.0, 0.0, 0.0]),
    }
    obs2, state2, *_ = simple_env.step(jax.random.PRNGKey(1), state, actions)

    prod_idx = simple_env.observation_mapping.PRODUCTION
    cons_idx = simple_env.observation_mapping.CONSUMPTION

    assert jnp.isclose(obs2[0, prod_idx], dataset.households.production[0, 1])
    assert jnp.isclose(obs2[0, cons_idx], dataset.households.consumption[0, 1])
    assert jnp.isclose(obs2[1, prod_idx], dataset.households.production[1, 1])
    assert jnp.isclose(obs2[1, cons_idx], dataset.households.consumption[1, 1])
