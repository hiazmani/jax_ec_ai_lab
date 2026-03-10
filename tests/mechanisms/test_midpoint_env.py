"""Unit tests for midpoint exchange environment."""

import jax
import jax.numpy as jnp

from jax_ec.data.types import (
    HouseholdData,
    JAXECDataset,
    Metadata,
    PricingData,
    TemporalData,
)
from jax_ec.environment.battery import get_disabled_battery_cfg
from jax_ec.environment.env import JAXECEnvironment

TOU = 0.30
FIT = 0.05
MID = 0.5 * (TOU + FIT)  # 0.175


def make_dataset(buyer_load, seller_surplus):
    """Construct 2‑agent 1‑step dataset.

    Agent 0 (buyer): consumption = buyer_load, production = 0
    Agent 1 (seller): consumption = 0, production = seller_surplus

    Args:
        buyer_load: Load of the buying agent.
        seller_surplus: Surplus of the selling agent.

    Returns:
        A JAXECDataset instance representing the scenario.
    """
    # Create consumption and production arrays
    cons = jnp.array([[buyer_load], [0.0]], dtype=jnp.float32)
    prod = jnp.array([[0.0], [seller_surplus]], dtype=jnp.float32)

    # Construct and return the dataset with metadata
    return JAXECDataset(
        # Dummy metadata
        metadata=Metadata(
            name="mid-env",
            description="midpoint env test",
            number_of_agents=2,
            pricing_data_included=True,
        ),
        # Single timestep data
        temporal=TemporalData(
            timestamps=["2025-01-01T00:00:00"],
            hour_of_day=jnp.array([0]),
            day_of_week=jnp.array([0]),
            month_of_year=jnp.array([1]),
            year=jnp.array([2025]),
        ),
        # Pricing data
        pricing=PricingData(
            feed_in_tariff=jnp.array([FIT]), time_of_use_tariff=jnp.array([TOU])
        ),
        # Two households: buyer and seller
        households=HouseholdData(
            id_to_index={"buyer": 0, "seller": 1},
            index_to_id={0: "buyer", 1: "seller"},
            consumption=cons,
            production=prod,
        ),
    )


def make_env(dataset):
    """Create a JAXECEnvironment with midpoint exchange and no grid charging.

    Args:
        dataset: The JAXECDataset to use in the environment.

    Returns:
        A JAXECEnvironment instance configured for midpoint exchange.
    """
    # Disable batteries for simplicity
    cfg = {aid: get_disabled_battery_cfg() for aid in dataset.households.id_to_index}
    # Create and return the environment
    return JAXECEnvironment(
        dataset=dataset,
        battery_configs=cfg,
        exchange_name="midpoint",
        allow_grid_charging=False,
    )


def test_midpoint_env_balanced():
    """Test midpoint environment with balanced buyer and seller."""
    # Create dataset and environment
    ds = make_dataset(buyer_load=2.0, seller_surplus=2.0)
    env = make_env(ds)

    # Reset environment and define actions
    obs, state = env.reset(jax.random.PRNGKey(0))
    actions = {
        "buyer": jnp.array([0.0, -1.0, 0.0]),
        "seller": jnp.array([0.0, 1.0, 0.0]),
    }
    # Step the environment
    _, _, rewards, _, info = env.step_env(jax.random.PRNGKey(1), state, actions)

    # Check that net grid energy is zero for both agents
    assert jnp.allclose(info["net_grid_energy"], jnp.array([0.0, 0.0]))
    # Check that internal trades sum to zero
    assert jnp.isclose(info["internal_trade"].sum(), 0.0)
    # Check that exchange price is the midpoint
    assert jnp.allclose(info["exch_price"], jnp.array([MID, MID]))
    # Check rewards are as expected
    expected = jnp.array([-2 * MID, +2 * MID])
    assert jnp.allclose(rewards, expected)


def test_midpoint_env_partial():
    """Test midpoint environment with partial buyer and seller."""
    # Create dataset and environment
    ds = make_dataset(buyer_load=2.0, seller_surplus=1.0)
    env = make_env(ds)

    # Reset environment and define actions
    obs, state = env.reset(jax.random.PRNGKey(0))
    actions = {
        "buyer": jnp.array([0.0, -1.0, 0.0]),
        "seller": jnp.array([0.0, 1.0, 0.0]),
    }
    # Step the environment
    _, _, rewards, _, info = env.step_env(jax.random.PRNGKey(1), state, actions)

    # Check net grid energy and exchange price
    assert jnp.isclose(info["net_grid_energy"][0], 1.0)
    assert jnp.isclose(info["net_grid_energy"][1], 0.0)
    assert jnp.isclose(info["exch_price"], MID)

    # Check rewards are as expected due to partial trade
    expected_reward_buyer = -(1.0 * TOU + 1.0 * MID)
    expected_reward_seller = +(1.0 * MID)
    assert jnp.isclose(rewards[0], expected_reward_buyer)
    assert jnp.isclose(rewards[1], expected_reward_seller)
