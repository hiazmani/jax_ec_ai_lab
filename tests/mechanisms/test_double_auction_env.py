"""End-to-end checks of the DoubleAuction exchange mechanism through JAXECEnvironment.

We build a tiny 2-agent, 1-hour dataset:
    * agent “buyer”   net-load  +2 kWh
    * agent “seller”  net-load  –2 kWh   (or –1 kWh for the partial test)

Actions:
    a_batt   = 0.0        (no battery)
    a_qty    = –1  / +1   (buy / sell the whole residual)
    a_price  = –1  / +1   → ask = FiT, bid = ToU
These extreme price limits guarantee price-overlap for the auction.
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
from jax_ec.environment.battery import get_disabled_battery_cfg
from jax_ec.environment.env import JAXECEnvironment

# price configuration
TOU = 0.30
FIT = 0.05
CLEAR = 0.5 * (TOU + FIT)  # 0.175 € — clearing price when ask=0.05, bid=0.30


# ---------- helpers --------------------------------------------------------


def _dataset(buyer_load: float, seller_surplus: float) -> JAXECDataset:
    cons = jnp.array([[buyer_load], [0.0]], dtype=jnp.float32)
    prod = jnp.array([[0.0], [seller_surplus]], dtype=jnp.float32)

    return JAXECDataset(
        metadata=Metadata(
            name="auction-env",
            description="double-auction env test",
            number_of_agents=2,
            pricing_data_included=True,
        ),
        temporal=TemporalData(
            timestamps=["2025-01-01T00:00:00"],
            hour_of_day=jnp.array([0]),
            day_of_week=jnp.array([0]),
            month_of_year=jnp.array([1]),
            year=jnp.array([2025]),
        ),
        pricing=PricingData(
            feed_in_tariff=jnp.array([FIT]), time_of_use_tariff=jnp.array([TOU])
        ),
        households=HouseholdData(
            id_to_index={"buyer": 0, "seller": 1},
            index_to_id={0: "buyer", 1: "seller"},
            consumption=cons,
            production=prod,
        ),
    )


def _env(ds: JAXECDataset) -> JAXECEnvironment:
    # every agent gets a disabled battery config
    bat_cfg = {aid: get_disabled_battery_cfg() for aid in ds.households.id_to_index}
    return JAXECEnvironment(
        dataset=ds,
        battery_configs=bat_cfg,
        exchange_name="double_auction",  # <-- only difference vs midpoint tests
        allow_grid_charging=False,
    )


def _auction_actions():
    """Return extreme auction actions for buyer and seller.

    Actions are arrays of shape (3,) for each agent: (a_batt , a_qty , a_price)

    • a_batt   = 0   (no battery use)
    • a_qty    = ±1 (trade full residual)
    • a_price  = ±1 (ask=FiT , bid=ToU) so prices overlap
    """
    return {
        "buyer": jnp.array([0.0, -1.0, +1.0]),  # bid full qty @ ToU
        "seller": jnp.array([0.0, +1.0, -1.0]),  # ask full qty @ FiT
    }


# ---------- tests ----------------------------------------------------------


def test_auction_env_balanced():
    """Seller can meet buyer entirely → zero grid exchange, uniform clearing price."""
    ds = _dataset(buyer_load=2.0, seller_surplus=2.0)
    env = _env(ds)

    _, state = env.reset(jax.random.PRNGKey(0))
    _, _, rewards, _, info = env.step_env(
        jax.random.PRNGKey(1), state, _auction_actions()
    )

    # energy balance --------------------------------------------------------
    assert jnp.allclose(info["net_grid_energy"], jnp.array([0.0, 0.0]))
    assert jnp.isclose(info["internal_trade"].sum(), 0.0)

    # price & reward check --------------------------------------------------
    assert jnp.allclose(info["exch_price"], jnp.array([CLEAR, CLEAR]))
    expected = jnp.array([-2.0 * CLEAR, +2.0 * CLEAR])  # buyer pays, seller earns
    assert jnp.allclose(rewards, expected)


def test_auction_env_partial():
    """Seller offers only 1 kWh, buyer still needs 1 kWh from the grid."""
    ds = _dataset(buyer_load=2.0, seller_surplus=1.0)
    env = _env(ds)

    _, state = env.reset(jax.random.PRNGKey(0))
    _, _, rewards, _, info = env.step_env(
        jax.random.PRNGKey(2), state, _auction_actions()
    )

    # grid interaction: buyer imports 1 kWh, seller none
    assert jnp.isclose(info["net_grid_energy"][0], 1.0)
    assert jnp.isclose(info["net_grid_energy"][1], 0.0)

    # clearing price is still 0.175 € (FiT/ToU midpoint with extreme bids)
    assert jnp.allclose(info["exch_price"], jnp.full(2, CLEAR))

    # check cost computation
    expected_buyer = -(1.0 * TOU + 1.0 * CLEAR)  # grid + internal
    expected_seller = +(1.0 * CLEAR)
    assert jnp.isclose(rewards[0], expected_buyer)
    assert jnp.isclose(rewards[1], expected_seller)
