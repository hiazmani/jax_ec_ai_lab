"""Tests for the MidpointPrice exchange mechanism."""

import jax.numpy as jnp

from jax_ec.environment.exchange_mechanisms.base import ExchangeInput
from jax_ec.environment.exchange_mechanisms.midpoint import MidpointPrice

TOU = 0.30
FIT = 0.05
MID = 0.5 * (TOU + FIT)  # 0.175


def make_input(net, trade):
    """Create ExchangeInput from net load and trade arrays.

    Makes it easier to create test inputs for unit tests.

    Args:
        net: Array of net load values for each agent.
        trade: Array of trade quantities for each agent.

    Returns:
        An ExchangeInput instance with the specified net load and trade quantities.
    """
    return ExchangeInput(
        net_load=jnp.array(net, dtype=jnp.float32),
        trade_quantity=jnp.array(trade, dtype=jnp.float32),
        trade_price=jnp.zeros_like(jnp.array(net)),
        tou_price=TOU,
        fit_price=FIT,
    )


def test_balanced_trade():
    """Test case where buyer and seller trade equal amounts.

    Buyer needs +2, seller surplus -2, they trade exactly 2 kWh.
    """
    # Create input where buyer needs +2 and seller has -2 surplus
    inp = make_input([+2.0, -2.0], [-2.0, +2.0])
    # Initialize the MidpointPrice mechanism
    mech = MidpointPrice()
    # Settle the trade
    out = mech.settle(inp)

    # Check that the internal trades cancel out
    assert jnp.isclose(out.internal_transfers.sum(), 0.0)
    # Ensure there is no grid energy needed
    assert jnp.allclose(out.net_grid_energy, jnp.array([0.0, 0.0]))
    # Check that the price is the midpoint for both agents
    assert jnp.allclose(out.prices, jnp.array([MID, MID]))


def test_no_sellers():
    """Test case where there are no sellers in the market.

    Both agents want to buy, so the grid must supply all energy.
    """
    # Create input where both agents want to buy
    inp = make_input([+1.0, +2.0], [-1.0, -2.0])
    # Initialize the MidpointPrice mechanism
    mech = MidpointPrice()
    # Settle the trade
    out = mech.settle(inp)

    # Ensure no internal trades occur
    assert jnp.allclose(out.internal_transfers, jnp.array([0.0, 0.0]))
    # Ensure both agents get their energy from the grid
    assert jnp.allclose(out.net_grid_energy, jnp.array([1.0, 2.0]))


def test_partial_supply():
    """Test case where seller can only partially fulfill buyer's demand.

    Tests that the mechanism correctly allocates internal trades and grid energy
    when the seller's surplus is less than the buyer's need.
    """
    # Create input where buyer needs +2 and seller has -1 surplus
    inp = make_input([+2.0, -1.0], [-2.0, +1.0])
    # Initialize the MidpointPrice mechanism
    mech = MidpointPrice()
    # Settle the trade
    out = mech.settle(inp)

    # Ensure internal transfers conserve energy
    assert jnp.isclose(out.internal_transfers.sum(), 0.0)

    # Ensure seller sells 1 internally → no grid interaction
    assert jnp.isclose(out.net_grid_energy[1], 0.0)
    # Ensure buyer still needs 1 kWh from grid
    assert jnp.isclose(out.net_grid_energy[0], 1.0)

    # Check prices
    assert jnp.isclose(out.prices, MID)


def test_midpoint_split_symmetric():
    """Test midpoint mechanism with symmetric trade split."""
    # Create input where buyer needs +4 and two sellers have -2 surplus each
    net = jnp.array([0.0, 0.0, 0.0])
    trade = jnp.array([+4.0, -2.0, -2.0])
    inp = ExchangeInput(
        net_load=net,
        trade_quantity=trade,
        trade_price=jnp.zeros_like(trade),
        tou_price=TOU,
        fit_price=FIT,
    )
    # Initialize the MidpointPrice mechanism
    mech = MidpointPrice()
    # Settle the trade
    out = mech.settle(inp)

    # Ensure internal trades match requested trades
    assert jnp.allclose(out.internal_transfers, trade)
    # Ensure energy is conserved
    assert jnp.isclose(out.internal_transfers.sum(), 0.0)
    # Check that all agents have the midpoint price
    assert jnp.allclose(out.prices, jnp.full(3, MID))


def test_midpoint_split_asymmetric():
    """Test midpoint mechanism with asymmetric trade split."""
    # Create input where buyer needs +4 and two sellers have -3 and -1 surplus
    net = jnp.array([0.0, 0.0, 0.0])
    trade = jnp.array([+4.0, -3.0, -1.0])
    inp = ExchangeInput(
        net_load=net,
        trade_quantity=trade,
        trade_price=jnp.zeros_like(trade),
        tou_price=TOU,
        fit_price=FIT,
    )
    # Initialize the MidpointPrice mechanism
    mech = MidpointPrice()
    # Settle the trade
    out = mech.settle(inp)

    # Ensure internal trades match requested trades
    assert jnp.allclose(out.internal_transfers, trade)
    # Ensure energy is conserved
    assert jnp.isclose(out.internal_transfers.sum(), 0.0)
    # Check that all agents have the midpoint price
    assert jnp.allclose(out.prices, jnp.full(3, MID))
