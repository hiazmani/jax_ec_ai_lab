"""Pytest suite for DoubleAuction exchange mechanism.

Runs seven deterministic scenarios to validate clearing logic plus JIT‑compatibility.
"""

import jax
import jax.numpy as jnp
import pytest

from jax_ec.environment.exchange_mechanisms.base import ExchangeInput
from jax_ec.environment.exchange_mechanisms.double_auction import DoubleAuction

auction = DoubleAuction()

# ---------- helpers --------------------------------------------------------


def _run(q, p, net_load=None, tou=0.10, fit=0.05):
    """Utility to call auction.settle with 1‑step scalar ToU / FiT."""
    q = jnp.array(q, dtype=jnp.float32)
    p = jnp.array(p, dtype=jnp.float32)
    N = q.size
    if net_load is None:
        # residual load = –q (i.e. pretend each order exactly matches load sign)
        net_load = -q
    net_load = jnp.array(net_load, dtype=jnp.float32)
    inp = ExchangeInput(
        net_load=net_load,
        trade_quantity=q,
        trade_price=p,
        tou_price=jnp.full(N, tou),
        fit_price=jnp.full(N, fit),
    )
    return auction.settle(inp)


# ---------- tests ----------------------------------------------------------


def test_perfect_match():
    """T1: 2 kWh ask @€0.06, 2 kWh bid @€0.08 → trade all, P*=0.07."""
    out = _run([+2, -2], [0.06, 0.08])
    assert jnp.allclose(out.internal_transfers, jnp.array([+2, -2]))
    assert jnp.allclose(out.net_grid_energy, jnp.zeros(2))
    assert pytest.approx(out.prices[0], rel=1e-6) == 0.07
    assert pytest.approx(out.prices[1], rel=1e-6) == 0.07


def test_over_supply():
    """T2: 5 kWh supply, 2 kWh demand → sellers scaled to 40 %."""
    out = _run([+5, -2], [0.06, 0.08])
    # Seller sends only 2 kWh (40 %), buyer gets 2 kWh
    assert pytest.approx(out.internal_transfers[0]) == 2.0
    assert pytest.approx(out.internal_transfers[1]) == -2.0
    # residual grid = +3 kWh surplus exported
    assert pytest.approx(out.net_grid_energy[0]) == -3.0
    assert pytest.approx(out.net_grid_energy[1]) == 0.0


def test_over_demand():
    """T3: 2 supply, 5 demand → buyers scaled to 40 %."""
    out = _run([+2, -5], [0.06, 0.08])
    assert pytest.approx(out.internal_transfers[0]) == 2.0  # seller full
    assert pytest.approx(out.internal_transfers[1]) == -2.0  # buyer partial
    # residual grid = +3 kWh import
    assert pytest.approx(out.net_grid_energy[0]) == 0.0
    assert pytest.approx(out.net_grid_energy[1]) == +3.0


def test_no_overlap():
    """T4: ask 0.12, bid 0.09 → no trade."""
    out = _run([+1, -1], [0.12, 0.09])
    assert jnp.allclose(out.internal_transfers, 0.0)
    # Each settles with grid price (import/export)
    assert out.prices[0] == 0.05  # export at FiT
    assert out.prices[1] == 0.10  # import at ToU


def test_zero_sellers():
    """T5: only buyers → grid import."""
    out = _run([-3, -2], [0.08, 0.07])
    assert jnp.allclose(out.internal_transfers, 0.0)
    assert jnp.all(out.net_grid_energy > 0)


def test_mass_balance():
    """T6: internal transfers sum to 0 and residual correct."""
    q = jnp.array([+4, +1, -3, -2])
    p = jnp.array([0.06, 0.065, 0.09, 0.085])
    net_load = -q
    out = _run(q, p)
    # Check sum to zero
    assert jnp.isclose(jnp.sum(out.internal_transfers), 0.0, atol=1e-6)
    assert jnp.allclose(
        out.net_grid_energy, net_load + out.internal_transfers, atol=1e-6
    )


def test_jit():
    """T7: JIT‑compile works."""
    q = jnp.array([+2, -2])
    p = jnp.array([0.06, 0.08])
    inp = ExchangeInput(
        net_load=-q,
        trade_quantity=q,
        trade_price=p,
        tou_price=jnp.array([0.10, 0.10]),
        fit_price=jnp.array([0.05, 0.05]),
    )
    compiled = jax.jit(auction.settle)
    out = compiled(inp)
    assert jnp.allclose(out.internal_transfers, jnp.array([2, -2]))
