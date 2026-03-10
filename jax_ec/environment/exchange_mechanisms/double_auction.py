"""Uniform‑price double auction exchange mechanism (JAX‑compatible).

Positive quantities = asks (sell), negative = bids (buy)

Outputs follow the project‑wide convention:
`internal_transfers` (+ buy / – sell) and `net_grid_energy = net_load − internal`.
This version is fully JIT‑compatible: every data‑dependent branch uses
`jax.lax.cond`; the only Python logic left is static (index manipulations).
"""

from functools import partial

import jax
import jax.numpy as jnp

from .base import BaseExchangeMechanism, ExchangeInput, ExchangeOutput

# ───────────────────────── helper primitives ──────────────────────────────


def _split_orders(q: jnp.ndarray, p: jnp.ndarray):
    ask_mask = q > 0
    bid_mask = q < 0
    ask_q = jnp.where(ask_mask, q, 0.0)
    ask_p = jnp.where(ask_mask, p, jnp.inf)
    bid_q = jnp.where(bid_mask, -q, 0.0)
    bid_p = jnp.where(bid_mask, p, -jnp.inf)
    return ask_q, ask_p, bid_q, bid_p


def _sort_asks(q, p):
    order = jnp.argsort(p)  # ascending price
    q_s = q[order]
    p_s = p[order]
    mask = q_s > 0  # asks
    ask_q = jnp.where(mask, q_s, 0.0)
    ask_p = jnp.where(mask, p_s, jnp.inf)
    return ask_q, ask_p, order  # order retains original indices


def _sort_bids(q, p):
    order = jnp.argsort(-p)  # descending price
    q_s = q[order]
    p_s = p[order]
    mask = q_s > 0  # bids already made positive
    bid_q = jnp.where(mask, q_s, 0.0)
    bid_p = jnp.where(mask, p_s, -jnp.inf)
    return bid_q, bid_p, order


def _pro_rata(q_vec, target):
    tot = jnp.sum(q_vec)
    scaled = jnp.where(tot > 0, q_vec * (target / (tot + 1e-10)), q_vec)
    # push residual (<1e‑6) into last filled element for exact sum
    resid = target - jnp.sum(scaled)
    last = jnp.max(jnp.where(q_vec > 0, jnp.arange(q_vec.size), 0))
    scaled = scaled.at[last].add(resid)
    return scaled


# ─────────────────────────── main mechanism ───────────────────────────────
class DoubleAuction(BaseExchangeMechanism):
    """Uniform‑price double auction with pro‑rata curtailment on the long side."""

    @partial(jax.jit, static_argnums=0)
    def settle(self, exchange_input: ExchangeInput) -> ExchangeOutput:  # noqa: D401
        """Settle the exchange mechanism.

        The double auction matches buy and sell orders based on submitted quantities
        and prices. It balances trades to exactly zero each step, with any excess or
        residual energy being traded with the grid.

        Args:
            exchange_input: ExchangeInput dataclass containing the input data.

        Returns:
            ExchangeOutput: ExchangeOutput dataclass containing the output data.
        """
        # q, p = exchange_input.trade_quantity, exchange_input.trade_price
        # net_load, tou, fit = (exchange_input.net_load,
        #                       exchange_input.tou_price,
        #                       exchange_input.fit_price)

        q = exchange_input.trade_quantity.astype(jnp.float32)
        p = exchange_input.trade_price.astype(jnp.float32)
        net_load = exchange_input.net_load.astype(jnp.float32)
        tou = exchange_input.tou_price.astype(jnp.float32)
        fit = exchange_input.fit_price.astype(jnp.float32)

        ask_q, ask_p, bid_q, bid_p = _split_orders(q, p)

        # branch 0: one side empty → everything with grid
        empty_side = jnp.logical_or(jnp.sum(ask_q) == 0.0, jnp.sum(bid_q) == 0.0)

        def _grid_only(_):
            base = jnp.where(net_load > 0, tou, fit)
            return ExchangeOutput(net_load, jnp.zeros_like(net_load), base)

        # full auction ------------------------------------------------------
        def _run_auction(_):
            ask_q_s, ask_p_s, ask_idx = _sort_asks(ask_q, ask_p)
            bid_q_s, bid_p_s, bid_idx = _sort_bids(bid_q, bid_p)
            cum_sup = jnp.cumsum(ask_q_s)
            cum_dem = jnp.cumsum(bid_q_s)

            def _assemble(filled_asks_s, filled_bids_s, P_star):
                internal = jnp.zeros_like(net_load)
                internal = internal.at[ask_idx].add(filled_asks_s)
                internal = internal.at[bid_idx].add(-filled_bids_s)
                internal = internal.at[jnp.argmax(jnp.abs(internal))].add(
                    -jnp.sum(internal)
                )
                residual = net_load + internal
                base = jnp.where(residual > 0, tou, fit)
                prices = jnp.where(internal != 0, P_star, base)
                return ExchangeOutput(residual, internal, prices)

            overlap = jnp.min(ask_p_s) <= jnp.max(bid_p_s)

            # branch 1a: no price overlap ----------------------------------
            def _no_overlap(_):
                base = jnp.where(net_load > 0, tou, fit)
                return ExchangeOutput(net_load, jnp.zeros_like(net_load), base)

            # branch 1b: have overlap --------------------------------------
            def _with_overlap(_):
                crosses = cum_sup >= cum_dem
                has_cross = jnp.any(crosses)

                def _cross_branch(_):
                    idx = jnp.argmax(crosses)
                    Q_star = jnp.minimum(cum_sup[idx], cum_dem[idx])
                    P_star = 0.5 * (ask_p_s[idx] + bid_p_s[idx])
                    return _assemble(
                        _pro_rata(ask_q_s, Q_star), _pro_rata(bid_q_s, Q_star), P_star
                    )

                def _unbalanced_branch(_):
                    tot_sup, tot_dem = cum_sup[-1], cum_dem[-1]
                    trade_vol = jnp.minimum(tot_sup, tot_dem)
                    filled_asks = jax.lax.select(
                        tot_sup < tot_dem, ask_q_s, _pro_rata(ask_q_s, trade_vol)
                    )
                    filled_bids = jax.lax.select(
                        tot_sup < tot_dem, _pro_rata(bid_q_s, trade_vol), bid_q_s
                    )
                    idx_last_ask = jnp.max(
                        jnp.where(filled_asks > 0, jnp.arange(filled_asks.size), 0)
                    )
                    idx_last_bid = jnp.max(
                        jnp.where(filled_bids > 0, jnp.arange(filled_bids.size), 0)
                    )
                    P_star = 0.5 * (ask_p_s[idx_last_ask] + bid_p_s[idx_last_bid])
                    return _assemble(filled_asks, filled_bids, P_star)

                return jax.lax.cond(has_cross, _cross_branch, _unbalanced_branch, None)

            return jax.lax.cond(overlap, _with_overlap, _no_overlap, None)

        return jax.lax.cond(empty_side, _grid_only, _run_auction, None)
