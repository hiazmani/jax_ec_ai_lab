import jax.numpy as jnp

from .base import BaseExchangeMechanism, ExchangeInput, ExchangeOutput


class MidpointPrice(BaseExchangeMechanism):
    """Internal trades are cleared at the midpoint of TOU and FIT.

    Each agent submits a signed kWh quantity; the mechanism balances
    trades to exactly zero each step (excess/residual -> grid).
    """

    def __init__(self, max_trade_kwh: float = 10.0):
        self.max_trade_kwh = max_trade_kwh

    def settle(self, exchange_input: ExchangeInput) -> ExchangeOutput:
        """Settle the exchange mechanism.

        The midpoint price sets a price for internal trades based on the average of the
        time-of-use (ToU) price and the feed-in tariff (FiT) price. Total trade quantity
        (sold by agents) is split evenly among all agents that are buying energy.

        Convention:
            * Negative trade quantity means the agent is buying energy.
            * Positive trade quantity means the agent is selling energy.
            * The net load is positive when the agent needs energy and negative when
              the agent has surplus energy.

        Args:
            exchange_input: ExchangeInput dataclass containing the input data.

        Returns:
            ExchangeOutput: ExchangeOutput dataclass containing the output data.
        """
        # Unpack the input data
        net_load = exchange_input.net_load
        trade_quantity = exchange_input.trade_quantity
        tou_price = exchange_input.tou_price
        fit_price = exchange_input.fit_price

        # Compute the midpoint price
        midpoint_price = 0.5 * (tou_price + fit_price)

        # Separate sellers and buyers
        sellers = jnp.where(trade_quantity > 0, trade_quantity, 0.0)
        buyers = jnp.where(trade_quantity < 0, -trade_quantity, 0.0)

        total_sold = jnp.sum(sellers)
        total_bought = jnp.sum(buyers)

        # Determine the amount of energy available for each buyer
        redistributed_energy = jnp.where(
            (total_sold > 0) & (total_bought > 0),
            jnp.minimum(buyers / (total_bought + 1e-10) * total_sold, buyers),
            jnp.zeros_like(buyers),
        )

        # Compute the final trade quantities
        # Original convention: positive means sold to community, negative means bought from community.
        trade = sellers - redistributed_energy

        # Compute the residual energy with the grid
        # net_load: +deficit, -surplus
        # trade: +sold, -bought
        # if trade is positive (sold), I have more to send to grid (or less deficit).
        # Actually, if net_load=-10 (surplus 10) and trade=2 (sold 2 internally), 
        # then net surplus for grid should be -8.
        # after_trade = net_load + trade = -10 + 2 = -8. CORRECT.
        after_trade = net_load + trade

        return ExchangeOutput(
            net_grid_energy=after_trade,
            internal_transfers=trade,
            prices=midpoint_price,
        )
