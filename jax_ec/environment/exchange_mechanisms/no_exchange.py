import jax.numpy as jnp

from .base import BaseExchangeMechanism, ExchangeInput, ExchangeOutput


class NoExchange(BaseExchangeMechanism):
    """Each household settles its residual load directly with the grid."""

    def settle(self, exchange_input: ExchangeInput) -> ExchangeOutput:
        """Settle the exchange mechanism.

        In this case, each household settles its residual load directly with the grid.

        Args:
            exchange_input: ExchangeInput dataclass containing the input data.

        Returns:
            ExchangeOutput: ExchangeOutput dataclass containing the output data.
        """
        net_grid_energy = exchange_input.net_load

        # Set prices based on whether energy is imported or exported
        tou_price = exchange_input.tou_price
        fit_price = exchange_input.fit_price
        prices = jnp.where(net_grid_energy > 0, tou_price, fit_price)

        # No internal transfers in this case
        internal_transfers = jnp.zeros_like(net_grid_energy)

        return ExchangeOutput(
            net_grid_energy=net_grid_energy,
            internal_transfers=internal_transfers,
            prices=prices,
        )
