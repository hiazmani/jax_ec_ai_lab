import jax.numpy as jnp
from .base import BaseExchangeMechanism, ExchangeInput, ExchangeOutput


class CentralAgentPricing(BaseExchangeMechanism):
    """Internal trades are cleared at a price determined by a central agent.

    This mechanism ignores the individual `trade_price` suggestions from buildings
    and instead uses a single price provided by the Pricing Agent (Agent N+1).
    """

    def settle(self, exchange_input: ExchangeInput) -> ExchangeOutput:
        """Settle the exchange mechanism using a central price.

        Args:
            exchange_input: ExchangeInput dataclass. 
                Expected to have trade_price as (N+1,) where index N is the global price.

        Returns:
            ExchangeOutput: Standardized output with internal transfers and grid interaction.
        """
        # Unpack the input data (first N entries are buildings)
        net_load = exchange_input.net_load
        trade_quantity = exchange_input.trade_quantity
        tou_price = exchange_input.tou_price
        fit_price = exchange_input.fit_price
        
        # The price is taken from the LAST entry of the trade_price array (Agent N)
        # We assume the caller (Environment) has packed N+1 actions.
        # However, ExchangeInput.net_load is typically shape (N,).
        # We need to be careful about shapes here.
        
        # In a N+1 setup:
        # trade_quantity: (N+1,) -> [b1_q, b2_q, ..., bN_q, price_agent_q (ignored)]
        # trade_price: (N+1,)    -> [b1_p, b2_p, ..., bN_p, global_price]
        
        # Use only building quantities for settlement
        N = net_load.shape[0]
        building_q = trade_quantity[:N]
        central_price = exchange_input.trade_price[N] # Index N is the (N+1)-th agent
        
        # Normalize/Scale the price from [-1, 1] to [FiT, ToU]
        # This ensures the price is always "rational" (better than grid for both sides)
        p_mid = 0.5 * (tou_price + fit_price)
        p_rng = 0.5 * (tou_price - fit_price)
        clearing_price = p_mid + central_price * p_rng

        # Standard settlement logic (same as Midpoint but with dynamic clearing_price)
        sellers = jnp.where(building_q > 0, building_q, 0.0)
        buyers = jnp.where(building_q < 0, -building_q, 0.0)

        total_sold = jnp.sum(sellers)
        total_bought = jnp.sum(buyers)

        redistributed_energy = jnp.where(
            total_sold > 0,
            jnp.minimum(buyers / (total_bought + 1e-10) * total_sold, buyers),
            jnp.zeros_like(buyers),
        )

        trade = sellers - redistributed_energy
        after_trade = net_load + trade

        return ExchangeOutput(
            net_grid_energy=after_trade,
            internal_transfers=trade,
            prices=jnp.full((N,), clearing_price),
        )
