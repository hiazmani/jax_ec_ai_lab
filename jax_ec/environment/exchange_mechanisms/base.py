"""Abstract interface for every energy‑exchange mechanism."""

import abc
from typing import Tuple

import jax
from flax import struct


@struct.dataclass(frozen=True)
class ActionSpec:
    """Box(-1,1) for a 3-slot superset vector."""

    low: float = -1.0
    high: float = 1.0
    shape: Tuple[int] = (3,)


@struct.dataclass
class ExchangeOutput:
    """Output of the exchange mechanism.

    All the exchange mechanisms must return this dataclass (i.e. standardized API).
    """

    net_grid_energy: jax.Array  # shape (N,)  (+ import, – export)
    internal_transfers: jax.Array  # shape (N,)  (+ bought, – sold)  at internal price
    prices: jax.Array  # shape (N,)  €/kWh  (+ cost, – revenue)


@struct.dataclass
class ExchangeInput:
    """Input to the exchange mechanism.

    All the exchange mechanisms must accept this dataclass (i.e. standardized API).
    """

    net_load: jax.Array  # shape (N,)  (+ need, – surplus)
    trade_quantity: jax.Array  # shape (N,)  signed kWh proposed by agent
    trade_price: jax.Array  # shape (N,)  €/kWh  limit price (may be unused)
    tou_price: jax.Array  # scalar or (N,)  €/kWh buy from grid
    fit_price: jax.Array  # scalar or (N,)  €/kWh sell to grid


class BaseExchangeMechanism(abc.ABC):
    """This is the abstract base class for all exchange mechanisms.

    It defines the interface for the exchange mechanism and expected input/output.
    """

    @abc.abstractmethod
    def settle(
        self,
        exchange_input: ExchangeInput,
    ) -> ExchangeOutput:
        """Settle the exchange mechanism.

        Args:
            exchange_input: ExchangeInput dataclass containing the input data.

        Returns:
            ExchangeOutput: ExchangeOutput dataclass containing the output data.
        """
        raise NotImplementedError("Subclasses must implement this method.")
