"""DataClasses for SimulationData."""

from typing import Dict, List

import jax.numpy as jnp
from chex import dataclass


@dataclass
class Metadata:
    """Metadata about the dataset."""

    name: str
    description: str
    number_of_agents: int
    pricing_data_included: bool


@dataclass
class TemporalData:
    """Temporal features for each timestep."""

    timestamps: List[str]
    hour_of_day: jnp.ndarray  # shape: (num_timesteps,)
    day_of_week: jnp.ndarray  # shape: (num_timesteps,)
    month_of_year: jnp.ndarray  # shape: (num_timesteps,)
    year: jnp.ndarray  # shape: (num_timesteps,)


@dataclass
class PricingData:
    """Pricing information for each timestep."""

    feed_in_tariff: jnp.ndarray  # shape: (num_timesteps,)
    time_of_use_tariff: jnp.ndarray  # shape: (num_timesteps,)


@dataclass
class HouseholdData:
    """Household consumption and production data."""

    id_to_index: Dict[str, int]  # Mapping from agent ID to array index
    index_to_id: Dict[int, str]  # Reverse mapping
    consumption: jnp.ndarray  # shape: (num_agents, num_timesteps)
    production: jnp.ndarray  # shape: (num_agents, num_timesteps)


@dataclass
class JAXECDataset:
    """Complete dataset for JAX-EC simulations."""

    metadata: Metadata
    temporal: TemporalData
    pricing: PricingData
    households: HouseholdData
