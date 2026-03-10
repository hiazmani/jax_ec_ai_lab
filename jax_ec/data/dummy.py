"""Create a dummy environment for debugging purposes."""

import jax.numpy as jnp
import numpy as np

from jax_ec.data.types import (
    HouseholdData,
    JAXECDataset,
    Metadata,
    PricingData,
    TemporalData,
)


def create_dummy_dataset(num_agents: int = 3, num_timesteps: int = 24):
    """Creates a dummy dataset for debugging purposes.

    The dataset uses a fixed number of agents and timesteps.
    Consumption patterns are sine waves and production patterns are cosine waves,
    with some added noise to make them look more realistic.

    Each agent has a different looking curve.

    Args:
        num_agents (int): Number of agents in the dataset. Default is 3.
        num_timesteps (int): Number of timesteps in the dataset. Default is 24.

    Returns:
        JAXECDataset: A dummy dataset with synthetic consumption and production data.
    """
    # Make production and consumption patterns for the num_agents.
    for i in range(num_agents):
        consumption = jnp.cos(
            jnp.linspace(0, 2 * jnp.pi, num_timesteps)
        ) + np.random.normal(0, 0.1, num_timesteps)
        production = jnp.sin(
            jnp.linspace(0, 2 * jnp.pi, num_timesteps)
        ) + np.random.normal(0, 0.1, num_timesteps)
        if i == 0:
            consumption_matrix = consumption
            production_matrix = production
        else:
            consumption_matrix = jnp.vstack((consumption_matrix, consumption))
            production_matrix = jnp.vstack((production_matrix, production))
    return JAXECDataset(
        metadata=Metadata(
            name="test",
            description="dummy dataset",
            number_of_agents=num_agents,
            pricing_data_included=False,
        ),
        temporal=TemporalData(
            timestamps=[
                f"2025-01-01T{hour:02d}:00:00" for hour in range(num_timesteps)
            ],
            hour_of_day=jnp.arange(num_timesteps),
            day_of_week=jnp.zeros(num_timesteps),
            month_of_year=jnp.ones(num_timesteps),
            year=2025 * jnp.ones(num_timesteps),
        ),
        pricing=PricingData(
            feed_in_tariff=jnp.zeros(num_timesteps),
            time_of_use_tariff=jnp.zeros(num_timesteps),
        ),
        households=HouseholdData(
            id_to_index={f"agent_{i}": i for i in range(num_agents)},
            index_to_id={i: f"agent_{i}" for i in range(num_agents)},
            consumption=consumption_matrix,
            production=production_matrix,
        ),
    )
