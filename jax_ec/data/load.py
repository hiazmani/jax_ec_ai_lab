"""Utilities to load the JSON data into the JAX-EC framework."""

import json

import jax.numpy as jnp

from jax_ec.data.types import (
    HouseholdData,
    JAXECDataset,
    Metadata,
    PricingData,
    TemporalData,
)


def load_simulation_data(json_file_or_dict) -> JAXECDataset:
    """Load the simulation data from a JSON file or a dictionary into the JAX-EC framework."""
    if isinstance(json_file_or_dict, str):
        with open(json_file_or_dict, "r") as f:
            data = json.load(f)
    else:
        data = json_file_or_dict

    metadata = Metadata(
        name=data["metadata"]["name"],
        description=data["metadata"]["description"],
        number_of_agents=data["metadata"]["number_of_agents"],
        pricing_data_included=data["metadata"].get("pricing_data_included", False),
    )

    temporal = TemporalData(
        timestamps=data["temporal"]["timestamps"],
        hour_of_day=jnp.array(data["temporal"]["hour_of_day"]),
        day_of_week=jnp.array(data["temporal"]["day_of_week"]),
        month_of_year=jnp.array(data["temporal"]["month_of_year"]),
        year=jnp.array(data["temporal"]["year"]),
    )

    pricing = PricingData(
        feed_in_tariff=jnp.array(data["pricing"]["feed_in_tariff"]),
        time_of_use_tariff=jnp.array(data["pricing"]["time_of_use_tariff"]),
    )

    household_ids = [hh["id"] for hh in data["households"]]
    print(household_ids)

    id_to_index = {agent_id: idx for idx, agent_id in enumerate(household_ids)}
    index_to_id = {idx: agent_id for agent_id, idx in id_to_index.items()}
    households = HouseholdData(
        id_to_index=id_to_index,
        index_to_id=index_to_id,
        consumption=jnp.stack(
            [jnp.array(hh["consumption"]) for hh in data["households"]]
        ),
        production=jnp.stack(
            [jnp.array(hh["production"]) for hh in data["households"]]
        ),
    )

    return JAXECDataset(
        metadata=metadata, temporal=temporal, pricing=pricing, households=households
    )


if __name__ == "__main__":
    test_json_file = "jax_ec/data/datasets/citylearn_challenge_2023_phase_1.json"
    data = load_simulation_data(test_json_file)
    print(data.households.consumption.shape)
    print(data.households.production.shape)
    print(data.households.id_to_index)
    print(data.households.index_to_id)
