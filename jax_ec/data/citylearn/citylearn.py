"""Contains utilities to convert CityLearn datasets into the format used by JAX-EC.

For a complete list of available CityLearn datasets see: https://github.com/intelligent-environments-lab/CityLearn/tree/master/data/datasets
"""

import argparse
import json
import logging
import os
import warnings

import numpy as np
import pandas as pd

from jax_ec.utils.logging.logger import CustomFormatter


def load_citylearn_dataset(dataset_path):
    """Load a CityLearn dataset and convert it to format used by JAX-EC.

    Args:
        dataset_path (str): Path to the CityLearn dataset directory.

    Returns:
        dict: A dictionary containing the standardized dataset.
    """
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    # Initialize the standardized data structure
    standardized_data = {
        "metadata": {
            "name": "citylearn_challenge_2020_climate_zone_1",
            "description": "CityLearn Challenge 2020 Climate Zone 1 Dataset",
            "number_of_agents": 0,
            "time_resolution": "hourly",
            "time_period": {"start_date": "", "end_date": ""},
            "pricing_data_included": False,
        },
        "temporal": {
            "timestamps": [],
            "hour_of_day": [],
            "day_of_week": [],
            "month_of_year": [],
            "year": [],
        },
        "pricing": {"feed_in_tariff": [], "time_of_use_tariff": []},
        "households": [],
    }

    # List all building CSV files
    building_files = [
        f
        for f in os.listdir(dataset_path)
        if f.startswith("Building") and f.endswith(".csv")
    ]
    standardized_data["metadata"]["number_of_agents"] = len(building_files)

    # Process each building file
    for building_file in building_files:
        building_path = os.path.join(dataset_path, building_file)
        df = pd.read_csv(building_path)

        # Extract temporal information if not already extracted
        if not standardized_data["temporal"]["timestamps"]:
            # Assuming the dataset starts on January 1st of a given year
            start_year = 2020  # Replace with the actual start year if known
            timestamps = pd.date_range(
                start=f"{start_year}-01-01", periods=len(df), freq="H"
            )
            standardized_data["temporal"]["timestamps"] = timestamps.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ).tolist()
            standardized_data["temporal"]["hour_of_day"] = timestamps.hour.tolist()
            standardized_data["temporal"]["day_of_week"] = timestamps.dayofweek.tolist()
            standardized_data["temporal"]["month_of_year"] = timestamps.month.tolist()
            standardized_data["temporal"]["year"] = timestamps.year.tolist()
            standardized_data["metadata"]["time_period"]["start_date"] = timestamps[
                0
            ].strftime("%Y-%m-%dT%H:%M:%S")
            standardized_data["metadata"]["time_period"]["end_date"] = timestamps[
                -1
            ].strftime("%Y-%m-%dT%H:%M:%S")

        # Extract consumption and production data
        production = df["solar_generation"] / 1000  # Convert Wh to kWh
        # Production also needs to be multiplied by the 'nominal power' of the PV panel
        with open(os.path.join(dataset_path, "schema.json"), "r") as f:
            schema = json.load(f)
            if "pv" in schema["buildings"][building_file.replace(".csv", "")]:
                nominal_power = schema["buildings"][building_file.replace(".csv", "")][
                    "pv"
                ]["attributes"]["nominal_power"]
                production = production * nominal_power
                production = production.tolist()
            else:
                building_file = building_file.replace(".csv", "")
                warnings.warn(
                    f"PV data not found for building {building_file}.\
                    Setting production to 0.",
                    stacklevel=2,
                )
                production = [0] * len(df)

            # Check in the

        household_data = {
            "id": building_file.replace(".csv", ""),
            "consumption": df["non_shiftable_load"].tolist(),
            "production": production,
            "has_pv": True if np.sum(production) > 0 else False,
        }
        standardized_data["households"].append(household_data)

    # Handle pricing data
    price_file_path = os.path.join(dataset_path, "pricing.csv")
    if os.path.exists(price_file_path):
        price_df = pd.read_csv(price_file_path)
        standardized_data["pricing"]["time_of_use_tariff"] = price_df[
            "electricity_pricing"
        ].tolist()
        # Assuming feed-in tariff is a fixed percentage of time-of-use tariff
        standardized_data["pricing"]["feed_in_tariff"] = [
            price * 0.5 for price in standardized_data["pricing"]["time_of_use_tariff"]
        ]
        standardized_data["metadata"]["pricing_data_included"] = True
    else:
        logger.warning(
            "Pricing data not found in the dataset.\
             Setting pricing_data_included to False and all prices to 0."
        )
        # If pricing data is not available, set tariffs to zero
        num_timesteps = len(standardized_data["temporal"]["timestamps"])
        standardized_data["pricing"]["time_of_use_tariff"] = [0.0] * num_timesteps
        standardized_data["pricing"]["feed_in_tariff"] = [0.0] * num_timesteps
        standardized_data["metadata"]["pricing_data_included"] = False

    return standardized_data


if __name__ == "__main__":
    # Setup parser
    parser = argparse.ArgumentParser()
    # Add arguments to be parsed
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        help="Path to the CityLearn dataset to be standardized.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Convert the given CityLearn dataset into the standardized format
    standardized_json = load_citylearn_dataset(args.dataset)
    # Write the standardized data to a JSON file
    dataset_name = args.dataset.split("/")[-1]
    json_output_path = f"jax_ec/data/datasets/{dataset_name}.json"
    with open(json_output_path, "w") as f:
        json.dump(standardized_json, f, indent=4)
