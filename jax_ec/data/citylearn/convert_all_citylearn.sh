#!/bin/bash

# Converts all locally stored CityLearn datasets to the format required by the JAX-EC simulation environment.
# The datasets are stored in the following directory structure:
# temp/CityLearn/data/datasets

# Each dataset is a directory containing the following files:
# - building_attributes.csv
# - pricing.csv
# - schema.json

# The citylearn.py file is used to convert the datasets to the JAX-EC format. We will call this file for each dataset.

for dataset in temp/CityLearn/data/datasets/*; do
    python jax_ec/data/citylearn/citylearn.py --dataset $dataset
done
