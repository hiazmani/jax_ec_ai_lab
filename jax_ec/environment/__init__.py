"""Contains the JAX-EC environment and its components."""

import os

import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

from jax_ec.data.load import load_simulation_data
from jax_ec.environment.battery import get_default_battery_cfg, get_disabled_battery_cfg
from jax_ec.environment.env import JAXECDataset, JAXECEnvironment

DATASET_DIR = "jax_ec/data/datasets"


def setup_environment(env_cfg: DictConfig) -> JAXECEnvironment:
    """Setup the JAX-EC environment."""
    print(f"[ENV] Setting up environment: {env_cfg}")

    # The path of the dataset is the dataset directory + the dataset name
    dataset_path: str = os.path.join(DATASET_DIR, env_cfg.dataset.name)

    data: JAXECDataset = load_simulation_data(dataset_path)

    # -- Pricing -- #
    # Create an array with the global FiT price
    n_readings = data.households.consumption.shape[1]
    # Check Feed-in Tariff (FiT) prices
    fit_sum = sum(data.pricing.feed_in_tariff)
    if fit_sum == 0:
        if env_cfg.global_fit_price is None:
            raise ValueError(
                "[ERROR] Feed-in tariff prices are all zero. \
                 Provide FiT prices in the dataset OR set global_fit_price in cfg."
            )
        else:
            data.pricing.feed_in_tariff = jnp.full(n_readings, env_cfg.global_fit_price)
    elif env_cfg.global_fit_price is not None:
        raise ValueError(
            "[ERROR] Dataset contains FiT prices, but global_fit_price also provided.\
             Please provide only one source of FiT pricing."
        )

    # Check Time-of-Use (ToU) prices
    tou_sum = sum(data.pricing.time_of_use_tariff)
    if tou_sum == 0:
        if env_cfg.global_tou_price is None:
            raise ValueError(
                "[ERROR] Time-of-use prices are all zero.\
                 Either provide ToU prices in the dataset OR global_tou_price in cfg."
            )
        else:
            data.pricing.time_of_use_tariff = jnp.full(
                n_readings, env_cfg.global_tou_price
            )
    elif env_cfg.global_tou_price is not None:
        raise ValueError(
            "[ERROR] Dataset contains ToU prices, but global_tou_price also provided.\
             Please provide only one source of ToU pricing."
        )

    if "battery_enabled" not in env_cfg:
        raise ValueError(
            "[ERROR] Battery configuration is missing. \
             Please provide battery_enabled in cfg."
        )
    # -- Battery Configuration -- #
    battery_configs = {}
    for a_id, is_enabled in env_cfg.battery_enabled.items():
        if is_enabled:
            _cfg = get_default_battery_cfg()
            _cfg = OmegaConf.create(_cfg)
            battery_configs[a_id] = _cfg
        else:
            _cfg = get_disabled_battery_cfg()
            _cfg = OmegaConf.create(_cfg)
            battery_configs[a_id] = _cfg

    return JAXECEnvironment(data, battery_configs=battery_configs)
