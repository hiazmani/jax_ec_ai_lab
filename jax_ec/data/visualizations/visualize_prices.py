import matplotlib.pyplot as plt

from jax_ec.data.load import load_simulation_data
from jax_ec.environment.battery import get_default_battery_cfg
from jax_ec.environment.env import JAXECEnvironment


def visualize_prices(dataset_name):
    """Visualize the feed-in-tariff and time-of-use pricing from the dataset.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        None
    """
    # Load in the dataset
    data = load_simulation_data(dataset_name)
    # Create a battery config
    battery_cfg = get_default_battery_cfg()
    # Make an environment with the data and battery config
    env = JAXECEnvironment(data, battery_cfg)

    # Create two vertically stacked subplots (price over dataset / average price/day)
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    # 1. Plot the feed-in-tariff and time-of-use pricing over the entire dataset
    axs[0].plot(env.dataset.pricing.feed_in_tariff, label="Feed-in Tariff")
    axs[0].plot(env.dataset.pricing.time_of_use_tariff, label="Time-of-Use Tariff")
    axs[0].set_title("Pricing Over the Entire Dataset")
    axs[0].set_xlabel("Timestep")
    axs[0].set_ylabel("Price")
    axs[0].legend()

    # 2. Plot the average feed-in-tariff and time-of-use pricing per day
    feed_in_tariff_avg = env.dataset.pricing.feed_in_tariff.reshape(-1, 24).mean(axis=0)
    time_of_use_tariff_avg = env.dataset.pricing.time_of_use_tariff.reshape(
        -1, 24
    ).mean(axis=0)
    axs[1].plot(feed_in_tariff_avg, label="Average Feed-in Tariff")
    axs[1].plot(time_of_use_tariff_avg, label="Average Time-of-Use Tariff")
    axs[1].set_title("Average Pricing Per Day")
    axs[1].set_xlabel("Hour of Day")
    axs[1].set_ylabel("Price")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_prices("jax_ec/data/datasets/citylearn_challenge_2023_phase_1.json")
