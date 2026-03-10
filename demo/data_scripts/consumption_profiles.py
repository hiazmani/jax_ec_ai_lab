import json
import random


def generate_hours():
    """Generate a list of hours from 1 to 24."""
    # Hours from 1 to 24.
    return list(range(1, 25))


def generate_9to5_worker_profile():
    """Generate a sample energy consumption profile for a 9-to-5 worker."""
    hours = generate_hours()
    y = []
    for h in hours:
        if h < 6:
            # Early morning: moderate usage.
            val = 1.2 + random.uniform(-0.1, 0.1)
        elif h < 9:
            # Morning peak.
            val = 2.2 + random.uniform(-0.2, 0.2)
        elif h < 18:
            # Daytime: low usage since they're away.
            val = 0.8 + random.uniform(-0.1, 0.1)
        elif h < 23:
            # Evening peak.
            val = 2.3 + random.uniform(-0.2, 0.2)
        else:
            # Late night: low/moderate usage.
            val = 1.0 + random.uniform(-0.1, 0.1)
        y.append(round(val, 2))
    return y


def generate_work_from_home_profile():
    """Generate a sample energy consumption profile for a work-from-home user."""
    hours = generate_hours()
    y = []
    for _ in hours:
        # Steady usage throughout the day with slight random variation.
        val = 1.5 + random.uniform(-0.15, 0.15)
        y.append(round(val, 2))
    return y


def generate_school_profile():
    """Generate a sample energy consumption profile for a school building."""
    hours = generate_hours()
    y = []
    for h in hours:
        if 8 <= h <= 15:
            # High usage during school hours.
            val = 2.0 + random.uniform(-0.2, 0.2)
        else:
            # Low usage outside school hours.
            val = 0.2 + random.uniform(-0.05, 0.05)
        y.append(round(val, 2))
    return y


def main():
    """Main function to generate the JSON file with energy consumption profiles."""
    profile_data = {
        "title": "Energy Profile",
        "x_label": "Time (hours)",
        "y_label": "Energy (kWh)",
        "description": "Sample energy consumption profile for various user types.",
        "data": [
            {
                "legend_title": "9-to-5 Worker",
                "color": "#6d78ad",
                "x": generate_hours(),
                "y": generate_9to5_worker_profile(),
            },
            {
                "legend_title": "Work-from-Home",
                "color": "#51cda0",
                "x": generate_hours(),
                "y": generate_work_from_home_profile(),
            },
            {
                "legend_title": "School Building",
                "color": "#a44c4c",
                "x": generate_hours(),
                "y": generate_school_profile(),
            },
        ],
    }

    filename = "energy_consumption_profiles"

    with open(f"{filename}.json", "w") as f:
        json.dump(profile_data, f, indent=2)

    print(f"JSON file generated: {filename}.json")


if __name__ == "__main__":
    main()
