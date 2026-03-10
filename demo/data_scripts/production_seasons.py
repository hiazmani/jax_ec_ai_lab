import json
import math
import random


def generate_hours():
    """Generate a list of hours from 1 to 24."""
    # Hours 1 through 24
    return list(range(1, 25))


def sunny_summer_day():
    """Generate a sunny summer day solar production profile."""
    hours = generate_hours()
    y = []
    for h in hours:
        # Assume production follows a roughly bell-shaped curve peaking at noon.
        # Use a Gaussian-like curve with peak around hour 12.
        peak = 12
        std = 2.5
        base = math.exp(-0.5 * ((h - peak) / std) ** 2)
        # Scale so that maximum production is around 3 kWh.
        val = 3 * base + random.uniform(-0.1, 0.1)
        y.append(round(val, 2))
    return y


def cloudy_day():
    """Generate a cloudy day solar production profile."""
    hours = generate_hours()
    y = []
    for h in hours:
        # Same bell-shape, but production reduced by a factor (e.g., 40% of sunny)
        peak = 12
        std = 2.5
        base = math.exp(-0.5 * ((h - peak) / std) ** 2)
        val = 3 * base * 0.4 + random.uniform(-0.05, 0.05)
        y.append(round(val, 2))
    return y


def winter_day():
    """Generate a winter day solar production profile."""
    hours = generate_hours()
    y = []
    for h in hours:
        # In winter, daylight is shorter.
        # Let production only occur from hour 9 to 16, with a lower peak.
        if 9 <= h <= 16:
            peak = 12
            std = 1.5
            base = math.exp(-0.5 * ((h - peak) / std) ** 2)
            # Peak production lower, say max 1.5 kWh.
            val = 1.5 * base + random.uniform(-0.05, 0.05)
        else:
            val = 0.0
        y.append(round(val, 2))
    return y


def main():
    """Generate a JSON file with synthetic solar production profiles."""
    profile = {
        "title": "Solar Production Profile",
        "x_label": "Time (hours)",
        "y_label": "Solar Production (kWh)",
        "description": "Synthetic PV profiles for different weather/season scenarios.",
        "data": [
            {
                "legend_title": "Sunny Summer Day",
                "color": "#f39c12",  # Orange-yellow
                "x": generate_hours(),
                "y": sunny_summer_day(),
            },
            {
                "legend_title": "Cloudy Day",
                "color": "#95a5a6",  # Gray
                "x": generate_hours(),
                "y": cloudy_day(),
            },
            {
                "legend_title": "Winter Day",
                "color": "#3498db",  # Blue
                "x": generate_hours(),
                "y": winter_day(),
            },
        ],
    }

    with open("solar_production_profile.json", "w") as f:
        json.dump(profile, f, indent=2)

    print("Generated solar_production_profile.json")


if __name__ == "__main__":
    main()
