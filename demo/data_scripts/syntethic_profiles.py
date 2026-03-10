import json
from datetime import datetime, timedelta

import numpy as np

# Constants
HOURS_PER_DAY = 24
DAYS = 7
TOTAL_HOURS = HOURS_PER_DAY * DAYS
START_TIME = datetime(2023, 1, 2)  # Monday
PROFILE_NAMES = [
    "9-to-5 Worker",
    "Retired Individual",
    "Student",
    "Elementary School",
    "Vacation Home",
    "Health Center",
]


def hourly_timestamps(start: datetime, total_hours: int):
    """Generate hourly timestamps starting from a given datetime.

    Args:
        start (datetime): The starting datetime.
        total_hours (int): The total number of hourly timestamps to generate.

    Returns:
        List[str]: A list of ISO 8601 formatted datetime strings.
    """
    return [(start + timedelta(hours=i)).isoformat() for i in range(total_hours)]


def generate_profile(pattern: str):
    """Generate a synthetic energy consumption profile based on a given pattern.

    Args:
        pattern (str): The consumption pattern to simulate. Must be one of:
            '9-to-5 Worker', 'Retired Individual', 'Student',
            'Elementary School', 'Vacation Home', 'Health Center'.

    Returns:
        np.ndarray: An array of shape (TOTAL_HOURS,) representing hourly consumption.
    """
    if pattern == "9-to-5 Worker":
        profile = []
        for day in range(DAYS):
            for hour in range(HOURS_PER_DAY):
                if 8 <= hour < 17 and day < 5:
                    profile.append(np.random.normal(0.1, 0.05))  # low usage at work
                elif 6 <= hour < 8 or 17 <= hour < 23:
                    profile.append(np.random.normal(0.5, 0.1))  # morning/evening use
                else:
                    profile.append(np.random.normal(0.2, 0.05))  # sleeping
        return np.clip(profile, 0, None)

    elif pattern == "Retired Individual":
        return np.clip(np.random.normal(0.4, 0.1, TOTAL_HOURS), 0, None)

    elif pattern == "Student":
        profile = []
        for day in range(DAYS):
            for hour in range(HOURS_PER_DAY):
                if day >= 5:
                    profile.append(np.random.normal(0.1, 0.05))  # away on weekends
                elif 8 <= hour < 17:
                    profile.append(np.random.normal(0.2, 0.05))  # classes
                else:
                    profile.append(np.random.normal(0.5, 0.1))  # home
        return np.clip(profile, 0, None)

    elif pattern == "Elementary School":
        profile = []
        for day in range(DAYS):
            for hour in range(HOURS_PER_DAY):
                if 8 <= hour < 15 and day < 5:
                    profile.append(np.random.normal(2.0, 0.3))  # school hours
                else:
                    profile.append(np.random.normal(0.2, 0.05))  # empty
        return np.clip(profile, 0, None)

    elif pattern == "Vacation Home":
        profile = []
        for day in range(DAYS):
            for hour in range(HOURS_PER_DAY):
                if day == 5 and 17 <= hour < 23:  # Saturday evening
                    profile.append(np.random.normal(0.5, 0.1))
                elif day == 6 and 7 <= hour < 15:  # Sunday morning
                    profile.append(np.random.normal(0.5, 0.1))
                else:
                    profile.append(np.random.normal(0.05, 0.02))  # mostly idle
        return np.clip(profile, 0, None)

    elif pattern == "Health Center":
        return np.clip(np.random.normal(1.5, 0.2, TOTAL_HOURS), 0, None)

    else:
        return np.zeros(TOTAL_HOURS)


# Generate profiles
output = {"profiles": []}

timestamps = hourly_timestamps(START_TIME, TOTAL_HOURS)

for name in PROFILE_NAMES:
    profile = generate_profile(name)
    output["profiles"].append(
        {"label": name, "consumption": profile.tolist(), "timestamps": timestamps}
    )

# Save to JSON
with open("synthetic_energy_profiles.json", "w") as f:
    json.dump(output, f, indent=2)
