# generate_energy_sharing_puzzle.py
import json
import math


# Define agent types and their consumption profiles
def generate_hourly_profile(pattern):
    """Generate a simple hourly consumption profile based on the pattern."""
    profile = []
    for h in range(24):
        if pattern == "9to5":
            profile.append(2 if h < 7 or h >= 18 else 0.8)
        elif pattern == "home_office":
            profile.append(1.2)
        elif pattern == "school":
            profile.append(2.5 if 8 <= h <= 15 else 0.2)
        elif pattern == "shop":
            profile.append(1.5 if 9 <= h <= 17 else 0.3)
        else:
            profile.append(1.0)
    return profile


# Define nodes with profile types
agents = [
    {"id": "home_1", "profile": "9to5"},
    {"id": "home_2", "profile": "home_office"},
    {"id": "school", "profile": "school"},
    {"id": "shop", "profile": "shop"},
    {"id": "home_3", "profile": "9to5"},
    {"id": "home_4", "profile": "home_office"},
]

# Layout in a circle
radius = 300
center_x = 400
center_y = 300

for i, agent in enumerate(agents):
    angle = 2 * math.pi * i / len(agents)
    x = center_x + radius * math.cos(angle)
    y = center_y + radius * math.sin(angle)
    agent["x"] = round(x, 2)
    agent["y"] = round(y, 2)
    agent["consumption"] = generate_hourly_profile(agent["profile"])

# Optionally: define solar profile used in simulation
solar_profile = [
    round(3 * math.exp(-0.5 * ((h - 12) / 2.5) ** 2), 2) for h in range(24)
]

puzzle = {"agents": agents, "solar_profile": solar_profile}

with open("energySharingPuzzle.json", "w") as f:
    json.dump(puzzle, f, indent=2)

print("energySharingPuzzle.json created!")
