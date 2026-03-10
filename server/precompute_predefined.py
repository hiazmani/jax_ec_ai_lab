"""Script to precompute results for all predefined scenarios.

Saves results as JSON files in server/results/. This way the server can
quickly load and return them without needing to run the simulation each time.
"""

import json
import os

from config_preconfigured import PREDEFINED_SCENARIOS

from jax_ec.sim_runner import run_full_simulation

OUT_DIR = "server/results"

os.makedirs(OUT_DIR, exist_ok=True)

for name, cfg in PREDEFINED_SCENARIOS.items():
    print("Running", name)
    res = run_full_simulation(cfg)  # returns dict
    with open(f"{OUT_DIR}/{name}.json", "w") as fp:
        json.dump(res, fp, indent=2)
