# Data generation module for JAX-EC.

This module contains the template used for data by the JAX-EC simulation as well as utilities to convert commonly used datasets to the right format.

## JSON Format
``jsonc
{
  "metadata": {
    "name": "community-name",
    "description": "Brief description of the energy community",
    "number_of_agents": 10,
    "time_resolution": "hourly", // can be 'hourly', 'daily', etc.
    "time_period": {
      "start_date": "2025-01-01T00:00:00",
      "end_date": "2025-12-31T23:00:00"
    }
  },
  "temporal": {
    "timestamps": [
      "2025-01-01T00:00:00",
      "2025-01-01T01:00:00"
      // ... all timestamps
    ],
    "hour_of_day": [0, 1, 2 /* ... */],
    "day_of_week": [0, 0, 0 /* ... */], // 0 = Monday, 6 = Sunday
    "month_of_year": [1, 1, 1 /* ... */],
    "year": [2025, 2025, 2025 /* ... */]
  },
  "pricing": {
    "feed_in_tariff": [0.10, 0.10 /* per timestep */],
    "time_of_use_tariff": [0.25, 0.25 /* per timestep */]
  },
  "households": [
    {
      "id": "household_1",
      "consumption": [1.2, 1.0 /* kWh, per timestep */],
      "production": [0.0, 0.5 /* kWh, per timestep */]
    },
    {
      "id": "household_2",
      "consumption": [0.9, 1.1],
      "production": [0.1, 0.0]
    }
    // ... other households
  ]
}
``
