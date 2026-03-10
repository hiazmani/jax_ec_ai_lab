# JAX-EC API Reference

JAX-EC provides a Flask-based REST API for running energy community simulations headlessly. By default, the server runs on port `5001`.

## Endpoints

### 1. `PUT /api/start_simulation`
Triggers a JAX-accelerated simulation based on the provided community configuration.

#### Request Body (JSON)
| Field | Type | Description |
| :--- | :--- | :--- |
| `community_name` | `string` | Name of the community (used for reporting). |
| `exchange_mechanism` | `string` | `none`, `midpoint`, `double_auction`, or `agent_pricing`. |
| `agents` | `array` | List of agent objects (see below). |
| `timestamps` | `array` | (Optional) ISO-8601 strings. Generated if omitted. |
| `pricing` | `object` | (Optional) `{ "time_of_use_tariff": [], "feed_in_tariff": [] }`. |

#### Agent Object
| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Unique identifier for the building. |
| `profile` | `string` | Predefined profile name (e.g., `9-to-5 Worker`). |
| `consumption` | `array` | (Optional) Custom hourly consumption values (kWh). |
| `production` | `array` | (Optional) Custom hourly production values (kWh). |
| `decision_making` | `string` | `rbc`, `ppo`, or `q_learning`. |
| `solar` | `object` | `{ "enabled": bool, "number_of_pvs": int }`. |
| `battery` | `object` | `{ "enabled": bool, "capacity": float }`. |

#### Response (JSON)
Returns a result object containing:
- `metrics`: Performance data for the chosen mechanism.
- `baseline`: Performance data for the "No Exchange" scenario.
- `delta`: Savings and improvements compared to baseline.
- `scenario`: Metadata about the run.

---

### 2. `GET /api/profiles`
Returns a list of all predefined energy profiles available on the server.

---

### 3. `GET /api/scenarios`
Returns a list of preconfigured benchmark scenarios (e.g., `citylearn2020`).
