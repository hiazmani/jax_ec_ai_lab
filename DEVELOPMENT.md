# Development Guide for JAX-EC

This guide provides detailed instructions for setting up the JAX-EC development environment, including bug fixes applied and the Dockerization strategy.

## 1. Local Environment Setup

### Prerequisites
- **Python 3.13+** (Required for JAX/Flax compatibility)
- **Node.js 20+**
- **npm**

### Backend Setup (Flask + JAX-EC)
1.  **Virtual Environment**:
    ```bash
    python3.13 -m venv venv
    source venv/bin/activate
    ```
2.  **Dependencies**:
    ```bash
    pip install --upgrade pip
    pip install flask-cors matplotlib pandas numpy pvlib flax chex
    pip install -e .
    ```
3.  **Run Server**:
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/server
    python server/main.py
    ```
    The server listens on `0.0.0.0:5001`.

### Frontend Setup (React + Vite)
1.  **Install Dependencies**:
    ```bash
    cd demo
    npm install
    ```
2.  **Run Dev Server**:
    ```bash
    npm run dev
    ```
    The demo is available at `http://localhost:5173`.

---

## 2. Docker Setup

The project is fully containerized for easy deployment.

### Services
- **backend**: Flask server (Port 5001)
- **frontend**: Vite dev server (Port 5173)

### Commands
- **Start everything**: `docker-compose up --build`
- **Stop everything**: `docker-compose down`

### Port Mapping
- The backend must be accessible on port **5001** because the frontend is hardcoded to communicate with `localhost:5001` in the user's browser.
- macOS AirPlay often occupies port 5000, which is why 5001 is used.

---

## 3. Architecture & Key Changes

### Bug Fixes Applied
- **Requirements Parsing**: Fixed `setup.py` to correctly identify dependencies.
- **Port Conflict**: Moved Flask from 5000 to 5001 to avoid macOS system conflicts.
- **API Integration**: Updated `start_simulation` in `server/main.py` to actually execute the simulation using `run_full_simulation` and return results, instead of just returning a success status.
- **Dependency Issues**: Upgraded `chex` to `0.1.91` to resolve `AttributeError: module 'jax' has no attribute 'util'` caused by JAX 0.5+ deprecations.
- **Import Paths**: Fixed several imports from `jax_ec.util` to `jax_ec.utils`.

### Simulation Flow
1.  Frontend sends a community configuration (agents, solar, batteries) to `/api/start_simulation`.
2.  Flask server converts this to a JAX-EC compatible dataset in-memory.
3.  `run_full_simulation` executes the JAX-based simulation.
4.  Results (metrics, baseline, deltas) are returned as JSON to the frontend for visualization.

---

## 4. Contributing

### Adding a New Exchange Mechanism
1.  **Create Mechanism**: Add a new class in `jax_ec/environment/exchange_mechanisms/` that inherits from `BaseExchangeMechanism`.
2.  **Implement `settle`**: Implement the logic to balance trades and calculate internal prices using JAX operations.
3.  **Register**: Add your mechanism name and class path to `_MECHS` in `jax_ec/environment/exchange_mechanisms/__init__.py`.

### Adding a New Agent Kind
1.  **Define Kind**: Add a new entry to the `AgentKind` enum in `jax_ec/agents/types.py`.
2.  **State Management**: If the agent has learnable parameters or a table (like Q-Learning), add a corresponding `@struct.dataclass` and include it in the unified `AgentState`.
3.  **Initialization**: Update `init_agent_state` in `jax_ec/agents/api.py` to handle the initialization of your new agent's state.
4.  **Action Selection**: Update `select_actions_generic` in `jax_ec/agents/api.py` to dispatch logic based on the new `AgentKind`. Ensure it returns actions in the standard `(N, 3)` format.
5.  **Training (Optional)**: If the agent is trainable, update `update_generic` or the post-rollout functions to apply learning updates.
