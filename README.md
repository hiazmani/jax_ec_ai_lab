# JAX_EC: A JAX-Accelerated Simulation Framework for Multi-Agent Energy Management in Energy Communities

[![Tests](https://github.com/hiazmani/jax_ec/actions/workflows/tests.yml/badge.svg)](https://github.com/hiazmani/jax_ec/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

JAX-EC is a high-performance simulation framework designed for researchers and stakeholders to prototype, evaluate, and understand energy exchange mechanisms within Energy Communities. It leverages JAX for vectorized computations, achieving massive speedups in multi-agent reinforcement learning (MARL) training and evaluation.

---

## 🚀 Quick Start (Docker)

The fastest way to get JAX-EC running is via Docker Compose. This starts both the **Flask Backend** (port 5001) and the **React Interactive Demo** (port 5173).

```bash
# Build and start all services
docker-compose up --build -d
```

Open [http://localhost:5173](http://localhost:5173) in your browser to start the interactive community builder.

---

## 🏗️ Key Features

*   **High-Performance Simulation**: Refactored with `jax.lax.scan` for a **31x speedup** compared to standard Python loops.
*   **Heterogeneous Agent Modeling**: Mix RBC, Q-Learning, and PPO agents in a single community.
*   **AI-Based Pricing**: A centralized RL Community Manager that dynamically sets internal prices to optimize community-level objectives.
*   **Flexible Exchange Mechanisms**: Includes Midpoint Pricing, Double Auction Marketplaces, and Agent-Based Pricing.
*   **Robust Data Handling**: Direct support for custom user-provided consumption, production, and pricing datasets via JSON. For synthetic data, we integrate with **PVLib** for solar profiles and recommend tools like **LoadProfileGenerator** for consumption profiles (ANTGen is no longer supported).
*   **Interactive Demonstration**: A user-friendly web app for non-technical stakeholders to explore energy community trade-offs.

---

## 📦 Manual Installation

If you prefer to run on bare metal, ensure you have **Python 3.12+** and **Node.js 20+**.

### 1. Setup Backend
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
export PYTHONPATH=$PYTHONPATH:$(pwd)/server
python server/main.py
```

### 2. Setup Frontend
```bash
cd demo
npm install
npm run dev
```

---

## 📁 Project Structure
*   `jax_ec/`: Core Python package (JAX environment, agents, mechanisms).
*   `server/`: Flask REST API for headless simulation control.
*   `demo/`: React-based interactive frontend.
*   `tests/`: Formal test suite including integration and mechanism checks.

---

## 📜 Citation

If you use JAX-EC in your research, please cite our AAMAS 2025 paper:

```bibtex
@inproceedings{azmani2025jaxec,
  author    = {Hicham Azmani and Andries Rosseau and Marjon Blondeel and Ann Nowé},
  title     = {A JAX-Accelerated Simulation Framework for Multi-Agent Energy Management in Energy Communities: Demonstration Track},
  booktitle = {Proc. of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025)},
  year      = {2025},
  month     = {May 19-23},
  address   = {Detroit, Michigan, USA},
  publisher = {IFAAMAS},
}
```

---

## 🤝 Contributing
Contributions are welcome! Please refer to [DEVELOPMENT.md](DEVELOPMENT.md) for setup instructions and contribution guidelines.

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
