# HyperCrystal v2.0 — Quantum-Inspired Novelty Engine

**A multi-dimensional concept exploration and optimisation system** using hyperbolic geometry, diffusion-based novelty injection, meta-learning (CMA-ES), Pareto optimisation, and real-time 3D visualisation.

**⚠️ Important Security & Setup Notice**  
- On first run, a **random admin password** is generated and printed to console. Change it immediately.  
- Default credentials are **no longer accepted**.  
- Always use HTTPS in production.  
- Never commit secrets (`users.json`, `.env`, `hypercrystal_config.json`).

---

## Overview

HyperCrystal generates, organises, and optimises concepts in a 3D goal space defined by:
- **Sophia** (coherence / wisdom)
- **Dark Wisdom** (contradiction / depth)
- **Paradox** (creative tension)

**Core features:**
- Hyperbolic / Möbius geometry for concept representation (Poincaré ball)
- Diffusion novelty injection with repulsion forces
- Meta-learning via CMA-ES
- Real-time Pareto front optimisation
- 3D interactive dashboard (Three.js + SocketIO + Flask)
- REST API with JWT/auth, quotas, credits, and artifact generation
- Marketplace simulation and business reporting tools
- Optional TDA (persistent homology), Torch-based scoring, HNSW indexing

---

## Quick Start (Local Development)

```bash
git clone https://github.com/GhostMeshIO/HyperCrystal.git
cd HyperCrystal

# Virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

pip install -r requirements.txt
pip install -r requirements-optional.txt   # recommended: cma, gudhi, torch, hnswlib

# Generate config + secrets
python -c "from utils.config import ensure_config; ensure_config()"

# Run core simulation + business plan report
python run.py --steps 100 --verbose --report business_plan

# Launch interactive 3D dashboard
python hypercrystal_dash.py

# Or start full API server
python run.py --serve-api --host 127.0.0.1 --port 5000
```

---

## Production Deployment (Docker + HTTPS)

```bash
cp .env.example .env
# Edit .env with strong secrets

# Generate certificates (or use Let's Encrypt in production)
mkdir -p certs
openssl req -x509 -nodes -days 90 -newkey rsa:2048 \
  -keyout certs/privkey.pem -out certs/fullchain.pem \
  -subj "/CN=your-domain.com"

docker-compose up -d --build
```

Access the API at `https://your-domain.com` and the dashboard accordingly.

**Security recommendations:**
- Rotate `admin_api_key` and JWT secrets regularly.
- Use a real database (PostgreSQL) in production instead of SQLite.
- Enable rate limiting and monitor for abuse.
- Never expose debug mode in production.

---

## Configuration

`hypercrystal_config.json` is auto-generated on first run. Key tunable parameters:

- `max_concepts`, `embedding_dim`, `ann_update_interval`
- `sophia_attractor_strength`, `paradox_rise_rate`, `repulsion_strength`
- Subscription tiers and credit system

Edit carefully — invalid values can cause crashes or degraded performance.

---

## API Endpoints (v1)

All endpoints (except `/health`) require authentication via **JWT** or **`X-API-Key`**.

See full OpenAPI docs at `/api/docs` when running.

---

## Known Shortcomings & Roadmap

A comprehensive **144-point audit** (critical bugs, logic issues, architecture flaws, performance bottlenecks, security gaps, and science-driven enhancement proposals) has been performed.

See [`SHORTCOMINGS_v1.md`](SHORTCOMINGS_v1.md) for the full detailed report.

**High-priority fixes needed before production use:**
- Fix index consistency after eviction/extinction events (critical data corruption risk)
- Replace O(n²) novelty computation with ANN/HNSW
- Secure credential handling (`users.json` must not be committed)
- Add proper locking for concurrent access
- Implement input validation and sanitisation
- Add database migrations and persistent storage

**Exciting science-driven enhancements proposed** (see audit):
- Natural gradients on Fisher-Rao manifold
- Topological Data Analysis (Mapper / persistence)
- Active Inference / Free Energy Principle framing
- Diffusion models for concept generation
- Hyperbolic Graph Neural Networks
- Stigmergic pheromone trails / swarm intelligence
- Lyapunov chaos control for "edge of chaos" operation

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). All PRs must pass:
- `pytest`
- `black` + `isort`
- `mypy` (when type coverage improves)

**Before submitting major changes**, review the 144-point audit and address relevant shortcomings.

---

## License

MIT License (see [`LICENSE`](LICENSE))

---

## Acknowledgements

Inspired by topological data analysis, information geometry, multi-objective optimisation, and concepts from Active Inference, hyperbolic geometry, and complex adaptive systems.

**Note:** This project is under active development. Many advanced features (TDA, Torch novelty, quantum-inspired modules) are optional and fall back gracefully when dependencies are missing.

**Status as of 2026-04-07:** The system runs and demonstrates core concepts, but contains several critical stability, security, and scalability issues identified in the audit. Use with caution in production.
```

---
