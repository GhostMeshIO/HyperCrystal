# HyperCrystal v2.0 — Quantum‑Inspired Novelty Engine

An AI system for generating, exploring, and optimizing concepts in a multi‑dimensional goal space (Sophia, Dark Wisdom, Paradox). Features diffusion‑based novelty, meta‑learning (CMA‑ES), Pareto optimization, 3D visualization, and a full product layer (users, credits, marketplace).

## Features
- Core simulation with hyperbolic/Möbius geometry
- Cognition engine with diffusion novelty + repulsion
- Real‑time Flask + SocketIO dashboard
- REST API with authentication & quotas
- Business report & artifact generation
- Multi‑objective optimization (Pareto front, fitness landscape)

## Quick Start

```bash
git clone https://github.com/yourusername/hypercrystal.git
cd hypercrystal

# Recommended: use virtualenv
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Run core simulation + report
python run.py --steps 50 --verbose --report business_plan

# Start the dashboard
python hypercrystal/dashboard/hypercrystal_dash.py

# Or start full API server
python run.py --serve-api --host 0.0.0.0 --port 5000
