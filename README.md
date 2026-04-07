# HyperCrystal v2.0 — Quantum‑Inspired Novelty Engine

An AI system for generating, exploring, and optimizing concepts in a multi‑dimensional goal space (Sophia, Dark Wisdom, Paradox). Features diffusion‑based novelty, meta‑learning (CMA‑ES), Pareto optimization, 3D visualization, and a full product layer (users, credits, marketplace).

**⚠️ Security Notice**  
This project has undergone major security and performance improvements. **Default credentials are no longer accepted** – on first start, a random admin password is generated and printed to the console. Always use HTTPS in production.

---

## Overview

HyperCrystal is an AI system for generating, exploring, and optimizing concepts in a multi‑dimensional goal space (Sophia, Dark Wisdom, Paradox). It features:

- Hyperbolic/Möbius geometry for concept representation
- Diffusion‑based novelty injection with repulsion forces
- Meta‑learning via CMA‑ES (Covariance Matrix Adaptation Evolution Strategy)
- Pareto front optimisation and fitness landscape analysis
- Real‑time 3D dashboard (SocketIO + Flask)
- REST API with user authentication, quotas, and artifact generation
- Business reporting and marketplace simulation

---

## Quick Start (Local Development)

```bash
# Clone the repository
git clone https://github.com/GhostMeshIO/HyperCrystal.git
cd HyperCrystal

# Set up a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# Install core dependencies
pip install -r requirements.txt

# (Optional) Install extra dependencies for advanced features
pip install -r requirements-optional.txt   # see below

# Create a default configuration file (automatically generates secrets)
python -c "from utils.config import ensure_config; ensure_config()"

# Run the core simulation with a business plan report
python run.py --steps 50 --verbose --report business_plan

# Start the interactive dashboard (3D visualization)
python HyperCrystal/dashboard/HyperCrystal_dash.py

# Or launch the full API server (REST + WebSocket)
python run.py --serve-api --host 127.0.0.1 --port 5000
```

---

## Production Deployment (Docker + HTTPS)

For production, use the provided Docker Compose stack with nginx and TLS.

```bash
# 1. Copy the environment template and edit secrets
cp .env.example .env
nano .env   # set strong random secrets

# 2. Generate self‑signed certificates (or use Let's Encrypt)
mkdir certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout certs/privkey.pem -out certs/fullchain.pem \
  -subj "/CN=your-domain.com"

# 3. Start the stack
docker-compose up -d

# 4. Access the API at https://your-domain.com
```

See `docker-compose.yml` and `nginx.conf` in the repository for details.

---

## Environment Variables

Create a `.env` file in the project root (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable Flask debug mode | `False` |
| `JWT_SECRET` | Secret for JWT tokens | auto‑generated |
| `DASH_SECRET_KEY` | Secret for dashboard sessions | auto‑generated |
| `HYPERCRYSTAL_ADMIN_KEY` | Admin API key (override config) | random on first run |
| `DATABASE_URL` | SQLite/PostgreSQL connection string | `sqlite:///data/hypercrystal.db` |

---

## Dependencies & Optional Features

**Required** (in `requirements.txt`):
- `flask`, `flask-socketio`, `python-socketio`
- `numpy`, `scipy`, `scikit-learn`
- `matplotlib`, `plotly`, `dash`
- `bcrypt`, `pyjwt`, `python-dotenv`

**Optional** (in `requirements-optional.txt`):
- `cma` – meta‑learning (CMA‑ES)
- `gudhi` – topological persistence (Betti numbers)
- `torch` – neural novelty scoring
- `hnswlib` – fast incremental ANN (replaces sklearn’s full rebuild)

If optional packages are missing, the system falls back to simpler implementations (Euclidean distance, constant metrics) and logs a warning.

---

## API Endpoints (v1)

All endpoints require either a **JWT bearer token** (from `/auth/login`) or an **`X-API-Key`** header (admin only).

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Create a new user (username, password) |
| POST | `/auth/login` | Obtain JWT token |
| GET | `/concepts` | List all concepts (paginated) |
| POST | `/concepts` | Store a new concept (embedding, metadata) |
| GET | `/concepts/<id>` | Retrieve a concept |
| POST | `/artifacts/generate` | Generate a report/diagram (uses credits) |
| GET | `/metrics` | Real‑time system metrics (novelty, Pareto front size) |
| GET | `/health` | Health check (no auth) |

Full OpenAPI documentation is available at `/api/docs` when the server is running.

---

## Configuration

The file `hypercrystal_config.json` is automatically created on first run. You can edit it to tune:

```json
{
  "admin_api_key": "auto_generated_random_key",
  "ann_update_interval": 100,
  "max_concepts": 10000,
  "history_size": 50,
  "credits_per_artifact": 10,
  "default_user_credits": 100
}
```

**Important**: Change the `admin_api_key` after deployment and store it securely (e.g., in your password manager).

---

## Security Best Practices

1. **Never commit `.env` or `hypercrystal_config.json`** – they contain secrets.
2. **Use HTTPS** in production – the provided nginx configuration enforces TLS.
3. **Regularly rotate API keys** – use the `/admin/rotate-key` endpoint (admin only).
4. **Passwords are hashed with bcrypt** – no plaintext storage.
5. **Run as non‑root user** inside the Docker container (already configured).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `KeyError: 'admin_api_key'` | Run `python -c "from utils.config import ensure_config; ensure_config()"` to create default config. |
| ANN rebuild is slow | Install `hnswlib` (optional) for incremental indexing. |
| Dashboard websocket disconnects | Ensure `proxy_http_version 1.1` and `Upgrade` headers are set in nginx (see example config). |
| “No module named 'rne_*'” | These are proprietary modules – the system falls back to Euclidean math. This is expected unless you have access to the RNE library. |

---

## Contributing

See `CONTRIBUTING.md` (to be added). For major changes, please open an issue first. All pull requests must pass:
- `pytest` (unit tests)
- `mypy` (type checking)
- `black` + `isort` (formatting)

---

## License

[Specify your license, e.g., MIT, GPL‑3.0, or Proprietary]

---

## Acknowledgements

Built with insights from topological data analysis, information geometry, and multi‑objective optimisation. The RNE (Recursive Neural Embedding) modules are optional and not required for basic functionality.
