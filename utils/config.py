# utils/config.py
import json
import secrets
from pathlib import Path

DEFAULT_CONFIG = {
    "seed": 42,
    "embedding_dim": 128,
    "memory_capacity": 10000,
    "history_size": 50,
    "ann_update_interval": 10,
    "metrics_compute_interval": 10,
    "sophia_attractor_strength": 0.2,
    "mutation_rate_base": 0.15,
    "novelty_acceptance_threshold": 0.08,
    "pareto_front_size": 100,
    "checkpoint_dir": "checkpoints",
    "verbose": False,
    "use_sqlite": False,
    "admin_api_key": None   # will be generated
}

def ensure_config(config_path: str = "hypercrystal_config.json"):
    """Create a default config file if it doesn't exist."""
    if not Path(config_path).exists():
        config = DEFAULT_CONFIG.copy()
        config["admin_api_key"] = secrets.token_urlsafe(32)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Created default configuration at {config_path}")
        print(f"Admin API key: {config['admin_api_key']}")
    else:
        print(f"Configuration already exists at {config_path}")

if __name__ == "__main__":
    ensure_config()
