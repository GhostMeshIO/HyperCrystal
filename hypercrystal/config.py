# config.py – Shared configuration and constants (v1.7)
# =======================================================
# Supports the full 144‑point blueprint with complete goal steering,
# novelty generation, meta‑learning, QNVM integration, and monetization.
# Sections:
#   I. Goal & Reward (points 1‑24)
#   II. Novelty Generation (25‑48)
#   III. Meta‑Intelligence (49‑72)
#   IV. QNVM Integration (73‑96)
#   V. Output Intelligence (97‑120)
#   VI. Productization & Monetization (121‑144)

import json
import os
import numpy as np

# -----------------------------------------------------------------------------
# Constants (from rne_alien_math and rne_core_types)
# -----------------------------------------------------------------------------
SOPHIA_POINT = (np.sqrt(5) - 1) / 2   # 1/φ ≈ 0.618
PHI = (1 + np.sqrt(5)) / 2            # ≈ 1.618
PHI_INV = 1 / PHI                     # ≈ 0.618
Z3_ROOTS = [1.0 + 0j, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)]
G_INFO = 1.0
DEFAULT_DIM = 128
EPS = 1e-8

# -----------------------------------------------------------------------------
# Default configuration (v1.7 – expanded for 144‑point blueprint)
# -----------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # --- Core / Runtime ---
    "seed": 42,
    "volume_cm3": 1.0,
    "steps": 100,
    "checkpoint_interval": 10,
    "embedding_dim": DEFAULT_DIM,
    "fractal_scale": PHI_INV,
    "retrocausal_strength": 0.1,
    "non_hermitian_gamma": 0.1,
    "stp_pruning": True,
    "memory_capacity": 10000,
    "swarm_agents": 30,
    "tda_enabled": True,
    "verbose": True,
    "checkpoint_dir": "checkpoints",

    # --- I. Goal & Reward (points 1‑24) ---
    "goal_learning_rate": 0.05,
    "reward_decay": 0.95,
    "fitness_weights": {
        "sophia": 1.0,
        "dark_wisdom": 1.0,
        "paradox": 0.5,
        "goal_alignment": 1.0
    },
    "goal_conditioned_embedding": True,
    "goal_history_length": 10,
    "embedding_steering_strength": 0.02,
    "pareto_front_size": 100,                 # new
    "subgoal_decomposition_depth": 3,         # new
    "goal_conflict_threshold": 0.3,           # new
    "expected_utility_discount": 0.9,         # new

    # --- II. Novelty Generation (points 25‑48) ---
    "mutation_rate_base": 0.05,
    "burst_probability": 0.1,
    "burst_factor": 3.0,
    "chaos_cycle_length": 20,
    "exploration_decay_steps": 1000,
    "annealing_steps": 5000,
    "max_universes": 5,
    "contradiction_probability": 0.15,        # new
    "anti_pattern_probability": 0.08,         # new
    "extinction_probability": 0.05,           # new
    "rare_feature_amp_prob": 0.1,             # new
    "hypothesis_stream_count": 4,             # new

    # --- III. Meta‑Learning (points 49‑72) ---
    "meta_learning_rate": 0.05,
    "benchmark_interval": 100,
    "max_recursion_depth": 5,
    "sophia_attractor_strength": 0.2,
    "convergence_threshold": 0.05,
    "convergence_steps": 3,
    "paradox_rise_rate": 0.02,
    "max_meta_depth": 10,
    "min_meta_depth": 1,
    "self_model_update_rate": 0.1,            # new
    "instability_variance_threshold": 0.2,    # new
    "heuristic_learning_rate": 0.05,          # new
    "rollback_probability": 0.02,             # new

    # --- IV. QNVM Integration (points 73‑96) ---
    "quantization_bits": 8,                   # new
    "soa_enabled": True,                      # new
    "predictive_prefetch": True,              # new

    # --- V. Output Intelligence (points 97‑120) ---
    "report_confidence_floor": 0.3,           # new

    # --- VI. Productization & Monetization (points 121‑144) ---
    "rate_limit": 100,
    "subscription_tiers": {
        "free": {
            "concepts": 100,
            "queries_per_day": 50,
            "price_usd": 0,
            "initial_credits": 100
        },
        "pro": {
            "concepts": 10000,
            "queries_per_day": 5000,
            "price_usd": 29,
            "initial_credits": 1000
        },
        "enterprise": {
            "concepts": 100000,
            "queries_per_day": 100000,
            "price_usd": 199,
            "initial_credits": 10000
        }
    },
    "api_key_prefix": "hc_",
    "viral_share_bonus_credits": 50,          # new
    "conversion_optimization_rate": 0.05,     # new
    "admin_api_key": "admin123",              # new
}


# -----------------------------------------------------------------------------
# Configuration loader with validation
# -----------------------------------------------------------------------------
def load_config(config_path: str = "hypercrystal_config.json") -> dict:
    """
    Load configuration from JSON file, merging with defaults.
    Raises KeyError if any required key is missing after merge,
    and validates numeric ranges for all new keys.
    """
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        config.update(user_config)

    # Required keys (must exist)
    required = [
        "seed",
        "embedding_dim",
        "memory_capacity",
        "sophia_attractor_strength",
        "goal_learning_rate",
        "reward_decay",
        "fitness_weights",
        "mutation_rate_base",
        "stagnation_threshold",
        "meta_learning_rate"
    ]
    for key in required:
        if key not in config:
            raise KeyError(f"Missing required configuration key: {key}")

    # Validate fitness_weights structure
    expected_weights = {"sophia", "dark_wisdom", "paradox", "goal_alignment"}
    if not expected_weights.issubset(config["fitness_weights"]):
        raise KeyError("fitness_weights must contain keys: sophia, dark_wisdom, paradox, goal_alignment")

    # Validate numeric ranges
    numeric_checks = [
        ("embedding_dim", 1, None),
        ("memory_capacity", 1, None),
        ("sophia_attractor_strength", 0, 1),
        ("mutation_rate_base", 0, 1),
        ("stagnation_threshold", 1, None),
        ("meta_learning_rate", 0, 1),
        ("goal_learning_rate", 0, 1),
        ("reward_decay", 0, 1),
        ("embedding_steering_strength", 0, 1),
        ("pareto_front_size", 1, None),
        ("subgoal_decomposition_depth", 1, None),
        ("goal_conflict_threshold", 0, 1),
        ("expected_utility_discount", 0, 1),
        ("contradiction_probability", 0, 1),
        ("anti_pattern_probability", 0, 1),
        ("extinction_probability", 0, 1),
        ("rare_feature_amp_prob", 0, 1),
        ("hypothesis_stream_count", 1, None),
        ("self_model_update_rate", 0, 1),
        ("instability_variance_threshold", 0, None),
        ("heuristic_learning_rate", 0, 1),
        ("rollback_probability", 0, 1),
        ("quantization_bits", 1, 64),
        ("report_confidence_floor", 0, 1),
        ("viral_share_bonus_credits", 0, None),
        ("conversion_optimization_rate", 0, 1),
    ]

    for key, min_val, max_val in numeric_checks:
        if key in config:
            val = config[key]
            if not isinstance(val, (int, float)):
                raise TypeError(f"{key} must be numeric")
            if min_val is not None and val < min_val:
                raise ValueError(f"{key} must be >= {min_val}")
            if max_val is not None and val > max_val:
                raise ValueError(f"{key} must be <= {max_val}")

    # Validate boolean flags (just existence, they can be bool or int)
    bool_keys = [
        "goal_conditioned_embedding",
        "stp_pruning",
        "tda_enabled",
        "soa_enabled",
        "predictive_prefetch",
    ]
    for key in bool_keys:
        if key in config and not isinstance(config[key], (bool, int)):
            raise TypeError(f"{key} must be boolean or integer")

    return config
