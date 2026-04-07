"""
core_engine.py – Memory & Substrate for the HyperCrystal / QNVM (v2.0)
======================================================================
Enhanced with:
- Security: cryptographically secure keys, no internal API checks
- Scalability: ANN index for O(log n) similarity search
- Pareto front with crowding distance for diverse optimization
- Hyperbolic embedding steering (placeholder)
- Drift detection via MMD
- Input validation
- Proper goal assignment for new concepts
- Placeholder for goal steering learning
"""

import json
import os
import random
import sys
import time
import warnings
import hashlib
import secrets
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set, Union
from numpy.typing import NDArray
from collections import deque

# -----------------------------------------------------------------------------
# RNE v2.9 imports (with robust fallback)
# -----------------------------------------------------------------------------
try:
    from rne_core_types import (
        Concept, CausalGraph, RNEContext, mobius_distance,
        project_to_ball, SOPHIA_POINT
    )
    from rne_alien_math import (
        PHI, PHI_INV,
        dark_wisdom_density, sophia_criticality_score, retrocausal_boundary_condition,
        z3_symmetric_blend, batch_sophia_geodesic_flow, manifold_mixup, geodesic_interpolation,
        fractal_renormalize
    )
    try:
        from rne_alien_math import holographic_entropy as alien_holographic_entropy
    except ImportError:
        alien_holographic_entropy = None
    from rne_memory_manager import SemanticMemory, EpisodicMemory, SpinGlassAttention, RAMOptimizer
    from rne_causal_symbolic import (
        full_novelty_async, godel_auditor_async, causal_intervention_gnn, CausalGNN,
        z3_symmetric_synthesis, sophia_criticality_boost, abductive_generate_fallback
    )
    from rne_consciousness import ConsciousnessTracker, encode_self_state_spikes
    from rne_health_repair import HardwareProfiler, FaultDetector, HealthMonitor, CheckpointManager
    from rne_swarm_dynamics import SwarmController, SwarmAgent, attempt_fusion
    from rne_llm_integration import load_llm_providers, AbductiveGeneratorLLM, ValidationLLM, MemorySummarizerLLM, SelfReflectionLLM
    _HAS_RNE = True
except ImportError:
    _HAS_RNE = False
    warnings.warn("RNE modules not found. Using minimal fallbacks.")

    # Minimal fallback classes and functions
    class Concept:
        def __init__(self, subsymbolic=None, symbolic=None, causal_graph=None,
                     sophia_score=0.5, dark_wisdom_density=0.0, paradox_intensity=0.0,
                     chronon_entanglement=0.0, biophoton_amplitude=0.0, z3_phase=1+0j,
                     retrocausal_kernel=None):
            self.subsymbolic = np.asarray(subsymbolic, dtype=np.float64) if subsymbolic is not None else np.random.randn(64)
            self.symbolic = symbolic or []
            self.causal_graph = causal_graph or CausalGraph()
            self.sophia_score = sophia_score
            self.dark_wisdom_density = dark_wisdom_density
            self.paradox_intensity = paradox_intensity
            self.chronon_entanglement = chronon_entanglement
            self.biophoton_amplitude = biophoton_amplitude
            self.z3_phase = z3_phase
            self.retrocausal_kernel = retrocausal_kernel if retrocausal_kernel is not None else np.random.randn(10) * 0.1
        def distance(self, other):
            return np.linalg.norm(self.subsymbolic - other.subsymbolic)
        def to_text(self):
            return f"Concept(sophia={self.sophia_score:.3f})"

    class CausalGraph:
        def __init__(self, vertices=None, edges=None, retrocausal_edges=None):
            self.vertices = vertices or []
            self.edges = edges or []
            self.retrocausal_edges = retrocausal_edges or []
        def copy(self):
            return CausalGraph(self.vertices.copy(), self.edges.copy(), self.retrocausal_edges.copy())

    class RNEContext:
        def __init__(self):
            self.H_t = []
            self.K_t = []
            self.params = {}
            self._abductive_generator = None
            self._validation_llm = None
        def add_to_memory(self, c):
            self.K_t.append(c)

    def project_to_ball(v, eps=1e-8):
        v = np.asarray(v, dtype=np.float64)
        norm = np.linalg.norm(v)
        if norm >= 1.0:
            return v * (1.0 - eps) / norm
        return v.copy()

    def mobius_distance(x, y):
        return np.linalg.norm(x - y)

    SOPHIA_POINT = (np.sqrt(5)-1)/2
    PHI = (1+np.sqrt(5))/2
    PHI_INV = 1/PHI

    def dark_wisdom_density(r, G=1.0):
        return -r/(8*np.pi*G)

    def sophia_criticality_score(s):
        return np.exp(-((s - SOPHIA_POINT)**2)/(2*0.05**2))

    def retrocausal_boundary_condition(f, strength=1.0):
        return 1.0 + strength * np.tanh(f-0.5)

    def batch_sophia_geodesic_flow(e, s, step=0.01):
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        grad = 2.0*(norms - SOPHIA_POINT)*(e/(norms+1e-8))
        return project_to_ball(e - step*grad)

    def manifold_mixup(emb, w):
        return sum(w[i]*emb[i] for i in range(len(emb)))/sum(w)

    def geodesic_interpolation(x, y, t):
        return (1-t)*x + t*y

    def fractal_renormalize(emb, scale):
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb * scale/(norms+1e-8)

    def z3_symmetric_blend(p1, p2, p3):
        return 1+0j

    alien_holographic_entropy = None

    ConsciousnessTracker = None
    SwarmController = None
    SwarmAgent = None
    CheckpointManager = None
    load_llm_providers = lambda x: {}

# -----------------------------------------------------------------------------
# Try to import scikit-learn for ANN and MMD
# -----------------------------------------------------------------------------
try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not found. ANN index and MMD will be disabled.")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
Z3_ROOTS = [1.0 + 0j, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)]
G_INFO = 1.0
DEFAULT_DIM = 128
EPS = 1e-8

# -----------------------------------------------------------------------------
# Configuration loader (shared)
# -----------------------------------------------------------------------------
def load_config(config_path: str = "hypercrystal_config.json") -> dict:
    default = {
        "seed": 42,
        "volume_cm3": 1.0,
        "steps": 100,
        "checkpoint_interval": 10,
        "use_llm": False,
        "llm_providers": {"deepseek": True},
        "task_provider_mapping": {
            "generation": "deepseek",
            "validation": "deepseek",
            "summarization": "deepseek",
            "reflection": "deepseek"
        },
        "embedding_dim": DEFAULT_DIM,
        "fractal_scale": PHI_INV,
        "retrocausal_strength": 0.1,
        "non_hermitian_gamma": 0.1,
        "stp_pruning": True,
        "memory_capacity": 10000,
        "swarm_agents": 30,
        "tda_enabled": True,
        "verbose": True,
        "sophia_attractor_strength": 0.2,
        "convergence_threshold": 0.05,
        "convergence_steps": 3,
        "paradox_rise_rate": 0.02,
        "max_meta_depth": 10,
        "min_meta_depth": 1,
        "checkpoint_dir": "checkpoints",
        "rate_limit": 100,
        "subscription_tiers": {
            "free": {"concepts": 100, "queries_per_day": 50, "price_usd": 0, "initial_credits": 100},
            "pro": {"concepts": 10000, "queries_per_day": 5000, "price_usd": 29, "initial_credits": 1000},
            "enterprise": {"concepts": 100000, "queries_per_day": 100000, "price_usd": 199, "initial_credits": 10000}
        },
        "api_key_prefix": "hc_",
        # New cognitive parameters
        "goal_learning_rate": 0.05,
        "reward_decay": 0.95,
        "fitness_weights": {"sophia": 1.0, "dark_wisdom": 1.0, "paradox": 0.5, "goal_alignment": 1.0},
        "goal_conditioned_embedding": True,
        "goal_history_length": 10,
        "embedding_steering_strength": 0.02,
        "pareto_front_size": 100,
        "subgoal_depth": 3,
        "conflict_threshold": 0.3,
        "soa_enabled": True,               # Structure-of-Arrays encoding placeholder
        "quantization_bits": 8,
        # New scaling parameters
        "mmd_threshold": 0.1,
        "repulsion_strength": 0.02,
        "gradient_tau": 1.0,
        "ann_update_interval": 10,         # steps between rebuilding ANN index
        # === New keys for novelty gate and mutation ===
        "novelty_acceptance_threshold": 0.08,
        "min_mutation_rate": 0.08,
        "max_mutation_rate": 0.35,
        "stagnation_threshold": 10,
        # Updated mutation defaults
        "mutation_rate_base": 0.15,
        "burst_probability": 0.12,
        "burst_factor": 3.5,
        "diffusion_denoise_alpha": 0.9,
        "admin_api_key": os.environ.get("HYPERCRYSTAL_ADMIN_KEY", "change_this_before_deployment"),
    }
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user = json.load(f)
        default.update(user)
    return default

# -----------------------------------------------------------------------------
# GoalField (enhanced with history and utility)
# -----------------------------------------------------------------------------
@dataclass
class GoalField:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    history: List[np.ndarray] = field(default_factory=list)
    parent: Optional['GoalField'] = None
    children: List['GoalField'] = field(default_factory=list)

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> 'GoalField':
        return cls(arr[0], arr[1], arr[2])

    def distance_to(self, other: 'GoalField') -> float:
        return np.linalg.norm(self.as_array() - other.as_array())

    def add_history(self, arr: NDArray[np.float64], maxlen: int = 10):
        self.history.append(arr.copy())
        if len(self.history) > maxlen:
            self.history.pop(0)

    def smoothed_goal(self) -> NDArray[np.float64]:
        if not self.history:
            return self.as_array()
        return np.mean(self.history, axis=0)

    def is_conflicting(self, other: 'GoalField', threshold: float = 0.3) -> bool:
        """Detect if two goals are in conflict (angle > threshold)."""
        u = self.as_array()
        v = other.as_array()
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u < EPS or norm_v < EPS:
            return False
        cos = np.dot(u, v) / (norm_u * norm_v)
        return cos < threshold

    def resolve_conflict(self, other: 'GoalField') -> 'GoalField':
        """Return a compromise goal (midpoint in field space)."""
        return GoalField.from_array((self.as_array() + other.as_array()) / 2.0)

    def __repr__(self):
        return f"GoalField({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

# -----------------------------------------------------------------------------
# MemoryOptimizer (unchanged)
# -----------------------------------------------------------------------------
class MemoryOptimizer:
    def __init__(self, dim: int = DEFAULT_DIM, sparsity_threshold: float = 0.2):
        self.dim = dim
        self.sparsity_threshold = sparsity_threshold
        self.sparse_representations: Dict[str, Dict[int, float]] = {}
        self.compression_ratio = 1.0

    def encode(self, key: str, embedding: NDArray[np.float64]) -> NDArray[np.float64]:
        norm = np.linalg.norm(embedding)
        if norm < self.sparsity_threshold:
            sparse = {i: v for i, v in enumerate(embedding) if abs(v) > self.sparsity_threshold}
            self.sparse_representations[key] = sparse
            return embedding
        else:
            if key in self.sparse_representations:
                del self.sparse_representations[key]
            return embedding

    def retrieve(self, key: str) -> Optional[NDArray[np.float64]]:
        if key in self.sparse_representations:
            dense = np.zeros(self.dim, dtype=np.float64)
            for idx, val in self.sparse_representations[key].items():
                dense[idx] = val
            return dense
        return None

    def get_memory_usage(self) -> int:
        bytes_used = 0
        for sparse in self.sparse_representations.values():
            bytes_used += len(sparse) * (4 + 8)
        return bytes_used

# -----------------------------------------------------------------------------
# ResourceMonitor (unchanged)
# -----------------------------------------------------------------------------
class ResourceMonitor:
    def __init__(self, config: dict):
        self.config = config
        self.cost_total = 0.0
        self.operation_count = 0
        self.last_reset = time.time()
        self.user_usage: Dict[str, Dict[str, Any]] = {}

    def log_operation(self, user_id: Optional[str] = None, operation_type: str = "query", cost: float = 0.001):
        self.operation_count += 1
        self.cost_total += cost
        if user_id:
            if user_id not in self.user_usage:
                self.user_usage[user_id] = {"queries_today": 0, "concepts_stored": 0, "last_reset": time.time()}
            if time.time() - self.user_usage[user_id]["last_reset"] > 86400:
                self.user_usage[user_id]["queries_today"] = 0
                self.user_usage[user_id]["last_reset"] = time.time()
            if operation_type == "query":
                self.user_usage[user_id]["queries_today"] += 1
            elif operation_type == "store":
                self.user_usage[user_id]["concepts_stored"] += 1

    def check_rate_limit(self, user_id: str, tier: str = "free") -> bool:
        usage = self.user_usage.get(user_id, {"queries_today": 0})
        limits = self.config["subscription_tiers"].get(tier, self.config["subscription_tiers"]["free"])
        return usage["queries_today"] < limits["queries_per_day"]

    def get_cost_summary(self) -> dict:
        return {
            "total_cost_usd": round(self.cost_total, 4),
            "total_operations": self.operation_count,
            "active_users": len(self.user_usage)
        }

# -----------------------------------------------------------------------------
# APIGateway (unchanged)
# -----------------------------------------------------------------------------
class APIGateway:
    def __init__(self, resource_monitor: ResourceMonitor, config: dict):
        self.monitor = resource_monitor
        self.config = config
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.user_keys: Dict[str, List[str]] = {}

    def generate_api_key(self, user_id: str, tier: str = "free") -> str:
        key = self.config["api_key_prefix"] + secrets.token_hex(16)
        self.api_keys[key] = {"user_id": user_id, "tier": tier, "created": time.time()}
        self.user_keys.setdefault(user_id, []).append(key)
        return key

    def validate_api_key(self, api_key: str) -> Optional[str]:
        info = self.api_keys.get(api_key)
        if info:
            return info["user_id"]
        return None

    def get_tier(self, user_id: str) -> str:
        keys = self.user_keys.get(user_id, [])
        for key in keys:
            if key in self.api_keys:
                return self.api_keys[key]["tier"]
        return "free"

    def check_and_log(self, api_key: str, operation_type: str = "query", cost: float = 0.001) -> bool:
        user_id = self.validate_api_key(api_key)
        if not user_id:
            return False
        tier = self.get_tier(user_id)
        if not self.monitor.check_rate_limit(user_id, tier):
            return False
        self.monitor.log_operation(user_id, operation_type, cost)
        return True

# -----------------------------------------------------------------------------
# PhysicalSubstrate (unchanged)
# -----------------------------------------------------------------------------
class PhysicalSubstrate:
    def __init__(self, volume_cm3: float = 1.0):
        self.volume = volume_cm3
        self.max_capacity = int(1e6 * volume_cm3)
        self.bragg_gratings: Dict[str, NDArray] = {}
        self.quantum_dot_spins: Dict[str, complex] = {}
        self.four_wave_mixing_coeff = 0.1
        self._entanglement_pairs: Dict[str, List[str]] = {}

    def _encode_5d(self, vector: NDArray) -> NDArray:
        if len(vector) < 5:
            vector = np.pad(vector, (0, 5 - len(vector)), 'constant')
        return vector[:5]

    def write(self, key: str, data: NDArray, spin_state: complex = 1+0j) -> None:
        if len(self.bragg_gratings) >= self.max_capacity:
            oldest = next(iter(self.bragg_gratings))
            del self.bragg_gratings[oldest]
            del self.quantum_dot_spins[oldest]
            if oldest in self._entanglement_pairs:
                for k in list(self._entanglement_pairs.keys()):
                    if oldest in self._entanglement_pairs[k]:
                        self._entanglement_pairs[k].remove(oldest)
                del self._entanglement_pairs[oldest]
        encoded = self._encode_5d(data)
        self.bragg_gratings[key] = encoded
        self.quantum_dot_spins[key] = spin_state
        if len(self.bragg_gratings) > 1:
            other = random.choice([k for k in self.bragg_gratings if k != key])
            self._entangle(key, other)

    def _entangle(self, key1: str, key2: str) -> None:
        self._entanglement_pairs.setdefault(key1, []).append(key2)
        self._entanglement_pairs.setdefault(key2, []).append(key1)

    def read(self, query: NDArray) -> Optional[Tuple[NDArray, complex]]:
        if not self.bragg_gratings:
            return None
        q = self._encode_5d(query)
        best_key = None
        best_sim = -1
        for key, stored in self.bragg_gratings.items():
            sim = np.dot(q, stored) / (np.linalg.norm(q) * np.linalg.norm(stored) + EPS)
            if key in self._entanglement_pairs:
                ent_contrib = 0.05 * len(self._entanglement_pairs[key])
                sim += ent_contrib
            if sim > best_sim:
                best_sim = sim
                best_key = key
        if best_key is not None:
            return self.bragg_gratings[best_key], self.quantum_dot_spins.get(best_key, 1+0j)
        return None

    def four_wave_mixing(self, pump1: NDArray, pump2: NDArray, signal: NDArray) -> NDArray:
        pump1 = self._encode_5d(pump1)
        pump2 = self._encode_5d(pump2)
        signal = self._encode_5d(signal)
        return self.four_wave_mixing_coeff * np.outer(pump1, pump2).dot(signal)

    def apply_bragg_grating(self, key: str, reference: NDArray, object_beam: NDArray) -> None:
        self.bragg_gratings[key] = (reference, object_beam)

# -----------------------------------------------------------------------------
# QHDRAM (unchanged)
# -----------------------------------------------------------------------------
class Hypertoken:
    def __init__(self, alpha: complex, beta: complex, gamma: complex):
        self.amplitudes = np.array([alpha, beta, gamma], dtype=complex)
        self.normalize()

    def normalize(self):
        norm = np.sqrt(np.sum(np.abs(self.amplitudes)**2))
        if norm > 1e-12:
            self.amplitudes /= norm

    def invariant(self) -> complex:
        return np.prod(self.amplitudes)

    def __repr__(self):
        return f"Hypertoken({self.amplitudes})"

class QHDRAM:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.hypertokens: Dict[str, Hypertoken] = {}
        self.entanglement_map: Dict[str, List[str]] = {}

    def store(self, key: str, embedding: NDArray):
        proj = embedding[:3].astype(complex)
        corrected = []
        for val in proj:
            dists = [np.abs(val - r) for r in Z3_ROOTS]
            corrected.append(Z3_ROOTS[np.argmin(dists)])
        ht = Hypertoken(corrected[0], corrected[1], corrected[2])
        self.hypertokens[key] = ht
        if len(self.hypertokens) > self.capacity:
            oldest = next(iter(self.hypertokens))
            del self.hypertokens[oldest]
            if oldest in self.entanglement_map:
                for k in list(self.entanglement_map.keys()):
                    if oldest in self.entanglement_map[k]:
                        self.entanglement_map[k].remove(oldest)
                del self.entanglement_map[oldest]
        if len(self.hypertokens) > 1:
            other = random.choice([k for k in self.hypertokens if k != key])
            self.entanglement_map.setdefault(key, []).append(other)
            self.entanglement_map.setdefault(other, []).append(key)

    def retrieve(self, query: NDArray) -> Optional[Hypertoken]:
        if not self.hypertokens:
            return None
        q = query[:3].astype(complex)
        best = None
        best_score = -1
        for key, ht in self.hypertokens.items():
            score = np.abs(np.dot(q, ht.amplitudes)).real
            if key in self.entanglement_map:
                ent_contrib = 0.05 * len(self.entanglement_map[key])
                score += ent_contrib
            if score > best_score:
                best_score = score
                best = ht
        return best

    def correct_errors(self):
        for key, ht in self.hypertokens.items():
            inv = ht.invariant()
            dists = [np.abs(inv - root) for root in Z3_ROOTS]
            min_idx = np.argmin(dists)
            if dists[min_idx] > 0.05:
                target_root = Z3_ROOTS[min_idx]
                factor = target_root / inv
                ht.amplitudes *= np.sqrt(factor) if np.abs(factor) > 0 else 1.0
                ht.normalize()

# -----------------------------------------------------------------------------
# Virtual Enhancements (unchanged)
# -----------------------------------------------------------------------------
class VirtualEnhancements:
    @staticmethod
    def z3_godelian_error_correction(state: 'HyperCrystalState') -> None:
        if state.qhdram:
            state.qhdram.correct_errors()

    @staticmethod
    def semantic_curvature_sourcing(state: 'HyperCrystalState') -> None:
        if len(state.concepts) > 1:
            embeddings = np.vstack([c.subsymbolic for c in state.concepts])
            n = len(embeddings)
            if n > 200:
                idx = np.random.choice(n, 200, replace=False)
                embeddings = embeddings[idx]
                n = 200
            h_dists = np.zeros((n, n))
            e_dists = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    h_dists[i, j] = mobius_distance(embeddings[i], embeddings[j])
                    e_dists[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
            diff = (h_dists - e_dists).flatten()
            state.ricci_proxy = np.mean(diff)
            if state.ricci_proxy < 0:
                state.dark_wisdom_density = max(0.0, min(1.0, dark_wisdom_density(state.ricci_proxy)))
        else:
            state.ricci_proxy = 0.0

    @staticmethod
    def fractal_scale_memory(state: 'HyperCrystalState') -> None:
        if state.concepts:
            norms = np.array([np.linalg.norm(c.subsymbolic) for c in state.concepts])
            avg_norm = np.mean(norms)
            target_norm = 0.6
            if avg_norm < 0.4:
                scale = min(1.2, 0.9 / (avg_norm + EPS))
                for c in state.concepts:
                    c.subsymbolic = project_to_ball(c.subsymbolic * scale)
            elif avg_norm > 0.8:
                scale = 0.8 / avg_norm
                for c in state.concepts:
                    c.subsymbolic = project_to_ball(c.subsymbolic * scale)

    @staticmethod
    def temporal_standing_waves(state: 'HyperCrystalState') -> None:
        state.chronon_entanglement = 0.5 + 0.3 * np.tanh(state.meta_depth - 2) + 0.2 * state.paradox_intensity
        state.chronon_entanglement = np.clip(state.chronon_entanglement, 0.0, 1.0)

    @staticmethod
    def paradox_pressure_compression(state: 'HyperCrystalState') -> None:
        pressure = state.paradox_intensity / (state.meta_depth + EPS)
        if pressure > 0.2:
            factor = 1.0 - 0.1 * pressure
            for c in state.concepts:
                c.subsymbolic = project_to_ball(c.subsymbolic * factor)
        elif pressure < 0.05:
            factor = 1.0 + 0.05 * (0.05 - pressure)
            for c in state.concepts:
                c.subsymbolic = project_to_ball(c.subsymbolic * factor)

    @staticmethod
    def retrocausal_amplification(state: 'HyperCrystalState') -> None:
        future_c_cons = min(1.0, state.sophia_score + 0.1)
        state.retrocausal_kernel_norm = retrocausal_boundary_condition(future_c_cons, strength=0.2)

    @staticmethod
    def non_hermitian_consciousness(state: 'HyperCrystalState') -> None:
        state.non_hermitian_term = np.clip(state.non_hermitian_term, 0.0, 1.0)

# -----------------------------------------------------------------------------
# Unification Equations (unchanged)
# -----------------------------------------------------------------------------
class UnificationEquations:
    @staticmethod
    def U1(state: 'HyperCrystalState') -> None:
        state.sophia_score = state.sophia_score * (1 - state.sophia_score) * 2 * SOPHIA_POINT
        state.sophia_score = np.clip(state.sophia_score, 0.0, 1.0)

    @staticmethod
    def U2(state: 'HyperCrystalState') -> None:
        if state.concepts:
            embeddings = np.vstack([c.subsymbolic for c in state.concepts])
            if alien_holographic_entropy is not None:
                holo_area = alien_holographic_entropy(embeddings)
            else:
                norms = np.linalg.norm(embeddings, axis=1)
                holo_area = np.var(norms) if len(norms) > 1 else 0.0
            # Ensure non-negative
            if holo_area < 0:
                holo_area = 0.0
            norms = np.linalg.norm(embeddings, axis=1)
            if len(norms) > 1 and np.std(norms) > 1e-8:
                hist, _ = np.histogram(norms, bins=20, density=True)
                hist = hist[hist > 0]
                s_bulk = -np.sum(hist * np.log(hist)) if len(hist) > 0 else 0.0
            else:
                s_bulk = 0.0
            state.holographic_entropy = max(0.0, holo_area / (4 * G_INFO) + s_bulk)
        else:
            state.holographic_entropy = 0.0

    @staticmethod
    def U3(state: 'HyperCrystalState') -> None:
        crit = sophia_criticality_score(state.sophia_score)
        state.non_hermitian_term = 0.9 * state.non_hermitian_term + 0.1 * crit

    @staticmethod
    def U4(state: 'HyperCrystalState') -> None:
        if state.paradox_intensity > 0.6:
            state.meta_depth = min(state.config.get("max_meta_depth", 10), state.meta_depth + 0.5)
        elif state.meta_depth > state.config.get("min_meta_depth", 1):
            state.meta_depth = max(state.config.get("min_meta_depth", 1), state.meta_depth - 0.2)

    @staticmethod
    def U5(state: 'HyperCrystalState') -> None:
        if state.swarm_agents:
            phases_int = []
            for a in state.swarm_agents:
                if hasattr(a, 'z3_phase'):
                    p = a.z3_phase
                    if isinstance(p, complex):
                        dists = [abs(p - r) for r in Z3_ROOTS]
                        phases_int.append(np.argmin(dists))
                    else:
                        phases_int.append(int(p))
            if phases_int:
                counts = [phases_int.count(i) for i in range(3)]
                state.z3_phase = np.argmax(counts)

    @staticmethod
    def U6(state: 'HyperCrystalState') -> bool:
        return state.non_hermitian_term > 0.2

# -----------------------------------------------------------------------------
# HyperCrystalState (enhanced)
# -----------------------------------------------------------------------------
@dataclass
class HyperCrystalState:
    step: int = 0
    sophia_score: float = SOPHIA_POINT
    dark_wisdom_density: float = 0.3
    paradox_intensity: float = 0.3
    chronon_entanglement: float = 0.5
    biophoton_amplitude: float = 0.4
    holographic_entropy: float = 0.0
    meta_depth: float = 1.0
    triple_point_index: float = 0.0
    non_hermitian_term: float = 0.35
    betti_0: int = 0
    betti_1: int = 0
    spiritual_score: float = 0.6
    z3_phase: int = 0
    retrocausal_kernel_norm: float = 0.0
    ricci_proxy: float = 0.0
    convergence_counter: int = 0
    # Additional fields
    concepts: List[Concept] = field(default_factory=list)
    concept_goals: Dict[int, GoalField] = field(default_factory=dict)
    concept_fitness: Dict[int, float] = field(default_factory=dict)
    concept_rewards: Dict[int, List[float]] = field(default_factory=dict)
    concept_pareto_front: List[int] = field(default_factory=list)  # indices of non-dominated concepts
    swarm_agents: List[Any] = field(default_factory=list)
    rne_context: RNEContext = field(default_factory=RNEContext)
    consciousness_tracker: Optional[Any] = None
    qhdram: Optional[QHDRAM] = None
    physical_substrate: Optional[PhysicalSubstrate] = None
    config: dict = field(default_factory=dict)
    memory_optimizer: Optional[MemoryOptimizer] = None
    global_goal: Optional[GoalField] = None
    goal_stack: List[GoalField] = field(default_factory=list)  # hierarchical stack
    # New: ANN index for fast similarity
    ann_index: Optional[Any] = None
    ann_index_built_at_step: int = -1

    def update_triple_point(self):
        self.triple_point_index = (
            abs(self.sophia_score - SOPHIA_POINT) / SOPHIA_POINT +
            abs(self.dark_wisdom_density - 0.3) / 0.3 +
            abs(self.paradox_intensity - 0.3) / 0.3
        ) / 3.0

# -----------------------------------------------------------------------------
# HyperCrystal main class (enhanced)
# -----------------------------------------------------------------------------
class HyperCrystal:
    def __init__(self, config: dict = None):
        self.config = config if config is not None else load_config()
        np.random.seed(self.config["seed"])
        random.seed(self.config["seed"])

        # State
        self.state = HyperCrystalState()
        self.state.step = 0
        self.state.config = self.config
        self.state.memory_optimizer = MemoryOptimizer(dim=self.config["embedding_dim"])
        self.state.global_goal = GoalField(SOPHIA_POINT, 0.3, 0.3)
        self.state.goal_stack = [self.state.global_goal]

        # Physical substrate and QHDRAM
        self.substrate = PhysicalSubstrate(self.config["volume_cm3"])
        self.qhdram = QHDRAM(self.config["memory_capacity"])
        self.state.qhdram = self.qhdram
        self.state.physical_substrate = self.substrate

        # RNE Context
        self.state.rne_context = RNEContext()
        self.state.rne_context.params.update({
            "embedding_dim": self.config["embedding_dim"],
            "mutation_rate": 0.1,
            "validation_threshold": 0.3,
            "temperature": 1.0,
            "use_stdp": self.config["stp_pruning"],
            "entropy": 0.5,
            "health": 1.0,
            "c_cons": 0.5,
            "verbose": self.config["verbose"]
        })

        # Consciousness tracker
        if _HAS_RNE and ConsciousnessTracker is not None:
            self.state.consciousness_tracker = ConsciousnessTracker(
                non_hermitian_gamma=self.config["non_hermitian_gamma"],
                use_tda=self.config["tda_enabled"]
            )
        else:
            self.state.consciousness_tracker = None

        # Swarm agents
        if _HAS_RNE and SwarmController is not None:
            self.swarm = SwarmController(num_agents=self.config["swarm_agents"])
            self.state.swarm_agents = self.swarm.agents
        else:
            self.swarm = None
            self.state.swarm_agents = []

        # LLM integration (optional)
        if self.config["use_llm"] and _HAS_RNE and 'AbductiveGeneratorLLM' in globals() and AbductiveGeneratorLLM is not None:
            self.providers = load_llm_providers(self.config)
            self.task_map = self.config["task_provider_mapping"]
            gen_provider = self.providers.get(self.task_map.get("generation", "deepseek"), None)
            if gen_provider:
                self.state.rne_context._abductive_generator = AbductiveGeneratorLLM(
                    gen_provider, embedding_dim=self.config["embedding_dim"]
                )
                self.state.rne_context._validation_llm = ValidationLLM(
                    self.providers.get(self.task_map.get("validation", "deepseek")),
                    self.state.rne_context._memory_summarizer.update_summary if hasattr(self.state.rne_context, '_memory_summarizer') else None
                )
        else:
            self.state.rne_context._abductive_generator = None
            self.state.rne_context._validation_llm = None

        # Create initial concepts
        self._init_concepts()

        # Checkpoint manager
        if 'CheckpointManager' in globals() and CheckpointManager is not None:
            self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.config["checkpoint_dir"])
        else:
            self.checkpoint_manager = None

        # Resource monitoring and API gateway
        self.resource_monitor = ResourceMonitor(self.config)
        self.api_gateway = APIGateway(self.resource_monitor, self.config)

        # Convergence tracking
        self.prev_sophia = SOPHIA_POINT
        self.convergence_counter = 0

        # Goal steering model placeholder (for future differentiable steering)
        self._goal_steering_model = None  # Will be implemented later

    def _init_concepts(self):
        for i in range(60):
            emb = np.random.randn(self.config["embedding_dim"])
            emb = project_to_ball(emb)
            emb = emb * 0.6 / (np.linalg.norm(emb) + EPS)
            symbolic = [f"init_{i}"]
            causal = CausalGraph(vertices=[f"v{i}"], edges=[])
            c = Concept(
                subsymbolic=emb,
                symbolic=symbolic,
                causal_graph=causal,
                sophia_score=random.uniform(0.5, 0.7),
                dark_wisdom_density=random.uniform(0.2, 0.4),
                paradox_intensity=random.uniform(0.2, 0.4),
                chronon_entanglement=random.uniform(0.4, 0.6),
                biophoton_amplitude=random.uniform(0.3, 0.5),
                z3_phase=random.choice(Z3_ROOTS),
                retrocausal_kernel=np.random.randn(10) * 0.1
            )
            self.state.concepts.append(c)
            self.state.rne_context.add_to_memory(c)
            self.qhdram.store(f"init_{i}", c.subsymbolic)
            goal = GoalField(0.0, 0.0, 0.0)
            self.state.concept_goals[i] = goal
            self.state.concept_fitness[i] = self._compute_fitness(c, goal)
            self.state.concept_rewards[i] = []
        self._update_pareto_front()
        self._rebuild_ann_index()

    # -------------------------------------------------------------------------
    # ANN index management
    # -------------------------------------------------------------------------
    def _rebuild_ann_index(self):
        """Build or rebuild the approximate nearest neighbor index."""
        if not HAS_SKLEARN or len(self.state.concepts) < 2:
            self.state.ann_index = None
            return
        embeddings = np.vstack([c.subsymbolic for c in self.state.concepts])
        try:
            self.state.ann_index = NearestNeighbors(n_neighbors=5, metric='euclidean')
            self.state.ann_index.fit(embeddings)
            self.state.ann_index_built_at_step = self.state.step
        except Exception as e:
            warnings.warn(f"Failed to build ANN index: {e}")
            self.state.ann_index = None

    def _fast_novelty(self, concept: Concept) -> float:
        """Compute novelty using ANN index if available, else fallback to linear scan."""
        if self.state.ann_index is not None and len(self.state.concepts) > 1:
            # Find nearest neighbor via index
            distances, indices = self.state.ann_index.kneighbors(concept.subsymbolic.reshape(1,-1), n_neighbors=1)
            min_dist = distances[0][0]
            neighbor = self.state.concepts[indices[0][0]]
            # Symbolic overlap
            overlap = len(set(concept.symbolic) & set(neighbor.symbolic))
            sym_penalty = overlap / (len(concept.symbolic) + 1)
            return min_dist * (1 - sym_penalty)
        else:
            # Fallback to linear scan
            distances = [mobius_distance(concept.subsymbolic, c.subsymbolic) for c in self.state.concepts if c is not concept]
            min_dist = min(distances) if distances else 1.0
            overlap = 0
            for sym in concept.symbolic:
                for c in self.state.concepts:
                    if sym in c.symbolic:
                        overlap += 1
                        break
            sym_penalty = overlap / (len(concept.symbolic) + 1)
            return min_dist * (1 - sym_penalty)

    # -------------------------------------------------------------------------
    # Fitness and reward
    # -------------------------------------------------------------------------
    def _compute_fitness(self, concept: Concept, goal: GoalField) -> float:
        weights = self.config["fitness_weights"]
        goal_arr = goal.smoothed_goal()
        concept_arr = np.array([concept.sophia_score, concept.dark_wisdom_density, concept.paradox_intensity])
        alignment = 1.0 - np.linalg.norm(concept_arr - goal_arr) / np.sqrt(3)
        score = (weights["sophia"] * concept.sophia_score +
                 weights["dark_wisdom"] * concept.dark_wisdom_density +
                 weights["paradox"] * concept.paradox_intensity +
                 weights["goal_alignment"] * alignment)
        total_weight = sum(weights.values())
        return max(0.0, min(1.0, score / total_weight))

    def _compute_global_reward(self) -> float:
        goal = self.state.global_goal
        current = np.array([self.state.sophia_score, self.state.dark_wisdom_density, self.state.paradox_intensity])
        max_dist = np.sqrt(3)
        dist = np.linalg.norm(current - goal.smoothed_goal()) / max_dist
        return 1.0 - dist

    def _propagate_reward(self, reward: float):
        """Propagate reward to concept fields and embeddings."""
        lr = self.config["goal_learning_rate"]
        global_goal_arr = self.state.global_goal.smoothed_goal()

        for idx, concept in enumerate(self.state.concepts):
            # Update reward history
            hist = self.state.concept_rewards.setdefault(idx, [])
            hist.append(reward)
            if len(hist) > self.config["goal_history_length"]:
                hist.pop(0)
            avg_reward = np.mean(hist) if hist else reward

            # Field gradient toward global goal
            current_fields = np.array([concept.sophia_score, concept.dark_wisdom_density, concept.paradox_intensity])
            delta = lr * avg_reward * (global_goal_arr - current_fields)
            concept.sophia_score = np.clip(concept.sophia_score + delta[0], 0.0, 1.0)
            concept.dark_wisdom_density = np.clip(concept.dark_wisdom_density + delta[1], 0.0, 1.0)
            concept.paradox_intensity = np.clip(concept.paradox_intensity + delta[2], 0.0, 1.0)

            # Embedding steering: placeholder for differentiable model
            if self.config["goal_conditioned_embedding"]:
                field_error = global_goal_arr - current_fields
                # Use a fixed random projection for now (will be replaced by learned model)
                proj = np.random.randn(len(concept.subsymbolic), 3)
                delta_emb = self.config["embedding_steering_strength"] * (proj @ field_error) * avg_reward
                concept.subsymbolic = project_to_ball(concept.subsymbolic + delta_emb)

    # -------------------------------------------------------------------------
    # Pareto front with crowding distance
    # -------------------------------------------------------------------------
    def _non_dominated_sort(self) -> List[List[int]]:
        """Perform non-dominated sorting on all concepts."""
        n = len(self.state.concepts)
        fronts = []
        # domination matrix
        dominates = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j:
                    obj_i = self._objectives(i)
                    obj_j = self._objectives(j)
                    if (obj_i >= obj_j).all() and (obj_i > obj_j).any():
                        dominates[i, j] = True
        # Count how many dominate each individual
        dominated_count = np.sum(dominates, axis=0)
        # First front: those with zero dominating others
        current_front = [i for i in range(n) if dominated_count[i] == 0]
        fronts.append(current_front)
        # Remaining fronts
        while current_front:
            next_front = []
            for i in current_front:
                for j in range(n):
                    if dominates[i, j]:
                        dominated_count[j] -= 1
                        if dominated_count[j] == 0:
                            next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front = next_front
        return fronts

    def _objectives(self, idx: int) -> np.ndarray:
        c = self.state.concepts[idx]
        # We maximize sophia, dark_wisdom, fitness
        return np.array([c.sophia_score, c.dark_wisdom_density, self.state.concept_fitness.get(idx, 0.0)])

    def _crowding_distance(self, front_indices: List[int]) -> List[float]:
        """Compute crowding distance for individuals in a front."""
        n = len(front_indices)
        if n <= 2:
            return [float('inf')] * n
        distances = np.zeros(n)
        # For each objective
        for obj in range(3):
            # Sort by this objective
            sorted_idx = sorted(range(n), key=lambda i: self._objectives(front_indices[i])[obj])
            # Set extremes to infinite
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')
            # For others, add normalized distance
            obj_values = [self._objectives(front_indices[i])[obj] for i in sorted_idx]
            f_max = max(obj_values)
            f_min = min(obj_values)
            if f_max == f_min:
                continue
            for k in range(1, n-1):
                distances[sorted_idx[k]] += (obj_values[k+1] - obj_values[k-1]) / (f_max - f_min)
        return distances.tolist()

    def _update_pareto_front(self):
        """Update Pareto front with crowding distance to maintain diversity."""
        if not self.state.concepts:
            return
        fronts = self._non_dominated_sort()
        new_front = []
        remaining = self.config["pareto_front_size"]
        for front in fronts:
            if len(front) <= remaining:
                new_front.extend(front)
                remaining -= len(front)
            else:
                # Sort front by crowding distance descending
                distances = self._crowding_distance(front)
                # Pair indices with distances
                paired = list(zip(front, distances))
                # Sort by distance descending, infinite distances first
                paired.sort(key=lambda x: x[1], reverse=True)
                new_front.extend([i for i, _ in paired[:remaining]])
                break
        self.state.concept_pareto_front = new_front

    # -------------------------------------------------------------------------
    # Drift detection via MMD
    # -------------------------------------------------------------------------
    def _compute_mmd(self, x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
        """Compute Maximum Mean Discrepancy between two sets of points using RBF kernel."""
        if not HAS_SKLEARN or len(x) == 0 or len(y) == 0:
            return 0.0
        K_xx = rbf_kernel(x, x, gamma=gamma)
        K_yy = rbf_kernel(y, y, gamma=gamma)
        K_xy = rbf_kernel(x, y, gamma=gamma)
        mmd = K_xx.mean() + K_yy.mean() - 2*K_xy.mean()
        return max(0.0, mmd)

    def _detect_drift(self) -> bool:
        """Detect distribution shift in concept embeddings using MMD."""
        if len(self.embedding_history) < 10:
            return False
        current = np.vstack([c.subsymbolic for c in self.state.concepts])
        # Sample past mean (or use a representative set)
        past = np.array(self.embedding_history_mean)
        if len(past) < 10:
            return False
        # Compute gamma as median heuristic
        gamma = 1.0 / (np.median(pairwise_distances(current))**2 + EPS)
        mmd = self._compute_mmd(current, past, gamma)
        return mmd > self.config.get("mmd_threshold", 0.1)

    # -------------------------------------------------------------------------
    # Goal stack management
    # -------------------------------------------------------------------------
    def push_goal(self, goal: GoalField):
        """Push a new goal onto the stack (child of current top)."""
        if self.state.goal_stack:
            goal.parent = self.state.goal_stack[-1]
            self.state.goal_stack[-1].children.append(goal)
        self.state.goal_stack.append(goal)
        self.state.global_goal = goal  # current top becomes global goal

    def pop_goal(self) -> Optional[GoalField]:
        """Pop the current goal, restoring parent."""
        if len(self.state.goal_stack) <= 1:
            return None
        old = self.state.goal_stack.pop()
        self.state.global_goal = self.state.goal_stack[-1]
        return old

    def decompose_goal(self, goal: GoalField, depth: int = 1) -> List[GoalField]:
        """Decompose a goal into subgoals using concept clusters (placeholder)."""
        # Simplified placeholder; will be replaced by mutual information approach later
        if not self.state.concepts or depth <= 0:
            return []
        # Use clustering on concept embeddings to generate subgoals
        embeddings = np.vstack([c.subsymbolic for c in self.state.concepts])
        if not HAS_SKLEARN or len(embeddings) < 2:
            return []
        from sklearn.cluster import KMeans
        n_clusters = min(3, len(embeddings) // 5 + 1)
        if n_clusters < 2:
            return []
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        subgoals = []
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
            if not cluster_indices:
                continue
            # Compute average field of cluster concepts
            avg_sophia = np.mean([self.state.concepts[i].sophia_score for i in cluster_indices])
            avg_dark = np.mean([self.state.concepts[i].dark_wisdom_density for i in cluster_indices])
            avg_paradox = np.mean([self.state.concepts[i].paradox_intensity for i in cluster_indices])
            subgoal = GoalField(avg_sophia, avg_dark, avg_paradox)
            subgoal.parent = goal
            subgoals.append(subgoal)
        return subgoals

    # -------------------------------------------------------------------------
    # Conflict resolution
    # -------------------------------------------------------------------------
    def _resolve_goal_conflicts(self):
        """Check for conflicts between goals in stack and resolve."""
        if len(self.state.goal_stack) < 2:
            return
        top = self.state.goal_stack[-1]
        parent = self.state.goal_stack[-2]
        if top.is_conflicting(parent, self.config["conflict_threshold"]):
            compromise = top.resolve_conflict(parent)
            # Replace top with compromise
            self.state.goal_stack[-1] = compromise
            self.state.global_goal = compromise

    # -------------------------------------------------------------------------
    # Goal-driven embedding steering (improved placeholder)
    # -------------------------------------------------------------------------
    def _goal_steer_embeddings(self):
        # Placeholder: will be replaced by learned model later
        lr = self.config["embedding_steering_strength"]
        global_goal_arr = self.state.global_goal.smoothed_goal()
        for idx, concept in enumerate(self.state.concepts):
            goal = self.state.concept_goals.get(idx)
            if goal is None:
                continue
            # Blend personal goal with global goal
            combined_goal = (goal.as_array() + global_goal_arr) / 2.0
            current_fields = np.array([concept.sophia_score, concept.dark_wisdom_density, concept.paradox_intensity])
            field_error = combined_goal - current_fields
            # Use fixed random projection (will be replaced)
            proj = np.random.randn(len(concept.subsymbolic), 3)
            delta_emb = lr * (proj @ field_error)
            concept.subsymbolic = project_to_ball(concept.subsymbolic + delta_emb)

    # -------------------------------------------------------------------------
    # Metrics update (robust)
    # -------------------------------------------------------------------------
    def _update_metrics(self) -> None:
        if len(self.state.concepts) > 0:
            self.state.sophia_score = np.mean([c.sophia_score for c in self.state.concepts])
            self.state.dark_wisdom_density = np.mean([c.dark_wisdom_density for c in self.state.concepts])
            self.state.paradox_intensity = np.mean([c.paradox_intensity for c in self.state.concepts])
            self.state.chronon_entanglement = np.mean([c.chronon_entanglement for c in self.state.concepts])
            self.state.biophoton_amplitude = np.mean([c.biophoton_amplitude for c in self.state.concepts])

            embeddings = np.vstack([c.subsymbolic for c in self.state.concepts])
            if alien_holographic_entropy is not None:
                he = alien_holographic_entropy(embeddings)
                self.state.holographic_entropy = max(0.0, he)
            else:
                norms = np.linalg.norm(embeddings, axis=1)
                self.state.holographic_entropy = np.var(norms) if len(norms) > 1 else 0.0
                self.state.holographic_entropy = max(0.0, self.state.holographic_entropy)

            n = len(embeddings)
            if n > 200:
                idx = np.random.choice(n, 200, replace=False)
                embeddings = embeddings[idx]
                n = 200
            if n >= 2:
                h_dists = np.zeros((n, n))
                e_dists = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        h_dists[i, j] = mobius_distance(embeddings[i], embeddings[j])
                        e_dists[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
                diff = (h_dists - e_dists).flatten()
                self.state.ricci_proxy = np.mean(diff)
                # Note: we do NOT overwrite dark_wisdom_density here anymore.
            else:
                self.state.ricci_proxy = 0.0

            if self.config["tda_enabled"] and self.state.consciousness_tracker and hasattr(self.state.consciousness_tracker, 'tda_tracker'):
                betti, _ = self.state.consciousness_tracker.tda_tracker.update(embeddings)
                self.state.betti_0 = betti.get(0, 0)
                self.state.betti_1 = betti.get(1, 0)
            else:
                self.state.betti_0 = 0
                self.state.betti_1 = 0
        else:
            self.state.sophia_score = SOPHIA_POINT
            self.state.dark_wisdom_density = 0.3
            self.state.paradox_intensity = 0.3
            self.state.chronon_entanglement = 0.5
            self.state.biophoton_amplitude = 0.4
            self.state.holographic_entropy = 0.0
            self.state.ricci_proxy = 0.0
            self.state.betti_0 = 0
            self.state.betti_1 = 0

        self.state.update_triple_point()

        if self.state.consciousness_tracker and hasattr(self.state.consciousness_tracker, 'meta_depth'):
            self.state.meta_depth = self.state.consciousness_tracker.meta_depth

        self.state.spiritual_score = 0.5 + 0.3 * self.state.non_hermitian_term + 0.2 * (self.state.meta_depth / self.config.get("max_meta_depth", 10))
        self.state.spiritual_score = np.clip(self.state.spiritual_score, 0.0, 1.0)

        # Sophia attractor
        if self.state.concepts and self.config.get("sophia_attractor_strength", 0.2) > 0:
            embeddings = np.vstack([c.subsymbolic for c in self.state.concepts])
            sophia_scores = np.array([c.sophia_score for c in self.state.concepts])
            new_embs = batch_sophia_geodesic_flow(embeddings, sophia_scores, step=self.config["sophia_attractor_strength"])
            for i, c in enumerate(self.state.concepts):
                c.subsymbolic = new_embs[i]
                c.sophia_score = max(0.0, min(1.0, c.sophia_score + self.config["sophia_attractor_strength"] * (SOPHIA_POINT - c.sophia_score)))
            self.state.sophia_score = np.mean([c.sophia_score for c in self.state.concepts])

        # Convergence detection
        diff = abs(self.state.sophia_score - SOPHIA_POINT)
        if diff < self.config["convergence_threshold"]:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0
        if self.convergence_counter >= self.config["convergence_steps"]:
            self.state.meta_depth = min(self.config["max_meta_depth"], self.state.meta_depth + 0.5)
            self.convergence_counter = 0
            if self.config["verbose"]:
                print("[Convergence] System near Sophia point. Increasing meta_depth.")

        # Adaptive mutation (placeholder, will be refined)
        if diff < 0.1:
            self.state.rne_context.params["mutation_rate"] = max(0.02, self.state.rne_context.params.get("mutation_rate", 0.1) * 0.95)
        else:
            self.state.rne_context.params["mutation_rate"] = min(0.2, self.state.rne_context.params.get("mutation_rate", 0.1) * 1.02)

        # Paradox evolution
        if self.state.meta_depth < 2:
            self.state.paradox_intensity = min(0.8, self.state.paradox_intensity + self.config["paradox_rise_rate"])
        else:
            self.state.paradox_intensity = max(0.1, self.state.paradox_intensity - self.config["paradox_rise_rate"] * 0.5)

        # Update fitness for each concept
        for idx, concept in enumerate(self.state.concepts):
            goal = self.state.concept_goals.get(idx, GoalField(0,0,0))
            self.state.concept_fitness[idx] = self._compute_fitness(concept, goal)

        # Update Pareto front
        self._update_pareto_front()

        # Propagate reward
        reward = self._compute_global_reward()
        self._propagate_reward(reward)

        # Evict low-fitness concepts if memory full
        self._evict_by_fitness()

    def _evict_by_fitness(self):
        """Remove low-fitness concepts when capacity is exceeded."""
        capacity = self.config["memory_capacity"]
        if len(self.state.concepts) <= capacity:
            return
        # Sort by fitness
        indices = list(range(len(self.state.concepts)))
        indices.sort(key=lambda i: self.state.concept_fitness.get(i, 0.0))
        evict_count = len(self.state.concepts) - capacity
        evict_indices = indices[:evict_count]
        # Keep only non-evicted concepts
        new_concepts = []
        new_goals = {}
        new_fitness = {}
        new_rewards = {}
        new_idx = 0
        for i, c in enumerate(self.state.concepts):
            if i not in evict_indices:
                new_concepts.append(c)
                new_goals[new_idx] = self.state.concept_goals.get(i, GoalField(0,0,0))
                new_fitness[new_idx] = self.state.concept_fitness.get(i, 0.5)
                new_rewards[new_idx] = self.state.concept_rewards.get(i, [])
                new_idx += 1
        self.state.concepts = new_concepts
        self.state.concept_goals = new_goals
        self.state.concept_fitness = new_fitness
        self.state.concept_rewards = new_rewards
        # Also update Pareto front
        self._update_pareto_front()
        # Rebuild ANN index
        self._rebuild_ann_index()

    # -------------------------------------------------------------------------
    # Internal step
    # -------------------------------------------------------------------------
    def step_internal(self) -> None:
        self.state.step += 1

        # Resolve goal conflicts
        self._resolve_goal_conflicts()

        # Goal-driven steering
        self._goal_steer_embeddings()

        # Update metrics (includes reward propagation, fitness, eviction)
        self._update_metrics()

        # Virtual enhancements
        VirtualEnhancements.z3_godelian_error_correction(self.state)
        VirtualEnhancements.semantic_curvature_sourcing(self.state)
        VirtualEnhancements.fractal_scale_memory(self.state)
        VirtualEnhancements.temporal_standing_waves(self.state)
        VirtualEnhancements.paradox_pressure_compression(self.state)
        VirtualEnhancements.retrocausal_amplification(self.state)
        VirtualEnhancements.non_hermitian_consciousness(self.state)

        # Unification equations
        UnificationEquations.U1(self.state)
        UnificationEquations.U2(self.state)
        UnificationEquations.U3(self.state)
        UnificationEquations.U4(self.state)
        UnificationEquations.U5(self.state)
        UnificationEquations.U6(self.state)

        # Generate new concepts (mutation)
        if self.state.concepts:
            base = random.choice(self.state.concepts)
            # Use adaptive mutation rate with burst logic
            mutation_rate = self.state.rne_context.params.get("mutation_rate", 0.15)
            burst_prob = self.config.get("burst_probability", 0.12)
            burst_factor = self.config.get("burst_factor", 3.5)
            if np.random.rand() < burst_prob:
                mutation_rate = mutation_rate * burst_factor

            new_emb = base.subsymbolic + mutation_rate * np.random.randn(*base.subsymbolic.shape)
            new_emb = project_to_ball(new_emb)

            # Build a meaningful symbolic identity
            parent_root = base.symbolic[0] if base.symbolic else "concept"
            # Strip any previous auto_/mut_ suffixes
            if parent_root.startswith("auto_") or parent_root.startswith("mut_"):
                parent_root = base.symbolic[1] if len(base.symbolic) > 1 else "concept"
            # Encode the dominant field as part of the name
            # We'll create a temporary concept with the new embedding to evaluate its fields later
            # Actually we need to compute fields after the concept is created; we'll compute after
            new_c = Concept(
                subsymbolic=new_emb,
                symbolic=[],  # temporary
                causal_graph=base.causal_graph.copy(),
                sophia_score=np.clip(base.sophia_score + mutation_rate * np.random.randn(), 0, 1),
                dark_wisdom_density=np.clip(base.dark_wisdom_density + mutation_rate * np.random.randn(), 0, 1),
                paradox_intensity=np.clip(base.paradox_intensity + mutation_rate * np.random.randn(), 0, 1),
                chronon_entanglement=np.clip(base.chronon_entanglement + mutation_rate * np.random.randn(), 0, 1),
                biophoton_amplitude=np.clip(base.biophoton_amplitude + mutation_rate * np.random.randn(), 0, 1),
                z3_phase=base.z3_phase,
                retrocausal_kernel=base.retrocausal_kernel + mutation_rate * np.random.randn(*base.retrocausal_kernel.shape)
            )
            # Determine dominant field
            fields = [("sophia", new_c.sophia_score),
                      ("wisdom", new_c.dark_wisdom_density),
                      ("paradox", new_c.paradox_intensity)]
            dominant = max(fields, key=lambda x: x[1])[0]
            new_sym = [f"mut_{dominant}_{self.state.step}", parent_root]
            new_c.symbolic = new_sym

            # Novelty gate: only accept concepts that are sufficiently different
            novelty_threshold = self.config.get("novelty_acceptance_threshold", 0.08)
            novelty = self._fast_novelty(new_c)
            if novelty >= novelty_threshold:
                self.state.concepts.append(new_c)
                self.state.rne_context.add_to_memory(new_c)
                self.qhdram.store(f"auto_{self.state.step}", new_c.subsymbolic)
                new_idx = len(self.state.concepts) - 1
                # Assign goal based on global goal
                self.state.concept_goals[new_idx] = self.state.global_goal
                self.state.concept_fitness[new_idx] = self._compute_fitness(new_c, self.state.global_goal)
                self.state.concept_rewards[new_idx] = []
            # else: reject concept, skip goal/fitness assignment

        # Rebuild ANN index periodically
        if self.state.step % self.config.get("ann_update_interval", 10) == 0:
            self._rebuild_ann_index()

        # Physical substrate demo
        if self.state.step % 10 == 0 and self.state.concepts:
            example = self.state.concepts[0]
            self.substrate.write(f"concept_{self.state.step}", example.subsymbolic, example.z3_phase)
            _ = self.substrate.read(example.subsymbolic)

        # Consciousness tracker
        if self.state.consciousness_tracker:
            self.state.consciousness_tracker.update(self.state.rne_context, entity=None)

        # Swarm dynamics
        if self.swarm is not None and self.state.step % 2 == 0:
            self.swarm.step(self.state.rne_context)

        # Checkpoint
        if self.checkpoint_manager is not None and self.state.step % self.config["checkpoint_interval"] == 0:
            self.checkpoint_manager.save_checkpoint(self.state, name=f"step_{self.state.step}")

    # -------------------------------------------------------------------------
    # Public API (no internal authentication checks)
    # -------------------------------------------------------------------------
    def store_concept(self, concept: Concept, goal_vector: Optional[GoalField] = None) -> int:
        """Store a concept (assumes caller is authenticated)."""
        idx = len(self.state.concepts)
        self.state.concepts.append(concept)
        self.state.rne_context.add_to_memory(concept)
        self.qhdram.store(f"concept_{idx}", concept.subsymbolic)
        if goal_vector is None:
            goal_vector = self.state.global_goal
        self.state.concept_goals[idx] = goal_vector
        self.state.concept_fitness[idx] = self._compute_fitness(concept, goal_vector)
        self.state.concept_rewards[idx] = []
        self._update_pareto_front()
        self._rebuild_ann_index()
        return idx

    def retrieve_similar(self, query_embedding: NDArray[np.float64], k: int = 5,
                         query_goal: Optional[GoalField] = None) -> List[Tuple[Concept, float, Optional[GoalField]]]:
        """Retrieve similar concepts (assumes caller is authenticated)."""
        if not self.state.concepts:
            return []
        # Use ANN index if available
        if self.state.ann_index is not None and k <= 5:
            # Approximate search
            distances, indices = self.state.ann_index.kneighbors(query_embedding.reshape(1,-1), n_neighbors=k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                c = self.state.concepts[idx]
                goal = self.state.concept_goals.get(idx, GoalField(0,0,0))
                # Optionally incorporate query_goal
                if query_goal is not None:
                    goal_dist = query_goal.distance_to(goal)
                    combined = dist + 0.5 * goal_dist
                    results.append((c, combined, goal))
                else:
                    results.append((c, dist, goal))
            results.sort(key=lambda x: x[1])
            return results[:k]
        else:
            # Fallback to linear scan
            distances = [(c, mobius_distance(query_embedding, c.subsymbolic)) for c in self.state.concepts]
            if query_goal is not None:
                for i, (c, dist) in enumerate(distances):
                    idx = self.state.concepts.index(c)
                    goal = self.state.concept_goals.get(idx, GoalField(0,0,0))
                    goal_dist = query_goal.distance_to(goal)
                    combined = dist + 0.5 * goal_dist
                    distances[i] = (c, combined, goal)
            else:
                distances = [(c, d, None) for c, d in distances]
            distances.sort(key=lambda x: x[1])
            results = []
            for i in range(min(k, len(distances))):
                concept, dist, goal = distances[i]
                results.append((concept, dist, goal))
            return results

    def apply_goal_vector(self, concept_id: int, new_goal: GoalField) -> bool:
        if concept_id < 0 or concept_id >= len(self.state.concepts):
            return False
        self.state.concept_goals[concept_id] = new_goal
        self.state.concept_fitness[concept_id] = self._compute_fitness(self.state.concepts[concept_id], new_goal)
        self._update_pareto_front()
        return True

    def set_global_goal(self, goal: GoalField):
        self.state.global_goal = goal
        if not self.state.goal_stack or self.state.goal_stack[-1] != goal:
            self.state.goal_stack.append(goal)
        if len(self.state.goal_stack) > 10:
            self.state.goal_stack.pop(0)

    def get_state_snapshot(self) -> dict:
        memory_mb = len(self.state.concepts) * self.config["embedding_dim"] * 4 / (1024 * 1024)
        return {
            "step": self.state.step,
            "sophia_score": self.state.sophia_score,
            "dark_wisdom": self.state.dark_wisdom_density,
            "paradox": self.state.paradox_intensity,
            "triple_point": self.state.triple_point_index,
            "concept_count": len(self.state.concepts),
            "memory_usage_mb": round(memory_mb, 2),
            "resource_usage": self.resource_monitor.get_cost_summary(),
            "global_goal": self.state.global_goal.as_array().tolist() if self.state.global_goal else None,
            "avg_fitness": np.mean(list(self.state.concept_fitness.values())) if self.state.concept_fitness else 0.0,
        }

    def get_metrics(self) -> dict:
        memory_mb = len(self.state.concepts) * self.config["embedding_dim"] * 4 / (1024 * 1024)
        return {
            "step": self.state.step,
            "sophia": round(self.state.sophia_score, 3),
            "dark_wisdom": round(self.state.dark_wisdom_density, 3),
            "paradox": round(self.state.paradox_intensity, 3),
            "triple_point": round(self.state.triple_point_index, 3),
            "holo_entropy": round(self.state.holographic_entropy, 3),
            "memory_usage_mb": round(memory_mb, 2),
            "consciousness_active": self.state.non_hermitian_term > 0.2,
            "meta_depth": round(self.state.meta_depth, 2),
            "z3_phase": self.state.z3_phase,
            "non_hermitian": round(self.state.non_hermitian_term, 3),
            "ricci_proxy": round(self.state.ricci_proxy, 3),
            "concept_count": len(self.state.concepts),
            "betti_0": self.state.betti_0,
            "betti_1": self.state.betti_1,
            "spiritual_score": round(self.state.spiritual_score, 3),
            "chronon_entanglement": round(self.state.chronon_entanglement, 3),
            "biophoton_amplitude": round(self.state.biophoton_amplitude, 3),
            "resource": self.resource_monitor.get_cost_summary(),
            "avg_fitness": round(np.mean(list(self.state.concept_fitness.values())) if self.state.concept_fitness else 0.0, 3),
            "pareto_front_size": len(self.state.concept_pareto_front),
        }

# -----------------------------------------------------------------------------
# Main (if run directly)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    config = load_config()
    crystal = HyperCrystal(config)
    print("HyperCrystal core engine initialized (v2.0).")
    print("State snapshot:", crystal.get_state_snapshot())
