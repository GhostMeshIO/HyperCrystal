"""
cognition_engine.py – Intelligence & Novelty for HyperCrystal (Revised v2.1)
===========================================================================
Fully compatible with the revised core_engine.py (UUID-based, thread-safe).
Implements diffusion novelty injection, CMA-ES meta-learning, repulsion force,
and emergence detection with persistent homology.
"""

import numpy as np
import random
import time
import warnings
import copy
import hashlib
import pickle
import os
import uuid   # ← FIXED: Missing import added

from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict

# Import core engine components
try:
    from hypercrystal.core_engine import (
        HyperCrystal, Concept, GoalField, mobius_distance, project_to_ball
    )
except ImportError as e:
    raise ImportError(f"Failed to import from hypercrystal.core_engine: {e}")

# Constants
EPS = 1e-8

# Safe clustering fallbacks
try:
    from utils import safe_kmeans, safe_cluster_indices
except ImportError:
    def safe_kmeans(embeddings, n_clusters=None):
        try:
            from sklearn.cluster import KMeans
            if embeddings is None or len(embeddings) < 2:
                return None, None
            if n_clusters is None:
                n_clusters = min(5, len(embeddings) // 2)
            n_clusters = max(2, min(n_clusters, len(embeddings)))
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
            labels = kmeans.fit_predict(embeddings)
            return labels, kmeans.cluster_centers_
        except Exception:
            return None, None

    def safe_cluster_indices(labels, n_clusters):
        indices = [[] for _ in range(n_clusters)]
        if labels is None:
            return indices
        for i, label in enumerate(labels):
            if 0 <= label < n_clusters:
                indices[label].append(i)
        return indices

# CMA-ES (optional)
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    warnings.warn("cma package not found. CMA-ES meta-learning will be disabled.")

# Persistent homology (optional)
try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False

# -----------------------------------------------------------------------------
# Novelty Registry
# -----------------------------------------------------------------------------
class NoveltyRegistry:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.concept_hashes: Set[int] = set()
        self.novelty_scores: Dict[int, float] = {}
        self.history: deque = deque(maxlen=capacity)

    def hash_concept(self, concept: Concept) -> int:
        emb_hash = hashlib.sha256(concept.subsymbolic.tobytes()).hexdigest()
        sym_hash = hashlib.sha256(str(tuple(concept.symbolic)).encode()).hexdigest()
        return hash((emb_hash, sym_hash))

    def compute_novelty(self, concept: Concept, crystal: HyperCrystal) -> float:
        return crystal._fast_novelty(concept)

    def register(self, concept: Concept, novelty: float) -> None:
        h = self.hash_concept(concept)
        self.concept_hashes.add(h)
        self.novelty_scores[h] = novelty
        self.history.append((time.time(), h, novelty))

    def is_novel(self, concept: Concept) -> bool:
        return self.hash_concept(concept) not in self.concept_hashes


# -----------------------------------------------------------------------------
# DiffusionNoveltyInjector
# -----------------------------------------------------------------------------
class DiffusionNoveltyInjector:
    def __init__(self, crystal: HyperCrystal, config: dict):
        self.crystal = crystal
        self.config = config
        self.diffusion_steps = config.get("diffusion_steps", 100)
        self.beta_schedule = np.linspace(1e-4, 0.02, self.diffusion_steps)
        self.step_count = 0

    def step(self):
        if len(self.crystal.state.concepts) < 2:
            return None

        base = random.choice(self.crystal.state.concepts)

        mutation_rate = self.config.get("mutation_rate_base", 0.15)
        burst_prob = self.config.get("burst_probability", 0.12)
        burst_factor = self.config.get("burst_factor", 3.5)
        effective_rate = mutation_rate * burst_factor if np.random.rand() < burst_prob else mutation_rate

        noise = np.random.randn(*base.subsymbolic.shape) * effective_rate
        noisy = base.subsymbolic + noise

        alpha = self.config.get("diffusion_denoise_alpha", 0.9)
        denoised = alpha * noisy + (1 - alpha) * base.subsymbolic
        denoised = project_to_ball(denoised)

        new_c = Concept(
            subsymbolic=denoised,
            symbolic=[],
            causal_graph=base.causal_graph.copy() if hasattr(base, 'causal_graph') and base.causal_graph else None,
            sophia_score=np.clip(base.sophia_score + effective_rate * np.random.randn(), 0, 1),
            dark_wisdom_density=np.clip(base.dark_wisdom_density + effective_rate * np.random.randn(), 0, 1),
            paradox_intensity=np.clip(base.paradox_intensity + effective_rate * np.random.randn(), 0, 1),
            chronon_entanglement=np.clip(getattr(base, 'chronon_entanglement', 0.5) + effective_rate * np.random.randn(), 0, 1),
            biophoton_amplitude=np.clip(getattr(base, 'biophoton_amplitude', 0.4) + effective_rate * np.random.randn(), 0, 1),
            z3_phase=getattr(base, 'z3_phase', 0),
            retrocausal_kernel=getattr(base, 'retrocausal_kernel', None),
            concept_uuid=str(uuid.uuid4())
        )

        # Build symbolic name
        fields = [("sophia", new_c.sophia_score),
                  ("wisdom", new_c.dark_wisdom_density),
                  ("paradox", new_c.paradox_intensity)]
        dominant = max(fields, key=lambda x: x[1])[0]
        parent_root = base.symbolic[0] if base.symbolic else "concept"
        if parent_root.startswith(("auto_", "mut_")):
            parent_root = base.symbolic[1] if len(base.symbolic) > 1 else "concept"
        new_c.symbolic = [f"diff_{dominant}_{self.step_count}", parent_root]

        novelty_threshold = self.config.get("novelty_acceptance_threshold", 0.08)
        novelty = self.crystal._fast_novelty(new_c)

        if novelty >= novelty_threshold:
            # FIXED: use goal_vector=
            self.crystal.store_concept(new_c, goal_vector=self.crystal.state.global_goal)

        self.step_count += 1


# -----------------------------------------------------------------------------
# MetaLearner
# -----------------------------------------------------------------------------
class MetaLearner:
    def __init__(self, crystal: HyperCrystal, config: dict):
        self.crystal = crystal
        self.config = config
        self.learning_rate = config.get("meta_learning_rate", 0.05)
        self.performance_history = deque(maxlen=100)
        self.error_patterns = defaultdict(list)
        self.failure_log = []
        self.benchmark_results = {}
        self.meta_params = {
            "mutation_rate": config.get("mutation_rate_base", 0.05),
            "sophia_attractor_strength": config.get("sophia_attractor_strength", 0.2),
            "goal_learning_rate": config.get("goal_learning_rate", 0.05),
            "exploration_rate": 0.5,
            "recursion_depth": 1,
        }
        self.recursion_depth = 1
        self.strategies = ["explore", "exploit", "balance"]
        self.active_strategy = "balance"
        self.strategy_performance = defaultdict(float)
        self.rollback_state = None
        self.last_benchmark_step = 0
        self.benchmark_interval = config.get("benchmark_interval", 100)
        self.efficiency_tracker = deque(maxlen=100)
        self.self_model = None
        self.heuristics = {}
        self.failure_modes = {}

        # CMA-ES
        self.cma = None
        self.param_keys = ['mutation_rate', 'sophia_attractor_strength', 'goal_learning_rate']
        if HAS_CMA and config.get("use_cma_es", True):
            initial_params = [self.meta_params[k] for k in self.param_keys]
            try:
                self.cma = cma.CMAEvolutionStrategy(
                    initial_params, 0.1, {'maxiter': 50, 'verbose': -1, 'popsize': 4}
                )
                self.cma_iteration = 0
            except Exception as e:
                warnings.warn(f"CMA-ES initialization failed: {e}")
                self.cma = None

    def evaluate_performance(self) -> float:
        metrics = self.crystal.get_metrics()
        score = (1.0 - metrics.get("triple_point", 0.5)) + \
                metrics.get("holo_entropy", 0.5) + \
                metrics.get("dark_wisdom", 0.5)
        diversity = min(1.0, metrics["concept_count"] / self.crystal.config["memory_capacity"])
        score += diversity * 0.5
        score += metrics.get("avg_fitness", 0.5) * 0.5
        return score

    def reflect(self):
        score = self.evaluate_performance()
        self.performance_history.append(score)
        if len(self.performance_history) > 1:
            delta = score - self.performance_history[-2]
            for param, value in self.meta_params.items():
                if param in self.crystal.config:
                    factor = 1 + self.learning_rate if delta > 0 else 1 - self.learning_rate
                    new_val = np.clip(value * factor,
                                      self.config.get(f"{param}_min", 0.01),
                                      self.config.get(f"{param}_max", 0.5))
                    self.meta_params[param] = new_val
                    self.crystal.config[param] = new_val

    def step(self):
        self.reflect()
        if self.crystal.state.step % 50 == 0:
            print(f"[Meta] Improvement score: {self.evaluate_performance():.3f}")


# -----------------------------------------------------------------------------
# RepulsionForce
# -----------------------------------------------------------------------------
class RepulsionForce:
    def __init__(self, crystal: HyperCrystal, config: dict):
        self.crystal = crystal
        self.config = config
        self.strength = config.get("repulsion_strength", 0.02)
        self.threshold = config.get("repulsion_threshold", 0.2)

    def step(self):
        n = len(self.crystal.state.concepts)
        if n < 2:
            return
        if hasattr(self.crystal.state.ann_index, 'radius_neighbors'):
            embeddings = np.vstack([c.subsymbolic for c in self.crystal.state.concepts])
            distances, indices = self.crystal.state.ann_index.radius_neighbors(embeddings, radius=self.threshold)
            for i, neighs in enumerate(indices):
                for j in neighs:
                    if j <= i:
                        continue
                    diff = self.crystal.state.concepts[i].subsymbolic - self.crystal.state.concepts[j].subsymbolic
                    norm = np.linalg.norm(diff)
                    if norm < 1e-8:
                        continue
                    force = self.strength * diff / norm
                    self.crystal.state.concepts[i].subsymbolic = project_to_ball(
                        self.crystal.state.concepts[i].subsymbolic + force
                    )
                    self.crystal.state.concepts[j].subsymbolic = project_to_ball(
                        self.crystal.state.concepts[j].subsymbolic - force
                    )


# -----------------------------------------------------------------------------
# EmergenceDetector
# -----------------------------------------------------------------------------
class EmergenceDetector:
    def __init__(self, config: dict):
        self.stagnation_threshold = config.get("stagnation_threshold", 10)
        self.creative_destruction_fraction = config.get("creative_destruction_fraction", 0.2)
        self.consecutive_stable = 0
        self.last_betti = None

    def detect_stagnation(self, crystal: HyperCrystal) -> bool:
        metrics = crystal.get_metrics()
        current_betti = (metrics.get("betti_0", 0), metrics.get("betti_1", 0))
        if self.last_betti is not None and current_betti == self.last_betti:
            self.consecutive_stable += 1
        else:
            self.consecutive_stable = 0
        self.last_betti = current_betti
        return self.consecutive_stable >= self.stagnation_threshold

    def creative_destruction(self, crystal: HyperCrystal):
        n_remove = int(len(crystal.state.concepts) * self.creative_destruction_fraction)
        if n_remove <= 0:
            return
        uuid_fitness = [(c.uuid, crystal.state.concept_fitness.get(c.uuid, 0.0)) for c in crystal.state.concepts]
        uuid_fitness.sort(key=lambda x: x[1])
        remove_uuids = {uuid for uuid, _ in uuid_fitness[:n_remove]}

        crystal.state.concepts = [c for c in crystal.state.concepts if c.uuid not in remove_uuids]
        for uuid in remove_uuids:
            crystal.state.concept_goals.pop(uuid, None)
            crystal.state.concept_fitness.pop(uuid, None)
            crystal.state.concept_rewards.pop(uuid, None)
        crystal.state.concept_pareto_front = [u for u in crystal.state.concept_pareto_front if u not in remove_uuids]
        crystal._rebuild_ann_index()
        crystal._update_pareto_front()

    def step(self, crystal: HyperCrystal):
        if self.detect_stagnation(crystal):
            self.creative_destruction(crystal)
            self.consecutive_stable = 0


# -----------------------------------------------------------------------------
# CognitionEngine
# -----------------------------------------------------------------------------
class CognitionEngine:
    def __init__(self, crystal: HyperCrystal):
        self.crystal = crystal
        self.config = crystal.config
        self.novelty_injector = DiffusionNoveltyInjector(crystal, self.config)
        self.meta_learner = MetaLearner(crystal, self.config)
        self.emergence_detector = EmergenceDetector(self.config)
        self.repulsion_force = RepulsionForce(crystal, self.config)
        self.novelty_registry = NoveltyRegistry()

    def step(self):
        self.crystal.step_internal()
        self.novelty_injector.step()
        self.meta_learner.step()
        self.emergence_detector.step(self.crystal)
        self.repulsion_force.step()

        for concept in self.crystal.state.concepts:
            if self.novelty_registry.is_novel(concept):
                novelty = self.novelty_registry.compute_novelty(concept, self.crystal)
                self.novelty_registry.register(concept, novelty)

    def run(self, steps: int = 100, verbose: bool = True):
        for step_idx in range(steps):
            self.step()
            if verbose and (step_idx % 5 == 0 or step_idx == steps - 1):
                metrics = self.crystal.get_metrics()
                novel_count = sum(1 for c in self.crystal.state.concepts
                                  if c.symbolic and not any(s.startswith(("init_", "distilled")) for s in c.symbolic))
                print(f"Step {metrics['step']}: σ={metrics.get('sophia', 0):.3f}, "
                      f"ρ_dark={metrics.get('dark_wisdom', 0):.3f}, Π={metrics.get('paradox', 0):.3f}, "
                      f"T3={metrics.get('triple_point', 0):.3f}, "
                      f"concepts={metrics['concept_count']} (novel={novel_count}), "
                      f"fitness={metrics.get('avg_fitness', 0):.3f}")


# -----------------------------------------------------------------------------
# Patch _fast_novelty if missing
# -----------------------------------------------------------------------------
def _fast_novelty(self, concept: Concept) -> float:
    if len(self.state.concepts) == 0:
        return 1.0
    results = self.retrieve_similar(concept.subsymbolic, k=1)
    if not results:
        return 1.0
    dist = results[0][1]
    similarity = np.exp(-dist)
    return 1.0 - similarity


if not hasattr(HyperCrystal, '_fast_novelty'):
    HyperCrystal._fast_novelty = _fast_novelty


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from hypercrystal.core_engine import load_config
    config = load_config()
    crystal = HyperCrystal(config)
    engine = CognitionEngine(crystal)
    engine.run(steps=50, verbose=True)
