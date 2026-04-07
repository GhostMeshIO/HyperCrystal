"""
cognition_engine.py – Intelligence & Novelty for HyperCrystal (Revised v2.1)
===========================================================================
Fully compatible with the revised core_engine.py (UUID-based, thread‑safe).
Implements diffusion novelty injection, CMA‑ES meta‑learning, repulsion force,
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
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict

# Import core engine components (absolute import)
try:
    from hypercrystal.core_engine import (
        HyperCrystal, Concept, GoalField, mobius_distance, project_to_ball
    )
except ImportError as e:
    raise ImportError(f"Failed to import from hypercrystal.core_engine: {e}")

# Constants
EPS = 1e-8

# Import safe clustering helpers – with fallback
try:
    from utils import safe_kmeans, safe_cluster_indices
except ImportError:
    # Simple fallback implementations
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

# For CMA‑ES (optional)
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    warnings.warn("cma package not found. CMA‑ES meta‑learning will be disabled.")

# For persistent homology (optional)
try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False

# For mutual information (optional)
try:
    from sklearn.feature_selection import mutual_info_regression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# -----------------------------------------------------------------------------
# Novelty Registry (using ANN via core_engine)
# -----------------------------------------------------------------------------
class NoveltyRegistry:
    """Keeps track of generated concepts and their novelty scores."""
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
        """Use core's fast novelty (uses ANN index)."""
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
    """Generates new concepts by denoising noisy copies of existing ones."""
    def __init__(self, crystal: HyperCrystal, config: dict):
        self.crystal = crystal
        self.config = config
        self.diffusion_steps = config.get("diffusion_steps", 100)
        self.beta_schedule = np.linspace(1e-4, 0.02, self.diffusion_steps)
        self.denoiser = None
        self.step_count = 0

    def step(self):
        if len(self.crystal.state.concepts) < 2:
            return
        base = random.choice(self.crystal.state.concepts)
        t = random.randint(1, self.diffusion_steps - 1)
        mutation_rate = self.config.get("mutation_rate_base", 0.15)
        burst_prob = self.config.get("burst_probability", 0.12)
        burst_factor = self.config.get("burst_factor", 3.5)
        effective_rate = mutation_rate * burst_factor if np.random.rand() < burst_prob else mutation_rate
        noise = np.random.randn(*base.subsymbolic.shape) * effective_rate
        noisy = base.subsymbolic + noise
        alpha = self.config.get("diffusion_denoise_alpha", 0.9)
        denoised = alpha * noisy + (1 - alpha) * base.subsymbolic
        denoised = project_to_ball(denoised)

        # Create temporary concept for field values
        new_c = Concept(
            subsymbolic=denoised,
            symbolic=[],
            causal_graph=base.causal_graph.copy() if base.causal_graph else None,
            sophia_score=np.clip(base.sophia_score + effective_rate * np.random.randn(), 0, 1),
            dark_wisdom_density=np.clip(base.dark_wisdom_density + effective_rate * np.random.randn(), 0, 1),
            paradox_intensity=np.clip(base.paradox_intensity + effective_rate * np.random.randn(), 0, 1),
            chronon_entanglement=np.clip(base.chronon_entanglement + effective_rate * np.random.randn(), 0, 1),
            biophoton_amplitude=np.clip(base.biophoton_amplitude + effective_rate * np.random.randn(), 0, 1),
            z3_phase=base.z3_phase,
            retrocausal_kernel=base.retrocausal_kernel + effective_rate * np.random.randn(*base.retrocausal_kernel.shape)
                if base.retrocausal_kernel is not None else None
        )
        # Build symbolic name
        fields = [("sophia", new_c.sophia_score),
                  ("wisdom", new_c.dark_wisdom_density),
                  ("paradox", new_c.paradox_intensity)]
        dominant = max(fields, key=lambda x: x[1])[0]
        parent_root = base.symbolic[0] if base.symbolic else "concept"
        if parent_root.startswith("auto_") or parent_root.startswith("mut_"):
            parent_root = base.symbolic[1] if len(base.symbolic) > 1 else "concept"
        new_c.symbolic = [f"diff_{dominant}_{self.step_count}", parent_root]

        novelty_threshold = self.config.get("novelty_acceptance_threshold", 0.08)
        novelty = self.crystal._fast_novelty(new_c)
        if novelty >= novelty_threshold:
            self.crystal.store_concept(new_c, goal=self.crystal.state.global_goal)
        self.step_count += 1

# -----------------------------------------------------------------------------
# MetaLearner – with CMA‑ES (updated for UUIDs)
# -----------------------------------------------------------------------------
class MetaLearner:
    """Recursive self‑evaluation, meta‑learning, and self‑improvement."""
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

        # CMA‑ES initialization
        self.cma = None
        self.param_keys = ['mutation_rate', 'sophia_attractor_strength', 'goal_learning_rate']
        if HAS_CMA and config.get("use_cma_es", True):
            initial_params = [self.meta_params[k] for k in self.param_keys]
            try:
                self.cma = cma.CMAEvolutionStrategy(
                    initial_params,
                    0.1,
                    {'maxiter': 50, 'verbose': -1, 'popsize': 4}
                )
                self.cma_iteration = 0
            except Exception as e:
                warnings.warn(f"CMA-ES initialization failed: {e}")
                self.cma = None

    def evaluate_performance(self) -> float:
        metrics = self.crystal.get_metrics()
        score = (1.0 - metrics.get("triple_point", 0.5)) + metrics.get("holo_entropy", 0.5) + metrics.get("dark_wisdom", 0.5)
        diversity = min(1.0, metrics["concept_count"] / self.crystal.config["memory_capacity"])
        score += diversity * 0.5
        avg_fitness = metrics.get("avg_fitness", 0.5)
        score += avg_fitness * 0.5
        return score

    def recursive_evaluate(self, depth: int = 1) -> float:
        base_score = self.evaluate_performance()
        if depth <= 0:
            return base_score
        fitness_vals = list(self.crystal.state.concept_fitness.values())
        consistency = 1.0 - np.std(fitness_vals) if fitness_vals else 0.5
        return base_score * (0.7 + 0.3 * consistency)

    def build_self_model(self) -> np.ndarray:
        metrics = self.crystal.get_metrics()
        return np.array([
            metrics.get("sophia", 0.5),
            metrics.get("dark_wisdom", 0.5),
            metrics.get("paradox", 0.5),
            metrics.get("triple_point", 0.5),
            metrics.get("meta_depth", 1),
            metrics.get("non_hermitian", 0.1),
            metrics.get("avg_fitness", 0.5),
            len(self.crystal.state.concepts) / self.crystal.config["memory_capacity"],
        ])

    def introspection_step(self):
        self.self_model = self.build_self_model()
        if self.self_model[0] < 0.5:
            self.active_strategy = "explore"
        elif self.self_model[0] > 0.8:
            self.active_strategy = "exploit"
        else:
            self.active_strategy = "balance"

    def reflect(self):
        score = self.evaluate_performance()
        self.performance_history.append(score)
        if len(self.performance_history) > 1:
            delta = score - self.performance_history[-2]
            for param, value in self.meta_params.items():
                if param in self.crystal.config:
                    if delta > 0:
                        new_val = value * (1 + self.learning_rate)
                    else:
                        new_val = value * (1 - self.learning_rate)
                    new_val = np.clip(new_val, self.config.get(f"{param}_min", 0.01), self.config.get(f"{param}_max", 0.5))
                    self.meta_params[param] = new_val
                    self.crystal.config[param] = new_val
                    if param == "mutation_rate" and hasattr(self.crystal.cognition, 'novelty_injector'):
                        self.crystal.cognition.novelty_injector.mutation_rate_base = new_val

    def detect_errors(self):
        low_fitness_count = sum(1 for f in self.crystal.state.concept_fitness.values() if f < 0.2)
        if low_fitness_count > len(self.crystal.state.concepts) * 0.3:
            self.error_patterns["low_fitness"].append(self.crystal.state.step)
            self.correct_error("low_fitness")
        for c in self.crystal.state.concepts:
            if np.any(np.isnan(c.subsymbolic)):
                self.error_patterns["nan_embedding"].append(self.crystal.state.step)
                self.correct_error("nan_embedding")

    def correct_error(self, error_type: str):
        if error_type == "low_fitness":
            self.active_strategy = "explore"
            self.meta_params["exploration_rate"] = min(0.9, self.meta_params.get("exploration_rate", 0.5) * 1.2)
        elif error_type == "nan_embedding":
            for c in self.crystal.state.concepts:
                if np.any(np.isnan(c.subsymbolic)):
                    c.subsymbolic = project_to_ball(np.random.randn(self.crystal.config["embedding_dim"]))

    def evolve_parameters(self):
        if not HAS_CMA or self.cma is None:
            return
        if self.crystal.state.step % 50 != 0:
            return
        try:
            candidates = self.cma.ask()
            scores = []
            for candidate in candidates:
                old_params = {k: self.meta_params[k] for k in self.param_keys}
                for i, key in enumerate(self.param_keys):
                    self.meta_params[key] = candidate[i]
                    self.crystal.config[key] = candidate[i]
                score = self.evaluate_performance()
                scores.append(score)
                for key, val in old_params.items():
                    self.meta_params[key] = val
                    self.crystal.config[key] = val
            self.cma.tell(candidates, scores)
            self.cma_iteration += 1
            if self.cma_iteration > 20 and self.cma.result.xbest is not None:
                best = self.cma.result.xbest
                for i, key in enumerate(self.param_keys):
                    self.meta_params[key] = best[i]
                    self.crystal.config[key] = best[i]
        except Exception as e:
            warnings.warn(f"CMA-ES error: {e}. Disabling.")
            self.cma = None

    def compression_score(self, concept: Concept) -> float:
        emb_entropy = -np.sum(np.abs(concept.subsymbolic) * np.log(np.abs(concept.subsymbolic) + EPS))
        sym_length = len(concept.symbolic)
        return emb_entropy / (sym_length + 1)

    def mutate_architecture(self):
        pass

    def adjust_recursion_depth(self):
        if len(self.performance_history) > 1:
            if self.performance_history[-1] > self.performance_history[-2]:
                self.recursion_depth = min(self.config.get("max_recursion_depth", 5), self.recursion_depth + 1)
            else:
                self.recursion_depth = max(1, self.recursion_depth - 1)

    def detect_instability(self) -> bool:
        if len(self.performance_history) < 10:
            return False
        recent = list(self.performance_history)[-10:]
        variance = np.var(recent)
        return variance > 0.2

    def stabilize(self):
        self.meta_params["mutation_rate"] = max(0.01, self.meta_params["mutation_rate"] * 0.5)
        self.active_strategy = "balance"
        if self.rollback_state is not None:
            # Rollback is a full copy of the state; assign safely
            self.crystal.state = self.rollback_state

    def switch_strategy(self):
        current_score = self.evaluate_performance()
        self.strategy_performance[self.active_strategy] = current_score
        best_strategy = max(self.strategy_performance, key=self.strategy_performance.get, default=self.active_strategy)
        self.active_strategy = best_strategy
        if self.active_strategy == "explore":
            self.meta_params["exploration_rate"] = min(0.9, self.meta_params.get("exploration_rate", 0.5) * 1.1)
        elif self.active_strategy == "exploit":
            self.meta_params["exploration_rate"] = max(0.1, self.meta_params.get("exploration_rate", 0.5) * 0.9)

    def adjust_learning_rate(self):
        if len(self.performance_history) > 1:
            if self.performance_history[-1] > self.performance_history[-2]:
                self.learning_rate = min(0.2, self.learning_rate * 1.05)
            else:
                self.learning_rate = max(0.01, self.learning_rate * 0.95)

    def learn_heuristic(self, pattern: str, action: str):
        self.heuristics[pattern] = action

    def distill_knowledge(self):
        concepts = self.crystal.state.concepts
        if len(concepts) < 10:
            return
        embeddings = np.vstack([c.subsymbolic for c in concepts])
        n_clusters = min(5, len(embeddings) // 2)
        if n_clusters < 2:
            return
        labels, _ = safe_kmeans(embeddings, n_clusters=n_clusters)
        if labels is None:
            return
        cluster_indices = safe_cluster_indices(labels, n_clusters)
        for cluster_id, indices in enumerate(cluster_indices):
            if len(indices) < 2:
                continue
            cluster_concepts = [concepts[i] for i in indices]
            avg_emb = np.mean([c.subsymbolic for c in cluster_concepts], axis=0)
            avg_emb = project_to_ball(avg_emb)
            avg_sophia = np.mean([c.sophia_score for c in cluster_concepts])
            avg_dark = np.mean([c.dark_wisdom_density for c in cluster_concepts])
            avg_paradox = np.mean([c.paradox_intensity for c in cluster_concepts])
            new_concept = Concept(
                subsymbolic=avg_emb,
                symbolic=["distilled", f"cluster_{cluster_id}"],
                sophia_score=avg_sophia,
                dark_wisdom_density=avg_dark,
                paradox_intensity=avg_paradox,
                chronon_entanglement=np.mean([c.chronon_entanglement for c in cluster_concepts]),
                biophoton_amplitude=np.mean([c.biophoton_amplitude for c in cluster_concepts]),
                z3_phase=cluster_concepts[0].z3_phase,
                retrocausal_kernel=np.mean([c.retrocausal_kernel for c in cluster_concepts], axis=0)
                    if cluster_concepts[0].retrocausal_kernel is not None else None
            )
            # store_concept now returns UUID; we ignore it
            self.crystal.store_concept(new_concept, goal=self.crystal.state.global_goal)

    def run_benchmark(self):
        if self.crystal.state.step - self.last_benchmark_step < self.benchmark_interval:
            return
        self.last_benchmark_step = self.crystal.state.step
        test_queries = [project_to_ball(np.random.randn(self.crystal.config["embedding_dim"])) for _ in range(10)]
        total_distance = 0.0
        for q in test_queries:
            results = self.crystal.retrieve_similar(q, k=1)
            if results:
                total_distance += results[0][1]
        avg_distance = total_distance / len(test_queries) if test_queries else 1.0
        self.benchmark_results["avg_retrieval_distance"] = avg_distance
        embeddings = np.vstack([c.subsymbolic for c in self.crystal.state.concepts])
        if len(embeddings) > 1:
            if len(embeddings) > 500:
                idx = np.random.choice(len(embeddings), 500, replace=False)
                embeddings = embeddings[idx]
            pairwise_dists = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    pairwise_dists.append(mobius_distance(embeddings[i], embeddings[j]))
            avg_pairwise = np.mean(pairwise_dists) if pairwise_dists else 0.5
            self.benchmark_results["avg_pairwise_distance"] = avg_pairwise
        else:
            self.benchmark_results["avg_pairwise_distance"] = 0.0

    def track_efficiency(self, cost: float, gain: float):
        self.efficiency_tracker.append((cost, gain))

    def get_efficiency_score(self) -> float:
        if not self.efficiency_tracker:
            return 0.0
        total_gain = sum(g for _, g in self.efficiency_tracker)
        total_cost = sum(c for c, _ in self.efficiency_tracker)
        return total_gain / (total_cost + EPS)

    def self_debug(self):
        for c in self.crystal.state.concepts:
            if np.any(np.isnan(c.subsymbolic)):
                c.subsymbolic = project_to_ball(np.random.randn(self.crystal.config["embedding_dim"]))

    def log_failure(self, failure_type: str, details: str):
        self.failure_log.append((self.crystal.state.step, failure_type, details))
        self.failure_modes[failure_type] = self.failure_modes.get(failure_type, 0) + 1

    def save_rollback_state(self):
        self.rollback_state = copy.deepcopy(self.crystal.state)

    def rollback(self):
        if self.rollback_state is not None:
            self.crystal.state = self.rollback_state

    def consolidate_memory(self):
        pass

    def improvement_score(self) -> float:
        if len(self.performance_history) < 2:
            return 0.0
        return self.performance_history[-1] - self.performance_history[0]

    def step(self):
        self.introspection_step()
        self.reflect()
        self.adjust_recursion_depth()
        self.detect_errors()
        if self.crystal.state.step % 50 == 0:
            self.self_debug()
        if self.crystal.state.step % self.benchmark_interval == 0:
            self.run_benchmark()
        self.evolve_parameters()
        self.switch_strategy()
        self.adjust_learning_rate()
        if self.crystal.state.step % 50 == 0:
            self.distill_knowledge()
        if self.detect_instability():
            self.stabilize()
        cost = 0.01
        gain = self.evaluate_performance() - (self.performance_history[-2] if len(self.performance_history) > 1 else 0)
        self.track_efficiency(cost, gain)
        if self.crystal.state.step % 100 == 0:
            print(f"[Meta] Improvement score: {self.improvement_score():.3f}, Efficiency: {self.get_efficiency_score():.3f}")

# -----------------------------------------------------------------------------
# RepulsionForce – uses ANN or fallback (works with UUIDs because it uses list indices)
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
        # Try to use radius_neighbors if available (sklearn)
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
        else:
            # Fallback: brute force pairwise
            for i in range(n):
                for j in range(i+1, n):
                    diff = self.crystal.state.concepts[i].subsymbolic - self.crystal.state.concepts[j].subsymbolic
                    dist = np.linalg.norm(diff)
                    if dist < self.threshold and dist > 1e-8:
                        force = self.strength * diff / dist
                        self.crystal.state.concepts[i].subsymbolic = project_to_ball(
                            self.crystal.state.concepts[i].subsymbolic + force
                        )
                        self.crystal.state.concepts[j].subsymbolic = project_to_ball(
                            self.crystal.state.concepts[j].subsymbolic - force
                        )

# -----------------------------------------------------------------------------
# EmergenceDetector – with persistent homology (updated for UUIDs)
# -----------------------------------------------------------------------------
class EmergenceDetector:
    def __init__(self, config: dict):
        self.stagnation_threshold = config.get("stagnation_threshold", 10)
        self.creative_destruction_fraction = config.get("creative_destruction_fraction", 0.2)
        self.consecutive_stable = 0
        self.last_betti = None
        self.last_triple_point = None
        self.triggered_destruction = False
        self.last_persistence_diagram = None

    @staticmethod
    def _diagram_to_array(diagram):
        arr = []
        for dim, (b, d) in diagram:
            if d == float('inf'):
                d = 1e10
            arr.append([b, d])
        return np.array(arr, dtype=np.float64)

    def detect_stagnation(self, crystal: HyperCrystal) -> bool:
        metrics = crystal.get_metrics()
        current_betti = (metrics.get("betti_0", 0), metrics.get("betti_1", 0))
        current_triple = metrics.get("triple_point", 0.5)

        if self.last_betti is not None and current_betti == self.last_betti:
            self.consecutive_stable += 1
        else:
            self.consecutive_stable = 0

        self.last_betti = current_betti
        self.last_triple_point = current_triple

        if current_triple < 0.05 and self.consecutive_stable > 5:
            return True

        if HAS_GUDHI and len(crystal.state.concepts) > 5:
            embeddings = np.vstack([c.subsymbolic for c in crystal.state.concepts])
            rips = gudhi.RipsComplex(points=embeddings)
            st = rips.create_simplex_tree(max_dimension=2)
            diagram = st.persistence()
            if self.last_persistence_diagram is not None:
                arr1 = self._diagram_to_array(self.last_persistence_diagram)
                arr2 = self._diagram_to_array(diagram)
                bottleneck = gudhi.bottleneck_distance(arr1, arr2)
                if bottleneck < 0.01 and self.consecutive_stable > 3:
                    return True
            self.last_persistence_diagram = diagram

        return self.consecutive_stable >= self.stagnation_threshold

    def creative_destruction(self, crystal: HyperCrystal) -> None:
        """Remove the lowest-fitness fraction of concepts, using UUIDs."""
        n_remove = int(len(crystal.state.concepts) * self.creative_destruction_fraction)
        if n_remove <= 0:
            return

        # Get list of (uuid, fitness) for all concepts
        uuid_fitness = [(c.uuid, crystal.state.concept_fitness.get(c.uuid, 0.0)) for c in crystal.state.concepts]
        # Sort by fitness (lowest first)
        uuid_fitness.sort(key=lambda x: x[1])
        remove_uuids = {uuid for uuid, _ in uuid_fitness[:n_remove]}

        # Keep only concepts not in remove_uuids
        new_concepts = [c for c in crystal.state.concepts if c.uuid not in remove_uuids]

        # Update state
        crystal.state.concepts = new_concepts
        # Remove metadata for evicted UUIDs
        for uuid in remove_uuids:
            crystal.state.concept_goals.pop(uuid, None)
            crystal.state.concept_fitness.pop(uuid, None)
            crystal.state.concept_rewards.pop(uuid, None)
        # Update Pareto front (filter out removed UUIDs)
        crystal.state.concept_pareto_front = [uuid for uuid in crystal.state.concept_pareto_front if uuid not in remove_uuids]

        # Rebuild ANN index and Pareto front
        crystal._rebuild_ann_index()
        crystal._update_pareto_front()
        self.triggered_destruction = True

    def step(self, crystal: HyperCrystal) -> None:
        if self.detect_stagnation(crystal):
            self.creative_destruction(crystal)
            self.consecutive_stable = 0
        else:
            self.triggered_destruction = False

# -----------------------------------------------------------------------------
# CognitionEngine – orchestrator (no changes needed, already uses UUID-compatible methods)
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
        # Attach to crystal for meta‑learner to update params
        self.crystal.cognition = self

    def step(self) -> None:
        self.crystal.step_internal()
        self.novelty_injector.step()
        self.meta_learner.step()
        self.emergence_detector.step(self.crystal)
        self.repulsion_force.step()
        # Update novelty registry
        for concept in self.crystal.state.concepts:
            if self.novelty_registry.is_novel(concept):
                novelty = self.novelty_registry.compute_novelty(concept, self.crystal)
                self.novelty_registry.register(concept, novelty)

    def run(self, steps: int = 100, verbose: bool = True):
        for _ in range(steps):
            self.step()
            if verbose and (_ % 5 == 0 or _ == steps-1):
                metrics = self.crystal.get_metrics()
                novel_count = sum(1 for c in self.crystal.state.concepts
                                  if c.symbolic and not c.symbolic[0].startswith("init_")
                                  and not c.symbolic[0].startswith("distilled"))
                print(f"Step {metrics['step']}: σ={metrics.get('sophia',0):.3f}, "
                      f"ρ_dark={metrics.get('dark_wisdom',0):.3f}, Π={metrics.get('paradox',0):.3f}, "
                      f"T3={metrics.get('triple_point',0):.3f}, "
                      f"concepts={metrics['concept_count']} (novel={novel_count}), "
                      f"fitness={metrics.get('avg_fitness',0):.3f}, strategy={self.meta_learner.active_strategy}")

# -----------------------------------------------------------------------------
# Add missing _fast_novelty method to HyperCrystal (monkey patch) – updated for UUIDs
# -----------------------------------------------------------------------------
def _fast_novelty(self, concept: Concept) -> float:
    """Compute novelty as 1 - maximum similarity to existing concepts."""
    if len(self.state.concepts) == 0:
        return 1.0
    # Use ANN to find nearest neighbor distance
    results = self.retrieve_similar(concept.subsymbolic, k=1)
    if not results:
        return 1.0
    dist = results[0][1]  # Euclidean distance
    # Convert distance to similarity in [0,1]
    similarity = np.exp(-dist)  # or 1/(1+dist)
    return 1.0 - similarity

# Apply patch if not already present
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
    engine.run(steps=20)
