"""
utils.py – Common helpers for HyperCrystal / QNVM (v2.0)
========================================================
Enhanced with:
- Safe KMeans clustering (with fallback)
- Timer decorator, memoization, progress bar wrapper
- Secure key generation using secrets
- MMD computation for drift detection
- ANN index builder wrapper
- Persistence diagram conversion (for gudhi)
- Viral share link generator
- Business plan formatter
- And more.
"""

import time
import logging
import functools
import secrets
import hashlib
import numpy as np
from typing import List, Optional, Callable, Any, Dict, Union, Tuple
from collections import defaultdict

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Configure a logger with console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper()))
    return logger

# -----------------------------------------------------------------------------
# Metric formatting
# -----------------------------------------------------------------------------
def format_metrics(metrics: dict, precision: int = 3) -> str:
    """Format a metrics dictionary into a compact string."""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.{precision}f}")
        elif isinstance(v, int):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)

# -----------------------------------------------------------------------------
# Progress bar wrapper (improved)
# -----------------------------------------------------------------------------
def progress_bar(iterable, desc: str = "", total: int = None,
                 print_metrics_every: Optional[int] = None,
                 metric_callback: Optional[Callable[[], str]] = None,
                 **kwargs):
    """
    Wrap tqdm if available, else a simple print.
    If print_metrics_every is set, print metrics via callback every N steps.
    """
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    if has_tqdm:
        pbar = tqdm(iterable, desc=desc, total=total, **kwargs)
        if print_metrics_every and metric_callback:
            for i, item in enumerate(pbar):
                if i % print_metrics_every == 0:
                    pbar.set_postfix_str(metric_callback())
                yield item
        else:
            yield from pbar
    else:
        # Simple fallback
        if total is None:
            total = len(iterable) if hasattr(iterable, '__len__') else 0
        for i, item in enumerate(iterable):
            if total > 0 and i % max(1, total // 10) == 0:
                print(f"{desc} {i+1}/{total}")
            if print_metrics_every and metric_callback and i % print_metrics_every == 0:
                print(metric_callback())
            yield item

# -----------------------------------------------------------------------------
# Timer decorator
# -----------------------------------------------------------------------------
def timer(func):
    """Decorator to measure execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

# -----------------------------------------------------------------------------
# Array helpers
# -----------------------------------------------------------------------------
def ensure_numpy(arr):
    """Convert to numpy array if not already."""
    if not isinstance(arr, np.ndarray):
        return np.array(arr, dtype=np.float64)
    return arr

def project_to_ball(v, eps: float = 1e-8):
    """Project a vector or batch onto the open unit ball (norm < 1)."""
    v = ensure_numpy(v)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm >= 1.0:
            return v * (1.0 - eps) / norm
        return v.copy()
    else:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        mask = norms >= 1.0
        if np.any(mask):
            v = v.copy()
            v[mask] = v[mask] * (1.0 - eps) / norms[mask]
        return v

# -----------------------------------------------------------------------------
# Memoization
# -----------------------------------------------------------------------------
def memoize(maxsize=128):
    """Simple memoization decorator."""
    def decorator(func):
        cache = {}
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            if len(cache) >= maxsize:
                cache.pop(next(iter(cache)))
            cache[key] = result
            return result
        return wrapper
    return decorator

# -----------------------------------------------------------------------------
# Safe KMeans clustering (graceful fallback)
# -----------------------------------------------------------------------------
def safe_kmeans(embeddings: np.ndarray, n_clusters: Optional[int] = None):
    """
    Perform KMeans clustering with fallback for edge cases.
    Returns (labels, centroids) or (None, None) if clustering fails.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return None, None

    if embeddings is None or len(embeddings) < 2:
        return None, None

    if n_clusters is None:
        n_clusters = min(5, len(embeddings) // 2)
    n_clusters = max(2, n_clusters)  # ensure at least 2 clusters
    n_clusters = min(n_clusters, len(embeddings))

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        return labels, kmeans.cluster_centers_
    except Exception:
        return None, None

def safe_cluster_indices(labels: np.ndarray, n_clusters: int) -> List[List[int]]:
    """
    Convert labels to list of index lists per cluster.
    Guarantees exactly n_clusters lists (some may be empty if labels missing).
    """
    indices = [[] for _ in range(n_clusters)]
    if labels is None:
        return indices
    for i, label in enumerate(labels):
        if 0 <= label < n_clusters:
            indices[label].append(i)
    return indices

# -----------------------------------------------------------------------------
# Fitness helper (exact match to core_engine._compute_fitness)
# -----------------------------------------------------------------------------
def compute_fitness_helper(concept, goal, weights: Dict[str, float]) -> float:
    """
    Compute fitness score using same formula as core_engine.
    Assumes concept has .sophia_score, .dark_wisdom_density, .paradox_intensity.
    Goal is a GoalField with .as_array() or .x, .y, .z.
    """
    # Get goal array
    if hasattr(goal, 'as_array'):
        goal_arr = goal.as_array()
    else:
        goal_arr = np.array([goal.x, goal.y, goal.z])
    concept_arr = np.array([concept.sophia_score, concept.dark_wisdom_density, concept.paradox_intensity])
    alignment = 1.0 - np.linalg.norm(concept_arr - goal_arr) / np.sqrt(3)
    score = (weights["sophia"] * concept.sophia_score +
             weights["dark_wisdom"] * concept.dark_wisdom_density +
             weights["paradox"] * concept.paradox_intensity +
             weights["goal_alignment"] * alignment)
    total_weight = sum(weights.values())
    return max(0.0, min(1.0, score / total_weight))

# -----------------------------------------------------------------------------
# Novelty scoring (reusable from NoveltyRegistry)
# -----------------------------------------------------------------------------
def novelty_score(concept, existing_concepts: List) -> float:
    """
    Compute novelty score based on embedding distance to nearest neighbor
    and symbolic overlap. This is a linear scan fallback; for large sets use ANN.
    """
    if not existing_concepts:
        return 1.0
    # Use Euclidean distance as a simple fallback (core_engine uses mobius_distance)
    try:
        from core_engine import mobius_distance
    except ImportError:
        def mobius_distance(x, y): return np.linalg.norm(x - y)
    distances = [mobius_distance(concept.subsymbolic, c.subsymbolic) for c in existing_concepts]
    min_dist = min(distances) if distances else 1.0
    # Symbolic overlap penalty
    overlap = 0
    for sym in concept.symbolic:
        for c in existing_concepts:
            if sym in c.symbolic:
                overlap += 1
                break
    sym_penalty = overlap / (len(concept.symbolic) + 1)
    novelty = min_dist * (1 - sym_penalty)
    return np.clip(novelty, 0.0, 1.0)

# -----------------------------------------------------------------------------
# Business plan formatter
# -----------------------------------------------------------------------------
def format_business_plan(plan: Dict) -> str:
    """
    Convert business_plan dictionary (as returned by ReportBuilder.business_plan)
    into a nicely formatted markdown string with sections and scores.
    """
    lines = []
    lines.append(f"# {plan.get('title', 'Business Plan')}")
    lines.append(f"*Generated: {plan.get('generated', 'N/A')}*")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    lines.append(plan.get('executive_summary', ''))
    lines.append("")

    # Market opportunity
    market = plan.get('market_opportunity', {})
    if market:
        lines.append("## Market Opportunity")
        lines.append(f"**Size & Growth:** {market.get('size_growth', 'N/A')}")
        lines.append(f"**Target Segments:** {', '.join(market.get('target_segments', []))}")
        lines.append(f"**Differentiation:** {market.get('differentiation', 'N/A')}")
        lines.append("")

    # Product
    product = plan.get('product', {})
    if product:
        lines.append("## Product")
        lines.append(f"**Core:** {product.get('core', 'N/A')}")
        lines.append(f"**Value Add:** {product.get('value_add', 'N/A')}")
        lines.append(f"**Future:** {product.get('future', 'N/A')}")
        lines.append("")

    # Technology
    tech = plan.get('technology', {})
    if tech:
        lines.append("## Technology")
        for k, v in tech.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    # Revenue model
    rev = plan.get('revenue_model', {})
    if rev:
        lines.append("## Revenue Model")
        lines.append(f"**Subscription Tiers:** {', '.join(rev.get('subscription_tiers', []))}")
        lines.append(f"**Pricing Model:** {rev.get('pricing_model', 'N/A')}")
        lines.append(f"**Marketplace Commission:** {rev.get('marketplace_commission', 'N/A')}")
        lines.append("")

    # Key insights
    insights = plan.get('key_insights', [])
    if insights:
        lines.append("## Key Insights")
        for ins in insights:
            lines.append(f"- {ins}")
        lines.append("")

    # Financial projections
    fin = plan.get('financial_projections', {})
    if fin:
        lines.append("## Financial Projections")
        lines.append(f"**Year 1 Revenue:** ${fin.get('year1_revenue_usd', 0):,}")
        lines.append(f"**Year 2 Revenue:** ${fin.get('year2_revenue_usd', 0):,}")
        lines.append(f"**Year 1 COGS:** ${fin.get('year1_cogs_usd', 0):,}")
        lines.append(f"**Year 2 COGS:** ${fin.get('year2_cogs_usd', 0):,}")
        lines.append("")

    # Risks
    risks = plan.get('risks', [])
    if risks:
        lines.append("## Risks")
        for r in risks:
            lines.append(f"- {r}")
        lines.append("")

    # Roadmap
    roadmap = plan.get('roadmap', [])
    if roadmap:
        lines.append("## Roadmap")
        for step in roadmap:
            lines.append(f"- {step}")
        lines.append("")

    # Scores
    lines.append("## Confidence & Actionability")
    lines.append(f"**Confidence:** {plan.get('confidence', 0.0):.2f}")
    lines.append(f"**Actionability:** {plan.get('actionability', 0.0):.2f}")
    lines.append(f"**Explainability:** {plan.get('explainability', 0.0):.2f}")

    return "\n".join(lines)

# -----------------------------------------------------------------------------
# Secure key generation (for API keys, etc.)
# -----------------------------------------------------------------------------
def generate_secure_key(prefix: str = "") -> str:
    """Generate a cryptographically secure random hex string, optionally with a prefix."""
    return prefix + secrets.token_hex(16)

# -----------------------------------------------------------------------------
# MMD computation for drift detection
# -----------------------------------------------------------------------------
def compute_mmd(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy between two sets of points using RBF kernel.
    x and y are arrays of shape (n, d) and (m, d).
    """
    try:
        from sklearn.metrics.pairwise import rbf_kernel
    except ImportError:
        return 0.0
    K_xx = rbf_kernel(x, x, gamma=gamma)
    K_yy = rbf_kernel(y, y, gamma=gamma)
    K_xy = rbf_kernel(x, y, gamma=gamma)
    mmd = K_xx.mean() + K_yy.mean() - 2*K_xy.mean()
    return max(0.0, mmd)

# -----------------------------------------------------------------------------
# ANN index builder wrapper
# -----------------------------------------------------------------------------
def build_ann_index(embeddings: np.ndarray, metric: str = 'euclidean', n_neighbors: int = 5):
    """Build a NearestNeighbors index for fast similarity search."""
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(embeddings)
        return nn
    except ImportError:
        return None

# -----------------------------------------------------------------------------
# Persistence diagram conversion (for gudhi)
# -----------------------------------------------------------------------------
def persistence_diagram_to_array(diagram: List[Tuple[int, Tuple[float, float]]]) -> np.ndarray:
    """
    Convert a gudhi persistence diagram (list of (dim, (birth, death))) to a numpy array
    of shape (n, 2) containing only the birth/death pairs, with inf replaced by a large number.
    """
    arr = []
    for dim, (b, d) in diagram:
        if d == float('inf'):
            d = 1e10
        arr.append([b, d])
    return np.array(arr, dtype=np.float64)

# -----------------------------------------------------------------------------
# Viral share hook – generates shareable URL with tracking
# -----------------------------------------------------------------------------
def viral_share_hook(artifact_id: str, user_id: str, base_url: str = "https://hypercrystal.ai") -> str:
    """
    Return a shareable URL for an artifact with user tracking.
    """
    return f"{base_url}/share/{artifact_id}?ref={user_id}"
