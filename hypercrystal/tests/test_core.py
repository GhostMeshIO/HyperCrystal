"""
Unit tests for HyperCrystal core engine.
Run with: pytest hypercrystal/tests/
"""

import pytest
import numpy as np
from hypercrystal.core_engine import (
    HyperCrystal, GoalField, Concept, load_config,
    project_to_ball, mobius_distance, SOPHIA_POINT
)


@pytest.fixture
def crystal():
    config = load_config()
    config["memory_capacity"] = 100
    config["embedding_dim"] = 32
    config["verbose"] = False
    return HyperCrystal(config)


def test_initialization(crystal):
    assert crystal.state.step == 0
    assert len(crystal.state.concepts) == 60  # default init count
    assert crystal.state.global_goal is not None
    assert crystal.state.global_goal.x == SOPHIA_POINT
    assert 0.0 <= crystal.state.sophia_score <= 1.0


def test_goal_field():
    g = GoalField(0.5, 0.3, 0.2)
    assert g.as_array().tolist() == [0.5, 0.3, 0.2]
    g.add_history(g.as_array())
    smoothed = g.smoothed_goal()
    assert np.allclose(smoothed, [0.5, 0.3, 0.2])
    
    g2 = GoalField(0.7, 0.4, 0.1)
    assert g.distance_to(g2) == pytest.approx(np.linalg.norm([0.2, 0.1, -0.1]))
    assert not g.is_conflicting(g2, threshold=0.3)  # small angle
    assert g.is_conflicting(GoalField(1, 0, 0), threshold=0.1)


def test_project_to_ball():
    v = np.array([2.0, 0.0, 0.0])
    proj = project_to_ball(v)
    assert np.linalg.norm(proj) < 1.0
    assert proj[0] > 0
    v2 = np.array([0.1, 0.2, 0.3])
    proj2 = project_to_ball(v2)
    assert np.allclose(proj2, v2)


def test_store_concept(crystal):
    emb = np.random.randn(32)
    emb = project_to_ball(emb)
    concept = Concept(
        subsymbolic=emb,
        symbolic=["test_concept"],
        sophia_score=0.7,
        dark_wisdom_density=0.4,
        paradox_intensity=0.2
    )
    idx = crystal.store_concept(concept, crystal.state.global_goal)
    assert idx == len(crystal.state.concepts) - 1
    assert crystal.state.concepts[idx].symbolic[0] == "test_concept"
    assert idx in crystal.state.concept_fitness
    assert 0.0 <= crystal.state.concept_fitness[idx] <= 1.0


def test_retrieve_similar(crystal):
    # Add a few concepts
    for i in range(5):
        emb = np.random.randn(32)
        emb = project_to_ball(emb)
        c = Concept(subsymbolic=emb, symbolic=[f"c{i}"])
        crystal.store_concept(c, crystal.state.global_goal)
    
    query = crystal.state.concepts[0].subsymbolic
    results = crystal.retrieve_similar(query, k=3)
    assert len(results) <= 3
    assert isinstance(results[0][0], Concept)
    assert isinstance(results[0][1], float)


def test_set_global_goal(crystal):
    new_goal = GoalField(0.8, 0.2, 0.1)
    crystal.set_global_goal(new_goal)
    assert crystal.state.global_goal.x == 0.8
    assert crystal.state.global_goal.y == 0.2
    assert crystal.state.global_goal.z == 0.1


def test_metrics(crystal):
    metrics = crystal.get_metrics()
    expected_keys = {"step", "sophia", "dark_wisdom", "paradox", "triple_point",
                     "concept_count", "avg_fitness", "pareto_front_size"}
    assert expected_keys.issubset(metrics.keys())
    assert metrics["concept_count"] == len(crystal.state.concepts)


def test_pareto_front(crystal):
    # Add diverse concepts
    for _ in range(20):
        emb = np.random.randn(32)
        emb = project_to_ball(emb)
        c = Concept(
            subsymbolic=emb,
            sophia_score=np.random.uniform(0.3, 0.9),
            dark_wisdom_density=np.random.uniform(0.2, 0.8),
            paradox_intensity=np.random.uniform(0.1, 0.7)
        )
        crystal.store_concept(c, crystal.state.global_goal)
    crystal._update_pareto_front()
    front = crystal.state.concept_pareto_front
    assert len(front) > 0
    assert len(front) <= crystal.config["pareto_front_size"]


def test_eviction(crystal):
    # Fill beyond capacity
    crystal.config["memory_capacity"] = 10
    for i in range(20):
        emb = np.random.randn(32)
        emb = project_to_ball(emb)
        c = Concept(subsymbolic=emb, symbolic=[f"evict_{i}"])
        crystal.store_concept(c, crystal.state.global_goal)
    assert len(crystal.state.concepts) <= 10
