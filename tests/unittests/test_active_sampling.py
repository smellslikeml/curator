"""Unit tests for the pool-based active sampling curation block."""

import pytest
from datasets import Dataset

from bespokelabs.curator.blocks.active_sampling import ActiveSampling
from bespokelabs.curator.blocks.raft import Raft
from bespokelabs.curator.blocks.simplestrat import StratifiedGenerator


def _build_pool():
    """A small synthetic pool with crafted uncertainties and embeddings.

    Item A is the highest-uncertainty candidate and item B is a near-duplicate
    of A (low diversity once A is selected). C/E are lower-uncertainty but far
    from A (high diversity). D is low-uncertainty and far away.
    """
    rows = [
        {"id": "A", "uncertainty": 0.95, "embedding": [1.0, 0.0]},
        {"id": "B", "uncertainty": 0.95, "embedding": [0.99, 0.01]},
        {"id": "C", "uncertainty": 0.60, "embedding": [0.0, 1.0]},
        {"id": "D", "uncertainty": 0.10, "embedding": [-1.0, 0.0]},
        {"id": "E", "uncertainty": 0.30, "embedding": [0.0, -1.0]},
    ]
    return Dataset.from_list(rows)


def test_active_sampling_is_a_blocks_peer():
    """ActiveSampling follows the same contract as the existing blocks."""
    for block_cls in (Raft, StratifiedGenerator, ActiveSampling):
        assert callable(block_cls)


def test_selects_batch_size_respecting_uncertainty_and_diversity():
    pool = _build_pool()
    sampler = ActiveSampling(batch_size=3, budget=10)
    selected = sampler(pool)

    assert isinstance(selected, Dataset)
    ids = selected["id"]
    scores = selected["active_sampling_score"]

    # Batch size is respected.
    assert len(ids) == 3
    # The highest-uncertainty item (A) is picked first because diversity
    # defaults to 1.0 before anything is selected.
    assert ids[0] == "A"
    # The near-duplicate of A (B) is excluded despite tying for highest
    # uncertainty, because its diversity to A collapses to ~0.
    assert "B" not in ids
    # The far-away, lower-uncertainty items C and E fill the batch.
    assert set(ids) == {"A", "C", "E"}
    # Greedy selection emits items in descending-score order.
    assert scores == sorted(scores, reverse=True)


def test_budget_caps_total_selection():
    pool = _build_pool()
    # batch_size larger than the pool, but budget caps the result at 2.
    sampler = ActiveSampling(batch_size=len(pool), budget=2)
    selected = sampler(pool)
    assert len(selected["id"]) == 2
    assert selected["id"][0] == "A"


def test_budget_already_exhausted_returns_empty():
    pool = _build_pool()
    sampler = ActiveSampling(batch_size=3, budget=2)
    selected = sampler(pool, already_selected=[0, 1])
    assert len(selected) == 0


def test_already_selected_seeds_diversity_and_counts_against_budget():
    pool = _build_pool()
    sampler = ActiveSampling(batch_size=2, budget=3)
    selected = sampler(pool, already_selected=[0])
    # One slot already used by the seed, so only 2 more may be selected.
    assert len(selected["id"]) == 2
    # The seed (A) is not re-emitted, and the near-duplicate B is still
    # avoided because A seeds the diversity computation.
    assert "A" not in selected["id"]
    assert "B" not in selected["id"]


def test_cannot_select_more_than_pool_size():
    pool = _build_pool()
    sampler = ActiveSampling(batch_size=100, budget=100)
    selected = sampler(pool)
    assert len(selected["id"]) == len(pool)


def test_euclidean_distance_metric_runs():
    pool = _build_pool()
    sampler = ActiveSampling(batch_size=2, budget=10, distance_metric="euclidean")
    selected = sampler(pool)
    assert selected["id"][0] == "A"
    assert "B" not in selected["id"]


def test_empty_pool_returns_empty_dataset():
    pool = _build_pool().select([])
    sampler = ActiveSampling(batch_size=3, budget=10)
    selected = sampler(pool)
    assert len(selected) == 0


def test_zero_batch_size_returns_empty_dataset():
    pool = _build_pool()
    sampler = ActiveSampling(batch_size=0, budget=10)
    selected = sampler(pool)
    assert len(selected) == 0


def test_descending_score_is_approximately_uncertainty_times_diversity():
    pool = _build_pool()
    sampler = ActiveSampling(batch_size=3, budget=10)
    selected = sampler(pool)
    # First score is uncertainty_A * 1.0 (nothing selected yet).
    assert selected["active_sampling_score"][0] == pytest.approx(0.95)
