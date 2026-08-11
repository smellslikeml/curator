"""Tests for the uncertainty x diversity active sampler block.

The pool fixtures are built with ``chunk_text`` from the pre-existing
``bespokelabs.curator.blocks.raft`` block, so this suite exercises both the new
``active_sampler`` module and an existing block rather than only self-testing the
new code. The annotator is a deterministic in-process callable, so no network or
LLM is required.
"""

import pytest
from datasets import Dataset

from bespokelabs.curator.blocks.active_sampler import ActiveCurationResult, ActiveSampler, annotate_with_consistency, select_batch
from bespokelabs.curator.blocks.raft import chunk_text


def _stable_annotator(label: str = "yes"):
    """Return an annotator that always emits the same label for every item."""

    def _ann(batch: Dataset):
        return [label for _ in batch]

    return _ann


def _flipping_annotator(flip_text: str):
    """Return an annotator that disagrees with itself on ``flip_text``.

    On the first pass over the batch it labels ``flip_text`` as ``"X"`` and on the
    second pass as ``"Y"``; every other item is always labeled ``"ok"``. This
    simulates two annotators disagreeing only on the targeted item.
    """

    counts: dict[str, int] = {}

    def _ann(batch: Dataset):
        out = []
        for item in batch:
            text = item["text"]
            counts[text] = counts.get(text, 0) + 1
            if text == flip_text and counts[text] == 1:
                out.append("X")
            elif text == flip_text:
                out.append("Y")
            else:
                out.append("ok")
        return out

    return _ann


@pytest.fixture
def raft_pool() -> Dataset:
    """Build a pool dataset using the existing ``raft.chunk_text`` block."""
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "Sphinx of black quartz, judge my vow. "
        "How vexingly quick daft zebras jump. "
        "The five boxing wizards jump quickly. "
        "Bright vixens jump; dozy fowl quack."
    )
    return chunk_text(text, chunk_size=40)


def test_select_batch_seeds_with_uncertainty_and_diversifies():
    """The seed is the most uncertain item; later picks favor far-apart items."""
    uncertainties = [0.9, 0.1, 0.8, 0.5]
    features = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    chosen = select_batch(uncertainties, features, k=2)
    # Most uncertain item (index 0) is seeded first.
    assert chosen[0] == 0
    # Index 2 is both uncertain (0.8) and far from the seed, so it beats index 1
    # (certain and identical to the seed) and index 3 (less uncertain than 2).
    assert chosen[1] == 2


def test_select_batch_degenerate_features_fall_back_to_uncertainty():
    """When all features are identical, selection orders by uncertainty."""
    uncertainties = [0.2, 0.9, 0.5]
    features = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    chosen = select_batch(uncertainties, features, k=2)
    assert chosen == [1, 2]


def test_select_batch_respects_pool_size():
    """``k`` larger than the pool returns every item once."""
    chosen = select_batch([0.1, 0.2, 0.3], [[1.0], [2.0], [3.0]], k=10)
    assert sorted(chosen) == [0, 1, 2]


def test_select_batch_empty_or_zero_k():
    """Empty pool or zero ``k`` returns an empty selection."""
    assert select_batch([], [], k=3) == []
    assert select_batch([0.5], [[1.0]], k=0) == []


def test_annotate_with_consistency_only_promotes_agreements():
    """Items where annotators disagree are flagged and given no label."""
    batch = Dataset.from_list([{"text": "p"}, {"text": "q"}, {"text": "flip"}])
    rows = annotate_with_consistency(_flipping_annotator("flip"), batch, n_annotators=2)
    agreement = {row["text"]: row["agreed"] for row in rows}
    assert agreement == {"p": True, "q": True, "flip": False}
    flip_row = next(row for row in rows if row["text"] == "flip")
    assert flip_row["label"] is None
    assert flip_row["labels"] == ["X", "Y"]


def test_active_sampler_runs_on_raft_pool(raft_pool):
    """End-to-end: the sampler consumes a pool built by the existing raft block."""
    sampler = ActiveSampler(budget=8, batch_size=4, annotator=_stable_annotator("relevant"))
    result = sampler(raft_pool)
    assert isinstance(result, ActiveCurationResult)
    # 4 items fit within a budget of 8 with a dual annotator (2 calls each).
    assert len(result.promoted) == 4
    assert all(row["agreed"] for row in result.promoted)
    assert all(row["label"] == "relevant" for row in result.promoted)
    # round metadata is attached.
    assert {row["round"] for row in result.promoted} == {0}


def test_active_sampler_budget_exhaustion_discards_remainder(raft_pool):
    """A tight budget evicts the unaffordable remainder; 'discard' drops it."""
    sampler = ActiveSampler(budget=4, batch_size=8, annotator=_stable_annotator("yes"), eviction_policy="discard")
    result = sampler(raft_pool)
    # Budget 4 / dual annotator = 2 affordable items.
    assert len(result.promoted) == 2
    assert len(result.deferred) == 0


def test_active_sampler_budget_exhaustion_defers_remainder(raft_pool):
    """The 'defer' policy returns the remainder marked as evicted."""
    sampler = ActiveSampler(budget=4, batch_size=8, annotator=_stable_annotator("yes"), eviction_policy="defer")
    result = sampler(raft_pool)
    assert len(result.promoted) == 2
    assert len(result.deferred) == len(raft_pool) - 2
    assert all(row["evicted"] for row in result.deferred)
    assert all(not row["agreed"] for row in result.deferred)


def test_active_sampler_runs_multiple_rounds(raft_pool):
    """A small batch size with enough budget spans multiple rounds."""
    sampler = ActiveSampler(budget=8, batch_size=2, annotator=_stable_annotator("yes"))
    result = sampler(raft_pool)
    # 8 budget / dual annotator = 4 items, two per round => rounds 0 and 1.
    assert len(result.promoted) == 4
    assert {row["round"] for row in result.promoted} == {0, 1}


def test_active_sampler_consistency_check_filters_disagreements(raft_pool):
    """Disagreements are not promoted even within a larger run."""
    flip_text = raft_pool[0]["content"]
    pool = Dataset.from_list([{"text": row["content"]} for row in raft_pool])
    sampler = ActiveSampler(budget=8, batch_size=4, annotator=_flipping_annotator(flip_text))
    result = sampler(pool)
    labels = {row["text"] for row in result.promoted}
    assert flip_text not in labels
    # The disagreeing item is dropped (not promoted), so only 3 survive.
    assert len(result.promoted) == 3


def test_active_sampler_rejects_invalid_config():
    """Invalid configuration raises immediately."""
    with pytest.raises(ValueError):
        ActiveSampler(budget=-1)
    with pytest.raises(ValueError):
        ActiveSampler(budget=4, batch_size=0)
    with pytest.raises(ValueError):
        ActiveSampler(budget=4, eviction_policy="keep")


def test_active_sampler_requires_annotator_or_model():
    """Without an annotator or model_name the sampler cannot annotate."""
    sampler = ActiveSampler(budget=4, annotator=None, model_name=None)
    pool = Dataset.from_list([{"text": "a"}])
    with pytest.raises(ValueError):
        sampler(pool)
