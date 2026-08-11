"""Unit tests for the semantic deduplication curation block."""

import numpy as np
import pytest
from datasets import Dataset

from bespokelabs.curator.blocks.semantic_dedup import HashingEmbedder, SemanticDedup


def _dataset(rows):
    """Build a ``Dataset`` from a list of ``{"text": ...}`` dicts."""
    return Dataset.from_list([{"text": r} for r in rows])


def test_dedup_removes_planted_duplicates_and_keeps_distinct():
    """Exact duplicates are removed while distinct items are all kept."""
    # Distinct items use fully disjoint vocabulary so their pairwise cosine
    # similarity is 0 (well below the conservative 0.95 threshold); each is
    # planted twice as an exact duplicate (cosine similarity 1.0).
    distinct = [
        "red apple sweet fruit harvest",
        "fast car engine highway road",
        "deep ocean blue whale diving",
        "tall mountain snow peak climbing",
    ]
    rows = [t for text in distinct for t in (text, text)]
    ds = _dataset(rows)

    # target_retention=0.0 disables the safety floor so dedup is unconstrained.
    out = SemanticDedup(text_field="text", target_retention=0.0)(ds)

    assert len(out) == len(distinct)
    assert sorted(out["text"]) == sorted(distinct)


def test_dedup_with_custom_embedder_removes_near_duplicates():
    """A pluggable embedder is used and its embedding space drives dedup."""
    # Four items: two distinct embedding directions, each duplicated. The
    # custom embedder pins embeddings exactly, so the result is deterministic
    # regardless of clustering.
    texts = ["a", "a", "b", "b"]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    class _FixedEmbedder:
        def __call__(self, batch):
            return embeddings

    out = SemanticDedup(text_field="text", embedder=_FixedEmbedder(), target_retention=0.0)(_dataset(texts))

    assert len(out) == 2
    assert sorted(out["text"]) == ["a", "b"]


def test_threshold_controls_aggressiveness():
    """A looser threshold removes a borderline pair that a tighter one keeps."""
    # Embeddings give cosine similarity 1/sqrt(2) ~= 0.707; n_clusters=1 forces
    # both items into one cluster so the threshold gate is actually exercised.
    embeddings = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    class _FixedEmbedder:
        def __call__(self, batch):
            return embeddings

    ds = _dataset(["alpha", "beta"])
    kept = SemanticDedup(text_field="text", embedder=_FixedEmbedder(), threshold=0.9, n_clusters=1, target_retention=0.0)(ds)
    removed = SemanticDedup(text_field="text", embedder=_FixedEmbedder(), threshold=0.5, n_clusters=1, target_retention=0.0)(ds)

    assert len(kept) == 2  # cosine ~0.707 < 0.9 -> both kept
    assert len(removed) == 1  # cosine ~0.707 > 0.5 -> duplicate removed


def test_target_retention_floor_is_respected():
    """The floor recovers examples when the threshold would over-remove."""
    # Five identical items: the threshold rule keeps only one, but the floor
    # demands at least ceil(0.6 * 5) == 3 retained.
    ds = _dataset(["same text"] * 5)
    out = SemanticDedup(text_field="text", threshold=0.95, target_retention=0.6)(ds)
    assert len(out) == 3


def test_preserves_columns_and_order():
    """Output schema matches the input and kept rows keep their relative order."""
    rows = [
        {"text": "alpha beta", "id": 0},
        {"text": "alpha beta", "id": 1},  # duplicate of id 0
        {"text": "gamma delta", "id": 2},
        {"text": "epsilon zeta", "id": 3},
    ]
    ds = Dataset.from_list(rows)
    out = SemanticDedup(text_field="text", target_retention=0.0)(ds)

    assert out.column_names == ds.column_names
    # The duplicate (id 1) is dropped; the rest survive in original order.
    assert out["id"] == [0, 2, 3]


def test_empty_and_single_row_pass_through():
    """Empty and single-row datasets are returned unchanged."""
    empty = Dataset.from_list([])
    assert len(SemanticDedup(text_field="text")(empty)) == 0

    single = _dataset(["only one example here"])
    out = SemanticDedup(text_field="text")(single)
    assert len(out) == 1
    assert out["text"] == single["text"]


def test_missing_text_field_raises():
    """A non-existent text_field raises KeyError."""
    ds = Dataset.from_list([{"content": "hello"}, {"content": "world"}])
    with pytest.raises(KeyError):
        SemanticDedup(text_field="text")(ds)


def test_invalid_threshold_raises():
    """Out-of-range thresholds are rejected."""
    ds = _dataset(["a", "b"])
    with pytest.raises(ValueError):
        SemanticDedup(threshold=1.5)(ds)
    with pytest.raises(ValueError):
        SemanticDedup(threshold=0.0)(ds)


def test_integration_with_raft_chunk_text():
    """Dedup composes with curator's existing ``blocks.raft.chunk_text`` helper.

    Exercises the new block against a real ``datasets.Dataset`` whose rows are
    produced by a pre-existing curator block helper (not just hand-rolled data),
    with an exact duplicate planted.
    """
    from bespokelabs.curator.blocks.raft import chunk_text

    # chunk_text is a pure, offline curator helper that returns a Dataset with a
    # 'content' column; use it to mint one real row, then plant its duplicate.
    chunked = chunk_text("curator blocks raft chunk text helper output", chunk_size=1000)
    assert len(chunked) == 1
    assert "content" in chunked.column_names
    chunk_value = chunked[0]["content"]

    distinct = [
        "photosynthesis converts sunlight into chemical energy",
        "quantum entanglement links particles across distance",
        "the treaty was signed in seventeen eighty three",
    ]
    rows = [{"content": chunk_value}, {"content": chunk_value}]  # exact duplicate
    rows += [{"content": text} for text in distinct]
    ds = Dataset.from_list(rows)

    out = SemanticDedup(text_field="content", target_retention=0.0)(ds)

    assert len(out) == 1 + len(distinct)
    assert sorted(out["content"]) == sorted([chunk_value, *distinct])


def test_hashing_embedder_deterministic_and_separates_disjoint_vocab():
    """The default embedder is deterministic and separates disjoint vocabulary.

    Identical inputs always embed identically; disjoint-vocabulary inputs land
    far below the conservative dedup threshold (collisions at a wide dim are
    negligible, and even one leaves cosine well under 0.95).
    """
    embedder = HashingEmbedder(dim=4096)
    a = embedder(["red apple", "red apple"])
    assert np.array_equal(a[0], a[1])  # deterministic for identical input

    va = a[0]
    vb = embedder(["blue ocean"])[0]
    va = va / (np.linalg.norm(va) or 1.0)
    vb = vb / (np.linalg.norm(vb) or 1.0)
    cosine = float(va @ vb)
    assert cosine < 0.95
