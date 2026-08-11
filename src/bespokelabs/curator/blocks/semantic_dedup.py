"""Semantic deduplication curation block.

Implements a SemDeDup-style (arXiv:2303.09540) deduplication pass over a
HuggingFace ``datasets.Dataset``: embed every example, cluster the embeddings
with k-means, then within each cluster greedily keep representatives and drop
examples whose cosine similarity to a kept representative exceeds a
configurable threshold.

Curator ships no embedding backend, so a pluggable :class:`Embedder` interface
is provided together with a dependency-free :class:`HashingEmbedder` default.
Pass a real semantic embedder (e.g. sentence-transformers or an API client)
via the ``embedder`` argument for production use.

This module follows the same block pattern as ``blocks.raft.Raft``: a dataclass
whose fields are configuration and whose ``__call__`` performs the work on a
``datasets.Dataset``. Curator has no strategy registry, so there is nothing to
register the block with; instantiate and call it directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from datasets import Dataset
from xxhash import xxh32


class Embedder(Protocol):
    """Embeds a batch of strings into a 2-D float array.

    The returned array must have shape ``(len(texts), d)`` for any ``d >= 1``.
    Rows need not be normalized; :class:`SemanticDedup` L2-normalizes internally
    before computing cosine similarity.
    """

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into an ``(len(texts), d)`` array."""


def _tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and split it into alphanumeric tokens."""
    scrubbed = "".join(ch if ch.isalnum() else " " for ch in text)
    return [tok for tok in scrubbed.lower().split() if tok]


class HashingEmbedder:
    """Deterministic, dependency-free bag-of-words embedder.

    Uses the hashing trick: each lowercased token is mapped (via ``xxhash``,
    already a curator dependency) to one column of a fixed-width vector and its
    count is accumulated there. This is not a true semantic model, but it places
    text with large token overlap close in cosine space while pushing
    disjoint-vocabulary text apart, which is enough to exercise the dedup
    pipeline with no network access and no extra dependencies. Identical strings
    always embed identically (cosine similarity 1.0).

    Args:
        dim: Width of the embedding vector. Larger values reduce hash
            collisions. Defaults to ``1024``.
    """

    def __init__(self, dim: int = 1024) -> None:
        """Store the embedding width."""
        self.dim = dim

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into an ``(len(texts), dim)`` float32 count matrix."""
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in _tokenize(text):
                col = xxh32(tok.encode("utf-8")).intdigest() % self.dim
                mat[row, col] += 1.0
        return mat


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalize ``mat``; all-zero rows are left untouched."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _kmeans(unit: np.ndarray, k: int, max_iter: int, seed: int) -> np.ndarray:
    """Cluster the unit-norm rows of ``unit`` into ``k`` groups with Lloyd's algorithm.

    Cosine similarity is computed as a dot product on the (already unit-norm)
    rows, so nearest-centroid assignment reduces to ``argmax`` of the dot product.

    Args:
        unit: ``(n, d)`` unit-norm matrix.
        k: Number of clusters. The caller clamps this to ``[1, n]``.
        max_iter: Maximum number of assignment/update steps.
        seed: Seed for deterministic centroid initialization.

    Returns:
        An ``(n,)`` ``int64`` array of cluster labels in ``[0, k)``.
    """
    n = unit.shape[0]
    if k >= n:
        # Each point in its own cluster: nothing to dedup within a cluster.
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    centroids = unit[rng.choice(n, size=k, replace=False)].astype(np.float32).copy()
    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(max_iter):
        sims = unit @ centroids.T  # cosine similarity to each centroid
        new_labels = np.argmax(sims, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = unit[labels == c]
            if members.shape[0] > 0:
                centroids[c] = _l2_normalize(members.mean(axis=0, keepdims=True))[0]
    return labels


def _dedup_within_cluster(unit: np.ndarray, threshold: float) -> np.ndarray:
    """Greedily drop near-duplicate rows within a single cluster.

    Iterates rows in order, keeping each one as a representative unless its
    cosine similarity to an already-kept representative exceeds ``threshold``.

    Args:
        unit: ``(m, d)`` unit-norm matrix for the cluster's members.
        threshold: Cosine similarity above which a row is dropped as a duplicate.

    Returns:
        A boolean ``(m,)`` mask that is ``True`` for kept rows.
    """
    m = unit.shape[0]
    keep = np.ones(m, dtype=bool)
    reps: list[int] = []
    for i in range(m):
        if not reps:
            reps.append(i)
            continue
        # Cosine similarity to every representative kept so far (unit-norm rows).
        max_sim = float(np.max(unit[i] @ unit[reps].T))
        if max_sim > threshold:
            keep[i] = False
        else:
            reps.append(i)
    return keep


@dataclass
class SemanticDedup:
    """SemDeDup-style semantic deduplication curation block.

    Operates on a ``datasets.Dataset`` and returns a filtered ``datasets.Dataset``
    containing the kept (de-duplicated) examples with their original column
    schema and relative order preserved.

    Args:
        text_field: Dataset column to embed and dedup on. Defaults to ``"text"``.
        threshold: Pairwise cosine similarity above which an example is treated
            as a near-duplicate of a kept representative within its cluster and
            removed. Higher is more conservative (removes fewer). Defaults to
            ``0.95`` so only clear near-duplicates are dropped.
        target_retention: Minimum fraction of examples to keep, as a safety
            floor. If the threshold rule would drop below this fraction, the
            most informative dropped examples (those least similar to any kept
            representative) are recovered until the floor is met. Defaults to
            ``0.8``. Set to a small value (e.g. ``0.0``) to disable the floor.
        n_clusters: Number of k-means clusters. ``None`` (default) uses
            ``ceil(sqrt(n))``, clamped to ``[1, n]``.
        embedder: Callable mapping a sequence of strings to an ``(n, d)`` array.
            ``None`` (default) uses :class:`HashingEmbedder`.
        dim: Embedding width for the default :class:`HashingEmbedder`. Ignored
            when ``embedder`` is supplied.
        max_iter: Maximum k-means iterations.
        seed: RNG seed for deterministic k-means initialization.
    """

    text_field: str = "text"
    threshold: float = 0.95
    target_retention: float = 0.8
    n_clusters: int | None = None
    embedder: Embedder | None = None
    dim: int = 1024
    max_iter: int = 25
    seed: int = 0

    def __call__(self, dataset: Dataset) -> Dataset:
        """Return ``dataset`` with semantic near-duplicates removed."""
        if not isinstance(dataset, Dataset):
            raise TypeError(f"SemanticDedup expects a datasets.Dataset, got {type(dataset).__name__}")
        n = len(dataset)
        if n <= 1:
            return dataset
        if self.text_field not in dataset.column_names:
            raise KeyError(f"text_field {self.text_field!r} not found in dataset columns {dataset.column_names}")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0.0, 1.0], got {self.threshold}")
        if not 0.0 <= self.target_retention <= 1.0:
            raise ValueError(f"target_retention must be in [0.0, 1.0], got {self.target_retention}")

        texts = [str(value) for value in dataset[self.text_field]]
        embedder = self.embedder or HashingEmbedder(dim=self.dim)
        raw = np.asarray(embedder(texts), dtype=np.float32)
        if raw.ndim != 2 or raw.shape[0] != n:
            raise ValueError(f"embedder must return an array of shape ({n}, d), got {raw.shape}")
        unit = _l2_normalize(raw)

        k = self.n_clusters if self.n_clusters is not None else max(1, int(math.ceil(math.sqrt(n))))
        k = max(1, min(k, n))
        labels = _kmeans(unit, k, self.max_iter, self.seed)

        keep = np.ones(n, dtype=bool)
        for c in range(int(labels.max()) + 1):
            members = np.where(labels == c)[0]
            if members.size == 0:
                continue
            keep[members] = _dedup_within_cluster(unit[members], self.threshold)

        kept = int(keep.sum())
        floor = max(1, int(math.ceil(self.target_retention * n)))
        if kept < floor:
            # Recover the dropped examples that are LEAST similar to any kept
            # representative (i.e. the most informative ones) until the
            # target-retention floor is met.
            dropped = np.where(~keep)[0]
            reps = unit[keep]
            sims = unit[dropped] @ reps.T if reps.shape[0] > 0 else np.zeros((dropped.size, 0), dtype=np.float32)
            worst = sims.max(axis=1) if sims.size else np.zeros(dropped.size, dtype=np.float32)
            for off in np.argsort(worst, kind="stable"):
                if kept >= floor:
                    break
                keep[dropped[off]] = True
                kept += 1

        return dataset.select(np.where(keep)[0].tolist())
