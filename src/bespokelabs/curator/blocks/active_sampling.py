"""Pool-based active sampling curation strategy.

This block implements pool-based active sampling for label-efficient
curation. Each candidate in an unlabeled pool is scored by
``uncertainty * diversity`` and a batch is selected greedily, re-scoring
diversity after every pick so the batch does not collapse onto a single
cluster. ``uncertainty`` is a caller-supplied model-confidence signal
(higher means more worth labeling) and ``diversity`` is the distance from
the candidate's embedding to the nearest already-selected candidate.

The selection core is domain-agnostic; the caller supplies per-candidate
``uncertainty`` and ``embedding`` values (for example from a confidence
estimator and an embedding model). This keeps the strategy free of heavy
modeling dependencies while composing with the rest of the ``blocks``
package, which shares the ``datasets.Dataset`` -> ``datasets.Dataset``
contract used by :mod:`bespokelabs.curator.blocks.raft` and
:mod:`bespokelabs.curator.blocks.simplestrat`.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import datasets
import numpy as np

_DistanceMetric = Literal["cosine", "euclidean"]


@dataclass
class ActiveSampling:
    """Pool-based active sampling curation block.

    Selects a batch of candidates from an unlabeled pool by greedily
    maximizing ``uncertainty * diversity``. Diversity is the distance from
    a candidate's embedding to the nearest already-selected candidate, so
    it is recomputed after every pick and the batch stays spread out in
    embedding space. An optional annotator-budget caps the total number of
    candidates selected across the whole campaign.

    Attributes:
        batch_size: Number of candidates to select per call.
        budget: Optional total annotator budget. Selection stops once the
            number of already-selected candidates reaches this value.
        uncertainty_field: Name of the column holding the per-candidate
            model-confidence signal (higher means more uncertain).
        embedding_field: Name of the column holding the per-candidate
            embedding vector used for the diversity term.
        distance_metric: Embedding distance used for diversity
            (``"cosine"`` or ``"euclidean"``).
    """

    batch_size: int = 10
    budget: Optional[int] = None
    uncertainty_field: str = "uncertainty"
    embedding_field: str = "embedding"
    distance_metric: _DistanceMetric = "cosine"

    def __call__(
        self,
        pool: datasets.Dataset,
        already_selected: Optional[Sequence[int]] = None,
    ) -> datasets.Dataset:
        """Select the next batch from ``pool``.

        Args:
            pool: Unlabeled candidate pool. Each row must contain the
                configured ``uncertainty_field`` and ``embedding_field``
                columns. Arbitrary additional columns are preserved on the
                returned rows.
            already_selected: Indices of candidates selected in previous
                rounds. They seed the diversity computation and count
                against ``budget`` but are not re-emitted.

        Returns:
            A ``datasets.Dataset`` containing the newly selected rows in
            descending selection-score order, with an extra
            ``active_sampling_score`` column. Empty if the batch size is
            zero, the pool is empty, or the budget is already exhausted.
        """
        indices, scores = self._select_indices(pool, already_selected)
        if not indices:
            return pool.select([])
        selected = pool.select(indices)
        return selected.add_column("active_sampling_score", scores)

    def _select_indices(
        self,
        pool: datasets.Dataset,
        already_selected: Optional[Sequence[int]] = None,
    ) -> tuple[list[int], list[float]]:
        """Greedily select candidate indices and their scores.

        Args:
            pool: Unlabeled candidate pool.
            already_selected: Indices selected in previous rounds.

        Returns:
            A ``(indices, scores)`` pair in selection order. Only newly
            selected indices are returned; seeds from ``already_selected``
            are consumed but not re-emitted.
        """
        n = len(pool)
        if n == 0 or self.batch_size <= 0:
            return [], []

        embeddings = np.asarray([np.asarray(row, dtype=float) for row in pool[self.embedding_field]])
        uncertainties = np.asarray(pool[self.uncertainty_field], dtype=float)

        selected: list[int] = []
        selected_mask = np.zeros(n, dtype=bool)
        for idx in already_selected or []:
            if 0 <= idx < n and not selected_mask[idx]:
                selected.append(int(idx))
                selected_mask[idx] = True
        n_seeds = len(selected)

        if self.budget is not None and len(selected) >= self.budget:
            return [], []

        # ``min_diversity[i]`` is the distance from candidate ``i`` to the
        # nearest selected candidate. Before anything is selected every
        # candidate is treated as maximally diverse (1.0), so the first pick
        # is driven purely by uncertainty.
        min_diversity = np.ones(n, dtype=float)

        remaining_budget = self.budget - len(selected) if self.budget is not None else self.batch_size
        n_to_select = min(self.batch_size, remaining_budget)

        scores: list[float] = []
        for _ in range(n_to_select):
            if selected_mask.all():
                break
            # Fold in the distance to the most recently added candidate so
            # ``min_diversity`` reflects the full selected set.
            if selected:
                min_diversity = np.minimum(min_diversity, self._distances_to(embeddings, selected[-1]))
            score = np.where(selected_mask, -np.inf, uncertainties * min_diversity)
            best_idx = int(np.argmax(score))
            selected.append(best_idx)
            selected_mask[best_idx] = True
            scores.append(float(score[best_idx]))

        return selected[n_seeds:], scores

    def _distances_to(self, embeddings: np.ndarray, idx: int) -> np.ndarray:
        """Distance from every candidate embedding to ``embeddings[idx]``.

        Args:
            embeddings: ``(n, d)`` array of candidate embeddings.
            idx: Index of the reference (already-selected) candidate.

        Returns:
            An ``(n,)`` array of distances from each candidate to the
            reference candidate.
        """
        reference = embeddings[idx]
        if self.distance_metric == "cosine":
            candidate_norm = np.linalg.norm(embeddings, axis=1)
            reference_norm = np.linalg.norm(reference)
            denom = candidate_norm * reference_norm
            denom = np.where(denom == 0.0, 1.0, denom)
            similarity = (embeddings @ reference) / denom
            return 1.0 - similarity
        return np.linalg.norm(embeddings - reference, axis=1)
