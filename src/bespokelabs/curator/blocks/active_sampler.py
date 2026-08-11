"""Pool-based active curation sampler.

This module is a self-contained "block" that follows the same recipe pattern as
`bespokelabs.curator.blocks.simplestrat` and `bespokelabs.curator.blocks.raft`.
It implements an uncertainty x diversity active sampler for data curation, in the
spirit of RTLCurator.

The design brief calls out three pieces, all implemented here:

  1. A pool-based active learner that selects each batch by scoring items with
     ``uncertainty * diversity`` (an entropy-style uncertainty times a greedy
     max-min distance in feature space, a.k.a. a core-set diversity term).
  2. A consistency ("dual annotator agreement") check: an item is only promoted
     to the curated set when ``n_annotators`` independent annotators agree on its
     label (``n_annotators=2`` recovers the dual-annotator case).
  3. A batch-eviction policy applied to the remaining pool once the annotator
     budget is exhausted (``discard`` drops the remainder, ``defer`` returns it
     marked as unannotated).

The sampler is language- and task-agnostic: uncertainty, feature representation,
and the annotator are all injectable, with sensible dependency-free defaults so
the block runs out of the box on any ``datasets.Dataset`` pool.

Reference:
- https://arxiv.org/abs/2607.29283
"""

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from datasets import Dataset
from pydantic import BaseModel, Field

from bespokelabs import curator

__all__ = [
    "ActiveSampler",
    "ActiveCurationResult",
    "AnnotatorFn",
    "select_batch",
    "annotate_with_consistency",
]

# Dimension of the default hashing-trick feature vector.
_DEFAULT_FEATURE_DIM = 64
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split ``text`` into alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def _safe_entropy(probabilities: Sequence[float]) -> float:
    """Return the normalized Shannon entropy of a distribution, in ``[0, 1]``.

    The input need not be pre-normalized; it is rescaled to sum to one. A peaked
    distribution yields ~0 and a uniform distribution yields ~1. ``log2`` base is
    used and the result is divided by ``log2(n_active)`` so it is bounded.
    """
    probs = [max(float(p), 0.0) for p in probabilities]
    total = sum(probs)
    if total <= 0:
        return 0.0
    norm = [p / total for p in probs]
    active = [p for p in norm if p > 0]
    if len(active) <= 1:
        return 0.0
    entropy = -sum(p * math.log2(p) for p in active)
    return entropy / math.log2(len(active))


def _default_uncertainty(item: dict) -> float:
    """Best-effort uncertainty for a pool item.

    Uses an explicit ``uncertainty`` field if present, otherwise the normalized
    entropy of a ``probabilities`` field, otherwise ``1 - confidence``, falling
    back to a uniform ``1.0`` so selection is driven purely by diversity.
    """
    if "uncertainty" in item:
        return float(item["uncertainty"])
    if "probabilities" in item:
        return _safe_entropy(item["probabilities"])
    if "confidence" in item:
        return max(0.0, 1.0 - float(item["confidence"]))
    return 1.0


def _hashing_vectorizer(item: dict, dim: int = _DEFAULT_FEATURE_DIM) -> list[float]:
    """Deterministic hashing-trick feature vector for an item's text.

    The text is taken from ``text`` (falling back to ``question``), tokenized, and
    projected into ``dim`` dimensions with signed counts, then L2-normalized so
    that Euclidean distances are stable regardless of document length. This keeps
    the block dependency-free while remaining language-agnostic.
    """
    text = item.get("text") or item.get("question") or ""
    vec = [0.0] * dim
    for token in _tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] & 1) else -1.0
        vec[index] += sign
    norm = math.sqrt(sum(value * value for value in vec))
    if norm > 0:
        vec = [value / norm for value in vec]
    return vec


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two equal-length vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def select_batch(uncertainties: Sequence[float], features: Sequence[Sequence[float]], k: int) -> list[int]:
    """Greedily select ``k`` item indices by uncertainty x diversity.

    The first pick maximizes uncertainty. Each subsequent pick maximizes
    ``uncertainty_i * min_distance(i, chosen)`` -- an uncertainty-weighted
    core-set (max-min) diversity score. When the remaining feature distances are
    all zero (a degenerate feature space), selection falls back to pure
    uncertainty ordering. Ties are broken by lower index for determinism.

    Args:
        uncertainties: Per-item uncertainty scores.
        features: Per-item feature vectors, aligned with ``uncertainties``.
        k: Number of items to select.

    Returns:
        The list of selected indices (length ``min(k, len(uncertainties))``).
    """
    n = len(uncertainties)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    chosen: list[int] = []
    chosen_set: set[int] = set()

    # Seed with the single most uncertain item.
    seed = max(range(n), key=lambda i: uncertainties[i])
    chosen.append(seed)
    chosen_set.add(seed)
    min_dist = [_euclidean(features[i], features[seed]) for i in range(n)]

    while len(chosen) < k:
        remaining = [i for i in range(n) if i not in chosen_set]
        if not remaining:
            break
        diverse = any(min_dist[i] > 0 for i in remaining)
        best_index = remaining[0]
        best_score = -1.0
        for i in remaining:
            score = uncertainties[i] * min_dist[i] if diverse else uncertainties[i]
            if score > best_score or (score == best_score and i < best_index):
                best_score = score
                best_index = i
        chosen.append(best_index)
        chosen_set.add(best_index)
        for i in range(n):
            if i not in chosen_set:
                distance = _euclidean(features[i], features[best_index])
                if distance < min_dist[i]:
                    min_dist[i] = distance
    return chosen


def annotate_with_consistency(annotator: Callable[[Dataset], Sequence[str]], batch: Dataset, n_annotators: int = 2) -> list[dict]:
    """Annotate ``batch`` ``n_annotators`` times and keep only unanimous labels.

    Each row of the returned list is the original item augmented with ``label``
    (the agreed label, or ``None`` if annotators disagreed), ``labels`` (the full
    list of per-annotator labels), and ``agreed`` (whether all annotators agreed).
    Promotion decisions in the active-learning loop are based on ``agreed``.

    Args:
        annotator: Callable mapping a ``Dataset`` batch to one label per item.
        batch: The batch of pool items to annotate.
        n_annotators: Number of independent annotation passes (2 = dual).

    Returns:
        The annotated rows with agreement metadata attached.
    """
    if n_annotators < 1:
        raise ValueError("n_annotators must be at least 1")
    label_sets = [list(annotator(batch)) for _ in range(n_annotators)]
    rows: list[dict] = []
    items = list(batch)
    for index, item in enumerate(items):
        labels = [label_set[index] for label_set in label_sets]
        agreed = len(set(labels)) == 1
        rows.append({**item, "label": labels[0] if agreed else None, "labels": labels, "agreed": agreed})
    return rows


class _Label(BaseModel):
    """A single predicted label for a pool item."""

    label: str = Field(description="The predicted label for the item.")


class _ActiveAnnotator(curator.LLM):
    """Default LLM-backed annotator that labels a pool item.

    Uses ``candidate_labels`` for classification when provided, otherwise assigns
    a free-form concise label. Annotators are intentionally simple; users with a
    domain-specific labeling task should pass their own ``annotator`` callable to
    :class:`ActiveSampler`.
    """

    response_format = _Label

    def prompt(self, input: dict) -> list[dict]:
        """Build the annotation prompt for a pool item."""
        text = input.get("text") or input.get("question") or ""
        candidates = input.get("candidate_labels")
        if candidates:
            instruction = f"Classify the item into exactly one of these labels: {candidates}.\nItem: {text}"
        else:
            instruction = f"Assign a single concise label to the following item.\nItem: {text}"
        return [
            {"role": "system", "content": "You are a careful, consistent data annotator."},
            {"role": "user", "content": instruction},
        ]

    def parse(self, input: dict, response: _Label) -> dict:
        """Attach the predicted label to the original item."""
        return {**input, "label": response.label}


# A callable that maps a batch (Dataset) to one label string per item.
AnnotatorFn = Callable[[Dataset], Sequence[str]]


@dataclass
class ActiveCurationResult:
    """Outcome of an active-sampling run.

    Attributes:
        promoted: Items that survived the consistency check and were labeled.
        deferred: Items evicted from the pool once the budget ran out. Non-empty
            only when ``eviction_policy='defer'``; each row is marked
            ``agreed=False`` and ``evicted=True``.
    """

    promoted: Dataset
    deferred: Dataset


@dataclass
class ActiveSampler:
    """Uncertainty x diversity active sampler for curation.

    The sampler iteratively selects a diverse, high-uncertainty batch from the
    pool, annotates it with ``n_annotators`` independent annotators, and promotes
    only the items whose annotators unanimously agree. The annotator budget is
    charged per annotation call (so a dual-annotator item costs 2 units); once the
    budget is exhausted the remaining pool is evicted according to
    ``eviction_policy``.

    All extension points have defaults so the block runs out of the box, but each
    can be overridden: ``annotator`` (e.g. an LLM or a custom model), the
    per-item ``uncertainty_fn``, and the ``feature_fn`` used for diversity.

    Attributes:
        budget: Total annotation calls the sampler may spend.
        batch_size: Maximum items annotated per round.
        annotator: Callable mapping a batch Dataset to one label per item. If
            ``None``, a default LLM annotator is built from ``model_name``.
        uncertainty_fn: Per-item uncertainty scorer. Defaults to entropy of
            ``probabilities`` / ``1 - confidence`` / ``1.0``.
        feature_fn: Per-item feature vector producer. Defaults to a deterministic
            hashing-trick vectorizer over the item's text.
        n_annotators: Independent annotation passes per item (2 = dual annotator).
        eviction_policy: ``'discard'`` (drop the remainder) or ``'defer'`` (return
            the remainder marked as evicted) when the budget is exhausted.
        feature_dim: Dimension of the default feature vector.
        model_name: Model for the default LLM annotator (ignored if ``annotator``
            is provided).
        backend: Optional curator backend for the default annotator.
        backend_params: Optional backend params for the default annotator.
        generation_params: Optional generation params for the default annotator.
    """

    budget: int
    batch_size: int = 8
    annotator: AnnotatorFn | None = None
    uncertainty_fn: Callable[[dict], float] | None = None
    feature_fn: Callable[[dict], Sequence[float]] | None = None
    n_annotators: int = 2
    eviction_policy: str = "discard"
    feature_dim: int = _DEFAULT_FEATURE_DIM
    model_name: str | None = None
    backend: str | None = None
    backend_params: dict | None = None
    generation_params: dict | None = None

    def __post_init__(self) -> None:
        """Validate configuration and resolve default callables."""
        if self.budget < 0:
            raise ValueError("budget must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.n_annotators < 1:
            raise ValueError("n_annotators must be at least 1")
        if self.eviction_policy not in ("discard", "defer"):
            raise ValueError("eviction_policy must be 'discard' or 'defer'")
        self._uncertainty: Callable[[dict], float] = self.uncertainty_fn or _default_uncertainty
        dim = self.feature_dim
        self._features: Callable[[dict], Sequence[float]] = self.feature_fn or (lambda item, _dim=dim: _hashing_vectorizer(item, _dim))

    def _resolve_annotator(self) -> AnnotatorFn:
        """Return the configured annotator, building the default LLM one if needed."""
        if self.annotator is not None:
            return self.annotator
        if self.model_name is None:
            raise ValueError("Provide either `annotator` or `model_name` for the default LLM annotator.")
        llm = _ActiveAnnotator(model_name=self.model_name, backend=self.backend, backend_params=self.backend_params, generation_params=self.generation_params)

        def _default(batch: Dataset) -> Sequence[str]:
            dataset = llm(batch).dataset
            if "label" in dataset.column_names:
                return dataset["label"]
            return [row.get("label") for row in dataset]

        return _default

    def sample(self, pool: Dataset) -> tuple[list[dict], list[dict]]:
        """Run the active-sampling loop over ``pool``.

        Args:
            pool: The unlabeled pool as a ``datasets.Dataset``.

        Returns:
            A ``(promoted, deferred)`` pair of row lists. ``promoted`` holds the
            agreed, labeled items; ``deferred`` holds the evicted remainder
            (non-empty only under the ``'defer'`` policy).
        """
        n = len(pool)
        if n == 0:
            return [], []
        items = [pool[i] for i in range(n)]
        uncertainties = [self._uncertainty(item) for item in items]
        features = [list(self._features(item)) for item in items]
        annotator = self._resolve_annotator()

        available = list(range(n))
        available_set = set(available)
        remaining_budget = self.budget
        promoted: list[dict] = []
        round_no = 0

        while available and remaining_budget >= self.n_annotators:
            affordable = remaining_budget // self.n_annotators
            k = min(self.batch_size, affordable, len(available))
            if k <= 0:
                break
            sub_uncertainties = [uncertainties[i] for i in available]
            sub_features = [features[i] for i in available]
            local_selection = select_batch(sub_uncertainties, sub_features, k)
            chosen = [available[j] for j in local_selection]
            batch = Dataset.from_list([items[i] for i in chosen])
            annotated = annotate_with_consistency(annotator, batch, n_annotators=self.n_annotators)
            for row in annotated:
                row["round"] = round_no
            promoted.extend(row for row in annotated if row["agreed"])
            remaining_budget -= k * self.n_annotators
            for i in chosen:
                available_set.discard(i)
            available = [i for i in available if i in available_set]
            round_no += 1

        deferred = self._evict(items, available)
        return promoted, deferred

    def _evict(self, items: list[dict], available: list[int]) -> list[dict]:
        """Apply the eviction policy to the unannotated remainder."""
        if not available or self.eviction_policy == "discard":
            return []
        deferred: list[dict] = []
        for i in available:
            row = {**items[i], "label": None, "labels": [], "agreed": False, "round": -1, "evicted": True}
            deferred.append(row)
        return deferred

    def __call__(self, pool: Dataset) -> ActiveCurationResult:
        """Run active sampling and return promoted and deferred datasets."""
        promoted, deferred = self.sample(pool)
        return ActiveCurationResult(promoted=_to_dataset(promoted), deferred=_to_dataset(deferred))


def _to_dataset(rows: list[dict]) -> Dataset:
    """Build a Dataset from rows, tolerating an empty list without a schema."""
    if not rows:
        return Dataset.from_dict({})
    return Dataset.from_list(rows)
