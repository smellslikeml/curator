"""Pool-based, label-efficient active-curation block.

This module is a `SimpleStrat`-style block that sits alongside the other curator
blocks (`raft`, `simplestrat`). It curates a labeled dataset from an unlabeled
pool using an active-sampling loop inspired by the RTLCurator data-curation
recipe:

  1. Score the remaining pool with the annotator; uncertainty is read off as
     ``(1 - confidence)``.
  2. Select the next batch by *uncertainty x diversity* (a greedy batch-mode
     active-learning rule).
  3. Run a second, independent annotation pass and promote a label only when the
     two passes agree -- the consistency / dual-annotator check.
  4. When the annotation budget is exhausted, evict the remaining pool in a batch
     instead of labeling it.

The block is intentionally language-agnostic: the annotator is a `curator.LLM`
that returns a free-form label plus a calibrated confidence, and the selection,
consistency and eviction policies are pure functions over plain records so they
can be exercised without an API.

Reference / inspiration:
- RTLCurator: a label-efficient, uncertainty x diversity data-curation recipe.
"""

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from datasets import Dataset
from pydantic import BaseModel, Field

from bespokelabs import curator

_DEFAULT_EMBED_DIM = 256


class Annotation(BaseModel):
    """Annotator response: a label plus a calibrated confidence.

    Attributes:
        label: The label assigned to the item.
        confidence: Confidence in the label, in ``[0, 1]``. Higher is better;
            uncertainty is ``1 - confidence``.
        reasoning: Short, free-form reasoning kept for traceability.
    """

    label: str = Field(description="The label assigned to the item.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the label, between 0 and 1.")
    reasoning: str = Field(default="", description="Short reasoning behind the label.")


class Annotator(Protocol):
    """Anything that can label a batch of pool items.

    `PoolAnnotator` satisfies this for real runs; tests inject a deterministic
    fake to exercise the policies without an API.
    """

    def annotate(self, items: Sequence[dict], working_dir: Optional[str] = None) -> List[dict]:
        """Annotate ``items`` and return one record per input item.

        Each returned record must carry ``id``, ``label`` and ``confidence``
        (and typically the originating ``text``). ``working_dir`` is an optional
        curator cache directory forwarded by the loop.
        """
        ...


def default_embedder(text: str, dim: int = _DEFAULT_EMBED_DIM) -> List[float]:
    """Deterministic, dependency-free text embedding.

    A hashed bag-of-characters vector used to measure diversity. It is stable
    across ``PYTHONHASHSEED`` (uses ``ord`` rather than the salted ``hash``) so
    batch selection is reproducible.

    Args:
        text: The text to embed.
        dim: Output dimensionality.

    Returns:
        A fixed-length vector of non-negative term counts.
    """
    vec = [0.0] * dim
    for ch in str(text):
        vec[ord(ch) % dim] += 1.0
    return vec


def _normalize(vec: Sequence[float]) -> List[float]:
    """L2-normalize a vector (returns it unchanged if it has zero norm)."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return list(vec)
    return [v / norm for v in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two (assumed normalized) vectors."""
    return sum(x * y for x, y in zip(a, b, strict=False))


def _combine(uncertainty: float, diversity: float, diversity_weight: float) -> float:
    """Blend uncertainty and diversity into a single batch score.

    Uses a weighted geometric mean, which is rank-equivalent to the
    ``uncertainty x diversity`` product at ``diversity_weight == 0.5``: ``0.0``
    selects on uncertainty alone, ``1.0`` on diversity alone.
    """
    if diversity_weight <= 0.0:
        return uncertainty
    if diversity_weight >= 1.0:
        return diversity
    return (uncertainty ** (1.0 - diversity_weight)) * (diversity**diversity_weight)


def labels_agree(a: dict, b: dict) -> bool:
    """Return True when two annotations agree on a (normalized) label.

    Labels are compared after lowercasing and collapsing whitespace so cosmetic
    differences (``" Cat "`` vs ``"cat"``) do not fail the consistency check.
    """
    return _normalize_label(a.get("label", "")) == _normalize_label(b.get("label", ""))


def _normalize_label(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


def _ensure_row(row: dict) -> dict:
    """Coerce a pool row into the canonical ``{id, text}`` form.

    Accepts rows keyed by ``text``/``content``/``input`` and synthesizes a stable
    ``id`` from the text when one is not supplied.
    """
    record = dict(row)
    text = record.get("text")
    if text is None:
        text = record.get("content", record.get("input", ""))
    record["text"] = str(text)
    if not record.get("id"):
        record["id"] = hashlib.sha1(record["text"].encode("utf-8")).hexdigest()[:16]
    else:
        record["id"] = str(record["id"])
    return record


def _promote(record: dict) -> dict:
    """Project an annotated record onto the promoted (output) schema."""
    return {
        "id": record["id"],
        "text": record.get("text", ""),
        "label": record["label"],
        "confidence": float(record["confidence"]),
        "reasoning": record.get("reasoning", ""),
    }


def select_batch(
    scored: Sequence[dict],
    k: int,
    embedder: Optional[Callable[[str], Sequence[float]]] = None,
    diversity_weight: float = 0.5,
) -> List[dict]:
    """Greedily select ``k`` items maximizing uncertainty x diversity.

    Uncertainty is ``1 - confidence``, min-max normalized across ``scored`` so
    the most uncertain item scores 1.0. Diversity is marginal novelty: ``1 -``
    the maximum cosine similarity to the already-selected set (``1.0`` for the
    first pick). Items are added one at a time, each time picking the highest
    ``_combine`` score, which yields a diverse batch rather than ``k`` near-
    duplicates of the single most uncertain item.

    Args:
        scored: Annotated records (each needs ``id``, ``text``, ``confidence``).
        k: Maximum batch size.
        embedder: Text -> vector callable. Defaults to `default_embedder`.
        diversity_weight: Blend passed to `_combine`.

    Returns:
        Up to ``k`` records in selection order.
    """
    if k <= 0 or not scored:
        return []
    embed = embedder or default_embedder
    records = list(scored)
    uncertainties = [1.0 - max(0.0, min(1.0, float(r.get("confidence", 0.0)))) for r in records]
    u_max = max(uncertainties)
    norm_u = [u / u_max if u_max > 0 else 0.0 for u in uncertainties]
    embeddings = {r["id"]: _normalize(embed(r.get("text", ""))) for r in records}
    by_id = {r["id"]: r for r in records}

    selected: List[str] = []
    remaining = [r["id"] for r in records]
    while remaining and len(selected) < k:
        best_id = None
        best_score = -1.0
        for rid in remaining:
            diversity = 1.0
            if selected:
                similarity = max(_cosine(embeddings[rid], embeddings[s]) for s in selected)
                diversity = max(0.0, 1.0 - similarity)
            idx = remaining.index(rid)
            score = _combine(norm_u[idx], diversity, diversity_weight)
            if score > best_score:
                best_score = score
                best_id = rid
        selected.append(best_id)
        remaining.remove(best_id)
    return [by_id[rid] for rid in selected]


class PoolAnnotator(curator.LLM):
    """Default `curator.LLM`-backed annotator for the active-curation loop.

    Labels each pool item with a free-form label and a calibrated confidence.
    Implements the `Annotator` protocol via `annotate`, which runs the item batch
    through curator and returns the parsed rows.
    """

    response_format = Annotation

    def __init__(
        self,
        *,
        model_name: str,
        task_prompt: str,
        backend: Optional[str] = None,
        backend_params: Optional[dict] = None,
        generation_params: Optional[dict] = None,
    ) -> None:
        """Initialize the annotator.

        Args:
            model_name: Model passed to `curator.LLM`.
            task_prompt: Instructions describing the labeling task and label space.
            backend: Optional curator backend.
            backend_params: Optional curator backend params.
            generation_params: Optional curator generation params.
        """
        super().__init__(
            model_name=model_name,
            backend=backend,
            backend_params=backend_params,
            generation_params=generation_params,
        )
        self.task_prompt = task_prompt

    def prompt(self, input: dict) -> List[dict]:
        """Build the annotator prompt for a single pool item."""
        text = input.get("text", "")
        return [
            {"role": "system", "content": self.task_prompt},
            {"role": "user", "content": f"Item to label:\n{text}\n\nReturn a label and your calibrated confidence in it."},
        ]

    def parse(self, input: dict, response: Annotation) -> dict:
        """Merge the structured response with the originating pool item."""
        return {
            "id": input.get("id"),
            "text": input.get("text", ""),
            "label": response.label,
            "confidence": float(response.confidence),
            "reasoning": response.reasoning,
        }

    def annotate(self, items: Sequence[dict], working_dir: Optional[str] = None) -> List[dict]:
        """Annotate a batch of items through curator and return parsed rows."""
        if not items:
            return []
        dataset = Dataset.from_list([_ensure_row(item) for item in items])
        return [dict(row) for row in self(dataset, working_dir=working_dir).dataset]


@dataclass
class ActiveCuration:
    """Uncertainty x diversity active-curation block.

    A `SimpleStrat`-style block that turns an unlabeled pool into a labeled
    dataset using a label-efficient active-sampling loop (see module docstring).

    Attributes:
        model: Model name forwarded to the default `PoolAnnotator`.
        task_prompt: Labeling instructions / label-space description.
        batch_size: Maximum items selected per round.
        annotator_budget: Total annotation calls the loop may spend.
        diversity_weight: Uncertainty/diversity blend (0.5 == uncertainty x diversity).
        consistency: When True (default), require dual-annotator agreement.
        backend / backend_params / generation_params: Forwarded to `PoolAnnotator`.
        embedder: Optional text-embedding callable (default: `default_embedder`).
        annotator: Optional pre-built annotator satisfying the `Annotator` protocol
            (e.g. a fake in tests). Takes precedence over `annotator_cls`.
        annotator_cls: Optional annotator class override (mirrors `Raft.answer_generator_cls`).
    """

    model: str
    task_prompt: str = "Assign the single most accurate label to the following item."
    batch_size: int = 8
    annotator_budget: int = 64
    diversity_weight: float = 0.5
    consistency: bool = True
    backend: Optional[str] = None
    backend_params: Optional[dict] = None
    generation_params: Optional[dict] = None
    embedder: Optional[Callable[[str], Sequence[float]]] = None
    annotator: Optional[Annotator] = None
    annotator_cls: Optional[type] = None

    def __post_init__(self) -> None:
        """Initialize per-run observability state."""
        self.last_stats: Dict[str, Any] = {}

    def _build_annotator(self) -> Annotator:
        if self.annotator is not None:
            return self.annotator
        annotator_cls = self.annotator_cls or PoolAnnotator
        return annotator_cls(
            model_name=self.model,
            task_prompt=self.task_prompt,
            backend=self.backend,
            backend_params=self.backend_params,
            generation_params=self.generation_params,
        )

    def __call__(self, pool: Sequence[dict], working_dir: Optional[str] = None) -> Dataset:
        """Run the active-curation loop over ``pool`` and return labeled items.

        Args:
            pool: Unlabeled items (each needs ``text`` or ``content``/``input``;
                ``id`` is synthesized when absent).
            working_dir: Accepted for API symmetry with the other blocks; the
                default annotator forwards it to curator via its own ``__call__``.

        Returns:
            A `datasets.Dataset` of promoted, consistency-checked labels.
        """
        embedder = self.embedder or default_embedder
        annotator = self._build_annotator()
        unlabeled = [_ensure_row(row) for row in pool]
        labeled: List[dict] = []
        evicted: List[dict] = []
        spent = 0
        rounds = 0

        while unlabeled and spent < self.annotator_budget:
            rounds += 1
            # 1. Score the remaining pool (one annotation pass); never overspend.
            affordable = self.annotator_budget - spent
            to_score, unlabeled = unlabeled[:affordable], unlabeled[affordable:]
            scored = annotator.annotate(to_score, working_dir=working_dir)
            spent += len(scored)

            # 2. Select the next batch by uncertainty x diversity.
            k = min(self.batch_size, len(scored))
            batch = select_batch(scored, k=k, embedder=embedder, diversity_weight=self.diversity_weight)

            # 3. Consistency / dual-annotator check on the batch.
            second_vote: Dict[str, dict] = {}
            if self.consistency and batch:
                affordable2 = self.annotator_budget - spent
                if affordable2 > 0:
                    second_vote = {r["id"]: r for r in annotator.annotate(batch[:affordable2], working_dir=working_dir)}
                    spent += len(second_vote)

            # 4. Promote only agreed labels; defer the rest back to the pool.
            promoted: set = set()
            for record in batch:
                if not self.consistency:
                    labeled.append(_promote(record))
                    promoted.add(record["id"])
                elif record["id"] in second_vote and labels_agree(record, second_vote[record["id"]]):
                    labeled.append(_promote(record))
                    promoted.add(record["id"])
            unlabeled = [r for r in scored if r["id"] not in promoted] + unlabeled

            # 5. Batch-eviction when the budget is exhausted.
            if spent >= self.annotator_budget:
                evicted.extend(unlabeled)
                unlabeled = []

        self.last_stats = {
            "labeled": len(labeled),
            "evicted": len(evicted),
            "spent": spent,
            "rounds": rounds,
        }
        return Dataset.from_list(labeled)
